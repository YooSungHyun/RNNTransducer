# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch import Tensor
from networks import AudioTransNet, TextPredNet
import copy
from transformers import Wav2Vec2CTCTokenizer
from pyctcdecode import LanguageModel
from pyctcdecode.language_model import HotwordScorer
from pyctcdecode.constants import DEFAULT_HOTWORD_WEIGHT
from typing import Optional, Iterable, Dict, List, Tuple, Union


class JointNet(nn.Module):
    def __init__(
        self,
        transnet_params: dict,
        prednet_params: dict,
        num_classes: int,
    ):
        super(JointNet, self).__init__()
        self.encoder = AudioTransNet(**transnet_params)
        self.decoder = TextPredNet(**prednet_params)
        self.num_classes = num_classes
        self.act_func = nn.GELU(approximate="tanh")
        self.fc = nn.Linear(transnet_params["output_size"] + prednet_params["output_size"], num_classes)

    def joint(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        """
        Joint `encoder_outputs` and `decoder_outputs`.

        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            decoder_outputs (torch.FloatTensor): A output sequence of decoder (added blank). `FloatTensor` of size
                ``(batch, seq_length+1(blank token), dimension)``

        Returns:
            * outputs (torch.FloatTensor): outputs of joint `encoder_outputs` and `decoder_outputs`..
        """
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            input_length = encoder_outputs.size(1)
            target_length = decoder_outputs.size(1)

            encoder_outputs = encoder_outputs.unsqueeze(2)
            decoder_outputs = decoder_outputs.unsqueeze(1)

            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
            decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)

        # 논문이나 몇몇 구현체는 요소합 하는 사례도 있는데, 직접 실험해본 결과 concat이 장기에포크 진행 시 더 유리했음.
        # 아무래도 볼 수 있는 shape이 늘어나기 때문이지 않을까 사료됨.
        outputs = self.act_func(outputs)
        outputs = self.fc(outputs)

        return outputs

    def forward(
        self, input_audios: Tensor, audio_lengths: Tensor, input_texts: Tensor, text_lengths: Tensor
    ) -> Tensor:
        """
        Forward propagate a `input_audios` and `input_texts` pair for training.

        Args:
            input_audios (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            audio_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            input_texts (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            text_lengths (torch.LongTensor): The length of target tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        # Use for inference only (separate from training_step)
        enc_state = self.encoder(input_audios, audio_lengths)
        dec_state, _ = self.decoder(input_texts, text_lengths)
        outputs = self.joint(enc_state, dec_state)
        return outputs

    @torch.no_grad()
    def recognize_greedy(
        self, inputs: Tensor, inputs_lengths: Tensor, blank_token_id: int, max_iters: int = 3
    ) -> Tensor:
        """
        Greedy Decoder (https://www.youtube.com/watch?v=dgsDIuJLoJU)
        This is more correctly than another git source.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            outputs (torch.LongTensor): tokens for sentence
        """
        outputs = list()
        encoder_outputs = self.encoder(inputs, inputs_lengths)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            pred_tokens = [blank_token_id]
            hidden_state = None
            decoder_input = torch.tensor([[blank_token_id]], dtype=torch.long, device=encoder_output.device)
            decoder_output, hidden_state = self.decoder(decoder_input, prev_hidden_state=hidden_state)

            for t in range(max_length):
                u = 0
                while u < max_iters:

                    step_output = self.joint(encoder_output[t].view(-1), decoder_output.view(-1))
                    step_output = step_output.softmax(dim=0)
                    pred_token = step_output.argmax(dim=0)
                    pred_token = int(pred_token.item())
                    if pred_token != blank_token_id:
                        if pred_tokens[-1] != pred_token:
                            # 최종 output의 경우 중복 제거
                            pred_tokens.append(pred_token)
                        # 텍스트인 경우, u 그래프 기준으로 위로 올라가야 하므로, 음성 들어있을 또 다른 텍스트를 찾아봅니다.
                        decoder_input = torch.tensor([[pred_token]], dtype=torch.long, device=decoder_input.device)
                        decoder_output, hidden_state = self.decoder(decoder_input, prev_hidden_state=hidden_state)
                        u = u + 1
                    else:
                        # blank는 삽입하지 않습니다.
                        break

            outputs.append(torch.LongTensor(pred_tokens[1:]))

        outputs = torch.stack(outputs, dim=1).transpose(0, 1)

        return outputs

    @torch.no_grad()
    def _get_lm_beams(
        self,
        lm: LanguageModel,
        beams: List[Dict],
        hotword_scorer: HotwordScorer,
        cached_lm_scores: Dict[str, Tuple[float, float, object]],
        cached_partial_token_scores: Dict[str, float],
        tokenizer: Wav2Vec2CTCTokenizer,
        is_eos: bool = False,
    ):
        def _merge_tokens(token_1: str, token_2: str) -> str:
            """Fast, whitespace safe merging of tokens."""
            if len(token_2) == 0:
                text = token_1
            elif len(token_1) == 0:
                text = token_2
            else:
                text = token_1 + " " + token_2
            return text

        if lm is None:
            for hyp in beams:
                text = tokenizer.decode(hyp["y_star"])
                if not text:
                    continue
                hyp["lm_score"] = (
                    hyp["asr_score"] + hotword_scorer.score(text) + hotword_scorer.score_partial_token(text)
                )
            return beams

        for hyp in beams:
            lm_score = 0.0
            text = tokenizer.decode(hyp["y_star"])
            if not text:
                continue
            current_text = " ".join(text.split()[:-1])
            next_word = text.split()[-1]
            new_text = _merge_tokens(current_text, next_word)
            if is_eos:
                # 문장이 다 만들어졌다고 판단된 경우. (B_hyps)
                # A_hyps에서는 맨 마지막 토큰을 고려하지 못하므로, 맨 마지막에 한번 추가로 돌려주어야함.
                flag = tokenizer.word_delimiter_token_id in hyp["y_star"]
            else:
                # 문장이 다 만들어지지 않은 경우.
                # 맨 마지막 토큰이 공백이라면, 무언가 단어가 완성됐다는 의미로 판단하고 lm을 진행한다.
                flag = tokenizer.word_delimiter_token_id == hyp["y_star"][-1]

            if flag:
                if new_text not in cached_lm_scores:
                    _, prev_raw_lm_score, start_state = cached_lm_scores[current_text]
                    score, end_state = lm.score(start_state, next_word, is_last_word=is_eos)
                    raw_lm_score = prev_raw_lm_score + score
                    lm_hw_score = raw_lm_score + hotword_scorer.score(new_text)
                    cached_lm_scores[new_text] = (lm_hw_score, raw_lm_score, end_state)
                lm_score, _, _ = cached_lm_scores[new_text]

            if next_word not in cached_partial_token_scores:
                # if prefix available in hotword trie use that, otherwise default to char trie
                if next_word in hotword_scorer:
                    cached_partial_token_scores[next_word] = hotword_scorer.score_partial_token(next_word)
                else:
                    cached_partial_token_scores[next_word] = lm.score_partial_token(next_word)
            lm_score += cached_partial_token_scores[next_word]
            hyp["lm_score"] = hyp["asr_score"] + lm_score
        # cached_lm_scores와 cached_partial_token_scores는 어짜피 주소참조이므로, 여기서 바뀌면 메소드 밖의 변수도 바뀜.
        return beams

    @torch.no_grad()
    def recognize_beams(
        self,
        inputs: Tensor,
        inputs_lengths: int,
        blank_token_id: int,
        beam_widths: int = 100,
        improved: bool = False,
        state_beam: float = 4.6,
        expand_beam: float = 2.3,
        lm: LanguageModel = None,
        tokenizer: Wav2Vec2CTCTokenizer = None,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: float = DEFAULT_HOTWORD_WEIGHT,
    ):
        """recognize_improved_beams Improved Beam Search Decoding (https://arxiv.org/abs/1911.01629)

        only except, prefix pseudocode. (prefix logic is condition for excepting while True loop)
        2 point is important.
            If beam tree node is deeper, each node's prob is smaller.
                1) if final 'B' probs is big enough, you don't need to look u+1. because, 'A' can not be bigger anymore
                2) if each vocabs prob is smaller than current max vocab, you don't need to look u+1 small prob vocab
        all prob is calculate log prob

        LM working like pyctcdecode.BeamSearchDecoderCTC._get_lm_beams

        Args:
            inputs (Tensor): audio input
            inputs_lengths (int): for encoder pack pad
            blank_token_id (int): blank token is needs for t+1 steping condition
            beam_widths (int, optional): how many B looking for? Defaults to 100.
            state_beam (float, optional): check 1) of 2 point's big enough. Defaults to 4.6.
            expand_beam (float, optional): check 2) of 2 point's smaller vocab. Defaults to 2.3.
            lm (_type_, optional): language_model (not implement) Defaults to None.

        Returns:
            list of tokens for sentence (count of beams's enough big prob sentences)
        """
        if hotwords is not None or lm is not None:
            compare_key = "lm_score"
        else:
            compare_key = "asr_score"
        # prepare hotword input
        hotword_scorer = HotwordScorer.build_scorer(hotwords, weight=hotword_weight)

        start_lm_state = None
        cached_lm_scores = None
        if lm is not None:
            start_lm_state = lm.get_start_state()
            cached_lm_scores = {"": (0.0, 0.0, start_lm_state)}
        cached_p_lm_scores = {}

        # 모델 학습은 batch단위로 했지만, 실시간의 inference는 batch가 있을 수 없다. 때문에 1배치만 들어올 것이기 때문에, 맨 앞에 것만 본다.
        encoder_outputs = self.encoder(inputs, inputs_lengths)[0]
        B_hyps = [
            {
                "asr_score": 0.0,
                "y_star": [blank_token_id],
                "hidden_state": None,
                "lm_score": 0.0,
                "lm_state": start_lm_state,
            }
        ]
        for encoder_idx, encoder_output in enumerate(encoder_outputs):
            A_hyps = B_hyps
            B_hyps = []
            # A에서 몇번째 골랐는지 판단함. (단순 로깅용)
            nth_extension_candidate_from_a = 0
            # A를 전부 확인했으면 더 이상 beam에 넣을 것도 없으므로, 무한루프에 빠지지 않도록 처리.
            while len(A_hyps) > 0:
                most_prob_A = max(A_hyps, key=lambda x: x[compare_key])

                # 변수명은 논문과 동일하게 처리
                a_best_hyp = max(A_hyps, key=lambda x: x[compare_key])[compare_key]
                if len(B_hyps) == 0:
                    # B_hyps이 하나도 없다는 것은 처음 들어온 음성이라는 의미이므로, 어떤 상황에서도 진행시켜야 되니, 엄청 작은 log_prob값을 적용함.
                    b_best_hyp = -9999.0
                else:
                    b_best_hyp = max(B_hyps, key=lambda x: x[compare_key])[compare_key]

                if improved and (b_best_hyp >= state_beam + a_best_hyp):
                    # 일반 beam을 돌리다보면 RNNT는 거의 비등비등하게 나오는데
                    # 확률상 쓰레시홀드를 줘서, A를 더 확장 시켜봐야 B보다 좋아질 여지가 거진 없으면, 그냥 다 뽑았다고 생각하고 out한다.
                    break

                A_hyps.remove(most_prob_A)

                # RNNT는 맨 마지막 것만 들어가야 한다.
                decoder_input = torch.tensor(
                    [most_prob_A["y_star"][-1]], dtype=torch.long, device=encoder_output.device
                )
                decoder_output, hidden_state = self.decoder(
                    decoder_input, prev_hidden_state=most_prob_A["hidden_state"]
                )
                y = self.joint(encoder_output.view(-1), decoder_output.view(-1))

                most_prob_a_next_scores = torch.log_softmax(y, dim=0)
                # 자모 서순 상 확률은, 띄어쓰기 단어 기준으로 N-Gram 되는 것과는 별개로 동작되어야 한다.
                best_prob = max(most_prob_a_next_scores[1:])
                nth_extension_candidate_from_a += 1
                for k, asr_score in enumerate(most_prob_a_next_scores):
                    beam_hyp = {
                        "asr_score": most_prob_A["asr_score"] + float(asr_score),
                        "y_star": copy.deepcopy(most_prob_A["y_star"]),
                        "hidden_state": most_prob_A["hidden_state"],
                        "lm_score": most_prob_A["lm_score"],
                        "lm_state": most_prob_A["lm_state"],
                    }
                    if k == blank_token_id:
                        # blank는 없는 단어이므로, lm 점수를 그냥 즉시 계산 가능하다.
                        # 직전 길이 문장의 lm_score + 현재 blank의 asr_score 및 직전 lm_state
                        # lm이나 hotword가 없어, text matching을 전혀 쓰지 않아도, 해당 score는 asr_score와 동치가 된다.
                        beam_hyp["lm_score"] = most_prob_A["lm_score"] + float(asr_score)
                        B_hyps.append(beam_hyp)
                    else:
                        # non_blank A의 경우는, lm_score를 이후에 계산하여야 한다.
                        # 띄어쓰기가 나올지, 자소가 나올지 현재는 모르기 때문.

                        if improved:
                            # 현재 예측값 (=best_prob)에서 패널티(=expand_beam)를 준것보다 작은 vocab의 prob은
                            # 확률 소수점 곱 연산으로 인해, 어짜피 진행될수록 더 작아질테니, 영향을 못줄 가능성이 높아서, prune 시켜버린다.
                            if asr_score >= best_prob - expand_beam:
                                if beam_hyp["y_star"][-1] != k:  # 중복제거
                                    # blank가 나오기 전 텍스트 예측은 직전값과 중복으로 들어가면 안됨.
                                    beam_hyp["y_star"].append(k)

                                beam_hyp["hidden_state"] = hidden_state
                                A_hyps.append(beam_hyp)
                        else:
                            # improved가 아니면, 모든 vocab을 검사한다.
                            if beam_hyp["y_star"][-1] != k:  # 중복제거
                                # blank가 나오기 전 텍스트 예측은 직전값과 중복으로 들어가면 안됨.
                                beam_hyp["y_star"].append(k)

                            beam_hyp["hidden_state"] = hidden_state
                            A_hyps.append(beam_hyp)

                A_hyps = self._get_lm_beams(
                    lm, A_hyps, hotword_scorer, cached_lm_scores, cached_p_lm_scores, tokenizer, False
                )
                most_prob_next_A = max(A_hyps, key=lambda a: a[compare_key])[compare_key]
                most_prob_next_B = max(B_hyps, key=lambda a: a[compare_key])[compare_key]
                if len(B_hyps) >= beam_widths and most_prob_next_B > most_prob_next_A:
                    break
        B_hyps = self._get_lm_beams(lm, B_hyps, hotword_scorer, cached_lm_scores, cached_p_lm_scores, tokenizer, True)
        nbest_hyps = sorted(B_hyps, key=lambda x: x[compare_key] / len(x["y_star"]), reverse=True)[:beam_widths]
        return [item["y_star"] for item in nbest_hyps]
