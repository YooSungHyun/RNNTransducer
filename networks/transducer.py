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


class JointNet(nn.Module):
    """
    Args:
        num_classes (int): number of classification
        input_dim (int): dimension of input vector (decoder_hidden_size)

    Inputs: inputs, input_lengths, targets, target_lengths
        inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
        target_lengths (torch.LongTensor): The length of target tensor. ``(batch)``

    Returns:
        * predictions (torch.FloatTensor): Result of model predictions.
    """

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
        self.fc = nn.Linear(transnet_params["output_size"] + prednet_params["output_size"], num_classes, bias=False)

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
        Forward propagate a `inputs` and `targets` pair for training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            target_lengths (torch.LongTensor): The length of target tensor. ``(batch)``

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
        Recognize input speech. This method consists of the forward of the encoder and the decode() of the decoder.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
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
                        # 최초 리스트의 경우에도 out of index 안나게 하기위해 [-1:] slice로 처리
                        if pred_tokens[-1] != pred_token:
                            # 최종 output의 경우 중복 제거
                            pred_tokens.append(pred_token)
                        # 텍스트인 경우, u 그래프 기준으로 위로 올라가야 하므로, 음성 들어있을 또 다른 텍스트를 찾아봅니다.
                        decoder_input = torch.tensor([[pred_token]], dtype=torch.long, device=decoder_input.device)
                        decoder_output, hidden_state = self.decoder(decoder_input, prev_hidden_state=hidden_state)
                        u = u + 1
                    else:
                        # blank여도 일단 토큰은 집어넣으나, 다음 예측을 위한 hidden은 건들지 않습니다.
                        break

            outputs.append(torch.LongTensor(pred_tokens[1:]))

        outputs = torch.stack(outputs, dim=1).transpose(0, 1)

        return outputs

    @torch.no_grad()
    def recognize_beams(
        self,
        inputs: Tensor,
        inputs_lengths: int,
        blank_token_id: int,
        beam_widths: int = 100,
        state_beam: float = 4.6,
        expand_beam: float = 2.3,
        lm=None,
    ):
        # 모델 학습은 batch단위로 했지만, 실시간의 inference는 batch가 있을 수 없다. 때문에 1배치만 들어올 것이기 때문에, 맨 앞에 것만 본다.
        encoder_outputs = self.encoder(inputs, inputs_lengths)[0]
        B_hyps = [{"score": 0.0, "y_star": [blank_token_id], "hidden_state": None}]
        for encoder_output in encoder_outputs:
            A_hyps = B_hyps
            B_hyps = []
            # A에서 몇번째 골랐는지 판단함.
            nth_extension_candidate_from_a = 0

            # A를 전부 확인했으면 더 이상 beam에 넣을 것도 없으므로, 무한루프에 빠지지 않도록 처리.
            while len(A_hyps) > 0:
                most_prob_A = max(A_hyps, key=lambda x: x["score"])
                a_best_hyp = max(A_hyps, key=lambda x: x["score"])["score"]
                if len(B_hyps) == 0:
                    b_best_hyp = -9999.0
                else:
                    b_best_hyp = max(B_hyps, key=lambda x: x["score"])["score"]

                # 일반 beam을 돌리다보면 RNNT는 거의 비등비등하게 나오는데, A를 더 확장 시켜봐야
                # 확률상 쓰레시홀드를 줘도, B보다 좋아질 여지가 거진 없으면, 그냥 다 뽑았다고 생각하고 out한다.
                # TODO 만약 요구사항상 꼭 3 beam이면 3 beam 뽑아야한다. 이러면 사용 못할수도 있다.
                if b_best_hyp >= state_beam + a_best_hyp:
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
                best_prob = max(most_prob_a_next_scores[1:])
                nth_extension_candidate_from_a += 1
                for k, k_score in enumerate(most_prob_a_next_scores):
                    beam_hyp = {
                        "score": most_prob_A["score"] + float(k_score),
                        "y_star": copy.deepcopy(most_prob_A["y_star"]),
                        "hidden_state": most_prob_A["hidden_state"],
                    }

                    if k == blank_token_id:
                        if len(B_hyps) == 0:
                            # B가 없고, blank라면 t1의 u0에서 u1을 판단하려는 경우이다.
                            # 이 경우 한번도 선택된 적이 없으니, 0.0+log_prob(blank) 된 값을 취한다.
                            # 비교하고자 하는 것은, 현재가 블랭크라고 가정할때, A너는 더 나아갈 여지가 있느냐? 를 판단하게 된다.
                            just_before_prob_B = beam_hyp["score"]
                        else:
                            # 아닌 경우라면, 무조건 삽입 전 마지막 놈과 비교한다.
                            just_before_prob_B = B_hyps[-1]["score"]
                        B_hyps.append(beam_hyp)
                    else:
                        # 현재 예측값 max(=best_prob)에서 패널티(=expand_beam)를 준것보다 작은 vocab의 prob은
                        # 확률 소수점 곱 연산으로 인해, 어짜피 진행될수록 더 작아질테니, 영향을 못줄 가능성이 높아서, prune 시켜버린다.
                        if k_score >= best_prob - expand_beam:
                            # 최초 리스트의 경우에도 out of index 안나게 하기위해 [-1:] slice로 처리
                            if beam_hyp["y_star"][-1] != int(k):
                                # blank가 나오기 전 텍스트 예측은 직전값과 중복으로 들어가면 안됨.
                                beam_hyp["y_star"].append(int(k))
                            beam_hyp["hidden_state"] = hidden_state

                            A_hyps.append(beam_hyp)

                most_prob_next_A = max(A_hyps, key=lambda a: a["score"])["score"]
                if len(B_hyps) >= beam_widths and just_before_prob_B >= most_prob_next_A:
                    break

        nbest_hyps = sorted(B_hyps, key=lambda x: x["score"] / len(x["y_star"]), reverse=True)[:beam_widths]
        return [item["y_star"] for item in nbest_hyps]
