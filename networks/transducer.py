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
    def decode(self, encoder_output: Tensor, max_length: int, blank_token_id: int, max_iters: int = 3) -> Tensor:
        """
        Decode `encoder_outputs`.

        Args:
            encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(seq_length, dimension)``
            max_length (int): max decoding time step

        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        pred_tokens, hidden_state = list(), None
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
                    if pred_tokens[-1:] != pred_token:
                        # 최종 output의 경우 중복 제거
                        pred_tokens.append(pred_token)
                    decoder_input = torch.tensor([[pred_token]], dtype=torch.long, device=decoder_input.device)
                    decoder_output, hidden_state = self.decoder(decoder_input, prev_hidden_state=hidden_state)
                    u = u + 1
                else:
                    break

        return torch.LongTensor(pred_tokens)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, inputs_lengths: Tensor, blank_token_id: int) -> Tensor:
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
            decoded_seq = self.decode(encoder_output, max_length, blank_token_id)
            outputs.append(decoded_seq)

        outputs = torch.stack(outputs, dim=1).transpose(0, 1)

        return outputs

    @torch.no_grad()
    def recognize_beams(
        self, inputs: Tensor, inputs_lengths: Tensor, blank_token_id: int, beam_widths: int = 100, lm=None
    ):
        encoder_outputs = self.encoder(inputs, inputs_lengths).squeeze()
        B_hyps = [{"score": 0.0, "y_star": [blank_token_id], "hidden_state": None}]

        # TODO encoder_output squeeze로 인해, batch는 고려하지 않아도 되는지 확인
        for encoder_output in encoder_outputs:
            A_hyps = B_hyps
            B_hyps = []

            # while True:
            while len(A_hyps) > 0:
                y_star = max(A_hyps, key=lambda x: x["score"])
                A_hyps.remove(y_star)

                decoder_input = torch.tensor([A_hyps["y_star"]], dtype=torch.long, device=encoder_output.device)
                decoder_output, hidden_state = self.decoder(decoder_input, prev_hidden_state=A_hyps["hidden_state"])
                # TODO decoder_output의 shape과 위치 적절한지 확인
                y = self.joint(encoder_output.view(-1), decoder_output.view(-1))

                score_y_star = torch.log_softmax(y, dim=0)

                for k, k_score in enumerate(score_y_star):
                    beam_hyp = {
                        "score": y_star["score"] + float(k_score),
                        "y_star": y_star["y_star"][:],
                        "hidden_state": y_star["hidden_state"],
                    }

                    if k == blank_token_id:
                        B_hyps.append(beam_hyp)
                    else:
                        beam_hyp["y_star"].append(int(k))
                        beam_hyp["hidden_state"] = hidden_state

                        A_hyps.append(beam_hyp)

                if len(B_hyps) >= beam_widths:
                    break

        nbest_hyps = sorted(B_hyps, key=lambda x: x["score"] / len(x["y_star"]), reverse=True)[:beam_widths]
        return nbest_hyps
