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
from typing import Tuple
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
        input_size: int,
        forward_output_size: int,
        act_func: str = "",
    ):
        super(JointNet, self).__init__()
        self.encoder = AudioTransNet(**transnet_params)
        self.decoder = TextPredNet(**prednet_params)
        self.num_classes = num_classes
        self.act_func_name = act_func.lower()
        if self.act_func_name == "glu":
            self.forward_layer = nn.Linear(input_size, forward_output_size * 2, bias=True)
            self.act_func = nn.GLU()
        elif self.act_func_name == "gelu":
            self.forward_layer = nn.Linear(input_size, forward_output_size, bias=True)
            self.act_func = nn.GELU(approximate="tanh")
        self.fc = nn.Linear(forward_output_size, num_classes, bias=False)

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
        # forward_layer는 논문에서 다뤄지는 내용은 아니나, 두개를 concat하면 길어지니, 예측에 중요한 특성을 한번 더 필터링 하기 위함
        # 그냥 다 넣는게 잘될지도 모르고 테스트 해봐야겠음.
        if self.act_func_name:
            outputs = self.forward_layer(outputs)
            outputs = self.act_func(outputs)
        # 마이너스로 많이 가는 값이 있으면 tanh가 더 안정적일 수 있음, 다만 tanh 근사 시키기때문에 gelu도 잘되지 않을까 판단해봄.
        # outputs = self.gelu(outputs)
        outputs = self.fc(outputs)

        return outputs

    def forward(self, input_audios: Tensor, input_texts: Tensor, text_lengths: Tensor) -> Tensor:
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
        enc_state, _ = self.encoder(input_audios)
        dec_state, _ = self.decoder(input_texts, text_lengths)
        outputs = self.joint(enc_state, dec_state)
        return outputs

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_length: int) -> Tensor:
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
        decoder_input = encoder_output.new_tensor([[self.decoder.bos_token_id]], dtype=torch.long)

        for t in range(max_length):
            decoder_output, hidden_state = self.decoder(decoder_input, prev_hidden_state=hidden_state)
            step_output = self.joint(encoder_output[t].view(-1), decoder_output.view(-1))
            step_output = step_output.softmax(dim=0)
            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = step_output.new_tensor([[pred_token]], dtype=torch.long)

        return torch.LongTensor(pred_tokens)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
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
        # TODO inputs를 padded_pack 시키거나, 이전에서 그렇게 해서 받아야함.
        encoder_outputs, encoder_hiddens = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            decoded_seq = self.decode(encoder_output, max_length)
            outputs.append(decoded_seq)

        outputs = torch.stack(outputs, dim=1).transpose(0, 1)

        return outputs
