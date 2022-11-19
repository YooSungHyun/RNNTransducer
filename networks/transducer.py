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

    def __init__(self, encoder: object, decoder: object, num_classes: int, input_size: int, forward_output_size: int):
        super(JointNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes
        self.forward_layer = nn.Linear(input_size, forward_output_size, bias=True)
        self.gelu = nn.GELU(approximate="tanh")

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
        outputs = self.forward_layer(outputs)
        # 마이너스로 많이 가는 값이 있으면 tanh가 더 안정적일 수 있음, 다만 tanh 근사 시키기때문에 gelu도 잘되지 않을까 판단해봄.
        outputs = self.gelu(outputs)
        outputs = self.fc(outputs)

        return outputs

    def forward(
        self, inputs: Tensor, inputs_lengths: Tensor, targets: Tensor, targets_lengths: Tensor, hiddens: Tuple[Tensor]
    ) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            target_lengths (torch.LongTensor): The length of target tensor. ``(batch)``
            hiddens(Tuple[Tensor]): (encoder_hiddens, decoder_hiddens) for BackPropagation Through Time

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        enc_hiddens = hiddens[0]
        dec_hiddens = hiddens[1]
        # Use for inference only (separate from training_step)
        # labels의 dim을 2차원으로 배치만큼 세움
        zero = torch.zeros((targets.shape[0], 1)).long().cuda()
        # 각 타겟별 맨 처음에 blank 토큰인 0을 채우게됨
        targets_add_blank = torch.cat((zero, targets), dim=1)

        enc_state, enc_hidden_states = self.encoder(inputs, inputs_lengths, enc_hiddens)
        dec_state, dec_hidden_states = self.decoder(targets_add_blank, targets_lengths + 1, dec_hiddens)
        outputs = self.joint(enc_state, dec_state)
        return outputs, (enc_hidden_states, dec_hidden_states)
