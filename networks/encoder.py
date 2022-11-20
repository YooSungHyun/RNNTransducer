# Copyright (c) 2022, SungHyun Yoo. All rights reserved.
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

import torch.nn as nn
from torch import Tensor
from typing import Tuple


class AudioTransNet(nn.Module):
    """
    Transcription Network of RNN-Transducer.
    Implements the functionalities of the transcription network in the model architecture, where
    the input is the speech features and projects it to high level feature representation.

    Args:
        input_size (int): dimension of input vector
        hidden_size (int, optional): hidden state dimension of encoder (default: 320)
        output_size (int, optional): output dimension of encoder and decoder (default: 512)
        num_layers (int, optional): number of encoder layers (default: 4)
        rnn_type (str, optional): type of rnn cell (default: lstm)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default: True)

    Inputs: inputs, input_lengths
        inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        (Tensor, Tensor)

        * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        * hidden_states (torch.FloatTensor): A hidden state of encoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    """

    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        rnn_type: str = "lstm",
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super(AudioTransNet, self).__init__()
        self.hidden_size = hidden_size
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.out_proj = nn.Linear(2 * hidden_size if bidirectional else hidden_size, output_size)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  encoder training.

        Args:
            inputs (torch.FloatTensor): pack_sequence(inputs) not padded, but run pad_packed_sequence() -> output padded automatical

        Returns:
            (Tensor, Tensor)

            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * hidden_state (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        # inputs shape: (batch_size, seq, features)의 pack_padded_sequence가 진행된 값.
        # DP를 사용하는경우 메모리 연속성을 유지해주기 위함. (메모리상 분산 저장되므로 Weight의 연속 무결성이 사라질 수 있음을 방지함.)
        self.rnn.flatten_parameters()
        outputs, hidden_states = self.rnn(inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        outputs = self.out_proj(outputs)
        """
        For bidirectional RNNs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when batch_first=False: output.view(seq_len, batch, num_directions, hidden_size).
        """
        return outputs, hidden_states
