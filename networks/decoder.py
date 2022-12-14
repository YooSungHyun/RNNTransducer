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
import torch


class TextPredNet(nn.Module):
    """
    Prediction Network of RNN-Transducer
    Implements the functionalities of the
    predict network in the model architecture

    Args:
        embedding_size (int): number of classification
        pad_token_id (int): embedding pad token id
        hidden_size (int): hidden state dimension of decoder (default: 512)
        output_size (int): output dimension of encoder and decoder (default: 512)
        num_layers (int): number of decoder layers (default: 1)
        rnn_type (str, optional): type of rnn cell (default: lstm)
        dropout (float, optional): dropout probability of decoder

    Inputs: inputs, input_lengths
        inputs (torch.LongTensor): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        hidden_states (torch.FloatTensor): A previous hidden state of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``

    Returns:
        (Tensor, Tensor):

        * decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        * hidden_states (torch.FloatTensor): A hidden state of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    """

    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
        self,
        embedding_size: int,
        pad_token_id: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        rnn_type: str = "lstm",
        dropout: float = 0.2,
    ):
        super(TextPredNet, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(embedding_size, hidden_size, padding_idx=pad_token_id)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=False,
        )
        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self, inputs: Tensor, input_lengths: Tensor = None, prev_hidden_state: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward propage a `inputs` (targets) for training.

        Args:
            inputs (torch.LongTensor): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            pred_hidden_state (torch.FloatTensor): A previous hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``

        Returns:
            (Tensor, Tensor):

            * decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * hidden_states (torch.FloatTensor): A hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        """
        embedded = self.embedding(inputs)
        if input_lengths is not None:
            # .cpu()??? ?????? ???????????? ?????? ???????????????, list??? ?????? ???, ?????? ???????????? ?????? cpu??? ?????? ???????????? ??????
            input_lengths = torch.tensor(data=input_lengths, device="cpu")
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            sorted_embedded = embedded[indices]
            # pack_padded_sequence output : (data, batch_sizes)
            # ??? ????????? ?????? ??????: https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html
            # ?????? ????????? ??????, blank??? ?????????????????? ?????? ????????? ??? ?????? ???????????? ????????????. ?????? pad_packed??? ?????? blank??? ????????????.
            packed_embedded = nn.utils.rnn.pack_padded_sequence(sorted_embedded, sorted_seq_lengths, batch_first=True)
            # next line just pad_packed tested source code
            # outputs = nn.utils.rnn.pad_packed_sequence(packed_embedded, batch_first=True)
            # self.rnn.flatten_parameters()
            outputs, hidden_states = self.rnn(packed_embedded, prev_hidden_state)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            # indices??? ???????????????, indices??? ????????? ?????? array??? index???????????? ????????????.
            _, desorted_indices = torch.sort(indices, descending=False)
            # pack_padded????????? ????????????
            outputs = outputs[desorted_indices]
        else:
            # self.rnn.flatten_parameters()
            outputs, hidden_states = self.rnn(embedded, prev_hidden_state)
        outputs = self.out_proj(outputs)

        return outputs, hidden_states
