import torch
from torch import Tensor
import pytorch_lightning as pl
from torchaudio import functional as F
import numpy as np
from networks import AudioTransNet, TextPredNet, JointNet
from argparse import Namespace


class RNNTransducer(pl.LightningModule):
    """
    RNN-Transducer are a form of sequence-to-sequence models that do not employ attention mechanisms.
    Unlike most sequence-to-sequence models, which typically need to process the entire input sequence
    (the waveform in our case) to produce an output (the sentence), the RNN-T continuously processes input samples and
    streams output symbols, a property that is welcome for speech dictation. In our implementation,
    the output symbols are the characters of the alphabet.
    """

    # 호출순서
    # datamodule setup 끝 -> configure_optimizers -> dataloader(collate_fn)
    def __init__(self, prednet_params: dict, transnet_params: dict, jointnet_params: dict, args: Namespace):
        super().__init__()
        self.transnet = AudioTransNet(**transnet_params)
        self.prednet = TextPredNet(**prednet_params)
        self.jointnet = JointNet(**jointnet_params)
        self.args = args
        # Truncated Backpropagation Through Time (https://pytorch-lightning.readthedocs.io/en/stable/guides/data.html)
        # Important: This property activates truncated backpropagation through time
        # Setting this value to 2 splits the batch into sequences of size 2
        # 배치들을 시간별로 자를 길이를 지정합니다. 자른만큼씩 역전파가 진행됩니다.
        # 해당 기능을 사용하려면 무조건 batch_first True여야 합니다.
        # batch의 split을 수정하려면, pytorch_lightning.core.module.LightningModule.tbptt_split_batch()를 재정의하세요.
        self.truncated_bptt_steps = 2

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
        decoder_input = encoder_output.new_tensor([[self.decoder.sos_id]], dtype=torch.long)

        for t in range(max_length):
            decoder_output, hidden_state = self.decoder(decoder_input, hidden_states=hidden_state)
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

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            decoded_seq = self.decode(encoder_output, max_length)
            outputs.append(decoded_seq)

        outputs = torch.stack(outputs, dim=1).transpose(0, 1)

        return outputs

    def forward(self, inputs, inputs_lengths, targets, targets_lengths):
        # Use for inference only (separate from training_step)
        zero = torch.zeros((targets.shape[0], 1)).long()
        targets_add_blank = torch.cat((zero, targets), dim=1)
        enc_state, _ = self.transnet(inputs, inputs_lengths)
        dec_state, _ = self.prednet(targets_add_blank, targets_lengths + 1)
        logits = self.jointnet(enc_state, dec_state)
        return logits

    def training_step(self, batch, batch_idx, hiddens):
        input_values, labels = batch
        # flatten any input
        input_values = input_values.view(input_values.size(0), -1)
        logits = self(input_values)
        # the training step must be updated to accept a ``hiddens`` argument
        # hiddens are the hiddens from the previous truncated backprop step
        out, hiddens = self.lstm(data, hiddens)
        return {"loss": ..., "hiddens": hiddens}

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps > 0:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_devices)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    @property
    def steps_per_epoch(self) -> int:
        return self.num_training_steps // self.trainer.max_epochs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.max_lr,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.args.max_epochs,
            pct_start=0.2,
        )
        return [optimizer], [scheduler]
