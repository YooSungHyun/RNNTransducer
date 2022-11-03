import torch
import pytorch_lightning as pl
from torch import Tensor, nn
from torch.nn import functional as F
import numpy as np
from networks import AudioTransNet, TextPredNet, JointNet


class RNNTransducer(pl.LightningModule):
    def __init__(
        self,
        prednet_params: dict,
        transnet_params: dict,
        joinnet_params: dict,
    ):
        super().__init__()
        self.transnet = AudioTransNet(**transnet_params)
        self.prednet = TextPredNet(**prednet_params)
        self.joinnet = JointNet(**joinnet_params)

    def forward(self, inputs, inputs_lengths, targets, targets_lengths):
        zero = torch.zeros((targets.shape[0], 1)).long()
        targets_add_blank = torch.cat((zero, targets), dim=1)
        enc_state, _ = self.transnet(inputs, inputs_lengths)
        dec_state, _ = self.prednet(targets_add_blank, targets_lengths + 1)
        logits = self.joinnet(enc_state, dec_state)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        # flatten any input
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction="sum")
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
