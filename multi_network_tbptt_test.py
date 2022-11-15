import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning import LightningModule, Trainer


class LSTMModel(LightningModule):
    """LSTM sequence-to-sequence model for testing TBPTT with automatic optimization."""

    def __init__(self, truncated_bptt_steps=2, input_size=1, hidden_size=8):
        super().__init__()
        torch.manual_seed(42)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size * 2, 1)
        self.truncated_bptt_steps = truncated_bptt_steps
        self.automatic_optimization = True

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx, hiddens):
        # batch: 실제 데이터
        # batch_idx: TBPTT가 아닌 실제 step의 인덱스 1 step == 1 batch
        # hiddens: TBPTT용 전달 데이터

        # tbptt 진행시, 긴 시퀀스의 chunk로 진행이 되므로 training_step은 실질적으로 200 seq에 100 step chunk시
        # 1배치를 수행하기위해 2번의 training_step이 요구됩니다. (0~99, 100~199를 수행하기 위함)
        # 매우 똑똑하게도, 새로운 배치에서의 hiddens은 이전과 연결되면 안되며, 그렇기 때문에 알아서 None으로 초기화해주는 것을 실험했습니다.
        x, y = batch
        if hiddens is not None:
            hiddens1 = hiddens
            hiddens2 = hiddens
        else:
            hiddens1 = None
            hiddens2 = None
        pred1, hiddens1 = self.lstm(x, hiddens1)
        pred2, hiddens2 = self.lstm2(x, hiddens2)
        logits = torch.concat([pred1, pred2], dim=-1)
        linear = self.linear(logits)
        loss = F.mse_loss(linear, y)
        # loss를 변화시키며 테스트하기 위해서는 해당 discussion을 참고하세요 (https://github.com/Lightning-AI/lightning/discussions/15643)
        return {"loss": loss, "hiddens": hiddens1}

    def train_dataloader(self):
        dataset = TensorDataset(torch.rand(50, 200, self.input_size), torch.rand(50, 200, self.input_size))
        return DataLoader(dataset=dataset, batch_size=4)


model = LSTMModel(truncated_bptt_steps=100)
trainer = Trainer(
    default_root_dir="./",
    max_epochs=2,
    log_every_n_steps=2,
    enable_model_summary=False,
    enable_checkpointing=False,
)
trainer.fit(model)
