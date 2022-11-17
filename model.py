import torch
from torch import Tensor
import pytorch_lightning as pl
from networks import AudioTransNet, TextPredNet, JointNet
from argparse import Namespace
from warprnnt_pytorch import RNNTLoss
import torchmetrics.functional as metric_f
from transformers import Wav2Vec2CTCTokenizer


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
        self.save_hyperparameters()
        self.args = args
        self.transnet = AudioTransNet(**transnet_params)
        self.prednet = TextPredNet(**prednet_params)
        self.jointnet = JointNet(**jointnet_params)
        self.rnnt_loss = RNNTLoss(blank=prednet_params["pad_token_id"], reduction="mean")
        # CTC 사용편의성이 좋은 HuggingFace Transformers를 활용하였습니다. (Tokenizer 만들기 귀찮...)
        self.tokenizer = Wav2Vec2CTCTokenizer(vocab_file=args.vocab_path)
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

    def forward(self, inputs, targets, inputs_lengths, targets_lengths, hiddens=(None, None)):
        enc_hiddens = hiddens[0]
        dec_hiddens = hiddens[1]
        # Use for inference only (separate from training_step)
        # labels의 dim을 2차원으로 배치만큼 세움
        zero = torch.zeros((targets.shape[0], 1)).long().cuda()
        # 각 타겟별 맨 처음에 blank 토큰인 0을 채우게됨
        targets_add_blank = torch.cat((zero, targets), dim=1)

        enc_state, enc_hidden_states = self.transnet(inputs, inputs_lengths, enc_hiddens)
        dec_state, dec_hidden_states = self.prednet(targets_add_blank, targets_lengths + 1, dec_hiddens)
        logits = self.jointnet(enc_state, dec_state)
        return logits, (enc_hidden_states, dec_hidden_states)

    def training_step(self, batch, batch_idx, optimizer_idx, hiddens=(None, None)):
        # batch: 실제 데이터
        # batch_idx: TBPTT가 아닌 실제 step의 인덱스 1 step == 1 batch
        # hiddens: TBPTT용 전달 데이터
        """
        If the following conditions are satisfied:
        1) cudnn is enabled,
        2) input data is on the GPU
        3) input data has dtype torch.float16
        4) V100 GPU is used,
        5) input data is not in PackedSequence format persistent algorithm
        can be selected to improve performance.
        """
        input_values, labels, seq_lengths, target_lengths = batch
        inputs_lengths = torch.IntTensor(seq_lengths)
        targets_lengths = torch.IntTensor(target_lengths)
        # tbptt 진행시, 긴 시퀀스의 chunk로 진행이 되므로 training_step은 실질적으로 200 seq에 100 step chunk시
        # 1배치를 수행하기위해 2번의 training_step이 요구됩니다. (0~99, 100~199를 수행하기 위함)
        # 하지만, 새로운 배치에서의 hiddens은 이전과 연결되면 안되며, 매우 똑똑하게도, 새로운 batch_idx의 스텝시작시에는 hiddens는 None으로 처리됩니다.
        # 실험은 ./multi_network_tbptt_test.py로 해볼 수 있음.

        # TODO: 뭔가 대충넣어도 bptt만 사용하면 가능한 시나리오에 loss가 전부 동일하게 떨어짐, 모델 학습 테스트 필요할듯
        # 예를들어, transcription만 bptt 써본다던가, 다 써본다던가 하는....
        logits, enc_dec_hiddens = self(input_values, labels, inputs_lengths, targets_lengths, hiddens)
        loss = self.rnnt_loss(logits, labels, inputs_lengths, targets_lengths)

        # sync_dist를 선언하는 것으로 ddp의 4장비에서 계산된 loss의 평균치를 async로 불러오도록 한다. (log할때만 집계됨)
        self.log("train_loss_step", loss, sync_dist=True)
        # the training step must be updated to accept a ``hiddens`` argument
        # hiddens are the hiddens from the previous truncated backprop step
        return {"loss": loss, "hiddens": enc_dec_hiddens}

    def validation_step(self, batch, batch_idx):
        # validation에서의 tbptt는 필요없습니다. (역전파를 진행하지 않으므로)
        # 때문에 한번의 valid step의 모든 seq가 들어가야 합니다.
        input_values, labels, seq_lengths, target_lengths = batch
        inputs_lengths = torch.IntTensor(seq_lengths)
        targets_lengths = torch.IntTensor(target_lengths)
        logits, _ = self(input_values, labels, inputs_lengths, targets_lengths)

        return {"logits": logits, "labels": labels}

    def validation_epoch_end(self, validation_step_outputs):
        # torch lightning은 약간의 데이터로 초기에 sanity eval step을 수행하고, training step에 돌입합니다.
        # 여기서도 기본적인 값은 찍어볼 수 있으므로, 1 에폭 간신히 돌려놓고 에러맞아서 멘붕오지말고 미리미리 체크하는 것도 좋겠네요
        # https://github.com/Lightning-AI/lightning/issues/2295 (trainer의 num_sanity_val_steps 옵션으로 끌 수도 있긴 함.)
        logits = list()
        labels = list()
        for out in validation_step_outputs:
            logits.append(out["logits"])
            labels.append(out["labels"])
        # gpu_0_prediction = predictions[0]
        # gpu_1_prediction = predictions[1]
        print(logits)
        print(labels)
        # TODO: Predictions list -> seq별 logits argmax -> tokenizer decode
        # TODO: labels list -> tokenizer decode
        # torchmetrics를 사용하는 경우, DistributedSampler가 각각 장비에서 동작하는 것으로 발생할 수 있는 비동기 문제에서 자유로워진다. (https://torchmetrics.readthedocs.io/en/stable/)
        # torchmetrics를 사용하지 않을경우, self.log(sync_dist) 등을 사용하여 따로 처리해줘야함. (https://github.com/Lightning-AI/lightning/discussions/6501)
        # wer = metric_f.word_error_rate(predictions, labels)
        # cer = metric_f.word_error_rate(predictions, labels)
        # self.log("val_wer", wer)
        # self.log("val_cer", cer)

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
