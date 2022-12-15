import torch
import pytorch_lightning as pl
from networks import JointNet
from argparse import Namespace
from warprnnt_pytorch import RNNTLoss as Warp_RNNTLoss
from torchaudio.transforms import RNNTLoss as Torch_RNNTLoss
from torchmetrics import WordErrorRate, CharErrorRate
from transformers import Wav2Vec2CTCTokenizer


class RNNTransducer(pl.LightningModule):
    """
    RNN-Transducer are a form of sequence-to-sequence models that do not employ attention mechanisms.
    Unlike most sequence-to-sequence models, which typically need to process the entire input sequence
    (the waveform in our case) to produce an output (the sentence), the RNN-T continuously processes input samples and
    streams output symbols, a property that is welcome for speech dictation. In our implementation,
    the output symbols are the characters of the alphabet.
    """

    def __init__(self, prednet_params: dict, transnet_params: dict, jointnet_params: dict, args: Namespace):
        super().__init__()
        self.save_hyperparameters(prednet_params, transnet_params, jointnet_params, args)
        self.args = args
        self.tokenizer = Wav2Vec2CTCTokenizer(vocab_file=args.vocab_path)
        self.blank_token_id = self.tokenizer.pad_token_id
        prednet_params["pad_token_id"] = self.tokenizer.pad_token_id
        self.jointnet = JointNet(transnet_params, prednet_params, **jointnet_params)
        if args.precision == 16:
            # precision 16의 경우 GPU 할당이 2배 여유가 생기므로 그냥 싹다 GPU에서 수행
            compute_on_cpu = False
            self.rnnt_loss = Torch_RNNTLoss(blank=self.blank_token_id, reduction="mean")
        else:
            if args.val_on_cpu:
                compute_on_cpu = True
            else:
                # precision이 16도 아니면서 val_on_cpu를 선택한다? (3090에서도 동작안함.)
                print("32비트 이상이라면, 무조건 compute_on_cpu는 True입니다. (리소스 효율을 위함)")
                compute_on_cpu = True
            self.rnnt_loss = Warp_RNNTLoss(blank=self.blank_token_id, reduction="mean")

        self.calc_wer = WordErrorRate(compute_on_cpu=compute_on_cpu)
        self.calc_cer = CharErrorRate(compute_on_cpu=compute_on_cpu)
        # !!!!!!!! bptt 미적용
        # 확실하진 않지만, RNNTLoss에서 각 인디코더의 grad를 확인한 것으로, 각각 BPTT 하듯이 적용해주지 않을까 사료됨.
        # self.truncated_bptt_steps = 5

    def forward(self, input_audios, audio_lengths, input_texts, text_lengths):

        logits = self.jointnet(input_audios, audio_lengths, input_texts, text_lengths)
        return logits

    def training_step(self, batch, batch_idx):
        assert not self.args.move_metrics_to_cpu, "DDP만을 지원하므로, cpu로 metric이동되면 gather가 동작하지 않습니다."
        input_audios, audio_lengths, tensor_audio_lengths, input_texts, text_lengths, targets, target_lengths = batch

        logits = self(input_audios, audio_lengths, input_texts, text_lengths)
        loss = self.rnnt_loss(logits, targets, tensor_audio_lengths, target_lengths)

        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_audios, audio_lengths, tensor_audio_lengths, input_texts, text_lengths, targets, target_lengths = batch

        if self.args.val_on_cpu or self.args.precision > 16:
            audio_lengths = audio_lengths.cpu()
            tensor_audio_lengths = tensor_audio_lengths.cpu()
            input_texts = input_texts.cpu()
            targets = targets.cpu()
            target_lengths = target_lengths.cpu()
            input_audios = input_audios.cpu()
            self.cpu()
        logits = self(input_audios, audio_lengths, input_texts, text_lengths)
        loss = self.rnnt_loss(logits, targets, tensor_audio_lengths, target_lengths)

        pred_tokens = self.jointnet.recognize(input_audios, audio_lengths, self.blank_token_id)
        pred_texts = self.tokenizer.batch_decode(pred_tokens)
        label_texts = self.tokenizer.batch_decode(targets)
        return {"loss": loss, "pred_texts": pred_texts, "label_texts": label_texts}

    def validation_epoch_end(self, validation_step_outputs):
        assert not self.args.move_metrics_to_cpu, "DDP만을 지원하므로, cpu로 metric이동되면 gather가 동작하지 않습니다."
        if self.precision == 16:
            # float16의 경우 torch audio를 활용하는데, 그 경우 torch value가 튀어나옴.
            loss_mean = torch.tensor([x["loss"] for x in validation_step_outputs], device=self.device).mean()
        else:
            # warp_transducer의 loss를 활용하는경우 torch 1차원 list가 튀어나옴
            loss_mean = torch.cat([x["loss"] for x in validation_step_outputs]).mean()
        preds = list()
        labels = list()
        for output in validation_step_outputs:
            preds.extend(output["pred_texts"])
            labels.extend(output["label_texts"])
        wer = self.calc_wer(preds, labels)
        cer = self.calc_cer(preds, labels)
        if not self.on_gpu:
            # 이 곳에 빠지는 경우는, val_on_cpu가 True인 경우여야만 한다. 혹은 precision이 16 이상임.
            # nccl의 경우, allreduce는 cuda에 있어야 정상동작하므로 cpu output을 cuda로 바꿈
            self.log("val_loss", loss_mean.cuda(), sync_dist=True)  # torchmetrics가 아니므로 sync_dist 필수
            self.log("val_wer", wer.cuda(), sync_dist=True)
            self.log("val_cer", cer.cuda(), sync_dist=True)
            # training_step에서는 gpu로 올려줘야, all_gather가 정상동작함.
            self.cuda()
        else:
            # 이미 cuda면 그냥 던짐
            self.log("val_loss", loss_mean, sync_dist=True)
            self.log("val_wer", wer, sync_dist=True)
            self.log("val_cer", cer, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [{"params": [p for p in self.parameters()], "name": "OneCycleLR"}],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.args.warmup_ratio,
            epochs=self.trainer.max_epochs,
            final_div_factor=self.args.final_div_factor
            # steps_per_epoch=self.steps_per_epoch,
        )
        lr_scheduler = {"interval": "step", "scheduler": scheduler, "name": "AdamW"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
