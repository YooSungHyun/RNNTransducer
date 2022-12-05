import torch
import pytorch_lightning as pl
from networks import JointNet
from argparse import Namespace
from warprnnt_pytorch import RNNTLoss as Warp_RNNTLoss
from torchaudio.transforms import RNNTLoss as Torch_RNNTLoss
from torchmetrics import WordErrorRate, CharErrorRate
from transformers import Wav2Vec2CTCTokenizer
from torch.nn.utils.rnn import pack_sequence


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
        self.save_hyperparameters(prednet_params, transnet_params, jointnet_params, args)
        self.args = args
        self.tokenizer = Wav2Vec2CTCTokenizer(vocab_file=args.vocab_path)
        self.blank_token_id = self.tokenizer.pad_token_id
        prednet_params["pad_token_id"] = self.tokenizer.pad_token_id
        # 다른 Transducer 쓰고싶으면 여기서 En,Decoder만 변경해서 사용
        self.jointnet = JointNet(transnet_params, prednet_params, **jointnet_params)
        # Joint Net의 repeat concat 때문에 발생하는 메모리 커지는 이슈를 조금이라도 쉽게 타개하기 위함.
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
        # CTC 사용편의성이 좋은 HuggingFace Transformers를 활용하였습니다. (Tokenizer 만들기 귀찮...)
        # Truncated Backpropagation Through Time (https://pytorch-lightning.readthedocs.io/en/stable/guides/data.html)
        # Important: This property activates truncated backpropagation through time
        # Setting this value to 2 splits the batch into sequences of size 2
        # 배치들을 시간별로 자를 길이를 지정합니다. 자른만큼씩 역전파가 진행됩니다.
        # 해당 기능을 사용하려면 무조건 batch_first True여야 합니다.
        # batch의 split을 수정하려면, pytorch_lightning.core.module.LightningModule.tbptt_split_batch()를 재정의하세요.

        # !!!!!!!! bptt 미적용
        # rnnt_loss를 사용하면 기본적으로 전체 타임 순서에 대한 reduction된 output이 나온다. (나는 -> 나는 밥을 -> 나는 밥을 먹었다 의 loss reduction)
        # bptt를 사용하려면, '나는'일때 역전파 1번, '나는 밥을'일때 역전파 1번, '나는 밥을 먹었다' 일때 역전파 1번해서 각각 3번의 역전파를 시켜야하는데,
        # 라이브러리를 쓰면서 활용하려면, input_lengths만큼 역전파가 진행되어야 해서, 실질적으로 optimizing step을 수동으로 돌려야할 것이다.
        # 수동으로 돌리려면, 그 만큼 성능도 저하될테니, 논문 그대로의 구현체를 만들어야 할 것이다.
        # RNNTloss github 소스를 보면, 내부에서 C로 trans_grad와 pred_grad를 다루는 부분이 있는데, bptt 형태로 알아서 적용해주나 싶기도 하다...?
        # 자세한 사항은 README 참고
        # self.truncated_bptt_steps = 1

    def forward(self, input_audios, input_texts, text_lengths):

        logits = self.jointnet(input_audios, input_texts, text_lengths)
        return logits

    def training_step(self, batch, batch_idx):
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
        # input_texts, targets, audio_lengths, target_lengths: on_cuda
        # input_audios, text_lengths : on_cpu or not tensor
        input_audios, audio_lengths, input_texts, text_lengths, targets, target_lengths = batch

        # tbptt 진행시, 긴 시퀀스의 chunk로 진행이 되므로 training_step은 실질적으로 200 seq에 100 step chunk시
        # 1배치를 수행하기위해 2번의 training_step이 요구됩니다. (0~99, 100~199를 수행하기 위함)
        # 하지만, 새로운 배치에서의 hiddens은 이전과 연결되면 안되며, 매우 똑똑하게도, 새로운 batch_idx의 스텝시작시에는 hiddens는 None으로 처리됩니다.
        # 실험은 ./multi_network_tbptt_test.py로 해볼 수 있음.

        input_audios = pack_sequence(input_audios)  # input_audio: list -> cuda
        logits = self(input_audios, input_texts, text_lengths)  # text_lengths must list or cpu cuda (list)
        loss = self.rnnt_loss(logits, targets, audio_lengths, target_lengths)

        # sync_dist를 선언하는 것으로 ddp의 4장비에서 계산된 loss의 평균치를 async로 불러오도록 한다. (log할때만 집계됨)
        # self.log("train_loss_step", loss, sync_dist=True)
        # the training step must be updated to accept a ``hiddens`` argument
        # hiddens are the hiddens from the previous truncated backprop step
        return {"loss": loss}

    def training_step_end(self, outputs):
        # only use when  on dp
        assert not self.args.move_metrics_to_cpu, "DDP만을 지원하므로, cpu로 metric이동되면 gather가 동작하지 않습니다."
        self.log("train_loss", outputs["loss"], sync_dist=True)

    def validation_step(self, batch, batch_idx):
        # validation에서의 tbptt는 필요없습니다. (역전파를 진행하지 않으므로)
        # 때문에 한번의 valid step의 모든 seq가 들어가야 합니다.
        input_audios, audio_lengths, input_texts, text_lengths, targets, target_lengths = batch

        # pad를 채우지 않고, 바로 pack 시키는 전략을 취함. audio는 스마트배칭으로, sort를 다시 할 필요가 없음
        input_audios = pack_sequence(input_audios)
        if self.args.val_on_cpu or self.args.precision > 16:
            audio_lengths = audio_lengths.cpu()
            input_texts = input_texts.cpu()
            targets = targets.cpu()
            target_lengths = target_lengths.cpu()
            input_audios = input_audios.cpu()
            self.cpu()

        logits = self(input_audios, input_texts, text_lengths)
        loss = self.rnnt_loss(logits, targets, audio_lengths, target_lengths)

        pred_tokens = self.jointnet.recognize(input_audios, self.blank_token_id)
        pred_texts = self.tokenizer.batch_decode(pred_tokens)
        label_texts = self.tokenizer.batch_decode(targets)
        return {"loss": loss, "pred_texts": pred_texts, "label_texts": label_texts}

    def validation_epoch_end(self, validation_step_outputs):
        assert not self.args.move_metrics_to_cpu, "DDP만을 지원하므로, cpu로 metric이동되면 gather가 동작하지 않습니다."
        # torch lightning은 약간의 데이터로 초기에 sanity eval step을 수행하고, training step에 돌입합니다.
        # 여기서도 기본적인 값은 찍어볼 수 있으므로, 1 에폭 간신히 돌려놓고 에러맞아서 멘붕오지말고 미리미리 체크하는 것도 좋겠네요
        # https://github.com/Lightning-AI/lightning/issues/2295 (trainer의 num_sanity_val_steps 옵션으로 끌 수도 있긴 함.)
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
        # torchmetrics를 사용하는 경우, DistributedSampler가 각각 장비에서 동작하는 것으로 발생할 수 있는 비동기 문제에서 자유로워진다. (https://torchmetrics.readthedocs.io/en/stable/)
        # torchmetrics를 사용하지 않을경우, self.log(sync_dist) 등을 사용하여 따로 처리해줘야함. (https://github.com/Lightning-AI/lightning/discussions/6501)
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
            max_lr=self.args.max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.args.warmup_ratio,
            epochs=self.trainer.max_epochs,
            final_div_factor=self.args.final_div_factor
            # steps_per_epoch=self.steps_per_epoch,
        )
        lr_scheduler = {"interval": "step", "scheduler": scheduler, "name": "AdamW"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    """ Do Not DELETE This Source
    This Used past on lr_scheduler.
    """
    # @property
    # def num_training_steps(self) -> int:
    #     """Total training steps inferred from datamodule and devices."""
    #     if self.trainer.max_steps > 0:
    #         return self.trainer.max_steps

    #     limit_batches = self.trainer.limit_train_batches
    #     batches = len(self.trainer.datamodule.train_dataloader())
    #     batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

    #     num_devices = max(1, self.trainer.num_devices)

    #     effective_accum = self.trainer.accumulate_grad_batches * num_devices
    #     return (batches // effective_accum) * self.trainer.max_epochs

    # @property
    # def steps_per_epoch(self) -> int:
    #     return self.num_training_steps // self.trainer.max_epochs
