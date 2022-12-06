import pytorch_lightning as pl
from model import RNNTransducer
from datamodule import RNNTransducerDataModule
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta
from setproctitle import setproctitle
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from simple_parsing import ArgumentParser
from utils import read_json, dataclass_to_namespace, LightningModuleArguments


def main(hparams):
    wandb_logger = WandbLogger(project="RNNTransducer", name="default", save_dir="./")
    setproctitle("bart_online_stt")
    pl.seed_everything(hparams.seed)

    # DataModule의 prepare_data가 Main CPU에서만 동작하므로, 나머지 컴퓨팅 대상은 대기상태에 들어간다.
    # torch.distributed.init_process_group의 timeout은 30분으로, 대기를 30분 이상하면 자동으로 종료된다.
    # 때문에 대용량 Data는 절대로 완성되지 않을 경우가 존재하는데, 따라서 임시적으로 해당 파라미터만 조절한다.
    # Trainer의 초기화가 from_argparse_args에서 hparams의 dict key, value 쌍으로 이루어지므로, 이렇게 구동하여도 문제는 없다. (개발자 가이드에도 제안되는 방법임)
    model_config_dict = read_json(hparams.model_config)["model"]
    data_config_dict = read_json(hparams.model_config)["data"]

    RNNT_datamodule = RNNTransducerDataModule(data_config_dict, hparams)
    # (원문이 재밌어서 그대로 따옴) There are no .cuda() or .to(device) calls required. Lightning does these for you.
    # DDP를 사용한다면, distributed sampler를 default로 선언하여 사용하게 해줍니다. 필요시 재정의도 가능합니다.
    # torch.nn.Module 그대로이며, 추가된 사항만 있습니다. 찾아서 즐기세요!
    model = RNNTransducer(
        model_config_dict["prednet"], model_config_dict["transnet"], model_config_dict["jointnet"], hparams
    )

    wandb_logger.watch(model, log="all")
    hparams.logger = wandb_logger
    # hparams.profiler = "simple"

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.output_dir,
        save_top_k=3,
        mode="min",
        monitor="val_cer",
        filename="bart-online-{epoch:02d}-{val_cer:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    hparams.callbacks = [checkpoint_callback, lr_monitor]

    hparams.strategy = DDPStrategy(timeout=timedelta(days=30))
    trainer = pl.Trainer.from_argparse_args(hparams)
    trainer.fit(model, datamodule=RNNT_datamodule)
    checkpoint_callback.best_model_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_arguments(LightningModuleArguments, dest="training_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "training_args")
    main(args)
