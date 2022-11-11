import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
import collections
import scipy
import random
from utils import get_concat_dataset, read_json
from pytorch_lightning import Trainer
from model import RNNTransducer
from datamodule import RNNTransducerDataModule
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta


def main(hparams):
    random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)

    # DataModule의 prepare_data가 Main CPU에서만 동작하므로, 나머지 컴퓨팅 대상은 대기상태에 들어간다.
    # torch.distributed.init_process_group의 timeout은 30분으로, 대기를 30분 이상하면 자동으로 종료된다.
    # 때문에 대용량 Data는 절대로 완성되지 않을 경우가 존재하는데, 따라서 임시적으로 해당 파라미터만 조절한다.
    # Trainer의 초기화가 from_argparse_args에서 hparams의 dict key, value 쌍으로 이루어지므로, 이렇게 구동하여도 문제는 없다. (개발자 가이드에도 제안되는 방법임)
    hparams.strategy = DDPStrategy(timeout=timedelta(days=30))

    model_config_dict = read_json(hparams.model_config)["model"]
    data_config_dict = read_json(hparams.model_config)["data"]

    RNNT_datamodule = RNNTransducerDataModule(data_config_dict, hparams)
    # (원문이 재밌어서 그대로 따옴) There are no .cuda() or .to(device) calls required. Lightning does these for you.
    # DDP를 사용한다면, distributed sampler를 default로 선언하여 사용하게 해줍니다. 필요시 재정의도 가능합니다.
    # torch.nn.Module 그대로이며, 추가된 사항만 있습니다. 찾아서 즐기세요!
    model = RNNTransducer(
        model_config_dict["prednet"], model_config_dict["transnet"], model_config_dict["jointnet"], hparams
    )
    trainer = Trainer.from_argparse_args(hparams)
    trainer.fit(model, datamodule=RNNT_datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", default=None, type=int, help="all seed")
    parser.add_argument("--local_rank", type=int, help="ddp local rank")
    parser.add_argument("--hf_data_dirs", nargs="+", default=[], type=str, help="source HuggingFace data dirs")
    parser.add_argument("--pl_data_dir", type=str, help="target pytorch lightning data dirs")
    parser.add_argument("--num_shards", type=int, help="target data shard cnt")
    parser.add_argument("--num_proc", type=int, default=None, help="how many proc map?")
    parser.add_argument("--cache_dir", type=str, default=None, help="datasets cache dir path")
    parser.add_argument("--model_config", type=str, help="data dirs")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument("--max_lr", default=0.01, type=float, help="lr_scheduler max learning rate")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="weigth decay")
    parser.add_argument(
        "--per_device_train_batch_size",
        default=1,
        type=int,
        help="The batch size per GPU/TPU core/CPU for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=1,
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    args = parser.parse_args()
    main(args)
