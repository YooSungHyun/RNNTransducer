import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
import collections
import scipy
import random
from utils import get_concat_dataset, comfy
from pytorch_lightning import Trainer
from model import RNNTransducer
from datamodule import RNNTransducerDataModule


def main(hparams):
    random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)

    model_config_dict = comfy.read_json(hparams.model_config)["model"]
    data_config_dict = comfy.read_json(hparams.model_config)["data"]
    model = RNNTransducer(
        model_config_dict["prednet"], model_config_dict["transnet"], model_config_dict["jointnet"], hparams
    )

    trainer = Trainer.from_argparse_args(args)
    RNNT_datamodule = RNNTransducerDataModule(data_config_dict, hparams)
    trainer.fit(model, datamodule=RNNT_datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", default=None, type=int, help="all seed")
    parser.add_argument("--local_rank", type=int, help="ddp local rank")
    parser.add_argument("--hf_data_dirs", nargs="+", default=[], type=str, help="source HuggingFace data dirs")
    parser.add_argument("--pl_data_dirs", nargs="+", default=[], type=str, help="target pytorch lightning data dirs")
    parser.add_argument("--model_config", type=str, help="data dirs")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="ddp local rank")
    parser.add_argument("--max_lr", default=0.01, type=float, help="ddp local rank")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="ddp local rank")
    args = parser.parse_args()
    main(args)
