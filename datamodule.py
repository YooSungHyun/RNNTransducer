import os
import torch
import pytorch_lightning as pl
from utils import get_concat_dataset
from torch.utils.data import random_split, DataLoader
import numpy as np
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking
import matplotlib.pyplot as plt
from argparse import Namespace
from datasets import concatenate_datasets

WINDOWS = {
    "hamming": torch.hamming_window,
    "hann": torch.hann_window,
    "blackman": torch.blackman_window,
    "bartlett": torch.bartlett_window,
}


class RNNTransducerDataModule(pl.LightningDataModule):
    # https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#datamodules
    def __init__(self, config: dict, args: Namespace):
        super().__init__()
        self.hf_data_dirs = args.hf_data_dirs
        self.pl_data_dirs = args.pl_data_dirs
        # 명심!!!!!
        self.train_shard_cnt = int(args.pl_data_dirs.split("-")[-1])
        self.seed = args.seed
        self.window_stride_sec = config["audio"]["window_stride_sec"]
        self.window_size_sec = config["audio"]["window_size_sec"]
        self.sample_rate = config["audio"]["sample_rate"]
        self.window = WINDOWS.get(config["audio"]["window"], WINDOWS["hamming"])
        self.normalize = config["audio"]["normalize"]
        self.spec_augment = config["audio"]["spec_augment"]
        self.freq_mask_para = config["audio"]["freq_mask_para"]
        self.time_mask_para = config["audio"]["time_mask_para"]
        self.n_mels = config["audio"]["n_mels"]

    def __parse_audio(self, batch, idx):
        # window_size는 통상 사람이 변화를 느끼는 한계인 25ms을 기본으로 합니다 (0.025)
        # 16000 * 0.025 = 400
        win_length = int(np.ceil(self.sample_rate * self.window_size_sec))
        # n_fft는 학습에 쓰이기 위한 max_length로 보면 됩니다. 해당 길이만큼 pad가 붙고, 모델과 관련이 있다는 점에서
        # 2의 n승 혹은 512, 1024 등의 값을 주로 쓰는데, win_length보다 크거나 같으면 되므로 저는 같게 설정.
        n_fft = win_length
        # 얼마만큼 밀면서 자를것이냐, (얼마만큼 겹치게 할 것이냐) 1부터 숫자에서 win_length가 10 hop_length를 2로 하면, 1~10 -> 3~13 과 같은 형태로 밀림.
        hop_length = int(self.sample_rate * self.window_stride_sec)

        x = batch["input_values"]
        if self.normalize:
            # HuggingFace Style MeanVarNorm
            x = [(x - x.mean()) / np.sqrt(x.var() + 1e-7)]

        # log_mel spec (channel(mono(1), 2~3 etc), n_mels, time)
        mel_spec = MelSpectrogram(
            sample_rate=self.sample_rate, win_length=win_length, n_fft=n_fft, hop_length=hop_length, n_mels=self.n_mels
        )(torch.Tensor(x))
        x = np.log1p(mel_spec)

        if self.spec_augment:
            torch.random.manual_seed(self.seed)
            x = FrequencyMasking(freq_mask_param=self.freq_mask_para)(x)
            x = TimeMasking(time_mask_param=self.time_mask_para)(x)

        if False:
            path = "./test_img"
            os.makedirs(path, exist_ok=True)
            plt.imsave("./test_img/" + str(idx) + ".png", x[0])
        batch["input_values"] = x
        return batch

    def save_raw_to_melspect_datasets(self, source_dataset, train_type):
        # 기존에 raw로 저장해놓은 huggingface datasets를 불러와, log melspectrogram 형태로 map합니다.
        # 성공 시, datasets를 shard하여 저장하고, 전체 datasets을 return합니다.
        datasets = get_concat_dataset(self.hf_data_dirs, train_type)
        if train_type == "train":
            shard_cnt = self.train_shard_cnt
        else:
            shard_cnt = 1

        dataset_lists = []
        for shard_idx in range(shard_cnt):
            shard_datasets = datasets.shard(num_shards=shard_cnt, index=shard_idx)
            shard_datasets = shard_datasets.map(self.__parse_audio, with_indices=True)
            shard_datasets.save_to_disk(os.path.join(self.pl_data_dirs, train_type, str(shard_idx)))
            dataset_lists.append(shard_datasets)

        concat_dataset = concatenate_datasets(dataset_lists)
        concat_dataset.set_format("torch")
        return concat_dataset

    def prepare_data(self):
        # prepare에서는 기존에 HuggingFace에서 load_dataset -> map 작업들을 하면 됩니다.
        # 여기서 진행하는 작업은 전체 데이터셋을 기준으로 합니다. 꼭 한번에 전체를 해야할지 잘 고민하십시오.
        # 음성은 collate에서 하면 더 오래걸릴 소요가 있어서 여기서 다하고 저장할란다...
        if os.path.isdir(self.pl_data_dirs):
            self.train_datasets = get_concat_dataset(self.pl_data_dirs, "train")
            self.train_datasets.set_format("torch")
            self.dev_datasets = get_concat_dataset(self.pl_data_dirs, "dev")
            self.dev_datasets.set_format("torch")
            self.clean_datasets = get_concat_dataset(self.pl_data_dirs, "clean")
            self.clean_datasets.set_format("torch")
            self.other_datasets = get_concat_dataset(self.pl_data_dirs, "other")
            self.other_datasets.set_format("torch")
        else:
            self.train_datasets = self.save_raw_to_melspect_datasets(self.hf_data_dirs, "train")
            self.dev_datasets = self.save_raw_to_melspect_datasets(self.hf_data_dirs, "dev")
            self.clean_datasets = self.save_raw_to_melspect_datasets(self.hf_data_dirs, "clean")
            self.other_datasets = self.save_raw_to_melspect_datasets(self.hf_data_dirs, "other")

    def setup(self, stage: str):
        # 각각의 split stage에 맞는 torch형 datasets를 구성합니다.
        # 모든 데이터셋을 한번에 읽어올 필요가 없다면, stage를 pass하거나 None으로 정의하십시오. (아마 콜레터를 쓰거나, train, test step에서 하고싶을 수 있으니까?)
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_datasets, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.dev_datasets, batch_size=32)

    def test_dataloader(self):
        return [DataLoader(self.clean_datasets), DataLoader(self.other_datasets)]
