import os
import torch
import pytorch_lightning as pl
from utils import get_concat_dataset
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
from datasets import Dataset

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
        self.pl_data_dir = args.pl_data_dir
        self.num_shards = args.num_shards
        self.seed = args.seed
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size
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

        raw_audio = torch.FloatTensor([batch["input_values"]])
        if self.normalize:
            # HuggingFace Style MeanVarNorm
            raw_audio = (raw_audio - raw_audio.mean()) / torch.sqrt(raw_audio.var() + 1e-7)

        # log_mel spec (channel(mono(1), 2~3 etc), n_mels, time)
        mel_spect = MelSpectrogram(
            sample_rate=self.sample_rate, win_length=win_length, n_fft=n_fft, hop_length=hop_length, n_mels=self.n_mels
        )(raw_audio)
        log_melspect = torch.log1p(mel_spect)

        if self.spec_augment:
            torch.random.manual_seed(self.seed)
            log_melspect = FrequencyMasking(freq_mask_param=self.freq_mask_para)(log_melspect)
            log_melspect = TimeMasking(time_mask_param=self.time_mask_para)(log_melspect)

        if False:
            path = "./test_img"
            os.makedirs(path, exist_ok=True)
            plt.imsave("./test_img/" + str(idx) + ".png", log_melspect[0])
        batch["input_values"] = log_melspect
        return batch

    def __save_raw_to_melspect_datasets(
        self, source_dataset_dir: str, target_dataset_dir: str, train_type: str
    ) -> Dataset:
        """
        기존에 raw로 저장해놓은 huggingface datasets를 불러와, log melspectrogram 형태로 map합니다. \n
        이미 log melspectrogram datasets가 있다면, 기존 것을 가져다 씁니다. \n
        성공 시, datasets를 shard하여 저장하고, 전체 datasets을 return합니다.
        """
        assert source_dataset_dir is not None, "log mel로 변환할 data source 경로가 없습니다!"
        assert target_dataset_dir is not None, "log mel을 저장할 data target 경로가 없습니다!"

        if os.path.isdir(target_dataset_dir):
            # target이 이미 존재하면, 새로 만들지 않고, 불러옵니다.
            print("이미 datasets이 존재합니다. save과정은 스킵됩니다!!!")
        else:
            datasets = get_concat_dataset(source_dataset_dir, train_type)

            if train_type == "train":
                num_shards = self.num_shards
            else:
                num_shards = 1

            dataset_lists = []
            for shard_idx in range(num_shards):
                shard_datasets = datasets.shard(num_shards=num_shards, index=shard_idx)
                shard_datasets = shard_datasets.map(self.__parse_audio, with_indices=True)
                shard_datasets.save_to_disk(os.path.join(target_dataset_dir, train_type, str(shard_idx)))
                dataset_lists.append(shard_datasets)

    def prepare_data(self):
        # prepare에서는 병렬처리 복잡도를 낮추고, 혹은 병렬처리 시 데이터가 손상될 위협을 방지하기위해 단일 CPU 프로세스로만 진행이 됩니다. (다중노드에서 사용하려면, prepare_data_per_node를 참고)
        # 따라서, prepare에서 class 변수에 특정 상태를 저장하지 마십시오. 단일 프로세스로만 진행되므로, 다른 프로세스에서 참고할 수 없습니다.

        # 데이터를 다운받고, 토크나이즈하고, 파일 시스템에 저장하는 task를 수행하십시오.
        # TODO: 개인적으로 궁금한게 Streaming으로 데이터를 다운로드받는 와중에 학습시킨다면, prepare_data는 pass되어야 할까요?

        # 여기서 진행하는 작업은 전체 데이터셋을 기준으로 합니다. 꼭 한번에 전체를 해야할지 잘 고민하십시오.
        # log mel spectrogram으로 변환하는게 마침 1회만 진행하면 되고, 있으면 무시되면 되서 여기에 딱 알맞는 Task입니다.
        # 사실 더 명확하게 하려면, 여기서는 raw_audio의 datasets를 다운받는 정도의 Task만 진행하고, log mel spectrogram 변환은 setup에서 하는게 더 나을지도 모르겠습니다. (GPU가 더 빠르려나?)
        self.__save_raw_to_melspect_datasets(self.hf_data_dirs, self.pl_data_dir, "train")
        self.__save_raw_to_melspect_datasets(self.hf_data_dirs, self.pl_data_dir, "dev")
        self.__save_raw_to_melspect_datasets(self.hf_data_dirs, self.pl_data_dir, "clean")
        self.__save_raw_to_melspect_datasets(self.hf_data_dirs, self.pl_data_dir, "other")

    def setup(self, stage: str):
        # prepare_data가 정상적으로 동작되면 호출됩니다. setup은 각 모든 GPU에서 호출됩니다.
        # 사용 가능한 모든 데이터들에 대한 모든 프로세스의 setup은 1번만 진행되도록 보장되어있습니다. (병렬처리하더라도, 1 data 1 setup을 보장한다는 의미같음.)
        # 분류할 개수를 세거나, vocab을 만들거나, train/val/test splits을 하거나, datasets를 만들(선언하)거나, transforms를 수행해야 하거나
        # 각각의 split stage에 맞는 torch형 datasets를 구성합니다.
        # 모든 데이터셋을 한번에 읽어올 필요가 없다면, stage를 pass하거나 None으로 정의하십시오. (아마 콜레터를 쓰거나, train, test step에서 하고싶을 수 있으니까?)
        # stage must like {fit,validate,test,predict}
        if stage == "fit":
            self.train_datasets = get_concat_dataset(self.pl_data_dir, "train")
            self.train_datasets = self.train_datasets.set_format("torch")
            self.val_datasets = get_concat_dataset(self.pl_data_dir, "dev")
            self.val_datasets = self.val_datasets.set_format("torch")

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.clean_datasets = get_concat_dataset(self.pl_data_dir, "eval_clean")
            self.clean_datasets = self.clean_datasets.set_format("torch")
            self.other_datasets = get_concat_dataset(self.pl_data_dir, "eval_other")
            self.other_datasets = self.other_datasets.set_format("torch")

    def train_dataloader(self):
        # setup에서 완성된 datasets를 여기서 사용하십시오. trainer의 fit() method가 사용합니다.
        return DataLoader(self.train_datasets, batch_size=self.per_device_train_batch_size)

    def val_dataloader(self):
        # setup에서 완성된 datasets를 여기서 사용하십시오. trainer의 fit(), validate() method가 사용합니다.
        return DataLoader(self.val_datasets, batch_size=self.per_device_eval_batch_size)

    def test_dataloader(self):
        # setup에서 완성된 datasets를 여기서 사용하십시오. trainer의 test() method가 사용합니다.
        return [
            DataLoader(self.clean_datasets, batch_size=1),
            DataLoader(self.other_datasets, batch_size=1),
        ]

    def predict_dataloader(self):
        # setup에서 완성된 datasets를 여기서 사용하십시오. trainer의 predict() method가 사용합니다.
        pass
