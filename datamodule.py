import os
import torch
import pytorch_lightning as pl
from utils import get_concat_dataset, get_cache_file_path, set_cache_log
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
from datasets import Dataset
from dataloader import AudioDataLoader

# 저는 1 batch로만 테스트되어서 group_by_length sampler가 오히려 느리기만 합니다.
# from transformers.trainer_pt_utils import DistributedLengthGroupedSampler

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
        self.num_proc = args.num_proc
        self.pad_token_id = config["audio"]["pad_token_id"]
        self.bos_token_id = config["text"]["bos_token_id"]
        self.window_stride_sec = config["audio"]["window_stride_sec"]
        self.window_size_sec = config["audio"]["window_size_sec"]
        self.sample_rate = config["audio"]["sample_rate"]
        self.window = WINDOWS.get(config["audio"]["window"], WINDOWS["hamming"])
        self.normalize = config["audio"]["normalize"]
        self.spec_augment = config["audio"]["spec_augment"]
        self.freq_mask_para = config["audio"]["freq_mask_para"]
        self.time_mask_para = config["audio"]["time_mask_para"]
        self.freq_mask_cnt = config["audio"]["freq_mask_cnt"]
        self.time_mask_cnt = config["audio"]["time_mask_cnt"]
        self.n_mels = config["audio"]["n_mels"]

    def raw_to_logmelspect(self, batch, idx):
        # window_size는 통상 사람이 변화를 느끼는 한계인 25ms을 기본으로 합니다 (0.025)
        # 16000 * 0.025 = 400
        win_length = int(np.ceil(self.sample_rate * self.window_size_sec))
        # n_fft는 학습에 쓰이기 위한 max_length로 보면 됩니다. 해당 길이만큼 pad가 붙고, 모델과 관련이 있다는 점에서
        # 2의 n승 혹은 512, 1024 등의 값을 주로 쓰는데, win_length보다 크거나 같으면 되므로 저는 같게 설정.
        n_fft = win_length
        # 얼마만큼 밀면서 자를것이냐, (얼마만큼 겹치게 할 것이냐) 1부터 숫자에서 win_length가 10 hop_length를 2로 하면, 1~10 -> 3~13 과 같은 형태로 밀림.
        hop_length = int(self.sample_rate * self.window_stride_sec)

        raw_audio = torch.FloatTensor([batch["input_values"]])

        # log_mel spec (channel(mono(1), 2~3 etc), n_mels, time)
        mel_spect = MelSpectrogram(
            sample_rate=self.sample_rate, win_length=win_length, n_fft=n_fft, hop_length=hop_length, n_mels=self.n_mels
        )(raw_audio)
        log_melspect = torch.log1p(mel_spect)

        if False:
            path = "./test_img"
            os.makedirs(path, exist_ok=True)
            plt.imsave("./test_img/" + str(idx) + ".png", log_melspect[0])

        batch["input_values"] = log_melspect
        return batch

    def spec_augmentation(self, batch):
        # torch.random.manual_seed(self.seed)
        # data shape: (channel, mel, seq)
        data = torch.FloatTensor(batch["input_values"])

        for _ in range(self.freq_mask_cnt):
            data = FrequencyMasking(freq_mask_param=self.freq_mask_para)(data)
        for _ in range(self.time_mask_cnt):
            data = TimeMasking(time_mask_param=self.time_mask_para)(data)
        # input_values shape: (channel, mel, seq)
        batch["input_values"] = data
        return batch

    def mean_var_norm(self, batch):
        data = np.array(batch["input_values"])
        batch["input_values"] = np.array([(data - data.mean()) / np.sqrt(data.var() + 1e-7)])[0]
        return batch

    def save_raw_to_logmelspect_datasets(
        self, source_dataset_dirs: list[str], target_dataset_dir: str, train_type: str
    ) -> Dataset:
        """
        기존에 raw로 저장해놓은 huggingface datasets를 불러와, log melspectrogram 형태로 map합니다. \n
        이미 log melspectrogram datasets가 있다면, 기존 것을 가져다 씁니다. \n
        성공 시, datasets를 shard하여 저장하고, 전체 datasets을 return합니다.
        """
        assert source_dataset_dirs, "log mel로 변환할 data source 경로가 없습니다!"
        assert target_dataset_dir, "log mel을 저장할 data target 경로가 없습니다!"

        print("check dir: ", os.path.join(target_dataset_dir, train_type))
        if os.path.isdir(os.path.join(target_dataset_dir, train_type)):
            # target이 이미 존재하면, 새로 만들지 않고, 불러옵니다.
            print(f"이미 datasets이 존재합니다. {train_type} save과정은 스킵됩니다!!!")
        else:
            datasets = get_concat_dataset(source_dataset_dirs, train_type)

            if train_type == "train":
                num_shards = self.num_shards
            else:
                num_shards = 1

            if self.normalize:
                normalize_task_name = "normalize"
                spec_aug_task_name = "norm_spec_augmentation"
                transpose_task_name = "norm_transpose"
                for source_dataset_dir in source_dataset_dirs:
                    cache_file_name = get_cache_file_path(source_dataset_dir, normalize_task_name, train_type)
                    # HuggingFace Style MeanVarNorm
                    datasets = datasets.map(
                        self.mean_var_norm, cache_file_name=cache_file_name, num_proc=self.num_proc
                    )
                    set_cache_log(
                        dataset_dir=source_dataset_dir,
                        num_proc=self.num_proc,
                        cache_task_func_name=normalize_task_name,
                        train_type=train_type,
                    )
            else:
                spec_aug_task_name = "spec_augmentation"
                transpose_task_name = "transpose"
            datasets = datasets.map(
                self.raw_to_logmelspect,
                num_proc=self.num_proc,
                with_indices=True,
                remove_columns=["length", "syllabel_labels"],
            )

            if train_type == "train":
                cache_file_name = get_cache_file_path(target_dataset_dir, spec_aug_task_name, train_type)
                datasets = datasets.map(
                    self.spec_augmentation, cache_file_name=cache_file_name, num_proc=self.num_proc
                )
                set_cache_log(
                    dataset_dir=target_dataset_dir,
                    num_proc=self.num_proc,
                    cache_task_func_name=spec_aug_task_name,
                    train_type=train_type,
                )

            cache_file_name = get_cache_file_path(self.pl_data_dir, transpose_task_name, train_type)
            datasets = datasets.map(
                lambda batch: {
                    "input_values": np.transpose(batch["input_values"][0]),
                    "input_ids": batch["grapheme_labels"]["input_ids"],
                    "audio_len": len(np.transpose(batch["input_values"][0])),
                    "label_len": len(batch["grapheme_labels"]["input_ids"]),
                },
                cache_file_name=cache_file_name,
                num_proc=self.num_proc,
                remove_columns=["grapheme_labels"],
            )
            set_cache_log(
                dataset_dir=target_dataset_dir,
                num_proc=self.num_proc,
                cache_task_func_name=transpose_task_name,
                train_type=train_type,
            )
            for shard_idx in range(num_shards):
                shard_datasets = datasets.shard(num_shards=num_shards, index=shard_idx)
                shard_datasets.save_to_disk(os.path.join(target_dataset_dir, train_type, str(shard_idx)))

    def prepare_data(self):
        if self.hf_data_dirs:
            self.save_raw_to_logmelspect_datasets(self.hf_data_dirs, self.pl_data_dir, "train")
            self.save_raw_to_logmelspect_datasets(self.hf_data_dirs, self.pl_data_dir, "dev")
            self.save_raw_to_logmelspect_datasets(self.hf_data_dirs, self.pl_data_dir, "eval_clean")
            self.save_raw_to_logmelspect_datasets(self.hf_data_dirs, self.pl_data_dir, "eval_other")
        else:
            print("hf_data_dirs 파라미터가 없습니다. prepare_data() 작업은 무시됩니다.")
            pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_datasets = get_concat_dataset([self.pl_data_dir], "train")
            self.train_datasets.set_format("torch", ["input_values", "input_ids"])
            self.val_datasets = get_concat_dataset([self.pl_data_dir], "dev")
            self.val_datasets.set_format("torch", ["input_values", "input_ids"])

        if stage == "test":
            self.clean_datasets = get_concat_dataset([self.pl_data_dir], "eval_clean")
            self.clean_datasets.set_format("torch", ["input_values", "input_ids"])
            self.other_datasets = get_concat_dataset([self.pl_data_dir], "eval_other")
            self.other_datasets.set_format("torch", ["input_values", "input_ids"])

    def train_dataloader(self):
        # setup에서 완성된 datasets를 여기서 사용하십시오. trainer의 fit() method가 사용합니다.
        return AudioDataLoader(
            dataset=self.train_datasets,
            batch_size=self.per_device_train_batch_size,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            n_mels=self.n_mels,
            num_workers=self.num_proc,
            pin_memory=True,
        )

    def val_dataloader(self):
        # setup에서 완성된 datasets를 여기서 사용하십시오. trainer의 fit(), validate() method가 사용합니다.
        return AudioDataLoader(
            dataset=self.val_datasets,
            batch_size=self.per_device_eval_batch_size,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            n_mels=self.n_mels,
            num_workers=self.num_proc,
            pin_memory=True,
        )

    def test_dataloader(self):
        # setup에서 완성된 datasets를 여기서 사용하십시오. trainer의 test() method가 사용합니다.
        return [
            AudioDataLoader(
                dataset=self.clean_datasets,
                batch_size=1,
                pad_token_id=self.pad_token_id,
                bos_token_id=self.bos_token_id,
                n_mels=self.n_mels,
                num_workers=self.num_proc,
                pin_memory=True,
            ),
            AudioDataLoader(
                dataset=self.other_datasets,
                batch_size=1,
                pad_token_id=self.pad_token_id,
                bos_token_id=self.bos_token_id,
                n_mels=self.n_mels,
                num_workers=self.num_proc,
                pin_memory=True,
            ),
        ]

    def predict_dataloader(self):
        # setup에서 완성된 datasets를 여기서 사용하십시오. trainer의 predict() method가 사용합니다.
        pass
