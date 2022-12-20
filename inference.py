import pytorch_lightning as pl
from model import RNNTransducer
from utils import read_json, InferenceArguments, get_concat_dataset, dataclass_to_namespace
import torch
from simple_parsing import ArgumentParser
from pyctcdecode import LanguageModel
import librosa
from time import time
import numpy as np
from torchaudio.transforms import MelSpectrogram


def main(hparams):
    # prednet_params, transnet_params, jointnet_params, args
    # prednet_params: dict, transnet_params: dict, jointnet_params: dict, args: Namespace
    datasets = get_concat_dataset(hparams.pl_data_dir, "eval_clean")
    datasets.set_format("torch", ["input_values", "input_ids"])
    model_config_dict = read_json(hparams.model_config)["model"]
    model = RNNTransducer.load_from_checkpoint(
        hparams.model_dir,
        prednet_params=model_config_dict["prednet"],
        transnet_params=model_config_dict["transnet"],
        jointnet_params=model_config_dict["jointnet"],
        args=hparams,
    )
    model.eval()
    start_time = time()
    # lm = None
    lm = LanguageModel.load_from_dir(hparams.lm_path)
    end_time = time()
    print(f"load_time: {end_time - start_time}")
    raw_audio, sr = librosa.load("./script1_i_0257-8001-01-01-KSH-F-07-A.wav", sr=16000)
    test = np.array([(raw_audio - raw_audio.mean()) / np.sqrt(raw_audio.var() + 1e-7)])
    # window_size는 통상 사람이 변화를 느끼는 한계인 25ms을 기본으로 합니다 (0.025)
    # 16000 * 0.025 = 400
    win_length = int(np.ceil(16000 * 0.025))
    # n_fft는 학습에 쓰이기 위한 max_length로 보면 됩니다. 해당 길이만큼 pad가 붙고, 모델과 관련이 있다는 점에서
    # 2의 n승 혹은 512, 1024 등의 값을 주로 쓰는데, win_length보다 크거나 같으면 되므로 저는 같게 설정.
    n_fft = win_length
    # 얼마만큼 밀면서 자를것이냐, (얼마만큼 겹치게 할 것이냐) 1부터 숫자에서 win_length가 10 hop_length를 2로 하면, 1~10 -> 3~13 과 같은 형태로 밀림.
    hop_length = int(16000 * 0.01)

    raw_audio = torch.FloatTensor([test[0]])

    # log_mel spec (channel(mono(1), 2~3 etc), n_mels, time)
    mel_spect = MelSpectrogram(
        sample_rate=16000, win_length=win_length, n_fft=n_fft, hop_length=hop_length, n_mels=80
    )(raw_audio)
    log_melspect = torch.log1p(mel_spect)[0]
    test_target = log_melspect.T
    input_audios = torch.stack([test_target], dim=0)
    # test_target = datasets[1]
    # input_audios = torch.stack([test_target["input_values"]], dim=0)
    input_lengths = [s.size(0) for s in input_audios]
    with torch.no_grad():
        pred_sentences = model.jointnet.recognize_beams(
            input_audios,
            input_lengths,
            model.tokenizer.pad_token_id,
            improved=True,
            beam_widths=5,
            lm=lm,
            tokenizer=model.tokenizer,
        )
    for pred_sentence in pred_sentences:
        print("출력:", model.tokenizer.decode(pred_sentence))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_arguments(InferenceArguments, dest="inference_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "inference_args")
    main(args)
