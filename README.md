# RNNTransducer
Streaming 가능한 RNN Transducer 모델을 PyTorch Lightning으로 구현해본다. (거의 Conversion에 가까울듯) <br />
paper reference: https://arxiv.org/abs/1211.3711 <br />
torch-lightning: https://www.pytorchlightning.ai/ <br />
torch-lightning dev guide: [https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html](https://pytorch-lightning.readthedocs.io/en/latest/) <br />
![1](https://user-images.githubusercontent.com/34292279/199925209-29902b23-1b8f-403e-88c5-439afa8a8165.png)


torch 1.11 이상 cuda 11이상 사용한다면 warprnnt_loss는 해당 git을 clone하여 설치하여야함. <br />
git clone -b espnet_v1.1 --single-branch https://github.com/YooSungHyun/torch1.11_warp-transducer.git
<br />
<br />

데이터는 KsponSpeech(음성전사)를 활용하였으며, KsponSpeech를 HuggingFace Datasets로 구성하여 사용되었습니다. <br />
개인적인 이유로 datasets를 공개해드릴 수는 없습니다. Pytorch Datasets을 사용해도 동일하니, 각자 입맛에 맞도록 처리하시는 것을 권장합니다. <br />
> Dataset({ <br />
> &nbsp;&nbsp;&nbsp;&nbsp;features: ['input_values', 'grapheme_labels', 'length'], <br />
> &nbsp;&nbsp;&nbsp;&nbsp;num_rows: 620000 <br />
> }) <br />

**input_values**: 평균-분산 정규화가 진행되지 **않은!** raw 음성입니다. librosa로 load한 16000 sr의 음성 float32값입니다. <br />
**grapheme_lables**: 자소 정답값입니다. unicodedata lib을 이용하여 전처리 한 후, vocab tokenize 진행하였습니다. <br />
&nbsp;&nbsp;&nbsp;&nbsp;**input_ids**: tokenize 진행된 자소값입니다.<br />
**length**: len(input_values) 입니다. HuggingFace에서 사용하던 SmartBatching용 데이터를 쓰다보니 들어있네요. <br />

> HuggingFace의 Audio FeatureExtractor와 Tokenizer를 사용해보신 분들이라면, 최대한 익숙할 수 있도록 naming하여 사용되었습니다. <br />

datamodule.py에서 하는 역할은 크게 2가지입니다.
1. 위의 raw 음성을, log melspectrogram으로 변환합니다. `load_raw_to_melspect_datasets`
   1. 해당 dataset을 저장합니다. (이미 경로가 있다면 불러서 사용)
2. 작성중
