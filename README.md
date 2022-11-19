# RNNTransducer
Streaming 가능한 RNN Transducer 모델을 PyTorch Lightning으로 구현해본다. (거의 Conversion에 가까울듯) <br />
paper reference: https://arxiv.org/abs/1211.3711 <br />
torch-lightning: https://www.pytorchlightning.ai/ <br />
torch-lightning dev guide: [https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html](https://pytorch-lightning.readthedocs.io/en/latest/) <br />
![1](https://user-images.githubusercontent.com/34292279/199925209-29902b23-1b8f-403e-88c5-439afa8a8165.png)


torch 1.11 이상 cuda 11이상 사용한다면 warprnnt_loss는 해당 git을 clone하여 설치하여야함. <br />
git clone -b espnet_v1.1 --single-branch https://github.com/YooSungHyun/warp-transducer.git
<br />
<br />

# Data & Datasets
데이터는 KsponSpeech(음성전사)를 활용하였으며, KsponSpeech를 HuggingFace Datasets로 구성하여 사용되었습니다. <br />
개인적인 이유로 datasets를 공개해드릴 수는 없습니다. Pytorch Datasets을 사용해도 동일하니, 각자 입맛에 맞도록 처리하시는 것을 권장합니다. <br />
해당 소스는 KsponSpeech로 아래와 같은 Datasets가 기 만들어져 있어야 합니다. <br />
> Dataset({ <br />
> &nbsp;&nbsp;&nbsp;&nbsp;features: ['input_values', 'grapheme_labels', 'length'], <br />
> &nbsp;&nbsp;&nbsp;&nbsp;num_rows: 620000 <br />
> }) <br />
해당 소스는, HuggingFace 기반의 Datasets으로만 Test 되었습니다. <br />

**input_values**: 평균-분산 정규화가 진행되지 **않은!** raw 음성입니다. librosa로 load한 16000 sr의 음성 float32값입니다. <br />
**grapheme_lables**: 자소 정답값입니다. unicodedata lib을 이용하여 전처리 한 후, vocab tokenize 진행하였습니다. <br />
&nbsp;&nbsp;&nbsp;&nbsp;**input_ids**: tokenize 진행된 자소값입니다.<br />
**length**: len(input_values) 입니다. HuggingFace에서 사용하던 SmartBatching용 데이터를 쓰다보니 들어있네요. <br />

- HuggingFace의 Audio FeatureExtractor와 Tokenizer를 사용해보신 분들이라면, 최대한 익숙할 수 있도록 naming하여 사용되었습니다. <br />
- 데이터는 Shard되어 처리되어 있어야 합니다. <br />
    - TB 단위 데이터를 학습시키다보면, shard 미진행 시, 학습간, IO 병목을 겪을 수 있습니다. <br />
- KsponSpeech는 20 shard정도가 적절 <br />
    - Train 20, Eval 1, Clean/Other 1 shard로 만들어 놨습니다. <br />
- Default Datasets(HF_DATA_DIRS) 은, 꼭! shard된 상위 구조의 Dataset을 먼저 만들어 주세요. <br />

# Scripts
필수적인 것만 설명하겠습니다. <br />
- **torchrun**: 가장 중요합니다. 해당 프로젝트는 DP를 지원하지 않습니다. 무조건 DDP로만 동작 가능합니다. <br />
- HF_DATA_DIRS: 위에 설명된 raw datasets 입니다. 각각 엔터로 구분하여 넣으면 되는데, KsponSpeech 단일 프로젝트만 Test 완료되었습니다. <br />
- PL_DATA_DIR: HF_DATA_DIRS로 log_melspect 변환과 spec_arguments를 진행할 전처리 output 경로입니다. <br />
- vocab_path: vocab_path 입니다. 모호한 숫자와 영어 등을 전부 제거하고 vocab을 뽑으면 72개가 나올 것입니다. <br />
- num_proc: 전처리를 진행할 프로세스 수입니다. cpu상황을 고려해서 선택하십시오. <br />
- num_shard: shard를 진행할 개수입니다. KsponSpeech의 경우 20이 적당합니다. <br />

# datamodule
1. prepare_data()
    - 위의 raw 음성 -> 평균분산 정규화 -> log melspectrogram(channel,mel,seq) -> spec aguments (channel,mel,seq) -> transpose (seq,mel)
    - Training Code의 일관성을 유지하기위해서 transpose까지 전부 시켜서 저장하며, spec_aguments와 실험용으로 plt 출력이 (c,m,s) 순이여야만 동작하므로 위와같이 처리합니다.
2. setup()
    - stage별로 데이터를 불러옵니다.
3. 각각 dataloader()
    - Dynamic Padding, Smart Batching을 구현하기위해, `DistributedSampler`를 상속받아 만들어진 HuggingFace Transformers의 `DistributedGroupedSampler`를 활용합니다.
    - 때문에 DP에서의 동작은 여기서부터 이상해질 수 있습니다.
    - `DistributedSampler`의 상속체면, `replace_sampler_ddp=False`를 사용하지 않아도, 재정의된 sampler를 찾아갑니다. (다만 조금 불안해서 scripts에는 확실하게 명명해놓음)

# model.py
Optimize, lr_scheduler, lr_scheduler를 위한 Epoch당 Steps 수 계산 등 학습에 실질적으로 필요한 소스들이 모여있습니다. <br />
`training_step` -> `self.forward()` -> `training_step_end`(사실 training_step이랑 합쳐도 될듯) -> `optimize_step` -> `validateion_step` -> `validation_epoch_step` <br />
순으로 진행됩니다. <br />
## BPTT (BackPropagation Through Time)
RNNT_loss를 사용하면 BPTT 역전파는 고려대상이 아닐 수 있습니다. <br />
https://github.com/fd873630/RNN-Transducer/issues/6 <br />
혹시 몰라 더럽더라도, BPTT와 관련된 주석은 최대한 남겨놓고, 소스는 주석처리하였습니다. <br />
<br />
아쉬운 점이 있다면, pad_packed의 lengths와 loss 계산의 lengths가 각각 cpu, gpu에서만 동작하여, 넣었다 뺐다 하는데 좀 손해를 보고, 코드를 잘 짠거같은데 GPU Utils가 크게 좋지는 않습니다.

# Inference

# Result

