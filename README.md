# RNNTransducer
Streaming 가능한 RNN Transducer 모델을 PyTorch Lightning으로 구현해본다. (거의 Conversion에 가까울듯)
https://arxiv.org/abs/1211.3711
![1](https://user-images.githubusercontent.com/34292279/199925209-29902b23-1b8f-403e-88c5-439afa8a8165.png)


torch 1.11 이상 cuda 11이상 사용한다면 warprnnt_loss는 해당 git을 clone하여 설치하여야함. <br />
git clone -b espnet_v1.1 --single-branch https://github.com/b-flo/warp-transducer.git
