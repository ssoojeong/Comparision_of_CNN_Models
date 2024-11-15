# Comparision_of_CNN_Models

## &#x1F4E2; Project Overview: 2310~2311

### 주요 기능
- 대표적인 4가지 CNN 모델 (AlexNet, VGG, GoogLeNet, ResNet) 성능 비교
- PyTorch를 활용하여 CIFAR-10 데이터셋에서 Classificaation 성능 실험

----

### &#x1F31F; CNN 모델 리스트
| 모델       | 발표 연도 | 발표 학회         | 논문 제목                                             |
|------------|-----------|------------------|-------------------------------------------------------|
| **AlexNet**   | 2012년    | NIPS (NeurIPS)   | *ImageNet Classification with Deep Convolutional Neural Networks* |
| **VGG**       | 2014년    | ICLR             | *Very Deep Convolutional Networks for Large-Scale Image Recognition* |
| **GoogLeNet** | 2014년    | CVPR             | *Going Deeper with Convolutions*                      |
| **ResNet**    | 2015년    | CVPR             | *Deep Residual Learning for Image Recognition*        |


### 🧪 평가 지표
- Accuracy (정확도): 이미지 분류 정확도
- Trian Time (소요시간/에폭): 한 에폭당 소요된 시간

### 📝 데이터셋
#### CIFAR-10
- 10개의 클래스
- 32x32 컬러 이미지
- Train dataset: 50,000장
- Test dataset: 10,000장

----

### 💻 실험 방법

#### 1. 가상 환경 설정

```bash
conda create -n cnns python=3.9
conda activate cnns

pip install -r requirements.txt
```

#### 2. 모델 훈련 및 인퍼런스
```bash
# 1. GLIDE
python train.py \
    --model {모델명} \ #e.g., alexnet, vgg16 ... \
    --epochs 10 #defualt: 25
    --gpus 0 \
```
> **모델명**: resnet18, resnet50, vgg16, alexnet, inception_v3, googlenet, mobilenet_v2, densenet121

#### 3. 모델 학습 log 및 성능 확인
log 저장 경로: ```./results/log/```

```bash
#e.g., AlexNet, 25epoch

############학습 시작############
Epoch 0/24
----------
train Loss: 9.1037 Acc: 0.1560
val Loss: 11.9101 Acc: 0.1925
Epoch 1/24
----------

...

############학습 끝############
Training complete in 3m 28s
Best val Acc: 0.449057
```

---

### 📄 프로젝트 요약 보고서

대표적인 합성곱 신경망 모델의 비교
[[paper]](./CNN_report.pdf) 

Comparision of representive Convolution Neural Network models

(권하연, 김소정, 배인우, 이강은)
