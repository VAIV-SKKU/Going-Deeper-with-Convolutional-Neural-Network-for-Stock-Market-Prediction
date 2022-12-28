# Going-Deeper-with-Convolutional-Neural-Network-for-Stock-Market-Prediction
> This is a validatioin experiment

> 논문 링크 : https://arxiv.org/pdf/1903.12258.pdf

> github 링크 : https://github.com/rosdyana/Going-Deeper-with-Convolutional-Neural-Network-for-Stock-Market-Prediction


## Dataset
+ Taiwan 50 종목에 대한 차트 이미지 생성 및 레이블링
    + Image features : OHLC, Volume
    + Image size : 50x50
    + Labeling(binary) : 상승 시 1, 하락 시 0으로 레이블링
    + Tradind period : 20 days
    + Forecasting interval : + 1 day    
+ Train 기간 : 2000.01.01 ~ 2016.12.31 (총 190,433개의 이미지)
    + Label 0 : 105,374개의 이미지
    + Label 1 : 85,059개의 이미지
+ Test 기간 : 2017.01.01 ~ 2018.06.14 (총 15,788개의 이미지)
    + Label 0 : 8,826개의 이미지
    + Label 1 : 6,962개의 이미지


## Train / Test
사용 모델 : VGG16, DeepCNN(15 Layers), ResNet50, Pretrained VGG16, Pretrained ResNet50

* Pretrained 모델의 경우 ImageNet으로 학습된 모델 사용


## 실험 결과
Taiwan50을 이용하여 DeepCNN, VGG16, ResNet50 세 가지 모델에 대하여 실험을 진행하였고, 동일 조건의 모든 결과에서 VGG16이 가장 좋은 성능을 보였음.

<img width="603" alt="스크린샷 2022-12-27 오전 2 21 16" src="https://user-images.githubusercontent.com/100757275/209571310-7b977ffe-b86f-442e-b8d7-146cbbf26317.png">

## Usage
### Environment
### Prepare Dataset
### Training
