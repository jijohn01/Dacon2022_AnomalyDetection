# Dacon2022_AnomalyDetection
Dacon / Computer Vision 이상치 탐지 알고리즘 경진대회/Private 9위

Public 점수 기준으로 가장 성능이 좋았던 모델은 model 폴더 안에 저장되어있습니다.

Public기준 가장 성능이 좋았던 모델은 01.EfficientNet_5fold.ipynb, 02.MixUp.ipynb로 학습시키면 됩니다.
03.Ensemble_predict.ipynb 실행 시 학습한 모델을 사용하여 동일한 결과를 생성할 수 있습니다.

다른 제출본은 이미지 사이즈를 (700,700)으로 키우면서 colab에서 메모리 초과 이슈가 발생해 local 환경에서 학습시켰으며 해당 코드는 "local" 폴더 안에 있습니다.

# Summary

## Baseline

제공된 baseline 코드를 기반으로 주로 Colab pro를 활용하였습니다.
Supervised Anomaly Detection문제이므로 Classification 문제 해결 방식으로 접근하였습니다.

## Data Size

Data는 load시 (512,512)로 resize하여 (498,498)로 randomcrop할 때 성능이 가장 좋았습니다.

(700,700)으로 resize하여 학습해보았지만 결과적으로 public 점수에서 이득을 얻지 못했습니다.
이미지 사이즈가 너무커서 Colab으로 돌리기 어려웠고 local에서 학습해봤지만 시간이 너무 오래 걸려 여러번 실험해 보지 못했습니다.

## Data Augmentation

기본적으로 Flip, Rotate, Randomcrop은 성능 향상에 도움이 되었습니다.

추가로 약간의 Distortion, Cutout을 소극적으로 적용했고, 6개의 model중 하나는 약간의 mixup을 추가하여 학습했을 때 가장 성능이 좋았습니다.
그 밖의 여러가지 방법으로 Augmentation을 시도했지만 Data특성상 Augmentation이 조금만 과해도 성능 저하가 나타났습니다.
(예를 들면, Cutmix, puzzlemix, scale 등)

 일부 class의 경우 Augmentation을 더욱 소극적으로 사용해야할 것으로 보였습니다.
 예를 들면, metal_nut의 경우 flip이 이상치 클래스 중에 있어 augmentation시 flip을 하면 학습에 방해가 될 것으로 보였습니다.
 두번째 제출본은 Class별로 다르게 Augmentation하도록 했지만 public에서 눈에 띄는 성능향상은 얻지 못했습니다. 실험 횟수가 부족한 탓이 있을 수 있을 것 같습니다.

## Data Imbalance

출처 : https://github.com/ufoym/imbalanced-dataset-sampler

이상치 데이터의 갯수가 정상에 비해 매우 적으므로 Oversampling, Undersampling을 적절하게 배분하는 ImbalanceSampler를 사용하였습니다. 
Loss function에 가중치를 주어 Weighted Cross entropy로 학습시켜보았으나 눈에 띄이는 성능향상은 얻지 못했습니다.

## Model

샘플이 5개 밖에 없는 class가 존재하였으므로 Stratified K-Fold를 사용하여 fold를 5개로 나눴습니다.

MixUp이 없이 5개의 모델을 각각 학습시켜 앙상블 했을 때 성능이 좋았습니다. 여기에 추가로 MixUp을 약하게 적용하여 학습한 모델 1개를 앙상블에 추가했을 때
Public 점수에서 눈에 띄는 이득이 있어 총 6개의 모델을 앙상블 하였습니다. 요약하면 아래와 같을 때 Public점수에서 가장 높았습니다.
- No MixUp, Efficientnet b4 모델 5개
- Mixup by 5 steps, Efficientnet b4 모델 1개

## TTA(Test Time Augmentation)

출처: https://github.com/qubvel/ttach

해당 패키지를 이용하여 Flip, Rotate(0,90,180), FiveCrop을 커스텀으로 생성하여 적용했습니다.
