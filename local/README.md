# local 이란?

Google colab에서 메모리 부족으로 인해 개인 컴퓨터(local) 환경에서 수행된 학습 및 테스트 이다.  
이 파이썬 파일로 학습 및 테스트 결과는 ***tta_g3.csv*** 파일로 최종 업로드 되었다.

# 학습 및 테스트 방법!

학습하길 원한다면 argparser의 mode의 default를 train으로 하여 사용.  
테스트를 원한다면 test로 하여라.  
```python
parser.add_argument('--mode', type=str,   help='training and test mode',    default='train')
```
이하 사용되는 학습 및 테스트 data augmentation 방법은 맨 앞에 있는 test관련 .ipynb와 동일하다.
