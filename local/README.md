# 학습 및 테스트 방법!

학습하길 원한다면 argparser의 mode의 default를 train으로 하여 사용.  
테스트를 원한다면 test로 하여라.  
```python
parser.add_argument('--mode', type=str,   help='training and test mode',    default='train')
```
이하 사용되는 학습 및 테스트 data augmentation 방법은 맨 
