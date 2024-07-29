import torch
import torch.nn as nn
import torch.nn.functional as F

# 신경망 모델을 정의하는 클래스
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        # 첫 번째 숨겨진 층: 입력 크기에서 100개의 뉴런으로 연결
        self.layer1 = nn.Linear(input_size, 100)
        # 두 번째 숨겨진 층: 100개의 뉴런에서 50개의 뉴런으로 연결
        self.layer2 = nn.Linear(100, 50)
        # 출력 층: 50개의 뉴런에서 출력 크기(클래스 수)로 연결
        self.output = nn.Linear(50, num_classes)

    # 데이터를 신경망을 통과시키는 함수
    def forward(self, x):
        # 첫 번째 숨겨진 층을 통과할 때 ReLU 활성화 함수 적용
        x = F.relu(self.layer1(x))
        # 두 번째 숨겨진 층을 통과할 때 ReLU 활성화 함수 적용
        x = F.relu(self.layer2(x))
        # 결과를 출력 층에서 반환
        return self.output(x)

