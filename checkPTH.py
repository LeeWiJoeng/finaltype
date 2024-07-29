import torch
from model import NeuralNet

# 모델의 입력 크기와 클래스 수는 데이터셋에 맞게 설정해야 합니다.
input_size = 63  # 예시로 사용된 입력 크기 (21개의 랜드마크 * 3개의 좌표)
num_classes = 6  # 예시로 사용된 클래스 수 (6개의 손 동작)

# 모델 인스턴스 생성
model = NeuralNet(input_size=input_size, num_classes=num_classes)

# 저장된 모델 가중치를 로드
model.load_state_dict(torch.load('hand_model.pth'))

# 모델을 평가 모드로 전환
model.eval()

# 모델의 구조 및 가중치 확인
print(model)
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# 예측 예시
# 예시 입력 데이터 (모양은 [1, input_size]여야 함)
example_input = torch.rand(1, input_size)

# 예측 수행
output = model(example_input)
print(output)
