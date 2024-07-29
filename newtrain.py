import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from model import NeuralNet
import pickle  # 라벨 인코더 저장을 위해 필요

# 데이터 로드
data = pd.read_csv('testdata.csv')  # CSV 파일에서 데이터셋을 읽어옴
X = data.drop('label', axis=1).values  # 'label' 열을 제외한 모든 특징 데이터를 가져옴
y = data['label'].values  # 'label' 열의 데이터를 가져옴

# 라벨 인코딩
label_encoder = LabelEncoder()  # 라벨 인코더 객체 생성
y_encoded = label_encoder.fit_transform(y)  # 라벨 데이터를 숫자로 변환
num_classes = len(label_encoder.classes_)  # 고유 라벨 수

# 라벨 인코더 저장
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)  # 라벨 인코더를 파일로 저장

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)  # 특징 데이터를 텐서로 변환
        self.labels = torch.tensor(labels, dtype=torch.int64)  # 라벨 데이터를 텐서로 변환
    
    def __len__(self):
        return len(self.features)  # 데이터셋의 크기를 반환
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]  # 인덱스에 해당하는 데이터 반환

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)  # 데이터를 훈련 세트와 테스트 세트로 분할

# 데이터 로더 설정
train_dataset = CustomDataset(X_train, y_train)  # 훈련 데이터셋 생성
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)  # 데이터 로더 생성

# 모델 생성
model = NeuralNet(input_size=X_train.shape[1], num_classes=num_classes)  # 모델 생성
criterion = nn.CrossEntropyLoss()  # 손실 함수 정의
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 옵티마이저 정의

# 모델 훈련
for epoch in range(100):  # 총 10번의 에포크 동안 훈련
    for features, labels in train_loader:
        optimizer.zero_grad()  # 옵티마이저 초기화
        outputs = model(features)  # 모델에 입력 데이터를 전달하여 출력값 계산
        loss = criterion(outputs, labels)  # 손실값 계산
        loss.backward()  # 손실값을 기준으로 역전파 수행
        optimizer.step()  # 가중치 업데이트
    print(f'Epoch {epoch+1} completed')  # 각 에포크가 완료될 때마다 출력

# 모델 저장
torch.save(model.state_dict(), 'hand_model.pth')  # 훈련된 모델의 가중치를 파일로 저장
print('Model training complete and saved.')  # 모델 저장 완료 메시지 출력
