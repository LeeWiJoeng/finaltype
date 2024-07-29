import torch
import cv2
import numpy as np
import mediapipe as mp
from model import NeuralNet
import pickle
import time

# 라벨 인코더 로드
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
num_classes = len(label_encoder.classes_)

# 모델 로드 및 설정
model = NeuralNet(input_size=63, num_classes=num_classes)  # 랜드마크 수에 따른 input_size 조정
model.load_state_dict(torch.load('hand_model.pth'))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
drawing_spec = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)  # 파란색으로 랜드마크 그리기

cap = cv2.VideoCapture(0)
prev_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    current_time = time.time()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, drawing_spec, drawing_spec)

        # 1초마다 인식 수행
        if current_time - prev_time > 1:
            prev_time = current_time
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [np.array([lm.x, lm.y, lm.z]) for lm in hand_landmarks.landmark]
                    processed_landmarks = np.array(landmarks).flatten()

                    # 모델 입력을 위한 형태 변환
                    input_tensor = torch.tensor([processed_landmarks], dtype=torch.float32)
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    label = predicted.item()

                    # 실제 라벨 이름으로 변환
                    detected_gesture = label_encoder.inverse_transform([label])[0]
                    print(f"Detected Gesture: {detected_gesture}")
            
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
