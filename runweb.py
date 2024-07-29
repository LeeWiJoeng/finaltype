# 필요한 라이브러리와 모듈을 임포트합니다.
from flask import Flask, render_template, Response
import torch  # PyTorch 라이브러리
import cv2  # OpenCV 라이브러리
import numpy as np  # numpy 라이브러리
import mediapipe as mp  # MediaPipe 라이브러리
from model import NeuralNet  # 모델 파일에서 NeuralNet 클래스를 임포트
import pickle  # 데이터 직렬화 라이브러리
import time  # 시간 관련 라이브러리

app = Flask(__name__)  # Flask 앱 인스턴스 생성

# pickle을 사용하여 미리 저장된 라벨 인코더 객체를 로드합니다.
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
num_classes = len(label_encoder.classes_)  # 라벨 인코더에서 클래스의 수를 확인합니다.

# NeuralNet 모델을 초기화하고, 사전 학습된 가중치를 로드합니다.
model = NeuralNet(input_size=63, num_classes=num_classes)
model.load_state_dict(torch.load('hand_model.pth'))
model.eval()  # 모델을 평가 모드로 설정

# MediaPipe를 사용하여 손 인식기를 초기화합니다.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
drawing_spec = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)  # 랜드마크 그리기 스펙 설정

# 현재 인식된 제스처와 그 제스처가 표시된 시간을 저장하는 전역 변수
current_gesture = ""
display_time = 0

# 웹캠에서 프레임을 실시간으로 생성하는 함수
def generate_frames():
    global current_gesture, display_time
    cap = cv2.VideoCapture(0)  # 웹캠 캡처 시작
    prev_time = 0  # 이전 제스처 인식 시간 초기화

    while cap.isOpened():
        success, image = cap.read()  # 웹캠에서 이미지를 읽어옵니다.
        if not success:
            break

        current_time = time.time()  # 현재 시간 측정
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # 이미지 전처리
        results = hands.process(image)  # 손의 랜드마크를 감지합니다.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 다시 BGR로 변환하여 디스플레이 준비

        # 감지된 손 랜드마크가 있다면, 각 랜드마크를 그립니다.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, drawing_spec, drawing_spec)

            # 1초마다 제스처를 인식하도록 합니다.
            if current_time - prev_time > 1:
                prev_time = current_time
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = [np.array([lm.x, lm.y, lm.z]) for lm in hand_landmarks.landmark]
                        processed_landmarks = np.array(landmarks).flatten()

                        input_tensor = torch.tensor([processed_landmarks], dtype=torch.float32)
                        output = model(input_tensor)
                        _, predicted = torch.max(output, 1)
                        label = predicted.item()

                        current_gesture = label_encoder.inverse_transform([label])[0]
                        display_time = current_time
                        print(f"Detected Gesture: {current_gesture}")

        # 인식된 제스처를 3초간 화면에 표시합니다.
        if current_gesture and current_time - display_time < 3:
            cv2.putText(image, current_gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 프레임을 HTTP 응답으로 반환

    cap.release()  # 카메라 자원 해제

# 루트 URL에 대한 라우트 및 뷰 함수
@app.route('/')
def index():
    return render_template('index.html')  # HTML 템플릿 렌더링

# 비디오 피드 URL에 대한 라우트 및 뷰 함수
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  # 실시간 비디오 스트림 응답

# 메인 실행 부분
if __name__ == '__main__':
    app.run(debug=True)  # 애플리케이션을 디버그 모드로 실행
