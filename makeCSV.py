import cv2  # OpenCV 라이브러리 임포트
import mediapipe as mp  # MediaPipe 라이브러리 임포트
import csv  # CSV 파일을 다루기 위한 라이브러리 임포트
import time  # 시간 관련 함수 사용을 위한 라이브러리 임포트

# MediaPipe의 손 인식 솔루션 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,  # 한 번에 한 손만 감지
    min_detection_confidence=0.5,  # 최소 감지 신뢰도
    min_tracking_confidence=0.5  # 최소 추적 신뢰도
)
mp_draw = mp.solutions.drawing_utils  # 랜드마크를 그리기 위한 도구 초기화

# 웹캠에서 비디오 캡처 시작
cap = cv2.VideoCapture(0)

# CSV 파일 생성 및 헤더 작성
with open('testdata.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    # 헤더 작성: 'label' 열과 21개의 랜드마크 데이터 (x, y, z 좌표)
    headers = ['label']
    for i in range(21):
        headers.extend([f'x{i}', f'y{i}', f'z{i}'])
    csv_writer.writerow(headers)

    # 레이블 입력 및 데이터 수집 반복
    for _ in range(6):  # 6개의 다른 레이블에 대해 반복
        input("준비가 되면 a입력: ")  # 새로운 레이블 수집 시작
        label = input("레이블을 입력: ")  # 레이블 이름 입력

        count = 0
        while count < 1000:  # 각 레이블에 대해 1000개의 데이터 수집
            success, image = cap.read()  # 웹캠에서 이미지 읽기
            if not success:  # 이미지 읽기에 실패하면 다음 반복으로 넘어감
                continue

            image = cv2.flip(image, 1)  # 이미지 좌우 반전 (거울 모드)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지를 RGB로 변환
            results = hands.process(image)  # 손 랜드마크 감지
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 이미지를 다시 BGR로 변환 (OpenCV 디스플레이 용)

            if results.multi_hand_landmarks:  # 손 랜드마크가 감지된 경우
                for hand_landmarks in results.multi_hand_landmarks:
                    row = [label]  # 현재 레이블 추가
                    # 각 랜드마크의 x, y, z 좌표를 row에 추가
                    for lm in hand_landmarks.landmark[:21]:
                        row.extend([lm.x, lm.y, lm.z])
                    csv_writer.writerow(row)  # CSV 파일에 한 줄 기록
                    count += 1  # 수집한 데이터 수 증가
                    time.sleep(0.1)  # 0.1초 대기 (0.1초마다 한 번씩 데이터 기록)

                # 이미지에 랜드마크 그리기
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 이미지 화면에 표시
            cv2.imshow("Hand Tracking", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 반복 종료
                break

        print(f"Completed recording 6 entries for label: {label}")  # 레이블에 대한 데이터 수집 완료 메시지 출력

# 자원 해제
cap.release()  # 웹캠 자원 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
