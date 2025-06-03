import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# ✅ Attention 레이어 (trainable, dtype 인자 받는 버전)
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.W = tf.keras.layers.Dense(1)

    def call(self, inputs):
        score = tf.nn.tanh(self.W(inputs))
        weights = tf.nn.softmax(score, axis=1)
        context_vector = weights * inputs
        return tf.reduce_sum(context_vector, axis=1)

# ✅ 모델 로드
print("✅ 모델 로드 중...")
model = load_model("best_lstm_model.keras", custom_objects={'Attention': Attention})
print("✅ 모델 로드 완료!")

# ✅ 클래스명 (한글 문장 리스트)
classes = [
    "안녕하세요", "뭐해요", "잘 지냈어요?", "고마워요", "미안해요",
    "괜찮아요", "좋아요", "싫어요", "주세요", "사랑해요",
    "축하해요", "안녕히 계세요", "안녕히 가세요", "생일 축하해요", "보고 싶어요",
    "배고파요", "졸려요", "추워요", "더워요", "도와주세요",
    "화이팅!", "멋져요", "조심하세요", "수고했어요", "행복하세요",
    "기다려 주세요", "조금만 더", "여기 있어요", "저기 있어요", "다시 한 번 말해 주세요"
]

# ✅ 한글 출력 함수
def put_text_kor(img, text, org, font_path='C:/Windows/Fonts/malgun.ttf', font_size=30, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(org, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ✅ 키포인트 추출 함수 (pose+face+hands)
def extract_keypoints(results):
    keypoints = []
    # pose
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        keypoints.extend([0] * 33 * 4)
    # face
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0] * 468 * 3)
    # left hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0] * 21 * 3)
    # right hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0] * 21 * 3)
    return np.array(keypoints)

# ✅ 카메라 열기
cap = cv2.VideoCapture(0)
print("✅ cap 객체 생성됨!")

if not cap.isOpened():
    print("❌ 카메라 열 수 없음!")
    exit()
print("✅ 카메라 열기 성공!")

# ✅ 시퀀스 초기화
sequence = deque(maxlen=259)
print("✅ 시퀀스 초기화!")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    print("✅ MediaPipe Holistic 준비 완료!")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임 읽기 실패")
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        if len(sequence) == 259:
            try:
                input_data = np.expand_dims(sequence, axis=0)
                predictions = model.predict(input_data, verbose=0)
                prob = np.max(predictions)
                predicted_idx = np.argmax(predictions)
                predicted_sentence = classes[predicted_idx]
                text = f"{predicted_sentence} ({prob:.2f})"
                print("✅ 예측 결과:", text)
                frame = put_text_kor(frame, text, (10, 30))
            except Exception as e:
                print("❌ 예측 중 오류:", e)

        # 랜드마크 그리기
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow("Sign Language Real-time", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 프로그램 종료")
            break

cap.release()
cv2.destroyAllWindows()
