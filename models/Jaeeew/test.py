import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import tensorflow as tf
import time
from PIL import ImageFont, ImageDraw, Image

data = np.load("C:/SignLang_3/processed_dataset/sign_keypoints.npz")
X = data['X']
print(X.shape)

# Attention 레이어 정의 (모델 로드에 필요)
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.W = tf.keras.layers.Dense(1)

    def call(self, inputs):
        score = tf.nn.tanh(self.W(inputs))
        weights = tf.nn.softmax(score, axis=1)
        context_vector = weights * inputs
        return tf.reduce_sum(context_vector, axis=1)
    
def put_text_kor(img, text, org, font_path='C:/Windows/Fonts/malgun.ttf', font_size=30, color=(0, 255, 0), thickness=2):
    # OpenCV 이미지 -> PIL 이미지 변환
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    draw.text(org, text, font=font, fill=color)
    # PIL 이미지 -> OpenCV 이미지 변환
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

# 모델 및 클래스 로드
model = load_model("sign_model_temporal_cnn_attention.h5", custom_objects={'Attention': Attention})
classes = np.load("C:/SignLang_3/processed_dataset/sign_keypoints.npz")["classes"]

print(f"[✅ 로드 완료] 모델 클래스 수: {model.output_shape[-1]}, 클래스 배열 수: {len(classes)}")
print(f"모델 입력 shape: {model.input_shape}")

# Mediapipe 설정
mp_holistic = mp.solutions.holistic

sequence_length = 100  # 모델 입력 시퀀스 길이
sequence = deque(maxlen=sequence_length)

def extract_keypoints(results):
    keypoints = []
    # pose (33 landmarks * 4 coords)
    if results.pose_landmarks:
        keypoints.extend([coord for lm in results.pose_landmarks.landmark for coord in (lm.x, lm.y, lm.z, lm.visibility)])
    else:
        keypoints.extend([0.0] * 33 * 4)
    # left hand (21 landmarks * 4 coords)
    if results.left_hand_landmarks:
        keypoints.extend([coord for lm in results.left_hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z, lm.visibility)])
    else:
        keypoints.extend([0.0] * 21 * 4)
    # right hand (21 landmarks * 4 coords)
    if results.right_hand_landmarks:
        keypoints.extend([coord for lm in results.right_hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z, lm.visibility)])
    else:
        keypoints.extend([0.0] * 21 * 4)
    # face (468 landmarks * 4 coords)
    if results.face_landmarks:
        keypoints.extend([coord for lm in results.face_landmarks.landmark for coord in (lm.x, lm.y, lm.z, lm.visibility)])
    else:
        keypoints.extend([0.0] * 468 * 4)

    return np.array(keypoints)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("🟢 실시간 수어 인식 시작 (ESC 누르면 종료)")

start_time = time.time()

with mp_holistic.Holistic(static_image_mode=False, model_complexity=2) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임 읽기 실패")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        if len(sequence) == sequence_length:
            print("[디버깅] 시퀀스 길이 259 도달, 예측 시작")
            input_data = np.expand_dims(sequence, axis=0)  # shape: (1, 259, 2640)
            try:
                prediction = model.predict(input_data, verbose=0)
                predicted_index = np.argmax(prediction)
                predicted_class = classes[predicted_index]
                prob = prediction[0][predicted_index]
                print(f"[예측 결과] {predicted_class} ({prob:.2f})")

                text = f'{predicted_class} ({prob:.2f})'
                org = (10, 50)
                image = put_text_kor(image, text, org)
                #font = cv2.FONT_HERSHEY_SIMPLEX
                #font_scale = 1.2
                #thickness = 3
                #text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                #cv2.rectangle(image, (org[0], org[1] - text_size[1] - 10),
                #              (org[0] + text_size[0], org[1] + 10), (0, 0, 0), -1)
                #cv2.putText(image, text, org, font, font_scale, (0, 255, 0), thickness)

            

            except Exception as e:
                print(f"❌ 예측 중 오류 발생: {e}")
        else:
            print(f"[디버깅] 시퀀스 길이 부족: {len(sequence)} / {sequence_length}")

        # 원하면 랜드마크 그리기
        # mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        # mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp.solutions.drawing_utils.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)

        cv2.imshow("Sign Language Recognition", image)
        key = cv2.waitKey(1)

        # 3초 후 ESC 키 누르면 종료
        if time.time() - start_time > 3 and key == 27:
            print("🛑 종료 신호 수신, 프로그램 종료")
            break

cap.release()
cv2.destroyAllWindows()
