# realtime_infer_transformer_resized.py

import cv2
import torch
import json
import numpy as np
from collections import deque

import mediapipe as mp

# ── 프로젝트 내부 모듈 import ──────────────────────────────
from models import SignTransformer     # models.py 안에 정의된 모델
from preprocess import extract_keypoints  # preprocess.py 안에 있는 함수
# ────────────────────────────────────────────────────────────

def load_label_map(path="label_map.json"):
    with open(path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    idx2label = {v: k for k, v in label_map.items()}
    return label_map, idx2label

def main():
    ###############################################
    # 1) 설정값 정의
    ###############################################
    SEQ_LEN        = 64
    INPUT_DIM      = 1662
    LABEL_MAP_PATH = "label_map.json"
    CKPT_PATH      = "best_model_fold1_0.8861.pt"
    DEVICE         = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # (A) 모델 추론용 해상도 (extract_keypoints는 224×224 영상 리스트를 가정)
    VIS_W, VIS_H = 224, 224

    # (B) 화면 출력용 해상도 (원하면 더 크게)
    DISPLAY_W, DISPLAY_H = 1080, 810

    # (C) FRAME_SKIP 설정: n프레임마다 extract_keypoints 호출
    FRAME_SKIP = 2

    # ── MediaPipe Holistic 초기화 ─────────────────────────────
    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    mp_draw = mp.solutions.drawing_utils
    # ─────────────────────────────────────────────────────────

    # ── DrawingSpec 정의 (점과 선을 작게) ───────────────────────
    pose_lm_style   = mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
    pose_conn_style = mp_draw.DrawingSpec(color=(0,200,0), thickness=1)

    hand_lm_style   = mp_draw.DrawingSpec(color=(0,128,255), thickness=1, circle_radius=1)
    hand_conn_style = mp_draw.DrawingSpec(color=(0,128,200), thickness=1)

    face_lm_style   = mp_draw.DrawingSpec(color=(255,0,128), thickness=1, circle_radius=1)
    face_conn_style = mp_draw.DrawingSpec(color=(200,0,80), thickness=1)
    # ─────────────────────────────────────────────────────────

    # ── 웹캠 열기 ─────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: 웹캠을 열 수 없습니다.")
        return

    # ─── 버퍼 및 상태 변수 ────────────────────────────────────
    frame_buffer = deque(maxlen=SEQ_LEN)  # 64장 추론용 영상(224×224)
    raw_buffer   = deque(maxlen=SEQ_LEN)  # 64장 raw 키포인트(1662 dim 전처리 전)
    seq_active   = False
    pred_buffer  = deque(maxlen=10)
    sentence     = []
    last_word    = None
    repeat_cnt   = 0
    frame_count  = 0
    # ─────────────────────────────────────────────────────────

    # ── 레이블맵 로드 ─────────────────────────────────────────
    label_map, idx2label = load_label_map(LABEL_MAP_PATH)
    NUM_CLASSES = len(label_map)
    # ─────────────────────────────────────────────────────────

    # ── SignTransformer 모델 로드 ─────────────────────────────
    model = SignTransformer(
        num_classes=NUM_CLASSES,
        input_dim=INPUT_DIM,
        # 명시하지 않으면 기본값 d_model=128, nhead=8, nlayers=3 사용
    ).to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    # ─────────────────────────────────────────────────────────

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1) 뒤집기, BGR→RGB
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2) 모델 추론용 리사이즈 (224×224)
            small = cv2.resize(img_rgb, (VIS_W, VIS_H))
            draw_frame = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

            # 3) MediaPipe Holistic 실행
            res = mp_holistic.process(small)

            # 4) 랜드마크 시각화 (작은 해상도에서)
            if res.pose_landmarks:
                mp_draw.draw_landmarks(
                    draw_frame,
                    res.pose_landmarks,
                    mp.solutions.holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=pose_lm_style,
                    connection_drawing_spec=pose_conn_style
                )
            if res.left_hand_landmarks:
                mp_draw.draw_landmarks(
                    draw_frame,
                    res.left_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_lm_style,
                    connection_drawing_spec=hand_conn_style
                )
            if res.right_hand_landmarks:
                mp_draw.draw_landmarks(
                    draw_frame,
                    res.right_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_lm_style,
                    connection_drawing_spec=hand_conn_style
                )
            if res.face_landmarks:
                mp_draw.draw_landmarks(
                    draw_frame,
                    res.face_landmarks,
                    mp.solutions.holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=face_lm_style,
                    connection_drawing_spec=face_conn_style
                )

            # 5) “raw 키포인트”만 뽑아서 raw_buffer에 저장
            pts = []
            # (A) PoseLandmarks (33×4)
            for i in range(33):
                if res.pose_landmarks and len(res.pose_landmarks.landmark) > i:
                    p = res.pose_landmarks.landmark[i]
                    pts.extend([p.x, p.y, p.z, p.visibility])
                else:
                    pts.extend([0.0, 0.0, 0.0, 0.0])
            # (B) Left hand (21×3)
            for i in range(21):
                if res.left_hand_landmarks and len(res.left_hand_landmarks.landmark) > i:
                    p = res.left_hand_landmarks.landmark[i]
                    pts.extend([p.x, p.y, p.z])
                else:
                    pts.extend([0.0, 0.0, 0.0])
            # (C) Right hand (21×3)
            for i in range(21):
                if res.right_hand_landmarks and len(res.right_hand_landmarks.landmark) > i:
                    p = res.right_hand_landmarks.landmark[i]
                    pts.extend([p.x, p.y, p.z])
                else:
                    pts.extend([0.0, 0.0, 0.0])
            # (D) Face landmarks (468×3)
            for i in range(468):
                if res.face_landmarks and len(res.face_landmarks.landmark) > i:
                    p = res.face_landmarks.landmark[i]
                    pts.extend([p.x, p.y, p.z])
                else:
                    pts.extend([0.0, 0.0, 0.0])

            raw_buffer.append(np.array(pts, dtype=np.float32))

            # 6) 모델 추론용 영상(224×224)도 frame_buffer에 저장
            frame_buffer.append(draw_frame.copy())

            # 7) 예측 활성화 상태에서만 동작
            if seq_active and len(raw_buffer) == SEQ_LEN:
                # FRAME_SKIP마다 한 번씩만 extract_keypoints 호출
                if frame_count % FRAME_SKIP == 0:
                    frames_list = list(frame_buffer)         # 64장 × (224×224 BGR)
                    kp_seq = extract_keypoints(frames_list)  # 반환: (64,1662)

                    x = torch.from_numpy(kp_seq).unsqueeze(0).to(DEVICE)  # (1,64,1662)
                    logits = model(x)  # (1, NUM_CLASSES)
                    pred_idx = logits.argmax(dim=1).item()
                    word = idx2label[pred_idx]

                    pred_buffer.append(word)
                    if word == last_word:
                        repeat_cnt += 1
                    else:
                        repeat_cnt = 0
                        last_word = word

                    if repeat_cnt >= 3:
                        if len(sentence) == 0 or sentence[-1] != word:
                            sentence.append(word)
                        repeat_cnt = 0

                #  화면 좌상단에 현재 예측 단어 그리기
                cv2.putText(
                    draw_frame,
                    f"{last_word}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

            # 8) 누적 문장(최근 7단어) 표시
            cv2.putText(
                draw_frame,
                " ".join(sentence[-7:]),
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # 9) 제어 안내 문구
            if seq_active:
                cv2.putText(
                    draw_frame,
                    "[ACTIVE] Press 'e' to pause, 'c' to clear, 'q' to quit",
                    (10, VIS_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2
                )
            else:
                cv2.putText(
                    draw_frame,
                    "Press 's'=start, 'e'=pause, 'c'=clear, 'q'=quit",
                    (10, VIS_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2
                )

            # 10) 화면 업스케일: 224×224 → DISPLAY_W×DISPLAY_H (예: 640×480)
            display_frame = cv2.resize(draw_frame, (DISPLAY_W, DISPLAY_H))
            cv2.imshow("Real-time Sign Translation", display_frame)

            frame_count += 1

            # 11) 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                seq_active = True
                frame_buffer.clear()
                raw_buffer.clear()
                pred_buffer.clear()
                sentence.clear()
                last_word = None
                repeat_cnt = 0
                print("🟢 예측 시작!")
            elif key == ord("e"):
                seq_active = False
                print("⛔ 예측 일시정지")
            elif key == ord("c"):
                sentence.clear()
                print("🧹 문장 초기화됨")

    cap.release()
    cv2.destroyAllWindows()
    mp_holistic.close()

if __name__ == "__main__":
    main()
