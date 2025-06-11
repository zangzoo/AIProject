import cv2
import mediapipe as mp
import os
import json

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

# 데이터셋 폴더 경로
DATASET_DIR = "./dataset"
OUTPUT_DIR = "./json_output"

# 결과 저장 경로 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 각 문장 폴더 순회
for sentence_dir in os.listdir(DATASET_DIR):
    sentence_path = os.path.join(DATASET_DIR, sentence_dir)
    if not os.path.isdir(sentence_path):
        continue

    sentence_keypoints = {}

    for video_file in os.listdir(sentence_path):
        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(sentence_path, video_file)
        cap = cv2.VideoCapture(video_path)

        frame_idx = 0
        video_keypoints = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            frame_keypoints = {}

            # 몸
            if results.pose_landmarks:
                frame_keypoints["pose"] = [
                    [lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark
                ]
            else:
                frame_keypoints["pose"] = []

            # 왼손
            if results.left_hand_landmarks:
                frame_keypoints["left_hand"] = [
                    [lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark
                ]
            else:
                frame_keypoints["left_hand"] = []

            # 오른손
            if results.right_hand_landmarks:
                frame_keypoints["right_hand"] = [
                    [lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark
                ]
            else:
                frame_keypoints["right_hand"] = []

            # 얼굴
            if results.face_landmarks:
                frame_keypoints["face"] = [
                    [lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark
                ]
            else:
                frame_keypoints["face"] = []

            video_keypoints[f"frame_{frame_idx}"] = frame_keypoints
            frame_idx += 1

        cap.release()
        sentence_keypoints[video_file] = video_keypoints
        print(f"{sentence_dir} / {video_file} 처리 완료!")

    # 문장별 JSON 저장
    output_file = os.path.join(OUTPUT_DIR, f"{sentence_dir}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sentence_keypoints, f, ensure_ascii=False, indent=4)

    print(f"✅ {sentence_dir} JSON 저장 완료!")

holistic.close()
print("=== 모든 문장 JSON 저장 완료! ===")
