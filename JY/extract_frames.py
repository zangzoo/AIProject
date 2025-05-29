import cv2
import mediapipe as mp
import json
import os
from glob import glob

# 경로 지정
dataset_dir = r"C:\SignLang_3\dataset"
output_dir = r"C:\SignLang_3\keypoints_json"

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic

def landmark_to_list(landmarks):
    if landmarks is None:
        return []
    return [coord for lm in landmarks.landmark for coord in (lm.x, lm.y, lm.z, lm.visibility)]

def extract_keypoints_from_video(video_path):
    print(f"[INFO] 처리 중: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 영상 열기 실패: {video_path}")
        return None

    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=2)
    frame_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        frame_info = {
            "frame": frame_idx,
            "pose": landmark_to_list(results.pose_landmarks),
            "left_hand": landmark_to_list(results.left_hand_landmarks),
            "right_hand": landmark_to_list(results.right_hand_landmarks),
            "face": landmark_to_list(results.face_landmarks)
        }
        frame_data.append(frame_info)
        frame_idx += 1

    cap.release()
    holistic.close()

    if len(frame_data) == 0:
        print(f"[WARNING] 키포인트 없음: {video_path}")
        return None

    return frame_data

def process_all_videos(dataset_dir, output_dir):
    print(f"[INFO] 데이터셋 루트 폴더: {dataset_dir}")
    os.makedirs(output_dir, exist_ok=True)

    phrase_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"[INFO] 총 문장 폴더 수: {len(phrase_dirs)}")

    for phrase in phrase_dirs:
        phrase_path = os.path.join(dataset_dir, phrase)
        video_files = glob(os.path.join(phrase_path, "*.mp4"))
        print(f"📁 문장: {phrase} / 영상 수: {len(video_files)}")

        phrase_output_dir = os.path.join(output_dir, phrase)
        os.makedirs(phrase_output_dir, exist_ok=True)

        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(phrase_output_dir, f"{video_name}.json")

            if os.path.exists(output_path):
                print(f"[-] 이미 존재: {output_path}")
                continue

            frame_data = extract_keypoints_from_video(video_path)
            if frame_data is None:
                continue

            with open(output_path, 'w') as f:
                json.dump(frame_data, f)
            print(f"[✓] 저장 완료: {output_path}")

if __name__ == "__main__":
    process_all_videos(dataset_dir, output_dir)
