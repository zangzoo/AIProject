import json
import numpy as np
import os
from glob import glob
from tqdm import tqdm

# 경로 설정
json_root = r"C:\SignLang_3\keypoints_json"
output_npz_path = r"C:\SignLang_3\processed_dataset\sign_keypoints.npz"

# 키포인트 개수
POSE_LANDMARKS = 33 * 4
LEFT_HAND_LANDMARKS = 21 * 4
RIGHT_HAND_LANDMARKS = 21 * 4
FACE_LANDMARKS = 468 * 4
TOTAL_FEATURES = POSE_LANDMARKS + LEFT_HAND_LANDMARKS + RIGHT_HAND_LANDMARKS + FACE_LANDMARKS

# 결과 저장 리스트
X = []
y = []

# 각 JSON 파일 순회
phrase_dirs = [d for d in os.listdir(json_root) if os.path.isdir(os.path.join(json_root, d))]

for phrase in tqdm(phrase_dirs, desc="문장 폴더 처리"):
    label = phrase.strip()
    phrase_dir = os.path.join(json_root, phrase)
    json_files = glob(os.path.join(phrase_dir, "*.json"))

    for json_file in json_files:
        with open(json_file, 'r') as f:
            frames = json.load(f)

        sequence = []
        for frame in frames:
            pose = frame.get("pose", [])
            left_hand = frame.get("left_hand", [])
            right_hand = frame.get("right_hand", [])
            face = frame.get("face", [])

            all_keypoints = pose + left_hand + right_hand + face

            # keypoint 누락될 경우 0으로 채움
            if len(all_keypoints) < TOTAL_FEATURES:
                all_keypoints += [0.0] * (TOTAL_FEATURES - len(all_keypoints))
            sequence.append(all_keypoints)

        # 프레임 수가 너무 적은 건 제외
        if len(sequence) < 10:
            continue

        X.append(np.array(sequence))
        y.append(label)

# 시퀀스 패딩: 모두 동일 길이로 맞추기
from tensorflow.keras.preprocessing.sequence import pad_sequences
X_padded = pad_sequences(X, padding='post', dtype='float32')

# 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 저장
os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
np.savez_compressed(output_npz_path, X=X_padded, y=y_encoded, classes=le.classes_)

print(f"✅ 저장 완료: {output_npz_path}")
