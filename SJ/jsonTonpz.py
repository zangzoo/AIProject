import os
import json
import numpy as np

JSON_DIR = "./json_output"
NPZ_DIR = "./npz_output"

# npz_output 폴더 없으면 생성
if not os.path.exists(NPZ_DIR):
    os.makedirs(NPZ_DIR)

# json_output 폴더의 모든 json 처리
for json_file in os.listdir(JSON_DIR):
    if not json_file.endswith(".json"):
        continue

    json_path = os.path.join(JSON_DIR, json_file)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_data_list = []
    for video_file, video_data in data.items():
        frame_data_list = []
        for frame_idx in sorted(video_data.keys(), key=lambda x: int(x.split("_")[1])):
            frame_keypoints = video_data[frame_idx]

            # 각 프레임의 keypoint를 하나로 묶기
            # pose (33개), left_hand (21개), right_hand (21개), face (468개)
            pose = np.array(frame_keypoints["pose"]) if frame_keypoints["pose"] else np.zeros((33, 4))
            left_hand = np.array(frame_keypoints["left_hand"]) if frame_keypoints["left_hand"] else np.zeros((21, 3))
            right_hand = np.array(frame_keypoints["right_hand"]) if frame_keypoints["right_hand"] else np.zeros((21, 3))
            face = np.array(frame_keypoints["face"]) if frame_keypoints["face"] else np.zeros((468, 3))

            # 프레임별 데이터를 하나의 배열로 결합
            frame_data = np.concatenate([
                pose.flatten(),
                left_hand.flatten(),
                right_hand.flatten(),
                face.flatten()
            ])
            frame_data_list.append(frame_data)

        video_data_list.append(np.array(frame_data_list))

    # 문장별로 npz 저장
    output_path = os.path.join(NPZ_DIR, f"{json_file.replace('.json', '.npz')}")
    np.savez_compressed(output_path, *video_data_list)
    print(f"{json_file} → {output_path} 변환 완료!")

print("=== 모든 JSON → NPZ 변환 완료! ===")
