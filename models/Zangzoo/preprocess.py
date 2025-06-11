# preprocess.py
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# MediaPipe Holistic 초기화
mp_holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

TARGET_FRAMES = 64  # 시퀀스 길이 고정
SIZE = (224, 224)


def extract_frames(video_path, target_frames=TARGET_FRAMES, size=SIZE):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, size))
    cap.release()

    if len(frames) >= target_frames:
        idxs = np.linspace(0, len(frames) - 1, target_frames).astype(int)
        return [frames[i] for i in idxs]
    elif frames:
        pad = [frames[-1]] * (target_frames - len(frames))
        return frames + pad
    else:
        return [np.zeros((size[1], size[0], 3), np.uint8)] * target_frames


def extract_keypoints(frames):
    """
    Holistic으로 포즈, 양손, 얼굴 랜드마크 추출 후
    선형 보간 및 정규화 적용
    항상 동일한 차원의 시퀀스 반환
    """
    seq = []
    for img in frames:
        res = mp_holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pts = []
        # Pose landmarks (33 points × 4 dims)
        for i in range(33):
            if res.pose_landmarks and len(res.pose_landmarks.landmark) > i:
                p = res.pose_landmarks.landmark[i]
                pts.extend([p.x, p.y, p.z, p.visibility])
            else:
                pts.extend([0.0, 0.0, 0.0, 0.0])
        # Left hand (21 points × 3 dims)
        for i in range(21):
            if res.left_hand_landmarks and len(res.left_hand_landmarks.landmark) > i:
                p = res.left_hand_landmarks.landmark[i]
                pts.extend([p.x, p.y, p.z])
            else:
                pts.extend([0.0, 0.0, 0.0])
        # Right hand (21 points × 3 dims)
        for i in range(21):
            if res.right_hand_landmarks and len(res.right_hand_landmarks.landmark) > i:
                p = res.right_hand_landmarks.landmark[i]
                pts.extend([p.x, p.y, p.z])
            else:
                pts.extend([0.0, 0.0, 0.0])
        # Face landmarks (468 points × 3 dims)
        for i in range(468):
            if res.face_landmarks and len(res.face_landmarks.landmark) > i:
                p = res.face_landmarks.landmark[i]
                pts.extend([p.x, p.y, p.z])
            else:
                pts.extend([0.0, 0.0, 0.0])
        seq.append(np.array(pts, dtype=np.float32))

    seq = np.stack(seq)  # (T, D) with D = 33*4 + 21*3 + 21*3 + 468*3
    T, D = seq.shape

    # 1) 선형 보간(interpolation)으로 결측치(0) 보완
    mask = np.any(seq != 0, axis=1)
    valid_idx = np.where(mask)[0]
    for d in range(D):
        seq[:, d] = np.interp(np.arange(T), valid_idx, seq[valid_idx, d])

    # 2) 포즈(coords) 중심화 및 스케일링
    pose_dim = 33 * 4
    for t in range(T):
        pose = seq[t, :pose_dim].reshape(33, 4)
        coords = pose[:, :3]
        mean = coords.mean(axis=0)
        coords -= mean
        max_dist = np.linalg.norm(coords, axis=1).max() + 1e-6
        coords /= max_dist
        pose[:, :3] = coords
        seq[t, :pose_dim] = pose.flatten()

    return seq


def cache_keypoints(video_path, cache_dir=None):
    """
    1) extract_frames → extract_keypoints 로 시퀀스 생성
    2) .npy 파일로 저장
    3) 배열 반환
    """
    frames = extract_frames(video_path)
    kp_seq = extract_keypoints(frames)            # shape: (64, D)

    # 저장 경로 결정
    video_path = Path(video_path)
    if cache_dir:
        out_dir = Path(cache_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / (video_path.stem + '.npy')
    else:
        save_path = video_path.with_suffix('.npy')

    # 저장
    np.save(str(save_path), kp_seq)
    # (다음번부터는) np.load(str(save_path))로 불러올 수 있습니다.
    return kp_seq