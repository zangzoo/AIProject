# augment.py
import random
import numpy as np


def random_temporal_crop(seq, min_ratio=0.8):
    """시퀀스의 일부분을 랜덤으로 잘라내고 다시 패딩"""
    T, D = seq.shape
    keep_len = int(T * random.uniform(min_ratio, 1.0))
    start = random.randint(0, T - keep_len)
    cropped = seq[start:start+keep_len]
    if keep_len < T:
        pad = np.zeros((T-keep_len, D), dtype=seq.dtype)
        cropped = np.vstack([cropped, pad])
    return cropped


def random_time_warp(seq, sigma=0.2):
    """시퀀스 속도를 랜덤하게 변화 (시간 왜곡)"""
    T, D = seq.shape
    tt = np.linspace(0, 1, T)
    tt_warp = np.interp(tt, 
                        np.linspace(0,1,int(T*(1-sigma))), 
                        np.sort(np.random.rand(int(T*(1-sigma)))))
    warped = np.array([np.interp(tt_warp, tt, seq[:,d]) for d in range(D)]).T
    return warped


def random_scale_rotate(seq, scale_range=(0.9,1.1), rot_range=(-10,10)):
    """포즈 좌표에 랜덤 스케일 및 Z축 회전 적용"""
    # 스케일
    scale = random.uniform(*scale_range)
    seq_scaled = seq.copy() * scale
    # 회전 (x,y 평면)
    theta = np.deg2rad(random.uniform(*rot_range))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    D = seq.shape[1]
    # pose 부분만 회전: 33 keypoints * (x,y,z,vis)
    pose_dim = 33*4
    for t in range(seq_scaled.shape[0]):
        pose = seq_scaled[t,:pose_dim].reshape(33,4)
        xy = pose[:,:2]
        xy_rot = np.dot(xy, np.array([[cos_t, -sin_t],[sin_t, cos_t]]))
        pose[:,:2] = xy_rot
        seq_scaled[t,:pose_dim] = pose.flatten()
    return seq_scaled


def random_flip(seq):
    """왼손/오른손 키포인트를 서로 교환하고 X 좌표 반전"""
    seq_flipped = seq.copy()
    T, D = seq_flipped.shape
    # 포즈 파트 x 좌표 인덱스 (0번째 요소부터 4단위)
    pose_len = 33 * 4
    pose_x_idxs = np.arange(0, pose_len, 4)
    # 왼손/오른손 시작 인덱스
    lh_start = pose_len
    rh_start = lh_start + 21*3
    # 손 파트 x 좌표 인덱스 (각 3단위마다 첫 요소)
    hand_x_idxs = np.arange(0, 21*3, 3)
    lh_x_idxs = lh_start + hand_x_idxs
    rh_x_idxs = rh_start + hand_x_idxs
    # 얼굴 파트 x 좌표 인덱스 (face: 468*3, but 얼굴 좌표 뒤에 x는 첫 요소)
    face_start = rh_start + 21*3
    face_x_idxs = face_start + np.arange(0, 468*3, 3)
    # 전체 x 좌표 인덱스
    all_x_idxs = np.concatenate([pose_x_idxs, lh_x_idxs, rh_x_idxs, face_x_idxs])

    # 1) x 좌표 반전
    seq_flipped[:, all_x_idxs] = 1.0 - seq_flipped[:, all_x_idxs]
    # 2) 왼손/오른손 스와핑
    left = seq_flipped[:, lh_start:lh_start+21*3].copy()
    right = seq_flipped[:, rh_start:rh_start+21*3].copy()
    seq_flipped[:, lh_start:lh_start+21*3] = right
    seq_flipped[:, rh_start:rh_start+21*3] = left
    return seq_flipped


def augment_sequence(seq, drop_rate=0.1, noise_std=0.02, apply_flip=0.5):
    """
    복합 증강: 시간 드롭, 노이즈, 크롭, 왜곡, 스케일·회전, 가끔 뒤집기
    """
    # 1) 랜덤 프레임 드롭 및 패딩
    T, D = seq.shape
    keep = max(1, int(T * (1 - drop_rate)))
    idxs = sorted(random.sample(range(T), keep))
    seq = seq[idxs]
    if keep < T:
        pad = np.zeros((T-keep, D), dtype=seq.dtype)
        seq = np.vstack([seq, pad])

    # 2) Gaussian 노이즈
    seq = seq + np.random.normal(0, noise_std, seq.shape)

    # 3) 랜덤 Temporal Crop
    seq = random_temporal_crop(seq)

    # 4) 랜덤 Time Warp
    if random.random() < 0.5:
        seq = random_time_warp(seq)

    # 5) 랜덤 Scale & Rotate
    if random.random() < 0.5:
        seq = random_scale_rotate(seq)

    # 6) 랜덤 Flip (좌/우 대칭)
    if random.random() < apply_flip:
        seq = random_flip(seq)

    return seq
