# dataset.py
import torch
import numpy as np
from preprocess import extract_frames, extract_keypoints

class SignDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, label_map, seq_len=64, augment=None):
        self.files = file_list
        self.label_map = label_map
        self.seq_len = seq_len
        self.augment = augment
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        vp = self.files[idx]
        npy_path = vp.with_suffix('.npy')
        if npy_path.exists():
            kp = np.load(str(npy_path))  # 캐시 로드
        else:
            frames = extract_frames(vp, self.seq_len)
            kp = extract_keypoints(frames)
            np.save(str(npy_path), kp)   # 새로 캐시
        label = self.label_map[vp.parent.name]
        frames = extract_frames(vp, self.seq_len)
        kp = extract_keypoints(frames)
        if self.augment:
            kp = self.augment(kp)
        return torch.tensor(kp, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.long)
