# save_npy.py
from pathlib import Path
from preprocess import cache_keypoints

if __name__ == '__main__':
    data_dir = Path('data')
    videos = list(data_dir.rglob('*.mp4'))
    for v in videos:
        npy_path = v.with_suffix('.npy')
        if npy_path.exists():
            print(f"Skip {npy_path.name} (already exists)")
            continue
        print(f"Caching {v.name} → {npy_path.name}")
        cache_keypoints(v)
