import numpy as np
import torch
from models import SignTransformer

# 1) 레이블맵 불러오기
import json
with open("label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
idx2label = {v:k for k,v in label_map.items()}

# 2) 모델 로드 (학습 때와 동일한 하이퍼파라미터)
NUM_CLASSES = len(label_map)
model = SignTransformer(num_classes=NUM_CLASSES, input_dim=1662).to("cpu")
state = torch.load("best_model_fold1_0.8861.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

# 3) 미리 저장된 npy 키포인트 불러오기
kp_seq = np.load("/Users/zangzoo/vscode/AIProject/data/29. 이름이 뭐예요?/이름이_뭐예요?_01.npy")  # shape = (64,1662)
print("kp_seq shape:", kp_seq.shape)

# 4) 예측
with torch.no_grad():
    x = torch.from_numpy(kp_seq).unsqueeze(0)  # (1,64,1662)
    logits = model(x)
    pred_idx = logits.argmax(dim=1).item()
    print("예측 인덱스:", pred_idx, "/ 레이블:", idx2label[pred_idx])
