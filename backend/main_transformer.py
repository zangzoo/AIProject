# backend/main.py

import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import os

# OpenAI 클라이언트 정의
from openai import OpenAI

# PositionalEncoding 및 SignLanguageTransformer 클래스 정의
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SignLanguageTransformer(nn.Module):
    def __init__(self, input_dim=274, d_model=384, nhead=8, num_layers=8, num_classes=980, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), # LayerNorm의 입력/출력 차원은 d_model과 같습니다.
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):  # x: (B, T, input_dim)
        # 이 forward 함수는 274 차원 입력을 기대합니다.
        B, T, _ = x.shape
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (B, T, d_model)
        x = self.dropout(x.mean(dim=1))  # (B, d_model) # 시퀀스의 평균을 사용
        return self.classifier(x)


app = FastAPI()

class SignLanguageRequest(BaseModel):
    keypoints_sequence: List[List[float]]
    relationships: List[str]

# 환경 변수에서 API 키 불러오기
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def convert_tone(text: str, relationship: str) -> str:
    prompt = f'"{text}"라는 문장을 "{relationship}" 관계에 맞게 자연스러운 말투로 바꿔줘.'
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":"너는 관계에 맞는 말투를 만들어 줘."},{"role":"user","content":prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except:
        return text

# 모델 로드
script_dir = os.path.dirname(__file__)

# PyTorch 모델 및 클래스 파일 경로 설정
model_file_name = "best_model_3e_flip_more_0.899.pt"
classes_file_name = "sign_keypoints.npz" # 클래스 파일 이름이 PyTorch 학습 시와 동일한지 확인 필요

# 프로젝트 루트 디렉토리를 기준으로 경로 설정
project_root = os.path.abspath(os.path.join(script_dir, ".."))
pytorch_model_path = os.path.join(project_root, "인식O_키포인트_전부", model_file_name)
pytorch_classes_path = os.path.join(project_root, "models", classes_file_name) # 가정: classes 파일은 models 디렉토리에 있음

try:
    # PyTorch 모델 로드
    # 모델 인스턴스 생성 (저장된 모델의 실제 아키텍처 파라미터 사용)
    model = SignLanguageTransformer(input_dim=274, d_model=384, num_layers=8, nhead=8, num_classes=980)
    # 모델 상태 사전 로드
    checkpoint = torch.load(pytorch_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval() # 평가 모드

    # 클래스 로드 (기존 numpy 파일을 그대로 사용)
    classes_data = np.load(pytorch_classes_path, allow_pickle=True)
    classes = classes_data["classes"] if "classes" in classes_data else classes_data["arr_0"] # .npz 파일 구조에 따라 키 조정

    print(f"✅ PyTorch 모델 로드 성공: {pytorch_model_path}")
    print(f"✅ 클래스 로드 성공: {pytorch_classes_path}, 클래스 수: {len(classes)}")

except Exception as e:
    print(f"❌ 모델 또는 클래스 로드 오류: {e}")
    model = None
    classes = []

# sequence_length는 동일하게 유지
sequence_length = 259

@app.post("/predict")
async def predict_sign_language(request: SignLanguageRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="모델 로드 오류")
    # 키포인트 차원 확인 (프레임당 150 차원) - 모델은 274 차원 기대. 여기서 불일치 발생.
    # 현재는 시퀀스 길이가 맞지 않으면 처리하지 않고 빈 결과 반환 (또는 오류 발생)
    # TODO: 키포인트 차원 불일치 (150 vs 274) 처리 로직 추가 (예: 150 -> 274 변환)
    if len(request.keypoints_sequence) == 0 or (len(request.keypoints_sequence) > 0 and len(request.keypoints_sequence[0]) != 150):
         print(f"⚠️ 경고: 수신된 키포인트 프레임 차원이 예상과 다릅니다. 예상: 150, 수신: {len(request.keypoints_sequence[0]) if len(request.keypoints_sequence) > 0 else 0}")
         return {"predicted_class": "처리중...", "probability": 0.0, "converted_text": "처리중..."}

    # 시퀀스 길이 확인
    if len(request.keypoints_sequence) != sequence_length:
         print(f"⚠️ 경고: 수신된 시퀀스 길이가 예상과 다릅니다. 예상: {sequence_length}, 수신: {len(request.keypoints_sequence)}")
         # 임시로 시퀀스 길이가 다르면 빈 결과 반환
         # TODO: 시퀀스 길이 불일치 시 패딩/잘라내기 로직 추가
         return {"predicted_class": "처리중...", "probability": 0.0, "converted_text": "처리중..."}


    # PyTorch 모델 입력 형식에 맞게 데이터 변환 (Batch_size, Sequence_length, Input_dim)
    # 모델은 274 차원 입력을 기대하지만, 현재 150 차원만 수신됨.
    # 이 데이터를 어떻게 274 차원으로 맞춰야 할지 결정해야 합니다.
    # 임시로 150 차원 데이터를 274 차원으로 '확장'하는 간단한 처리를 추가합니다. (정확한 방법은 아님)
    # 실제 사용 시에는 학습에 사용된 274차원 키포인트를 추출하거나, 150차원에 맞게 모델을 수정해야 합니다.
    padded_data = []
    for frame_keypoints in request.keypoints_sequence:
        # 150 차원 키포인트 뒤에 0으로 채워서 274 차원으로 만듭니다.
        padded_frame = frame_keypoints + [0.0] * (274 - 150)
        padded_data.append(padded_frame)

    data = torch.tensor(padded_data, dtype=torch.float32).unsqueeze(0) # 배치 차원 추가

    with torch.no_grad(): # 추론 시에는 grad 계산 안 함
        pred = model(data) # 모델 예측

    # 예측 결과 처리
    prob = torch.softmax(pred, dim=1) # 확률 계산
    idx = int(torch.argmax(prob, dim=1).item()) # 가장 높은 확률의 클래스 인덱스

    if not classes or idx < 0 or idx >= len(classes):
         sign = "알 수 없음"
         predicted_probability = 0.0
    else:
         sign = classes[idx]
         predicted_probability = float(prob[0][idx].item())

    # 관계 정보를 가져와서 GPT 변환 (기존 로직 유지)
    rel = request.relationships[0] if request.relationships else "상대"
    conv_text = convert_tone(sign, rel)

    # 디버깅 로그 추가
    print(f"📥 /predict 요청 수신. 시퀀스 길이: {len(request.keypoints_sequence)}, 관계: {request.relationships}, 첫 프레임 차원: {len(request.keypoints_sequence[0]) if len(request.keypoints_sequence) > 0 else 0}")
    print(f"📊 모델 예측 결과: 클래스={sign}, 확률={predicted_probability:.4f}")
    print(f"🗣️ GPT 변환 결과: {conv_text}")


    return {"predicted_class": sign, "probability": predicted_probability, "converted_text": conv_text}

@app.get("/")
def root():
    return {"message": "수어 인식 + GPT API"}
