# backend/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from openai import OpenAI
import os

app = FastAPI()

class SignLanguageRequest(BaseModel):
    keypoints_sequence: List[List[float]]
    relationships: List[str]

# 환경 변수에서 API 키 불러오기
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def convert_tone(text: str, relationship: str) -> str:
    system_message = {
        "role": "system",
        "content": (
            "당신은 주어진 문장을 상대방과의 관계에 맞추어 자연스럽고 읽기 편한 말투로 바꿔주는 언어 전문가입니다. 주어진 문장만을 바꿔주세요."
        )
    }
    user_message = {
        "role": "user",
        "content": (
            f"다음 문장을 \"{relationship}\"에게 보내는 메시지라고 생각하고, "
            f"어울리는 말투로 바꿔주세요:\n\n\"{text}\""
        )
    }
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_message, user_message],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text


class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.W = tf.keras.layers.Dense(1)
    def call(self, inputs):
        score = tf.nn.tanh(self.W(inputs))
        weights = tf.nn.softmax(score, axis=1)
        context = weights * inputs
        return tf.reduce_sum(context, axis=1)

# 모델 로드
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "..", "models", "sign_model_temporal_cnn_attention.h5")
classes_path = os.path.join(script_dir, "..", "models", "sign_keypoints.npz")

try:
    model = load_model(model_path, custom_objects={"Attention": Attention})
    classes = np.load(classes_path)["classes"]
except:
    model = None
    classes = []

sequence_length = 259

@app.post("/predict")
async def predict_sign_language(request: SignLanguageRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="모델 로드 오류")
    if len(request.keypoints_sequence) != sequence_length:
        raise HTTPException(status_code=400, detail=f"시퀀스 길이는 {sequence_length}")
    data = np.expand_dims(np.array(request.keypoints_sequence), axis=0)
    pred = model.predict(data, verbose=0)
    idx = int(np.argmax(pred))
    sign = classes[idx]
    rel = request.relationships[0] if request.relationships else "상대"
    conv_text = convert_tone(sign, rel)
    return {"predicted_class": sign, "probability": float(pred[0][idx]), "converted_text": conv_text}

@app.get("/")
def root():
    return {"message": "수어 인식 + GPT API"}
