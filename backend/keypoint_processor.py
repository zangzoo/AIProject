# backend/keypoint_processor.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import mediapipe as mp
from collections import deque
import cv2
import json
import requests

app = FastAPI()
mp_holistic = mp.solutions.holistic.Holistic
sequence_length = 259
active_sequences = {}

def extract_keypoints(results):
    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark[:33]:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        keypoints.extend([0.0] * 33 * 4)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark[:21]:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        keypoints.extend([0.0] * 21 * 4)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark[:21]:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        keypoints.extend([0.0] * 21 * 4)
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark[:468]:
            visibility = getattr(lm, 'visibility', 0.0)
            keypoints.extend([lm.x, lm.y, lm.z, visibility])
    else:
        keypoints.extend([0.0] * 468 * 4)
    return np.array(keypoints)

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    if client_id not in active_sequences:
        active_sequences[client_id] = { 'keypoints': deque(maxlen=sequence_length), 'relationships': [] }

    try:
        with mp_holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                msg = await websocket.receive()
                if 'text' in msg and msg['text'] is not None:
                    try:
                        obj = json.loads(msg['text'])
                        if obj.get('type') == 'relationships':
                            active_sequences[client_id]['relationships'] = obj.get('data', [])
                    except:
                        pass
                elif 'bytes' in msg and msg['bytes'] is not None:
                    frame_bytes = msg['bytes']
                    np_arr = np.frombuffer(frame_bytes, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    keypoints = extract_keypoints(results)
                    active_sequences[client_id]['keypoints'].append(keypoints.tolist())

                    if len(active_sequences[client_id]['keypoints']) == sequence_length:
                        backend_url = "http://127.0.0.1:8000/predict"
                        try:
                            resp = requests.post(
                                backend_url,
                                json={
                                    "keypoints_sequence": list(active_sequences[client_id]['keypoints']),
                                    "relationships": active_sequences[client_id]['relationships']
                                },
                                timeout=5
                            )
                            if resp.status_code == 200:
                                backend_result = resp.json()
                                await websocket.send_json({ "translated_text": backend_result.get("converted_text", "") })
                            else:
                                await websocket.send_json({ "error": f"Backend error {resp.status_code}" })
                        except:
                            await websocket.send_json({ "error": "Backend request failed" })
                        active_sequences[client_id]['keypoints'] = deque(maxlen=sequence_length)
                else:
                    pass
    except WebSocketDisconnect:
        if client_id in active_sequences:
            del active_sequences[client_id]
