# backend/keypoint_processor.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import mediapipe as mp
from collections import deque
import cv2
import json
import requests

app = FastAPI()

# MediaPipe Pose와 Hands 모듈만 사용 (Holistic 대신)
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# 한 시퀀스에 모을 프레임 개수
sequence_length = 259

# 클라이언트별로 {'keypoints': deque, 'relationships': List[str]} 형태로 저장
active_sequences = {}


def extract_keypoints(pose_results, hands_results):
    """
    MediaPipe Pose와 Hands 결과에서 (x, y) 좌표만 추출하여
    150차원(포즈 33개×2 + 왼손 21개×2 + 오른손 21개×2)으로 반환.
    """
    keypoints = []

    # 1) Pose landmarks (33개) - (x, y)만 추출
    if pose_results and pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
    else:
        # Pose 랜드마크가 감지되지 않으면 0으로 채움 (33 * 2 = 66 차원)
        keypoints.extend([0.0] * 33 * 2)

    # 2) Hands landmarks (왼손 21개 + 오른손 21개) - (x, y)만 추출
    left_hand_keypoints = []
    right_hand_keypoints = []

    if hands_results and hands_results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
            hand_type = handedness.classification[0].label # 'Left' or 'Right'
            if hand_type == 'Left':
                for lm in hand_landmarks.landmark:
                    left_hand_keypoints.extend([lm.x, lm.y])
            elif hand_type == 'Right':
                for lm in hand_landmarks.landmark:
                    right_hand_keypoints.extend([lm.x, lm.y])

    # 왼손 또는 오른손이 감지되지 않았으면 해당 키포인트 자리를 0으로 채움 (21 * 2 = 42 차원)
    if not left_hand_keypoints:
        keypoints.extend([0.0] * 21 * 2)
    else:
        keypoints.extend(left_hand_keypoints)

    if not right_hand_keypoints:
        keypoints.extend([0.0] * 21 * 2)
    else:
        keypoints.extend(right_hand_keypoints)

    # 총 66 (Pose) + 42 (Left Hand) + 42 (Right Hand) = 150 차원
    return np.array(keypoints, dtype=np.float32)


@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket 엔드포인트
    - client_id 쿼리 파라미터로 각 클라이언트 구분
    - 첫 연결 시 관계 정보(text JSON)를 수신하여 저장
    - 바이너리로 전달된 프레임을 OpenCV로 디코딩 → MediaPipe Pose+Hands로 키포인트 추출
    - 키포인트 시퀀스가 sequence_length만큼 쌓이면, 메인 백엔드(/predict)로 HTTP POST
    - 백엔드 응답(번역된 텍스트)을 다시 WebSocket으로 클라이언트에 전달
    """
    await websocket.accept()
    print(f"[WebSocket] Client #{client_id} connected")

    if client_id not in active_sequences:
        active_sequences[client_id] = {
            'keypoints': deque(maxlen=sequence_length),
            'relationships': []
        }
        print(f"[WebSocket] Client #{client_id} 시퀀스 및 관계 초기화")

    # Pose 및 Hands 모델 인스턴스 생성
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
        connection_active = True

        try:
            # WebSocket 메시지 수신 루프
            while connection_active:
                try:
                    # 메시지 수신 시도. 연결 끊어지면 WebSocketDisconnect 발생
                    message = await websocket.receive()
                except WebSocketDisconnect:
                    print(f"[WebSocket] 클라이언트 {client_id} 연결 해제 감지 (receive)")
                    connection_active = False # 루프 종료 플래그 설정
                    break # 메시지 수신 try 블록에서 빠져나옴

                # 메시지 타입 확인 및 처리
                if message['type'] == 'websocket.disconnect':
                    print(f"[WebSocket] 클라이언트 {client_id} 연결 해제 감지 (disconnect message)")
                    connection_active = False # 루프 종료 플래그 설정
                    break # while 루프 종료

                # 'websocket.receive' 타입 메시지 처리 (텍스트 또는 바이너리 데이터 포함)
                elif message['type'] == 'websocket.receive':
                    if 'text' in message and message['text'] is not None:
                        # 텍스트 메시지 처리 (관계 정보 등)
                        try:
                            obj = json.loads(message['text'])
                            if obj.get('type') == 'relationships':
                                active_sequences[client_id]['relationships'] = obj.get('data', [])
                                print(f"[WebSocket] 클라이언트 {client_id} 관계 업데이트: {active_sequences[client_id]['relationships']}")
                        except Exception as e:
                            print(f"[WebSocket] 클라이언트 {client_id} 텍스트 메시지 처리 오류: {e}")

                    elif 'bytes' in message and message['bytes'] is not None:
                        # 바이너리 메시지 처리 (영상 프레임)
                        frame_bytes = message['bytes']
                        np_arr = np.frombuffer(frame_bytes, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if img is None:
                            print(f"[WebSocket] 클라이언트 {client_id} 유효하지 않은 이미지 데이터 수신")
                            continue # 유효하지 않은 이미지 건너뛰기

                        # 이미지 처리 및 키포인트 추출
                        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pose_results = pose.process(image_rgb)
                        hands_results = hands.process(image_rgb)

                        # 키포인트 추출 함수 호출
                        keypoints = extract_keypoints(pose_results, hands_results)

                        active_sequences[client_id]['keypoints'].append(keypoints.tolist())

                        # 디버깅 로그: 수신된 키포인트 차원 확인
                        if active_sequences[client_id]['keypoints']:
                            print(f"[WebSocket] 클라이언트 {client_id} 프레임 처리됨. 시퀀스 길이: {len(active_sequences[client_id]['keypoints'])}, 현재 프레임 키포인트 차원: {len(active_sequences[client_id]['keypoints'][-1])}")

                        # 시퀀스 길이가 차면 백엔드로 전송
                        if len(active_sequences[client_id]['keypoints']) == sequence_length:
                            print(f"[WebSocket] 클라이언트 {client_id} 시퀀스 길이 {sequence_length} 달성. 백엔드로 전송.")
                            backend_url = "http://127.0.0.1:8000/predict"
                            try:
                                resp = requests.post(
                                    backend_url,
                                    json={
                                        "keypoints_sequence": list(active_sequences[client_id]['keypoints']),
                                        "relationships": active_sequences[client_id]['relationships']
                                    },
                                    timeout=10 # 타임아웃을 충분히 설정
                                )
                                if resp.status_code == 200:
                                    backend_result = resp.json()
                                    translated_text = backend_result.get("converted_text", backend_result.get("predicted_class", "..."))
                                    print(f"[WebSocket] 클라이언트 {client_id} 백엔드 응답 수신: {translated_text}")
                                    await websocket.send_json({ "translated_text": translated_text })
                                else:
                                    print(f"[WebSocket] 클라이언트 {client_id} 백엔드 오류 응답: 상태 코드 {resp.status_code}, 응답: {resp.text}")
                                    await websocket.send_json({ "error": f"백엔드 오류: {resp.status_code}" })
                            except requests.exceptions.RequestException as e:
                                print(f"[WebSocket] 클라이언트 {client_id} 백엔드 요청 실패: {e}")
                                await websocket.send_json({ "error": "백엔드 요청 실패" })

                            # 전송 후 시퀀스 초기화
                            active_sequences[client_id]['keypoints'] = deque(maxlen=sequence_length)

                    # 'websocket.receive' 타입이지만 'text' 또는 'bytes' 키가 없는 경우
                    else:
                        print(f"[WebSocket] 클라이언트 {client_id} 예상치 못한 'websocket.receive' 메시지 형식: {message}")

                # 그 외 예상치 못한 메시지 타입 (connect 등)
                else:
                    print(f"[WebSocket] 클라이언트 {client_id} 예상치 못한 메시지 타입: {message.get('type', 'Unknown')}")

        except Exception as e:
            # 그 외 예상치 못한 오류 처리
            print(f"[WebSocket] 클라이언트 {client_id} 처리 중 오류 발생: {e}")
            try:
                # 클라이언트에 오류 메시지 전송 시도
                await websocket.send_json({ "error": f"서버 오류: {e}" })
            except:
                # 오류 메시지 전송 중 다시 오류 발생 시 무시
                pass

    # try/except 블록 종료 후 클라이언트 데이터 정리 (WebSocketDisconnect 발생 시에도 실행)
    if client_id in active_sequences:
        del active_sequences[client_id]

    print(f"[WebSocket] 클라이언트 {client_id} 처리 종료")
