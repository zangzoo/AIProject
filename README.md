# 👋 수어 TALK

## ✨ 개요
**수어 TALK**는 카메라 영상을 실시간으로 분석해 수어(수화)를 텍스트로 변환하고,  
관계 기반 자연스러운 말투로 재가공해 보여주는 크로스플랫폼 앱입니다.

---

## 📂 데이터
- **30개 수어 문장 클래스** (일상 대화용)  
- **총 600개 영상** (클래스당 20개씩) 직접 촬영  
- **데이터 다양성**  
  - 여러 사람, 조명, 배경, 각도  
  - 증강: 밝기 조절 · 노이즈 추가 · 좌우 반전  
- **전처리 파이프라인**  
  1. MediaPipe Holistic → 3D 손·몸 키포인트 추출  
  2. JSON → NumPy 배열 변환 → `.npz` 포맷 저장  

---

## 🚀 주요 기능
- 🎥 **실시간 수어 인식**  
  MediaPipe로 손·몸 키포인트 추출 → 딥러닝 모델 예측  
- 💬 **관계 기반 텍스트 변환**  
  GPT-3.5-turbo로 부모님·친구·동료 등 어투 자동 변환  
- 🌐 **크로스플랫폼 지원**  
  React Native + Expo로 iOS·Android·Web 앱 제공  

---

## 🏗️ 프로젝트 구조

프로젝트는 다음과 같이 명확하게 구분된 세 가지 핵심 폴더로 구성됩니다.

-   `backend/`
    : 수어 키포인트 데이터 처리, 모델 예측 실행, OpenAI API 연동을 담당하는 **Python FastAPI 서버** 코드입니다.
    -   `keypoint_processor.py`: 웹소켓을 통해 영상 프레임을 수신하고 MediaPipe를 이용해 키포인트를 추출한 뒤, 메인 백엔드로 전송하는 역할을 합니다.
    -   `main.py`: 수신된 키포인트 시퀀스를 로드된 딥러닝 모델로 예측하고 OpenAI API를 호출하여 최종 텍스트를 변환하는 핵심 서버 로직입니다.
    -   `requirements.txt`: 백엔드 실행에 필요한 모든 Python 라이브러리 목록입니다.

-   `frontend/SignApp/`
    : 사용자와 상호작용하는 **Expo 기반 React Native 애플리케이션** 코드입니다.
    -   카메라 접근 및 미리보기, 관계 선택 UI, 실시간 번역 결과 표시 등 사용자 경험과 관련된 기능을 구현합니다.
    -   백엔드와 웹소켓 통신을 통해 실시간으로 영상 데이터를 스트리밍하고 번역 결과를 수신합니다.

-   `models/`
    : 수어 인식 모델의 **개발 및 추론 관련 코드**가 포함된 폴더입니다. 여기에는 모델 정의, 학습 스크립트, 전처리 및 추론 로직 등이 있습니다. 실제 학습된 대용량 모델 파일(`.h5`, `.pt`)은 Git LFS를 통해 별도로 관리됩니다.

---

## 🧠 사용한 AI 모델

저희 팀은 여러 AI 모델 중 **LSTM** 모델을 최종적으로 채택했습니다. 이는 실시간 수어 인식 프로젝트의 특성을 고려하여, 데이터의 안정성과 처리 속도, 그리고 실제 성능을 종합적으로 판단한 결과입니다.

![AI 모델 채택 과정](assets/images/ai_model_adoption_process.png)

### LSTM 선택 이유
1.  **가장 작은 Loss 값으로 인해 가장 안정성이 높은 모델이라고 판단.**
2.  **Transformer의 느린 속도: 실시간 변환이 불가능하다고 판단.**

---

## 🛠️ 기술 스택
| 분야        | 기술                                 |
|-----------|------------------------------------|
| Frontend  | React Native, Expo, TypeScript     |
| Backend   | Python, FastAPI, WebSocket, NumPy, OpenCV, MediaPipe |
| AI/ML     | TensorFlow, PyTorch, LSTM, Transformer, CNN |
| NLP       | OpenAI GPT-3.5-turbo               |

---

## ▶️ 설치 & 실행

### 백엔드
```bash
cd backend
python -m venv venv
source venv/bin/activate    # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
uvicorn main:app --reload --port 8000
uvicorn keypoint_processor:app --reload --port 8001
```

### 프론트엔드
```bash
    cd frontend/SignApp
    npm install # 또는 yarn install
    npx expo start
```

---

## 👥 팀원 소개

| 프로필 | 이름·역할 | GitHub |
| :----: | ---------------- | ------------------------------------ |
| <img src="https://github.com/sujin7167.png?size=100" width="60"/> | **장수진**<br>기획 · AI 모델 개발 · UI/UX 디자인 | [@sujin7167](https://github.com/sujin7167) |
| <img src="https://github.com/zangzoo.png?size=100" width="60"/> | **장지우**<br>기획 · AI 모델 개발 · UI/UX 디자인 · 백엔드·프론트엔드 개발 | [@zangzoo](https://github.com/zangzoo) |
| <img src="https://github.com/jaeeew.png?size=100" width="60"/> | **황재윤**<br>기획 · AI 모델 개발 · 백엔드 개발| [@jaeeew](https://github.com/jaeeew) |

---

## 📄 라이센스

이 프로젝트는 [MIT 라이센스](https://opensource.org/licenses/MIT) 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참고해주세요. (LICENSE 파일 생성 예정)

---

## 📬 문의

프로젝트에 대한 질문이나 제안이 있으시면 언제든지 GitHub 저장소의 [Issues] 탭을 통해 문의해주세요.

[Issues]: https://github.com/zangzoo/AIProject/issues