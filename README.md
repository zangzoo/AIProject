# 수어 TALK 프로젝트

## 프로젝트 소개

이 프로젝트는 실시간 수어(수화) 인식을 위한 애플리케이션입니다. 사용자의 카메라 영상을 분석하여 수어 동작의 핵심 키포인트를 추출하고, 머신러닝 모델을 통해 해당 수어가 어떤 단어나 문장인지 예측합니다. 예측된 결과는 대화 상대방과의 관계에 맞춰 더 자연스러운 말투로 변환되어 사용자에게 보여집니다.

주요 기능:
- 실시간 카메라 영상 기반 수어 인식
- MediaPipe를 활용한 정밀한 키포인트 추출
- 학습된 딥러닝 모델(TensorFlow/PyTorch)을 이용한 수어 단어/문장 예측
- OpenAI API를 활용한 관계별(부모님, 친구, 동료 등) 텍스트 톤 변환
- 웹 환경에서 작동하는 프론트엔드 애플리케이션

## 프로젝트 구조

프로젝트는 크게 세 가지 주요 폴더로 구성됩니다.

- `backend/`: 수어 키포인트를 처리하고, 모델 예측을 수행하며, 텍스트 변환 로직을 담당하는 FastAPI 서버 코드입니다.
  - `keypoint_processor.py` / `keypoint_processor_transformer.py`: 웹소켓을 통해 프레임을 수신하고 MediaPipe로 키포인트를 추출하여 메인 백엔드로 전송하는 프로세서입니다.
  - `main.py` / `main_transformer.py`: 수신된 키포인트 시퀀스를 모델로 예측하고 OpenAI API를 호출하여 텍스트를 변환하는 메인 백엔드 서버입니다.
  - `requirements.txt`: 백엔드 실행에 필요한 Python 라이브러리 목록입니다.

- `frontend/SignApp/`: 사용자 인터페이스를 제공하는 Expo 기반의 React Native 애플리케이션 코드입니다.
  - 카메라 접근, 영상 스트리밍, 관계 선택, 번역 결과 표시 등의 기능을 구현합니다.
  - 웹소켓을 통해 백엔드와 통신하여 실시간으로 영상 데이터를 보내고 번역 결과를 수신합니다.

- `models/`: 수어 인식을 위해 학습된 딥러닝 모델 파일과 클래스(단어/문장 레이블) 정보가 저장된 폴더입니다. 다양한 버전의 모델이 포함될 수 있습니다.

- `data/`: 각 수어 단어/문장에 해당하는 영상 데이터 또는 관련 정보가 저장될 수 있는 폴더입니다.

## 기술 스택

- **프론트엔드**: React Native (with Expo), TypeScript
- **백엔드**: Python (FastAPI, NumPy, OpenCV, MediaPipe, Requests)
- **머신러닝**: TensorFlow, PyTorch
- **텍스트 변환**: OpenAI API
- **통신**: WebSocket, HTTP

## 설치 및 실행 방법

프로젝트를 실행하기 위해서는 백엔드와 프론트엔드를 각각 설정하고 실행해야 합니다.

### 필수 요구사항

- Python 3.8+
- Node.js 및 npm 또는 yarn
- Expo CLI (`npm install -g expo-cli`)
- OpenAI API Key (텍스트 변환 기능을 사용하려면 필요)

### 1. 백엔드 설정 및 실행

1.  백엔드 디렉토리로 이동합니다.
    ```bash
    cd backend
    ```
2.  Python 가상 환경을 설정하고 활성화합니다 (선택 사항이지만 권장).
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  필요한 Python 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```
4.  OpenAI API 키를 환경 변수로 설정하거나, `backend/main.py` 파일 내에 직접 입력합니다. (보안을 위해 환경 변수 사용 권장)
    ```bash
    # macOS/Linux 예시 (세션 유지)
    export OPENAI_API_KEY='여러분의_API_키'
    # Windows 예시
    set OPENAI_API_KEY='여러분의_API_키'
    ```
5.  FastAPI 서버를 실행합니다.
    ```bash
    uvicorn main:app --reload --port 8000
    ```
    또는 Transformer 모델을 사용하려면:
    ```bash
    uvicorn main_transformer:app --reload --port 8000
    ```
    키포인트 프로세서 서버도 별도로 실행해야 합니다.
    ```bash
    uvicorn keypoint_processor:app --reload --port 8001
    ```
    또는 Transformer 모델용 키포인트 프로세서를 사용하려면:
    ```bash
    uvicorn keypoint_processor_transformer:app --reload --port 8001
    ```
    (두 개의 서버 - `main` 또는 `main_transformer`와 `keypoint_processor` 또는 `keypoint_processor_transformer` - 가 동시에 실행되어야 합니다. 포트 번호가 8000과 8001인지 확인하세요.)

### 2. 프론트엔드 설정 및 실행 (`frontend/SignApp`)

1.  프론트엔드 앱 디렉토리로 이동합니다.
    ```bash
    cd ../frontend/SignApp
    ```
2.  JavaScript 종속성을 설치합니다.
    ```bash
    npm install
    # 또는 yarn install
    ```
3.  Expo 앱을 실행합니다.
    ```bash
    npx expo start
    ```
4.  터미널에 표시되는 옵션 중 웹 브라우저에서 실행(보통 `w` 키 입력)을 선택합니다. (현재 카메라 기능은 웹 환경에서만 구현되어 있습니다.)

## 기여 방법

프로젝트에 기여하고 싶으시다면 언제든지 환영입니다. Pull Request를 보내주시거나 Issue를 등록해주세요.

## 라이센스

이 프로젝트는 [MIT 라이센스](https://opensource.org/licenses/MIT) 하에 제공됩니다. (TBD - 라이센스 파일 생성 필요)

## 연락처

문의사항이 있으시면 [프로젝트 저장소의 Issues]를 이용해주세요.