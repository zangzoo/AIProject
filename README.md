# 👋 수어 TALK 프로젝트

## ✨ 프로젝트 소개

이 프로젝트는 **실시간 수어(수화) 인식**을 위한 혁신적인 애플리케이션입니다. 사용자의 카메라 영상을 실시간으로 분석하여 수어 동작의 핵심 키포인트를 정밀하게 추출하고, 최신 머신러닝 모델을 활용하여 해당 수어가 어떤 단어나 문장인지 예측합니다.

**약 30개의 자주 사용되는 문장**에 대해 각 문장당 20개씩 직접 촬영한 데이터를 사용하여 모델을 학습시켰습니다. 이를 통해 일상적인 대화에서 필요한 기본적인 수어를 효과적으로 인식할 수 있습니다.

더 나아가, 예측된 수어 텍스트는 대화 상대방과의 관계(부모님, 친구, 동료 등)에 맞춰 **가장 자연스럽고 어울리는 말투**로 자동 변환되어 사용자에게 제공됩니다. 수어를 모르는 사람도 수어 사용자과 더욱 쉽게 소통할 수 있도록 돕는 것을 목표로 합니다.

**주요 기능:**

-   🎥 실시간 카메라 영상 기반 수어 인식
-   🦴 MediaPipe를 활용한 정확한 신체 및 손 키포인트 추출
-   🧠 학습된 딥러닝 모델 (TensorFlow / PyTorch) 기반의 높은 정확도 수어 예측
-   💬 OpenAI API를 활용한 맥락 및 관계 기반 텍스트 톤 변환
-   🌐 웹 환경에서 즉시 실행 가능한 사용자 친화적인 프론트엔드

---

## 🏗️ 프로젝트 구조

프로젝트는 다음과 같이 명확하게 구분된 세 가지 핵심 폴더로 구성됩니다.

-   `backend/`
    : 수어 키포인트 데이터 처리, 모델 예측 실행, OpenAI API 연동을 담당하는 **Python FastAPI 서버** 코드입니다.
    -   `keypoint_processor.py` / `keypoint_processor_transformer.py`: 웹소켓을 통해 영상 프레임을 수신하고 MediaPipe를 이용해 키포인트를 추출한 뒤, 메인 백엔드로 전송하는 역할을 합니다.
    -   `main.py` / `main_transformer.py`: 수신된 키포인트 시퀀스를 로드된 딥러닝 모델로 예측하고 OpenAI API를 호출하여 최종 텍스트를 변환하는 핵심 서버 로직입니다.
    -   `requirements.txt`: 백엔드 실행에 필요한 모든 Python 라이브러리 목록입니다.

-   `frontend/SignApp/`
    : 사용자와 상호작용하는 **Expo 기반 React Native 애플리케이션** 코드입니다.
    -   카메라 접근 및 미리보기, 관계 선택 UI, 실시간 번역 결과 표시 등 사용자 경험과 관련된 기능을 구현합니다.
    -   백엔드와 웹소켓 통신을 통해 실시간으로 영상 데이터를 스트리밍하고 번역 결과를 수신합니다.

-   `models/`
    : 수어 인식을 위해 **사전에 학습된 딥러닝 모델 파일**(`.h5` 또는 `.pt`) 및 모델이 예측할 수 있는 **수어 클래스(단어/문장 레이블) 정보**가 저장된 공간입니다. 다양한 모델 실험 결과가 포함될 수 있습니다.

-   `data/` (선택 사항)
    : 특정 수어 단어/문장에 해당하는 원본 영상 데이터 또는 추가 학습 관련 데이터가 저장될 수 있는 폴더입니다.

---

## 🛠️ 기술 스택

이 프로젝트는 다음과 같은 기술들로 개발되었습니다.

-   **프론트엔드**:
    -   ⚛️ **React Native (with Expo):** 크로스 플랫폼 모바일 및 웹 앱 개발
    -   🟦 **TypeScript:** 정적 타입 체크를 통한 코드 안정성 확보

-   **백엔드**:
    -   🐍 **Python:** 핵심 로직 구현 및 데이터 처리
    -   🚀 **FastAPI:** 빠른 비동기 웹 프레임워크
    -   🔢 **NumPy:** 계산 및 데이터 처리
    -   🖥️ **OpenCV:** 실시간 영상 처리
    -   ✋ **MediaPipe:** 고성능 키포인트 추출 솔루션
    -   🌐 **Requests:** HTTP 클라이언트 통신

-   **머신러닝 & AI**:
    -   🧠 **TensorFlow / PyTorch:** 딥러닝 모델 개발 및 실행
    -   🤖 **OpenAI API:** 자연스러운 텍스트 톤 변환

-   **통신 프로토콜**:
    -   📡 **WebSocket:** 실시간 양방향 통신 (영상 스트리밍 및 결과 수신)
    -   🔗 **HTTP:** 일반적인 API 통신

---

## ▶️ 설치 및 실행 방법

프로젝트를 로컬 환경에서 실행하기 위해서는 백엔드 서버와 프론트엔드 애플리케이션을 각각 설정하고 실행해야 합니다.

### 필수 요구사항

-   ✅ **Python 3.8+**: 백엔드 실행 환경
-   ✅ **Node.js 및 npm 또는 yarn**: 프론트엔드 실행 환경
-   ✅ **Expo CLI**: 프론트엔드 개발 도구 (`npm install -g expo-cli`)
-   ✅ **OpenAI API Key**: 텍스트 변환 기능 활성화 (`sk-...` 형태)
-   ✅ **Git LFS**: 대용량 모델 파일 관리에 필요 (설치 가이드: [https://git-lfs.github.com/](https://git-lfs.github.com/))

### 1. 백엔드 설정 및 실행 (`backend/`)

1.  프로젝트 루트 디렉토리에서 백엔드 디렉토리로 이동합니다.
    ```bash
    cd backend
    ```

2.  Python 가상 환경을 설정하고 활성화하는 것을 권장합니다.
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

3.  `requirements.txt`에 명시된 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

4.  OpenAI API 키를 **환경 변수**로 설정합니다. (코드를 직접 수정하는 대신 이 방법을 강력히 권장합니다.)
    ```bash
    # macOS/Linux 터미널 (현재 세션 유지)
    export OPENAI_API_KEY='여러분의_실제_API_키'
    # Windows 명령 프롬프트
    set OPENAI_API_KEY='여러분의_실제_API_키'
    # 또는 .env 파일을 사용하고 싶다면 python-dotenv 등을 활용하세요.
    ```

5.  **두 개의 백엔드 서버**를 각각 다른 터미널에서 실행합니다.
    -   **메인 백엔드 (모델 예측 및 GPT 변환):**
        ```bash
        uvicorn main:app --reload --port 8000
        # 또는 Transformer 모델 사용 시: uvicorn main_transformer:app --reload --port 8000
        ```
    -   **키포인트 프로세서 (프레임 수신 및 키포인트 추출):**
        ```bash
        uvicorn keypoint_processor:app --reload --port 8001
        # 또는 Transformer 모델용 프로세서 사용 시: uvicorn keypoint_processor_transformer:app --reload --port 8001
        ```
    💡 포트 번호 `8000`과 `8001`이 사용 가능한지 확인해주세요.

### 2. 프론트엔드 설정 및 실행 (`frontend/SignApp/`)

1.  프로젝트 루트 디렉토리에서 프론트엔드 앱 디렉토리로 이동합니다.
    ```bash
    cd ../frontend/SignApp
    ```

2.  JavaScript 종속성을 설치합니다.
    ```bash
    npm install
    # 또는 yarn install
    ```

3.  Expo 개발 서버를 실행합니다.
    ```bash
    npx expo start
    ```

4.  터미널에 표시되는 QR 코드나 링크를 통해 개발 빌드, 에뮬레이터/시뮬레이터 또는 **웹 브라우저**로 앱을 실행합니다. (현재 카메라 기능은 웹 환경에서 가장 잘 작동합니다.)


---

## 📄 라이센스

이 프로젝트는 [MIT 라이센스](https://opensource.org/licenses/MIT) 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참고해주세요. (LICENSE 파일 생성 예정)

---

## 📬 문의

프로젝트에 대한 질문이나 제안이 있으시면 언제든지 GitHub 저장소의 [Issues] 탭을 통해 문의해주세요.

[Issues]: https://github.com/zangzoo/AIProject/issues