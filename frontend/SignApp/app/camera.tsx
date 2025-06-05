// app/camera.tsx (또는 CameraScreen.tsx)
import { View, Text, StyleSheet, Platform } from 'react-native';
import { Camera, useCameraPermissions } from 'expo-camera';
import { useState, useEffect, useRef } from 'react';
import { useLocalSearchParams } from 'expo-router';

// ─────────────────────────────────────────────────────────────────────
// 1) 웹에서만 사용되는 WebCamera 컴포넌트
//    → HTML <video> 태그와 getUserMedia를 이용해 브라우저 카메라를 표시
// ─────────────────────────────────────────────────────────────────────
function WebCamera({ style }: { style: any }) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    let currentStream: MediaStream | null = null;

    // 웹 전용: 브라우저의 mediaDevices API로 카메라 스트림 요청
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((stream) => {
        currentStream = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => {
        console.error('웹 카메라 접근 실패:', err);
      });

    // 언마운트 시 스트림 정리
    return () => {
      if (currentStream) {
        currentStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <video
      ref={videoRef}
      style={style}
      autoPlay
      playsInline
      muted
    />
  );
}

// ─────────────────────────────────────────────────────────────────────
// 2) CameraScreen 컴포넌트 (웹 + 모바일 공용)
// ─────────────────────────────────────────────────────────────────────
export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [selectedRelationships, setSelectedRelationships] = useState<string[]>([]);
  const [translatedText, setTranslatedText] = useState<string>('');
  const params = useLocalSearchParams();

  // 페이지가 마운트되면 카메라 권한 요청하고,
  // URL 파라미터(selectedRelationships)가 있으면 파싱해서 상태에 저장
  useEffect(() => {
    (async () => {
      if (!permission) {
        await requestPermission();
      }
    })();

    if (params.selectedRelationships) {
      try {
        const rels = JSON.parse(params.selectedRelationships as string);
        if (Array.isArray(rels)) {
          setSelectedRelationships(rels);
        }
      } catch (error) {
        console.error('Failed to parse relationships parameter:', error);
      }
    }
  }, [permission, params.selectedRelationships]);

  // (예시) 백엔드로 수어 데이터를 전송하는 함수 (사용 예시는 별도로 추가)
  const sendSignLanguageData = async (signLanguageData: any) => {
    const backendUrl = 'YOUR_BACKEND_API_URL/process_sign_language';
    try {
      const response = await fetch(backendUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sign_language_data: signLanguageData,
          relationships: selectedRelationships,
        }),
      });
      if (response.ok) {
        const result = await response.json();
        if (result && result.translated_text) {
          setTranslatedText(result.translated_text);
        }
      } else {
        console.error('Backend request failed:', response.status, response.statusText);
      }
    } catch (error) {
      console.error('Error sending data to backend:', error);
    }
  };

  // ────────────────────────────────────────────────────────────
  // 권한 로딩 중
  if (!permission) {
    return (
      <View style={styles.loadingContainer}>
        <Text>Loading camera permissions…</Text>
      </View>
    );
  }

  // 권한 거부 상태
  if (!permission.granted) {
    return (
      <View style={styles.permissionDeniedContainer}>
        <Text style={{ textAlign: 'center' }}>카메라 접근 권한이 필요합니다.</Text>
      </View>
    );
  }

  // 웹(Web) 환경에서는 <Camera>를 사용하지 못하므로, WebCamera(HTML<video>)를 렌더링
  if (Platform.OS === 'web') {
    return (
      <View style={styles.container}>
        {/* 웹용 카메라 영역: 흰색 배경, 상단 둥근 모서리 */}
        <View style={styles.cameraContainerWebOuter}>
          <View style={styles.cameraContainerWebInner}>
            <WebCamera style={styles.webVideo} />
          </View>
        </View>
        {/* 번역 결과 영역 */}
        <View style={styles.translationContainer}>
          <Text style={styles.relationshipText}>
            선택된 관계: {selectedRelationships.join(', ')}
          </Text>
          <Text style={styles.translatedText}>
            {translatedText || '번역된 텍스트 출력'}
          </Text>
        </View>
      </View>
    );
  }

  // Android / iOS 환경 (expo-camera의 <Camera> 사용)
  return (
    <View style={styles.container}>
      {/* 모바일용 카메라 영역: 흰색 배경, 상단 둥근 모서리 */}
      <View style={styles.cameraContainerOuter}>
        <View style={styles.cameraContainerInner}>
          {/* CameraType.back 사용 방식이 Expo 버전에 따라 다를 수 있어 Camera.Constants.Type.back으로 수정 */}
          <Camera style={styles.camera} type={Camera.Constants.Type.back} />
        </View>
      </View>
      {/* 번역 결과 영역 */}
      <View style={styles.translationContainer}>
        <Text style={styles.relationshipText}>
          선택된 관계: {selectedRelationships.join(', ')}
        </Text>
        <Text style={styles.translatedText}>
          {translatedText || '번역된 텍스트 출력'}
        </Text>
      </View>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────
// 3) 스타일 정의
// ─────────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#AEE1D4',
    padding: 20, // 전체 컨테이너에 패딩 추가
    paddingBottom: 0, // 하단 번역 영역과의 간격 조절
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  permissionDeniedContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  // ─ 웹 비디오용 컨테이너 (바깥쪽: 녹색 배경) ─
  cameraContainerWebOuter: {
    flex: 2, // 상단 카메라 영역 비율
    backgroundColor: '#AEE1D4', // 바깥 배경 색상 (컨테이너와 동일)
    marginBottom: 10, // 하단 영역과의 간격
  },
  // ─ 웹 비디오용 컨테이너 (안쪽: 흰색 배경, 둥근 모서리) ─
  cameraContainerWebInner: {
    flex: 1,
    backgroundColor: '#FFFFFF', // 흰색 배경
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    overflow: 'hidden', // 둥근 모서리 밖으로 나가는 내용 자르기
  },
  webVideo: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  // ─ 모바일 카메라용 컨테이너 (바깥쪽: 녹색 배경) ─
  cameraContainerOuter: {
    flex: 2, // 상단 카메라 영역 비율
    backgroundColor: '#AEE1D4', // 바깥 배경 색상 (컨테이너와 동일)
    marginBottom: 10, // 하단 영역과의 간격
  },
  // ─ 모바일 카메라용 컨테이너 (안쪽: 흰색 배경, 둥근 모서리) ─
  cameraContainerInner: {
    flex: 1,
    backgroundColor: '#FFFFFF', // 흰색 배경
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    overflow: 'hidden', // 둥근 모서리 밖으로 나가는 내용 자르기
  },
  camera: {
    flex: 1,
    width: '100%',
  },
  // ─ 번역 결과 영역 ─
  translationContainer: {
    flex: 1, // 하단 번역 영역 비율
    backgroundColor: '#FFFFFF', // 흰색 배경
    alignItems: 'center',
    justifyContent: 'center',
    padding: 10,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    // marginTop: -20, // 이전 코드의 겹침 효과. 필요에 따라 조정 또는 제거
  },
  relationshipText: {
    fontSize: 14,
    color: '#333333',
    marginBottom: 5,
  },
  translatedText: {
    fontSize: 18,
    color: '#1D3D47',
    textAlign: 'center',
  },
});
