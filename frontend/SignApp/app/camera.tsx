// app/camera.tsx

import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Platform,
  Dimensions,
  TouchableOpacity,
} from 'react-native';
import { useLocalSearchParams } from 'expo-router';
import { Buffer } from 'buffer'; // base64 → 바이너리 변환

// 웹 전용 WebSocket 서버 주소 (실제 환경에 맞게 변경)
const KEYPOINT_PROCESSOR_WS_URL = 'ws://127.0.0.1:8001/ws/video';

export default function CameraScreen() {
  const { selectedRelationships: relationshipsJson } = useLocalSearchParams();
  const relationships: string[] = relationshipsJson
    ? JSON.parse(relationshipsJson as string)
    : [];

  // 웹캠(video) 관련
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // 화면에 표시할 번역 결과
  const [translatedText, setTranslatedText] = useState<string>(
    '번역된 텍스트가 여기에 표시됩니다.'
  );
  // 웹캠 스트림이 준비되었는지 여부
  const [streamReady, setStreamReady] = useState<boolean>(false);
  // 프레임 전송 타이머 ID
  const frameIntervalRef = useRef<number | null>(null);

  // 1) 웹(Web)일 때에만 아래 effect 실행
  useEffect(() => {
    if (Platform.OS !== 'web') {
      // Expo Go(iOS/Android) 환경에서는 Web쪽 로직을 사용하지 않음
      return;
    }

    let localStream: MediaStream | null = null;

    // WebSocket 연결
    const clientId = Date.now().toString();
    const wsUrl = `${KEYPOINT_PROCESSOR_WS_URL}?client_id=${clientId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[CameraScreen] WebSocket 연결됨 →', wsUrl);
      // 첫 연결 시 “관계 정보” JSON으로 전송
      const relMsg = {
        type: 'relationships',
        data: relationships,
      };
      ws.send(JSON.stringify(relMsg));
      console.log('[CameraScreen] 관계 정보 전송 →', relMsg);
    };

    ws.onmessage = (event) => {
      console.log('[CameraScreen] WebSocket 메시지 수신 →', event.data);
      try {
        const msg = JSON.parse(event.data);
        if (msg.translated_text) {
          setTranslatedText(msg.translated_text);
        }
      } catch (e) {
        console.warn('[CameraScreen] 수신 메시지 파싱 오류:', e);
      }
    };

    ws.onerror = (ev) => {
      console.error('[CameraScreen] WebSocket 오류 →', ev);
      setTranslatedText('WebSocket 에러 발생');
    };

    ws.onclose = (ev) => {
      console.log(
        `[CameraScreen] WebSocket 연결 해제 (code=${ev.code}, reason=${ev.reason})`
      );
      stopFrameCapture();
      wsRef.current = null;
    };

    // getUserMedia로 웹캠 스트림 가져오기
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: 'environment' }, audio: false })
      .then((stream) => {
        localStream = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
          setStreamReady(true);
        }
      })
      .catch((err) => {
        console.error('[CameraScreen] webcam 접근 실패 →', err);
        setTranslatedText('웹캠 접근 권한이 필요합니다.');
      });

    return () => {
      // 언마운트 시: 스트림 해제, WebSocket 닫기, 타이머 중지
      if (localStream) {
        localStream.getTracks().forEach((t) => t.stop());
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
      stopFrameCapture();
    };
  }, []);

  // 2) streamReady가 true(웹캠 준비됨)이면, 프레임 전송 시작
  useEffect(() => {
    if (Platform.OS !== 'web') return;
    if (streamReady && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      startFrameCapture();
    }
  }, [streamReady]);

  // 3) 프레임 캡처 및 WebSocket 전송 (웹 전용)
  const startFrameCapture = () => {
    if (frameIntervalRef.current !== null) return;

    frameIntervalRef.current = window.setInterval(() => {
      if (
        !videoRef.current ||
        videoRef.current.readyState < HTMLVideoElement.HAVE_CURRENT_DATA
      ) {
        return;
      }
      if (!canvasRef.current) return;
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        stopFrameCapture();
        return;
      }

      // 캔버스에 현재 비디오 프레임 그리기
      const videoEl = videoRef.current;
      const canvasEl = canvasRef.current;
      const ctx = canvasEl.getContext('2d');
      if (!ctx) return;

      const width = videoEl.videoWidth;
      const height = videoEl.videoHeight;

      // 캔버스 크기를 비디오와 동일하게 설정
      canvasEl.width = width;
      canvasEl.height = height;
      ctx.drawImage(videoEl, 0, 0, width, height);

      // 캔버스 → base64 데이터 URL
      const dataUrl = canvasEl.toDataURL('image/jpeg', 0.6); // JPEG 압축율 0.6
      // dataUrl: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
      // prefix 제거 후 base64만 남김
      const base64 = dataUrl.split(',')[1];
      // base64 → 바이너리(Buffer) 변환
      const binary = Buffer.from(base64, 'base64');
      // WebSocket 바이너리 전송
      wsRef.current.send(binary);
      // console.log('[CameraScreen] 프레임 전송 (바이너리 길이=', binary.length, ')');
    }, 100); // 100ms 간격
  };

  const stopFrameCapture = () => {
    if (frameIntervalRef.current !== null) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
      console.log('[CameraScreen] 프레임 전송 중지');
    }
  };

  // 4) Expo Go(iOS/Android) 환경 안내
  if (Platform.OS !== 'web') {
    return (
      <View style={styles.centerContainer}>
        <Text>현재 웹 브라우저 환경에서만 테스트 가능합니다.</Text>
        <Text>Expo Go 앱(iOS/Android)에서는 별도 네이티브 구현이 필요합니다.</Text>
      </View>
    );
  }

  // 5) 웹(Web) 화면 렌더링
  const screenW = Dimensions.get('window').width;
  // 비율 4:3로 높이 계산
  const videoH = (screenW * 4) / 3;

  return (
    <View style={styles.container}>
      <View style={[styles.videoContainer, { height: videoH }]}>
        {/* HTML5 <video> 태그 */}
        <video
          ref={videoRef}
          style={{ width: '100%', height: '100%', borderRadius: 12 }}
          muted
          playsInline
        />
        {/* 보이지 않는 <canvas> (프레임 캡처용) */}
        <canvas
          ref={canvasRef}
          style={{ display: 'none' }}
        />
      </View>
      <View style={styles.textContainer}>
        <Text style={styles.translatedText}>{translatedText}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  container: {
    flex: 1,
    backgroundColor: '#AEE1D4',
    alignItems: 'center',
    paddingTop: 30,
  },
  videoContainer: {
    width: '95%',
    backgroundColor: '#000',
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 20,
  },
  textContainer: {
    width: '95%',
    backgroundColor: '#FFF',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
  },
  translatedText: {
    fontSize: 18,
    color: '#333',
    textAlign: 'center',
  },
});
