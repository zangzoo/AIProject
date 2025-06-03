import { Image } from 'expo-image';
import { Platform, StyleSheet, TouchableOpacity, View } from 'react-native';
import { useState, useEffect } from 'react';
import { Link, useRouter } from 'expo-router';

import { HelloWave } from '@/components/HelloWave';
import ParallaxScrollView from '@/components/ParallaxScrollView';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';

// 이미지 파일 임포트
const splashImage1 = require('@/assets/images/start_2s.png');
const splashImage2 = require('@/assets/images/start_camera.png');

export default function HomeScreen() {
  const [showSplashScreen1, setShowSplashScreen1] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowSplashScreen1(false);
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  if (showSplashScreen1) {
    // 첫 번째 시작 화면 렌더링
    return (
      <ThemedView style={styles.container}>
        <ThemedText type="title" style={styles.splashText}>수어</ThemedText>
        <ThemedText type="title" style={styles.splashText}>TALK</ThemedText>
        <Image source={splashImage1} style={styles.splashImage} contentFit="contain" />
      </ThemedView>
    );
  } else {
    // 두 번째 시작 화면 렌더링
    return (
      <ThemedView style={styles.container}>
        <ThemedText type="title" style={styles.splashText}>수어톡</ThemedText>
        <TouchableOpacity onPress={() => router.push('/relationship-select')} style={styles.splashImage}>
          <View style={styles.splashImage}>
            <Image source={splashImage2} style={styles.splashImage} contentFit="contain" />
          </View>
        </TouchableOpacity>
        <ThemedText style={styles.cameraText}>카메라를 들고있는 가락이를 클릭해 주세요!</ThemedText>
      </ThemedView>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#AEE1D4', // 연한 하늘색 배경
    paddingBottom: 50, // 하단 여백 추가
  },
  splashText: { // 시작 화면 텍스트 스타일
    fontSize: 50,
    fontWeight: 'bold',
    color: '#264D4D', // 어두운 색상
    marginVertical: 5,
  },
  splashImage: { // 시작 화면 이미지 스타일
    width: '100%', // 너비 조정
    height: '75%', // 높이 조정
    marginVertical: 20,
  },
  cameraText: { // 두 번째 화면 하단 텍스트 스타일
    fontSize: 18,
    color: '#264D4D', // 어두운 색상
    textAlign: 'center',
    marginTop: 20,
  },
  titleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  stepContainer: {
    gap: 8,
    marginBottom: 8,
  },
  reactLogo: {
    height: 178,
    width: 290,
    bottom: 0,
    left: 0,
    position: 'absolute',
  },
});
