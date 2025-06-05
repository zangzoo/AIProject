import React, { useState, useEffect } from 'react';
import { View, StyleSheet, TouchableOpacity, Dimensions } from 'react-native';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';

// 이미지 파일 임포트
const splashImage1 = require('@/assets/images/start_2s.png');
const splashImage2 = require('@/assets/images/start_camera.png');

export default function HomeScreen() {
  const [showSplashScreen1, setShowSplashScreen1] = useState(true);
  const router = useRouter();
  const insets = useSafeAreaInsets();

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowSplashScreen1(false);
    }, 2000);
    return () => clearTimeout(timer);
  }, []);

  const { width: SCREEN_WIDTH } = Dimensions.get('window');

  if (showSplashScreen1) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <ThemedView style={styles.container}>
          {/* 텍스트 영역: flex 1.2 */}
          <View style={[styles.firstTitleContainer, { flex: 2 }]}>
            <ThemedText type="title" style={styles.firstSplashText}>
              수어
            </ThemedText>
            <ThemedText type="title" style={styles.firstSplashText}>
              TALK
            </ThemedText>
          </View>

          {/* 이미지 영역: flex 8.8 */}
          <View style={[styles.firstImageContainer, { flex: 8.8 }]}>
            <Image
              source={splashImage1}
              style={{
                width: SCREEN_WIDTH * 1.2,
                height: '100%',
              }}
              contentFit="contain"
            />
          </View>
        </ThemedView>
      </SafeAreaView>
    );
  }

  // 두 번째 스플래시 화면
  return (
    <SafeAreaView style={styles.safeArea}>
      <ThemedView style={styles.container}>
        {/* 상단 텍스트: 화면 높이의 20% */}
        <View style={[styles.secondTitleContainer, { flex: 2 }]}>
          <ThemedText type="title" style={styles.secondSplashText}>
            수어톡
          </ThemedText>
        </View>

        {/* 중간 이미지: 화면 높이의 60% */}
        <TouchableOpacity
          onPress={() => router.push('/relationship-select')}
          style={[styles.secondImageWrapper, { flex: 4 }]}
        >
          <Image
            source={splashImage2}
            style={{
              width: SCREEN_WIDTH * 0.7,
              height: '100%',
            }}
            contentFit="contain"
          />
        </TouchableOpacity>

        {/* 하단 안내문구: 화면 높이의 20% */}
        <View
          style={[
            styles.secondCaptionContainer,
            {
              flex: 2,
              paddingBottom: insets.bottom + 60,
            },
          ]}
        >
          <ThemedText style={styles.cameraText}>
            카메라를 들고있는 가락이를 클릭해 주세요!
          </ThemedText>
        </View>
      </ThemedView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#AEE1D4',
  },
  container: {
    flex: 1,
    backgroundColor: '#AEE1D4',
    alignItems: 'center',
    justifyContent: 'center',
  },

  // 첫 번째 스플래시 스타일
  firstTitleContainer: {
    justifyContent: 'flex-end',   // 텍스트를 아래로 붙임
    alignItems: 'center',
    marginBottom:  -20,           // 이미지와 더 가까워지도록 살짝 겹치게
  },
  firstSplashText: {
    fontSize: 60,
    fontWeight: 'bold',
    color: '#264D4D',
    lineHeight: 40,
    marginBottom: 15,
  },
  firstImageContainer: {
    justifyContent: 'flex-start', // 이미지를 위로 붙임
    alignItems: 'center',
    marginTop: -90,               // 텍스트와 더 가까워지도록 위로 올림
  },

  // 두 번째 스플래시 스타일
  secondTitleContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  secondSplashText: {
    fontSize: 52,
    fontWeight: 'bold',
    color: '#264D4D',
  },
  secondImageWrapper: {
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
  },
  secondCaptionContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  cameraText: {
    fontSize: 23,
    color: '#264D4D',
    textAlign: 'center',
    lineHeight: 24,
  },
});