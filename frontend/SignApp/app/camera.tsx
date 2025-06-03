import { View, Text, StyleSheet } from 'react-native';
import { Camera, CameraType, useCameraPermissions } from 'expo-camera';
import { useState, useEffect } from 'react';
import { useLocalSearchParams } from 'expo-router';

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [selectedRelationships, setSelectedRelationships] = useState<string[]>([]);
  const params = useLocalSearchParams();

  useEffect(() => {
    (async () => {
      if (!permission) {
        await requestPermission();
      }
    })();

    if (params.selectedRelationships) {
      try {
        const relationships = JSON.parse(params.selectedRelationships as string);
        if (Array.isArray(relationships)) {
          setSelectedRelationships(relationships);
        }
      } catch (error) {
        console.error('Failed to parse relationships parameter:', error);
      }
    }

  }, [permission, params.selectedRelationships]);

  const sendSignLanguageData = async (signLanguageData: any) => {
    const backendUrl = 'YOUR_BACKEND_API_URL/process_sign_language';
    try {
      const response = await fetch(backendUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sign_language_data: signLanguageData,
          relationships: selectedRelationships,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Backend response:', result);
      } else {
        console.error('Backend request failed:', response.status, response.statusText);
      }
    } catch (error) {
      console.error('Error sending data to backend:', error);
    }
  };

  if (!permission) {
    // Camera permissions are still loading
    return <View />;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: 'center' }}>We need your permission to show the camera</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera style={styles.camera} type={CameraType.back} />
      <Text style={{ position: 'absolute', top: 50, color: 'white' }}>
        선택된 관계: {selectedRelationships.join(', ')}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  camera: {
    flex: 1,
    width: '100%',
  },
});