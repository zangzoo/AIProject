import { View, Text, StyleSheet, TouchableOpacity, TextInput } from 'react-native';
import { Link, useRouter } from 'expo-router'; // useRouter 임포트
import React, { useState } from 'react'; // React 임포트 추가

// 관계 객체 타입 정의
interface Relationship {
  name: string;
  isCustom: boolean;
}

export default function RelationshipSelectScreen() {
  const [relationships, setRelationships] = useState<Relationship[]>([
    { name: '부모님', isCustom: false },
    { name: '친구', isCustom: false },
    { name: '동료', isCustom: false },
    { name: '선생님', isCustom: false },
  ]); // 관계 목록 상태 (객체 배열)
  const [customRelationship, setCustomRelationship] = useState('');
  const [selectedRelationships, setSelectedRelationships] = useState<string[]>([]);

  const router = useRouter(); // useRouter 훅 호출

  // 관계 추가 함수
  const addRelationship = () => {
    if (customRelationship.trim() !== '' && !relationships.some(rel => rel.name === customRelationship.trim())) {
      const newRelationshipName = customRelationship.trim();
      const newRelationship: Relationship = { name: newRelationshipName, isCustom: true }; // 직접 추가된 관계 표시
      setRelationships([...relationships, newRelationship]);
      setSelectedRelationships([...selectedRelationships, newRelationshipName]); // 추가된 관계 자동 선택
      setCustomRelationship('');
    }
  };

  // 관계 삭제 함수
  const removeRelationship = (name: string) => {
    setRelationships(relationships.filter(rel => rel.name !== name));
    setSelectedRelationships(selectedRelationships.filter(item => item !== name));
  };

  // 관계 선택/해제 함수
  const toggleSelectRelationship = (name: string) => {
    if (selectedRelationships.includes(name)) {
      setSelectedRelationships(selectedRelationships.filter(item => item !== name));
    } else {
      setSelectedRelationships([...selectedRelationships, name]);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>상대방과의 관계를 선택해주세요</Text>

      <View style={styles.optionsContainer}>
        {/* 관계 선택 옵션들 및 삭제 버튼 */}
        {relationships.map((relationship, index) => (
          <View key={index} style={styles.optionRow}> {/* 새로운 컨테이너 추가 */}
            <TouchableOpacity
              style={styles.option}
              onPress={() => toggleSelectRelationship(relationship.name)}>
              <View style={[styles.checkbox, selectedRelationships.includes(relationship.name) && styles.checkboxSelected]} />
              <Text style={styles.optionText}>{relationship.name}</Text>
            </TouchableOpacity>
            {relationship.isCustom && ( // 직접 추가된 관계인 경우에만 삭제 버튼 표시
              <TouchableOpacity onPress={() => removeRelationship(relationship.name)} style={styles.deleteButton}> {/* 삭제 버튼 */}
                <Text style={styles.deleteButtonText}>X</Text>
              </TouchableOpacity>
            )}
          </View>
        ))}

        {/* 직접 입력 필드와 추가 버튼 */}
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            placeholder="선택지에 없다면 직접 입력해주세요"
            placeholderTextColor="#264D4D"
            value={customRelationship}
            onChangeText={setCustomRelationship}
            onSubmitEditing={addRelationship}
          />
          <TouchableOpacity onPress={addRelationship} style={styles.addButton}>
            <Text style={styles.addButtonText}>✓</Text>
          </TouchableOpacity>
        </View>

      </View>

      {/* 촬영 START 버튼 - 카메라 화면으로 연결 */}
      <TouchableOpacity style={styles.startButton} onPress={() => {
        // 선택된 관계 값을 파라미터로 전달하며 카메라 화면으로 이동
        router.push({
          pathname: '/camera',
          params: { selectedRelationships: JSON.stringify(selectedRelationships) },
        });
      }}>
        <Text style={styles.startButtonText}>촬영 START</Text>
      </TouchableOpacity>

      {/* 임시: 선택된 관계 확인 */}
      {/* <Text>선택된 관계: {selectedRelationships.join(', ')}</Text> */}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#AEE1D4', // 배경색
    alignItems: 'center',
    paddingTop: 50, // 상단 여백
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#264D4D',
    marginBottom: 30,
  },
  optionsContainer: {
    backgroundColor: '#FFFFFF', // 흰색 배경
    borderRadius: 20,
    padding: 20,
    width: '90%', // 너비 조정
    marginBottom: 30,
  },
  optionRow: { // 관계 항목과 삭제 버튼을 담는 컨테이너 스타일
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between', // 양쪽 정렬
    marginBottom: 15,
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  checkbox: { // 체크박스 임시 스타일
    width: 24,
    height: 24,
    backgroundColor: '#E0E0E0', // 회색 체크박스
    marginRight: 15,
    borderRadius: 4,
  },
  optionText: {
    fontSize: 18,
    color: '#333333',
  },
  inputContainer: { // 직접 입력 필드와 버튼 컨테이너 스타일
    flexDirection: 'row', // 가로 방향 배치
    alignItems: 'center',
    marginTop: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#cccccc',
    paddingVertical: 5, // 세로 패딩 추가
  },
  input: { // 직접 입력 필드 스타일
    flex: 1, // 남은 공간 모두 차지
    fontSize: 18,
    color: '#264D4D',
    paddingVertical: 0, // 기본 패딩 제거
  },
  addButton: { // 추가 버튼 스타일
    marginLeft: 10,
    padding: 5,
    //backgroundColor: '#E0E0E0', // 버튼 배경색
    borderRadius: 5,
  },
  addButtonText: { // 추가 버튼 텍스트 스타일
    fontSize: 18,
    color: '#264D4D',
    fontWeight: 'bold',
  },
  startButton: {
    backgroundColor: '#FFFFFF', // 흰색 배경
    borderRadius: 30,
    paddingVertical: 15,
    paddingHorizontal: 40,
    elevation: 3, // 안드로이드 그림자
    shadowColor: '#000', // iOS 그림자
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  startButtonText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1D3D47',
  },
  checkboxSelected: { // 체크박스 선택 시 스타일
    backgroundColor: '#264D4D', // 선택된 체크박스 색상
  },
  deleteButton: { // 삭제 버튼 스타일
    marginLeft: 10,
    padding: 5,
    //backgroundColor: '#E0E0E0', // 버튼 배경색
    borderRadius: 5,
  },
  deleteButtonText: { // 삭제 버튼 텍스트 스타일
    fontSize: 14,
    color: '#264D4D', // 빨간색
    fontWeight: 'bold',
  },
});
