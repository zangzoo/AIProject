import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from scipy.interpolate import interp1d
import random

# --- ✅ 데이터 로드 ---
data = np.load("C:/SignLang_3/processed_dataset/sign_keypoints.npz")
X = data['X']
y = data['y']
classes = data['classes']

# --- ✅ 증강 함수 ---
def add_noise(X, noise_level=0.05):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=X.shape)
    return X + noise

def time_warp(X, max_warp=0.1):
    timesteps = X.shape[0]
    warp_factor = 1 + np.random.uniform(-max_warp, max_warp)
    new_timesteps = int(timesteps * warp_factor)
    time_old = np.linspace(0, 1, timesteps)
    time_new = np.linspace(0, 1, new_timesteps)
    f = interp1d(time_old, X, axis=0, kind='linear')
    X_warped = f(time_new)
    if new_timesteps > timesteps:
        X_warped = X_warped[:timesteps]
    else:
        pad_len = timesteps - new_timesteps
        X_warped = np.pad(X_warped, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
    return X_warped

def horizontal_flip(X):
    X_flipped = X.copy()
    X_flipped[:, ::2] = -X_flipped[:, ::2]
    return X_flipped

def augment_sample(X_sample):
    X_aug = add_noise(X_sample)
    X_aug = time_warp(X_aug)
    X_aug = horizontal_flip(X_aug)
    return X_aug

# --- ✅ 클래스별 증강 개수 조절 함수 ---
def augment_to_balance(X, y, min_samples_per_class=300):
    X_augmented = list(X)
    y_augmented = list(y)
    unique_classes, counts = np.unique(y, return_counts=True)
    class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}

    for cls, count in zip(unique_classes, counts):
        needed = max(0, min_samples_per_class - count)
        for _ in range(needed):
            idx = random.choice(class_indices[cls])
            augmented = augment_sample(X[idx])
            X_augmented.append(augmented)
            y_augmented.append(cls)

    return np.array(X_augmented, dtype=np.float32), np.array(y_augmented)

# --- ✅ 학습/검증 데이터 분리 ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- ✅ 클래스별 증강 적용 ---
X_train_aug, y_train_aug = augment_to_balance(X_train, y_train, min_samples_per_class=100)
print(f"훈련 데이터: 원본 {len(X_train)} -> 증강 후 {len(X_train_aug)}")

# --- ✅ 클래스 가중치 계산 ---
class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_aug), y=y_train_aug)
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

# --- ✅ 커스텀 Attention 레이어 ---
class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()
        self.W = tf.keras.layers.Dense(1)

    def call(self, inputs):
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

# --- ✅ 모델 구성 ---
input_layer = Input(shape=(X.shape[1], X.shape[2]))
x = Masking(mask_value=0.0)(input_layer)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dropout(0.3)(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.3)(x)
x = Attention()(x)
x = Dense(128, activation='relu')(x)
output_layer = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- ✅ 콜백 설정 ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

# --- ✅ 모델 학습 ---
history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    class_weight=class_weights,
    callbacks=callbacks
)

# --- ✅ 평가 ---
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"\n✅ 검증 정확도: {val_acc:.2f}")

y_pred = np.argmax(model.predict(X_val), axis=1)
print("\n📊 분류 리포트:")
print(classification_report(y_val, y_pred, target_names=classes))

# model.save("sign_model_lstm.h5")