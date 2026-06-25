import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==========================================
# 1. PERSIAPAN DATA
# ==========================================
DATA_FILE = 'processed_data/landmarks_numbers.csv'
MODEL_DIR = 'models'

MODEL_FILE = os.path.join(MODEL_DIR, 'numbers_cnn1d_model.h5')
ENCODER_FILE = os.path.join(MODEL_DIR, 'number_cnn1d_encoder.pkl')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("Membaca data CSV angka...")
df = pd.read_csv(DATA_FILE)

X = df.drop('label', axis=1).values
y_text = df['label'].values

# ==========================================
# 2. PREPROCESSING LABEL
# ==========================================
print("Mengubah label angka ke format numerik...")
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_text)

num_classes = len(encoder.classes_)

with open(ENCODER_FILE, 'wb') as f:
    pickle.dump(encoder, f)

print("Kelas angka:", encoder.classes_)

# ==========================================
# 3. SPLIT DATA
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ==========================================
# 4. RESHAPE UNTUK 1D CNN
# ==========================================
# Dari bentuk: (jumlah_data, jumlah_fitur)
# Menjadi: (jumlah_data, jumlah_fitur, 1)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# ==========================================
# 5. MEMBANGUN MODEL 1D CNN
# ==========================================
print("Membangun model 1D CNN untuk angka...")

model = Sequential([
    Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        input_shape=(X_train.shape[1], 1)
    ),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(filters=256, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 6. CALLBACK
# ==========================================
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_FILE,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ==========================================
# 7. TRAINING
# ==========================================
print("Memulai training 1D CNN angka...")

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
)

# ==========================================
# 8. EVALUASI
# ==========================================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n" + "=" * 40)
print(f"AKURASI MODEL ANGKA 1D CNN: {accuracy * 100:.2f}%")
print("=" * 40)

print(f"Model tersimpan di: {MODEL_FILE}")
print(f"Encoder tersimpan di: {ENCODER_FILE}")
print("Training 1D CNN angka selesai!")