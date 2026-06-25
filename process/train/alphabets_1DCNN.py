import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten

# ==========================================
# 1. PERSIAPAN DATA DAN RELATIVE PATH
# ==========================================
DATA_FILE = '../processed_data/landmarks_bisindo.csv'
MODEL_DIR = '../models'

# Simpan dengan nama khusus CNN
MODEL_FILE = os.path.join(MODEL_DIR, 'bisindo_cnn1d_model.h5') 
ENCODER_FILE = os.path.join(MODEL_DIR, 'cnn1d_label_encoder.pkl')  

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("Membaca data CSV...")
df = pd.read_csv(DATA_FILE)

X = df.drop('label', axis=1).values
y_text = df['label'].values

# ==========================================
# 2. PREPROCESSING & RESHAPING (KUNCI UTAMA 1D CNN)
# ==========================================
print("Memproses label data menjadi numerik...")
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_text)
num_classes = len(encoder.classes_)

# Simpan encoder untuk nanti dipakai di proses inference
with open(ENCODER_FILE, 'wb') as f:
    pickle.dump(encoder, f)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ---  RESHAPE DATA ---
# Syarat mutlak Conv1D: Input harus berbentuk 3D Array (Jumlah Data, Jumlah Langkah, Jumlah Fitur per Langkah).
# Data aslinya 2D diubah jadi 3D.
X_train = np.expand_dims(X_train, axis=2) # Menambahkan dimensi baru di akhir untuk fitur tunggal (1D)
X_test = np.expand_dims(X_test, axis=2)

# ==========================================
# 3. MEMBANGUN ARSITEKTUR 1D CNN
# ==========================================
print("Membangun model 1D Convolutional Neural Network...")

model = Sequential([
    # LAYER CONVOLUTION 1: 
    # Membaca data menggunakan 'jendela' berukuran 3 titik (kernel_size=3) yang bergeser.
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    
    # MAX POOLING: Menyusutkan data dengan mengambil nilai fitur terpenting saja agar komputasi ringan.
    MaxPooling1D(pool_size=2),
    
    # LAYER CONVOLUTION 2: Mengekstrak pola yang lebih rumit.
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    
    # FLATTEN: Meratakan kembali data 3D menjadi 1D agar bisa masuk ke Dense layer (seperti ANN biasa)
    Flatten(),
    
    # FULLY CONNECTED LAYER (ANN)
    Dense(128, activation='relu'), # Menambahkan lapisan ANN untuk belajar pola yang lebih kompleks setelah fitur diekstrak oleh Conv1D
    Dropout(0.3), # Mematikan 30% neuron secara acak untuk mencegah Overfitting
    
    # OUTPUT LAYER
    Dense(num_classes, activation='softmax') # Softmax untuk klasifikasi multi-kelas, menghasilkan probabilitas untuk setiap kelas
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Menampilkan ringkasan arsitektur model untuk memastikan semuanya benar sebelum training
model.summary()

# ==========================================
# 4. MULAI TRAINING
# ==========================================
print("Memulai proses belajar (Training)...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# ==========================================
# 5. EVALUASI DAN SIMPAN
# ==========================================
print("\nMenyimpan model ke folder models/...")
model.save(MODEL_FILE)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\n" + "="*40)
print(f"AKURASI MODEL 1D CNN: {accuracy * 100:.2f}%")
print("="*40)
print("Proses training 1D CNN selesai!")