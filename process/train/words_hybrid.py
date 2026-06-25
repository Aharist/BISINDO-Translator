import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==========================================
# 1. MEMUAT DATA HASIL EKSTRAKSI VIDEO
# ==========================================
print("Memuat data video (.npy)...")

# Folder tempat menyimpan hasil ekstraksi landmark video
DATA_DIR = './processed_data'

# Folder untuk menyimpan model hasil training dan label encoder
MODEL_DIR = 'isolated_models'

# File output model dan encoder
MODEL_FILE = os.path.join(MODEL_DIR, 'kata_hybrid7_model.h5')
ENCODER_FILE = os.path.join(MODEL_DIR, 'kata_hybrid7_encoder.pkl')

# Membuat folder model jika belum tersedia
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Memuat data fitur video dan label kata
# X berisi sequence landmark video, sedangkan y_text berisi label kata
X = np.load(os.path.join(DATA_DIR, 'X_kata7.npy'))
y_text = np.load(os.path.join(DATA_DIR, 'y_kata7.npy'))

# ==========================================
# 2. PREPROCESSING LABEL DAN PEMBAGIAN DATA
# ==========================================
# Mengubah label kata dari teks menjadi angka agar dapat diproses oleh model
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_text)

# Menghitung jumlah kelas kata yang akan diprediksi
num_classes = len(encoder.classes_)

# Menyimpan encoder agar nanti hasil prediksi angka bisa dikembalikan ke label kata asli
with open(ENCODER_FILE, 'wb') as f:
    pickle.dump(encoder, f)

# Membagi data menjadi data training dan data testing dengan rasio 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# ==========================================
# 3. MEMBANGUN ARSITEKTUR MODEL HYBRID CNN-LSTM
# ==========================================
print("Membangun model Hybrid (1D CNN + LSTM)...")

model = Sequential([
    # Conv1D digunakan untuk mengekstraksi pola fitur landmark dari setiap sequence video
    # Input model berbentuk: jumlah frame x jumlah fitur
    Conv1D(
        filters=64, # Jumlah filter untuk menangkap pola
        kernel_size=3, # Ukuran jendela untuk membaca pola dalam sequence
        activation='relu', 
        input_shape=(X_train.shape[1], X_train.shape[2])
    ),

    # MaxPooling1D digunakan untuk mengambil fitur penting dan mengurangi kompleksitas data
    MaxPooling1D(pool_size=2),

    # LSTM digunakan untuk mempelajari urutan gerakan dari frame ke frame
    # Bagian ini penting karena data kata berupa video yang memiliki pola temporal
    LSTM(
        64, # Jumlah unit LSTM untuk mempelajari pola temporal
        return_sequences=False,
        activation='tanh'
    ),

    # Dropout digunakan untuk mengurangi risiko overfitting
    Dropout(0.3),

    # Dense layer digunakan untuk memproses fitur akhir sebelum klasifikasi
    Dense(64, activation='relu'),

    # Output layer menggunakan softmax untuk klasifikasi multi-kelas
    Dense(num_classes, activation='softmax')
])

# Compile model dengan optimizer Adam dan loss untuk klasifikasi multi-kelas
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Menampilkan ringkasan arsitektur model
model.summary()

# ==========================================
# 4. TRAINING MODEL DENGAN CALLBACK
# ==========================================
print("\nMemulai proses belajar...")

# ModelCheckpoint menyimpan model terbaik berdasarkan validation accuracy
checkpoint = ModelCheckpoint(
    MODEL_FILE,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# EarlyStopping menghentikan training jika performa validasi tidak meningkat
# restore_best_weights=True mengembalikan bobot terbaik selama training
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True
)

# Melatih model menggunakan data training dan validasi
history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop]
)

# ==========================================
# 5. EVALUASI MODEL
# ==========================================
# Menguji performa model menggunakan data testing
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n" + "=" * 40)
print(f"AKURASI MODEL HYBRID: {accuracy * 100:.2f}%")
print("=" * 40)