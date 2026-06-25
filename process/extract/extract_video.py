import cv2
import mediapipe as mp
import numpy as np
import os

# ==========================================
# 1. KONFIGURASI DATASET DAN OUTPUT
# ==========================================
# Direktori dataset berisi folder per kata, misalnya: halo, saya, apa, dst.
DATASET_DIR = 'word2_dataset'

# Direktori untuk menyimpan hasil ekstraksi dalam format numpy array
OUTPUT_DIR = 'processed_data'

# File output untuk menyimpan fitur video dan label kata
X_OUTPUT = os.path.join(OUTPUT_DIR, 'X_kata7.npy')
Y_OUTPUT = os.path.join(OUTPUT_DIR, 'y_kata7.npy')

# Setiap video akan direpresentasikan menjadi 30 frame agar panjang data seragam
SEQUENCE_LENGTH = 30

# Membuat folder output jika belum tersedia
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. INISIALISASI MEDIAPIPE HANDS
# ==========================================
# MediaPipe digunakan untuk mendeteksi landmark tangan pada setiap frame video
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,        # False karena input berupa video/frame berurutan
    max_num_hands=2,                # Maksimal mendeteksi dua tangan
    min_detection_confidence=0.5    # Batas minimum kepercayaan deteksi tangan
)

# List untuk menyimpan data fitur dan label
X_data = []
y_data = []

# ==========================================
# 3. PROSES EKSTRAKSI VIDEO
# ==========================================
# Membaca semua folder kata yang ada di dalam dataset
words = os.listdir(DATASET_DIR)

for word_label in words:
    word_path = os.path.join(DATASET_DIR, word_label)

    # Lewati jika yang terbaca bukan folder
    if not os.path.isdir(word_path):
        continue

    print(f"Mengekstrak video untuk kata: {word_label}...")

    # Membaca seluruh file video pada folder kata tertentu
    for video_name in os.listdir(word_path):

        # Memastikan hanya file video yang diproses
        if not video_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue
        
        video_full_path = os.path.join(word_path, video_name)
        cap = cv2.VideoCapture(video_full_path)

        # Mengambil jumlah total frame dalam video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Video yang terlalu pendek atau rusak akan dilewati
        if total_frames < 5:
            print(f"  > Video {video_name} di folder {word_label} terlalu pendek/error, dilewati.")
            cap.release()
            continue

        # Menentukan interval frame agar setiap video diambil menjadi 30 frame
        skip_frames = max(int(total_frames / SEQUENCE_LENGTH), 1)

        sequence_data = []
        frame_count = 0

        # Membaca frame video sampai mencapai 30 frame
        while cap.isOpened() and len(sequence_data) < SEQUENCE_LENGTH:
            ret, frame = cap.read()

            if not ret:
                break

            # Hanya memproses frame tertentu berdasarkan interval skip_frames
            if frame_count % skip_frames == 0:

                # OpenCV membaca gambar dalam BGR, sedangkan MediaPipe membutuhkan RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Deteksi landmark tangan pada frame
                results = hands.process(frame_rgb)

                hands_data = []

                if results.multi_hand_landmarks:
                    # Memproses setiap tangan yang terdeteksi
                    for hand_landmarks in results.multi_hand_landmarks:

                        # Landmark 0 atau pergelangan tangan dijadikan titik pusat
                        base_x = hand_landmarks.landmark[0].x
                        base_y = hand_landmarks.landmark[0].y
                        base_z = hand_landmarks.landmark[0].z

                        temp_hand = []
                        max_val = 0

                        # Mengambil 21 landmark tangan, masing-masing memiliki koordinat X, Y, Z
                        for lm in hand_landmarks.landmark:
                            rx = lm.x - base_x
                            ry = lm.y - base_y
                            rz = lm.z - base_z

                            temp_hand.extend([rx, ry, rz])

                            # Nilai terbesar digunakan untuk normalisasi skala
                            max_val = max(max_val, abs(rx), abs(ry), abs(rz))

                        # Normalisasi skala agar ukuran tangan lebih seragam
                        if max_val > 0:
                            temp_hand = [val / max_val for val in temp_hand]

                        hands_data.extend(temp_hand)

                    # Jika hanya satu tangan terdeteksi, fitur tangan kedua diisi nol
                    if len(results.multi_hand_landmarks) == 1:
                        hands_data.extend([0.0] * 63)

                else:
                    # Jika tidak ada tangan terdeteksi pada frame, seluruh fitur diisi nol
                    hands_data.extend([0.0] * 126)

                # Setiap frame harus memiliki 126 fitur: 2 tangan x 21 landmark x 3 koordinat
                sequence_data.append(hands_data[:126])

            frame_count += 1

        cap.release()

        # Jika frame yang berhasil diproses kurang dari 30, lakukan padding
        # Padding menggunakan frame terakhir agar panjang sequence tetap konsisten
        while len(sequence_data) < SEQUENCE_LENGTH:
            if len(sequence_data) > 0:
                sequence_data.append(sequence_data[-1])
            else:
                sequence_data.append([0.0] * 126)

        # Satu video menjadi satu sequence data dengan label sesuai nama folder
        X_data.append(sequence_data)
        y_data.append(word_label)

# Menutup proses MediaPipe setelah ekstraksi selesai
hands.close()

# ==========================================
# 4. MENYIMPAN HASIL EKSTRAKSI
# ==========================================
# Mengubah list menjadi numpy array agar siap digunakan untuk training CNN + LSTM
X_data = np.array(X_data)
y_data = np.array(y_data)

if len(X_data) > 0:
    print("\n" + "=" * 40)
    print(f"Bentuk data X: {X_data.shape}")
    print(f"Bentuk data Y: {y_data.shape}")
    print("=" * 40)

    np.save(X_OUTPUT, X_data)
    np.save(Y_OUTPUT, y_data)

    print(f"Berhasil! Data tersimpan di: {OUTPUT_DIR}")
else:
    print("\n[GAGAL] Tidak ada data yang berhasil diekstrak. Cek kembali path foldernya.")