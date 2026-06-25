import cv2
import mediapipe as mp
import os
import csv

# ==========================================
# 1. PERSIAPAN MEDIAPIPE & FOLDER (SETUP)
# ==========================================
# Inisialisasi modul MediaPipe untuk deteksi tangan
mp_hands = mp.solutions.hands

# Mengatur parameter deteksi:
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Menentukan letak folder dataset (sumber) dan folder hasil ekstraksi (tujuan)
DATASET_DIR = 'num_dataset' 
OUTPUT_DIR = 'processed_data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'landmarks_numbers.csv')

# Jika folder output belum ada, program akan membuatnya secara otomatis
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Membatasi gambar yang diproses maksimal 500 per angka agar data seimbang (balance)
MAX_IMAGES_PER_CLASS = 500 

# ==========================================
# 2. MEMBUAT HEADER UNTUK FILE CSV (KERANGKA DATA)
# ==========================================
# Kolom pertama diisi dengan 'label' (target angka yang akan diprediksi)
header = ['label']

# Membuat kolom untuk 126 fitur:
# Berasal dari 2 tangan x 21 titik sendi x 3 koordinat (X, Y, Z)
for hand_idx in range(2): 
    for i in range(21):
        header.extend([f'h{hand_idx}_x_{i}', f'h{hand_idx}_y_{i}', f'h{hand_idx}_z_{i}'])

# Membuka file CSV dan menuliskan baris judul (header) tersebut
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    # ==========================================
    # 3. PROSES EKSTRAKSI (INTI PROGRAM)
    # ==========================================

    classes = os.listdir(DATASET_DIR)
    
    for class_name in classes:
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir): continue # Lewati jika ternyata bukan folder
            
        print(f"Mengekstrak angka: {class_name}...")
        image_count = 0 
        
        # Mulai membaca satu per satu gambar di dalam folder angka tersebut
        for image_name in os.listdir(class_dir):
            if image_count >= MAX_IMAGES_PER_CLASS: break
                
            # Membaca gambar menggunakan OpenCV
            img = cv2.imread(os.path.join(class_dir, image_name))
            if img is None: continue
                
            # OpenCV menggunakan format warna BGR, sedangkan MediaPipe butuh RGB. 
            # Jadi kita konversi warnanya di sini.
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Meminta MediaPipe untuk mencari posisi tangan di dalam gambar
            results = hands.process(img_rgb)
            
            # Jika MediaPipe berhasil menemukan tangan:
            if results.multi_hand_landmarks:
                row = [class_name] # Siapkan baris data baru, diawali dengan label angka
                hands_data = []
                
                # Proses setiap tangan yang terdeteksi
                for hand_landmarks in results.multi_hand_landmarks:
                    # TRANSLASI: Menjadikan pergelangan tangan (titik 0) sebagai titik pusat (0,0,0)
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y
                    base_z = hand_landmarks.landmark[0].z
                    
                    temp_hand = []
                    max_val = 0
                    
                    # Menghitung jarak tiap titik jari terhadap pergelangan tangan
                    for lm in hand_landmarks.landmark:
                        rx, ry, rz = lm.x - base_x, lm.y - base_y, lm.z - base_z
                        temp_hand.extend([rx, ry, rz])
                        # Mencari nilai jarak terjauh untuk keperluan normalisasi
                        max_val = max(max_val, abs(rx), abs(ry), abs(rz))
                    
                    # NORMALISASI: Membagi semua nilai dengan jarak terjauh
                    # Tujuannya agar ukuran tangan (dekat/jauh dari kamera) tidak merusak prediksi model
                    if max_val > 0:
                        temp_hand = [val / max_val for val in temp_hand]
                        
                    hands_data.extend(temp_hand)
                
                # ZERO PADDING: Penanganan khusus jika hanya 1 tangan yang muncul
                # Isyarat angka biasanya hanya pakai 1 tangan. Agar data tetap muat di 126 kolom, 
                # bagian tangan kedua yang kosong kita isi dengan angka 0.
                if len(results.multi_hand_landmarks) == 1:
                    hands_data.extend([0.0] * 63) 
                    
                # Menggabungkan label dengan data koordinat, memastikan maksimal 126 fitur, lalu simpan ke CSV
                row.extend(hands_data[:126]) 
                writer.writerow(row)
                image_count += 1

# ==========================================
# 4. PENUTUP & PEMBERSIHAN MEMORI
# ==========================================
print("==========================================")
print(f"Selesai! Data angka disimpan di: {OUTPUT_FILE}")
print("==========================================")
hands.close() 