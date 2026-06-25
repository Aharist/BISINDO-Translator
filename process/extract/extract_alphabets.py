import cv2
import mediapipe as mp
import os
import csv

# ==========================================
# 1. PERSIAPAN MEDIAPIPE & FOLDER
# ==========================================
# Inisialisasi modul MediaPipe untuk deteksi tangan
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Menentukan lokasi folder dataset (Kaggle BISINDO) dan tempat hasil ekstraksi (CSV)
DATASET_DIR = 'dataset/images/train' 
OUTPUT_DIR = 'processed_data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'landmarks_bisindo.csv')

# Buat folder output otomatis jika belum ada di dalam sistem
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Batasan gambar per kelas yang diekstrak
MAX_IMAGES_PER_CLASS = 350 

# ==========================================
# 2. MEMBUAT HEADER UNTUK FILE CSV
# ==========================================
header = ['label']

# Looping untuk membuat judul kolom
# h0 melambangkan Tangan Pertama, h1 melambangkan Tangan Kedua
for hand_idx in range(2): 
    for i in range(21):
        header.extend([f'h{hand_idx}_x_{i}', f'h{hand_idx}_y_{i}', f'h{hand_idx}_z_{i}'])

# Membuat file CSV dan menulis baris pertama tersebut
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    # ==========================================
    # 3. PROSES EKSTRAKSI (LOOPING DATASET)
    # ==========================================
    # Membaca semua nama folder di dalam direktori dataset (A, B, C, dst)
    classes = os.listdir(DATASET_DIR)
    
    for class_name in classes:
        class_dir = os.path.join(DATASET_DIR, class_name)
        # Abaikan jika yang terbaca bukan folder
        if not os.path.isdir(class_dir): continue
            
        print(f"Mengekstrak kelas: {class_name}...")
        image_count = 0 
        
        # Masuk ke dalam folder huruf tertentu dan baca semua fotonya
        for image_name in os.listdir(class_dir):
            if image_count >= MAX_IMAGES_PER_CLASS: break # Berhenti jika sudah 350 gambar
                
            # Membaca gambar menggunakan OpenCV
            img = cv2.imread(os.path.join(class_dir, image_name))
            if img is None: continue # Lewati jika file bukan gambar atau korup
                
            # MediaPipe mewajibkan format warna RGB, sedangkan OpenCV membaca dengan format BGR.
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Inti program: Menyuruh MediaPipe mencari letak titik tangan di gambar
            results = hands.process(img_rgb)
            
            # Jika MediaPipe berhasil menemukan tangan di gambar tersebut...
            if results.multi_hand_landmarks:
                row = [class_name] # Kolom pertama diisi nama hurufnya
                hands_data = [] # List sementara untuk menampung 126 angka koordinat
                
                # Looping untuk setiap tangan yang terdeteksi
                for hand_landmarks in results.multi_hand_landmarks:
                    
                    # --- TEKNIK NORMALISASI ---
                    # 1. NORMALISASI POSISI (Koordinat Relatif)
                    # jadikan pergelangan tangan (Landmark 0) sebagai titik pusat/Nol.
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y
                    base_z = hand_landmarks.landmark[0].z
                    
                    temp_hand = []
                    max_val = 0 # Variabel untuk mencari jarak jari yang paling jauh
                    
                    for lm in hand_landmarks.landmark:
                        # Mengurangi semua koordinat jari dengan koordinat pusat (pergelangan)
                        rx, ry, rz = lm.x - base_x, lm.y - base_y, lm.z - base_z
                        temp_hand.extend([rx, ry, rz])
                        
                        # Mencari angka terbesar (absolut) untuk bahan pembagi di tahap selanjutnya
                        max_val = max(max_val, abs(rx), abs(ry), abs(rz))
                    
                    # 2. NORMALISASI SKALA (Scale Invariant)
                    # Membagi semua titik dengan nilai terbesar agar rentang ukurannya seragam.
                    if max_val > 0:
                        temp_hand = [val / max_val for val in temp_hand]
                        
                    # Masukkan data 1 tangan yang sudah dinormalisasi ini ke array utama
                    hands_data.extend(temp_hand)
                
                # --- TEKNIK ZERO PADDING ---
                # JIKA GAMBAR HANYA MENAMPILKAN 1 TANGAN (misal huruf I atau U):
                # Karena Neural Network kita menuntut input yang pasti berjumlah 126 kolom,
                # kita harus mengisi sisa kekosongan tangan kedua dengan nilai nol (0.0).
                if len(results.multi_hand_landmarks) == 1:
                    hands_data.extend([0.0] * 63) 
                    
                # Gabungkan Label (huruf) dengan array koordinatnya
                row.extend(hands_data[:126]) # [:126] untuk memastikan data tidak kelebihan batas
                
                # Tulis data tersebut sebagai satu baris ke dalam file CSV
                writer.writerow(row)
                image_count += 1

print("==========================================")
print(f"Selesai! Data disimpan di: {OUTPUT_FILE}")
print("==========================================")
# Bebaskan memori RAM setelah tugas selesai
hands.close()