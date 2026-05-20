import cv2
import mediapipe as mp
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================
# 1. LOAD KEDUA MODEL & ENCODER (PENGGABUNGAN)
# ==========================================
print("Memuat Model Abjad dan Angka... Harap tunggu sebentar.")

model_abjad = load_model('models/bisindo_cnn1d_model.h5')
with open('models/cnn1d_label_encoder.pkl', 'rb') as f:
    encoder_abjad = pickle.load(f)

model_angka = load_model('models/numbers_cnn1d_model.h5')
with open('models/number_cnn1d_encoder.pkl', 'rb') as f:
    encoder_angka = pickle.load(f)

MODE_AKTIF = "ABJAD" 

# ==========================================
# 2. SETUP MEDIAPIPE
# ==========================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# static_image_mode=False: Dioptimalkan untuk video streaming
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# ==========================================
# 3. AKSES WEBCAM & PIPELINE UTAMA
# ==========================================
cap = cv2.VideoCapture(0)
print("Kamera siap! Tekan 'M' untuk ganti mode, dan 'Q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret: break
        
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        hands_data = []
        
        # PROSES EKSTRAKSI REAL-TIME
        for hand_landmarks in results.multi_hand_landmarks:
            # Menggambar kerangka tangan di layar
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Normalisasi koordinat agar sesuai dengan data yang dipelajari model saat training
            base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
            temp_hand = []
            max_val = 0
            
            for lm in hand_landmarks.landmark:
                rx, ry, rz = lm.x - base_x, lm.y - base_y, lm.z - base_z
                temp_hand.extend([rx, ry, rz])
                max_val = max(max_val, abs(rx), abs(ry), abs(rz))
            
            if max_val > 0:
                temp_hand = [val / max_val for val in temp_hand]
            hands_data.extend(temp_hand)
            
        # Penanganan Isyarat 1 Tangan: Menambahkan nol agar total input tetap 126
        if len(results.multi_hand_landmarks) == 1:
            hands_data.extend([0.0] * 63)
            
        X_input = np.array([hands_data[:126]])
        
        # ==========================================
        # 4. PREDIKSI DINAMIS (LOGIKA PEMILIHAN MODEL)
        # ==========================================
        # Program memilih model mana yang digunakan berdasarkan pilihan user (M)
        if MODE_AKTIF == "ABJAD":
            predictions = model_abjad.predict(X_input, verbose=0)[0]
            max_prob = np.max(predictions)
            predicted_index = np.argmax(predictions)
            huruf_hasil = encoder_abjad.inverse_transform([predicted_index])[0]
        else:
            predictions = model_angka.predict(X_input, verbose=0)[0]
            max_prob = np.max(predictions)
            predicted_index = np.argmax(predictions)
            huruf_hasil = encoder_angka.inverse_transform([predicted_index])[0]
        
        # CONFIDENCE THRESHOLD (Batas Keyakinan)
        # Jika keyakinan model di bawah 75%, kita anggap "Tidak Yakin" untuk menghindari salah prediksi
        if max_prob > 0.75:
            teks_tampil = f"Hasil: {huruf_hasil} ({max_prob*100:.0f}%)"
            warna_kotak = (0, 255, 0) # Hijau jika yakin
        else:
            teks_tampil = "Hasil: [Tidak Yakin]"
            warna_kotak = (0, 165, 255) # Oranye jika ragu

        # Menampilkan hasil prediksi ke layar
        cv2.rectangle(frame, (10, 60), (450, 130), warna_kotak, cv2.FILLED)
        cv2.putText(frame, teks_tampil, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # ==========================================
    # 5. UI INDICATOR (PENUNJUK MODE)
    # ==========================================
    # Menampilkan status mode aktif di pojok kiri atas
    cv2.rectangle(frame, (10, 10), (550, 50), (50, 50, 50), cv2.FILLED)
    teks_mode = f"MODE: {MODE_AKTIF} (Tekan 'M' untuk ganti)"
    cv2.putText(frame, teks_mode, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Sign Language Recognition', frame)
    
    # ==========================================
    # 6. KONTROL INTERAKTIF (KEYBOARD)
    # ==========================================
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'): # Menutup aplikasi
        break
    elif key == ord('m'):
        if MODE_AKTIF == "ABJAD":
            MODE_AKTIF = "ANGKA"
            print("--> BERUBAH KE MODE ANGKA")
        else:
            MODE_AKTIF = "ABJAD"
            print("--> BERUBAH KE MODE ABJAD")

cap.release()
cv2.destroyAllWindows()