import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque, Counter
from tensorflow.keras.models import load_model

# ==========================================
# 1. LOAD MODEL HYBRID 3
# ==========================================
print("Memuat Model Kata Hybrid...")
MODEL_FILE = 'server/models/kata_hybrid3_model.h5'
ENCODER_FILE = 'server/models/kata_hybrid3_encoder.pkl' 

model = load_model(MODEL_FILE)
with open(ENCODER_FILE, 'rb') as f:
    encoder = pickle.load(f)

sequence = deque(maxlen=30)
prediction_history = deque(maxlen=15)

kalimat_terangkai = []
kata_terakhir_ditambahkan = "" 

# ==========================================
# 2. SETUP MEDIAPIPE
# ==========================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# ==========================================
# 3. SETUP KAMERA DAN LAYAR (DIPERBESAR)
# ==========================================
cap = cv2.VideoCapture(0)

# Membuat jendela yang ukurannya bisa kita atur sendiri
cv2.namedWindow('Sistem Penerjemah Kalimat (BISINDO)', cv2.WINDOW_NORMAL)
# Mengubah ukuran layar (Lebar: 1024, Tinggi: 768)
cv2.resizeWindow('Sistem Penerjemah Kalimat (BISINDO)', 1024, 768)

print("\n" + "="*40)
print("KAMERA SIAP!")
print("Tekan 'Backspace' untuk Reset Kalimat, 'Q' untuk Keluar.")
print("="*40)

kata_stabil = "[Menunggu]"
warna_kotak = (50, 50, 50)

while True:
    ret, frame = cap.read()
    if not ret: break
        
    frame = cv2.flip(frame, 1) 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    hands_data = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
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
            
        if len(results.multi_hand_landmarks) == 1:
            hands_data.extend([0.0] * 63)
        hands_data = hands_data[:126]
    else:
        hands_data = [0.0] * 126

    # ==========================================
    # 4. PREDIKSI & ANTI-SPAM
    # ==========================================
    sequence.append(hands_data)
    
    if len(sequence) == 30:
        X_input = np.array([sequence])
        predictions = model.predict(X_input, verbose=0)[0]
        max_prob = np.max(predictions)
        predicted_idx = np.argmax(predictions)
        
        if max_prob > 0.85: 
            kata_hasil = encoder.inverse_transform([predicted_idx])[0]
            prediction_history.append(kata_hasil)
        else:
            prediction_history.append("Unknown")
            
        if len(prediction_history) == 15:
            most_common_word, count = Counter(prediction_history).most_common(1)[0]
            
            if most_common_word != "Unknown" and count >= 10:
                kata_stabil = f"Kata: {most_common_word}"
                warna_kotak = (255, 0, 0)
                
                # LOGIKA ANTI-SPAM (Penyusun Kalimat):
                if most_common_word != kata_terakhir_ditambahkan:
                    kalimat_terangkai.append(most_common_word)
                    kata_terakhir_ditambahkan = most_common_word
                    
            elif most_common_word == "Unknown":
                kata_stabil = "[Mendeteksi Gerak...]"
                warna_kotak = (0, 165, 255)
                
                # MELEPAS ANTI-SPAM (Reset) ketika tangan diam/turun
                if count >= 10: 
                    kata_terakhir_ditambahkan = ""
    else:
        kata_stabil = f"Mengumpulkan frame... ({len(sequence)}/30)"

    # ==========================================
    # 5. RENDER UI
    # ==========================================
    # Kotak Kalimat (Atas)
    cv2.rectangle(frame, (10, 10), (630, 80), (20, 20, 20), cv2.FILLED)
    
    teks_kalimat = " ".join(kalimat_terangkai) 
    if not teks_kalimat:
        teks_kalimat = "Mulai peragakan isyarat..."
        
    cv2.putText(frame, teks_kalimat, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Kotak Prediksi Kata Satuan (Bawah)
    cv2.rectangle(frame, (10, 380), (630, 450), warna_kotak, cv2.FILLED)
    cv2.putText(frame, kata_stabil, (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imshow('Sistem Penerjemah Kalimat (BISINDO)', frame)
    
    # ==========================================
    # 6. KEYBOARD LISTENER (BACKSPACE & Q)
    # ==========================================
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'): 
        break
    elif key == 8 or key == 127: 
        kalimat_terangkai.clear()
        kata_terakhir_ditambahkan = ""
        prediction_history.clear()
        print("--> KALIMAT DIRESET")

cap.release()
cv2.destroyAllWindows()