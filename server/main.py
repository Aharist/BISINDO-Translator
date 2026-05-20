import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
import json
from collections import deque, Counter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Memuat Model AI ke dalam Server...")

# 1. Model Abjad
model_abjad = load_model('models/bisindo_cnn1d_model.h5')
with open('models/cnn1d_label_encoder.pkl', 'rb') as f:
    encoder_abjad = pickle.load(f)

# 2. Model Angka
model_angka = load_model('models/numbers_cnn1d_model.h5')
with open('models/number_cnn1d_encoder.pkl', 'rb') as f:
    encoder_angka = pickle.load(f)

# 3. Model Kata
model_kata = load_model('models/kata_hybrid3_model.h5')
with open('models/kata_hybrid3_encoder.pkl', 'rb') as f:
    encoder_kata = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

print("Server FastAPI Siap Beroperasi! 🚀")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    sequence = deque(maxlen=30) # Simpan 30 frame terakhir untuk prediksi kata
    prediction_history = deque(maxlen=15) # Simpan 15 prediksi terakhir untuk voting kata
    
    while True:
        try:
            data = await websocket.receive_text()   # Menerima data gambar dalam format base64 dari klien
            payload = json.loads(data)
            
            mode_aktif = payload.get("mode", "KATA")
            image_base64 = payload.get("image", "")
            
            if payload.get("reset"):
                sequence.clear()
                prediction_history.clear()
                await websocket.send_json({"hasil": None, "confidence": 0.0, "mode": mode_aktif, "reset_kata": True})
                continue

            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            img_bytes = base64.b64decode(image_base64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            # Mirror Kamera
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            hands_data = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y
                    base_z = hand_landmarks.landmark[0].z
                    temp_hand = []
                    max_val = 0
                    for lm in hand_landmarks.landmark:
                        rx = lm.x - base_x
                        ry = lm.y - base_y
                        rz = lm.z - base_z
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

            confidence = 0.0
            hasil_prediksi = None
            reset_kata = False
            
            if not results.multi_hand_landmarks:
                hasil_prediksi = None
                confidence = 0.0
            
            # ==========================================
            # FIX KECEPATAN: GANTI .predict() JADI model(X, training=False)
            # ==========================================
            elif mode_aktif == "ABJAD" and results.multi_hand_landmarks:
                X_input = np.array([hands_data[:126]], dtype=np.float32)
                X_input = np.expand_dims(X_input, axis=2) 
                # EKSEKUSI SUPER CEPAT
                preds = model_abjad(X_input, training=False).numpy()[0] 
                confidence = float(np.max(preds))
                if confidence > 0.75:
                    hasil_prediksi = str(encoder_abjad.inverse_transform([np.argmax(preds)])[0])

            elif mode_aktif == "ANGKA" and results.multi_hand_landmarks:
                X_input = np.array([hands_data[:126]], dtype=np.float32)
                X_input = np.expand_dims(X_input, axis=2) 
                # EKSEKUSI SUPER CEPAT
                preds = model_angka(X_input, training=False).numpy()[0]
                confidence = float(np.max(preds))
                if confidence > 0.75:
                    hasil_prediksi = str(encoder_angka.inverse_transform([np.argmax(preds)])[0])

            elif mode_aktif == "KATA":
                sequence.append(hands_data[:126])
                if len(sequence) == 30:
                    seq_array = np.array(list(sequence), dtype=np.float32)
                    X_input = np.expand_dims(seq_array, axis=0) 
                    
                    # EKSEKUSI SUPER CEPAT
                    preds = model_kata(X_input, training=False).numpy()[0]
                    confidence = float(np.max(preds))
                    
                    if confidence > 0.85:
                        kata_tebakan = str(encoder_kata.inverse_transform([np.argmax(preds)])[0])
                        prediction_history.append(kata_tebakan)
                    else:
                        prediction_history.append("Unknown")
                        
                    if len(prediction_history) == 15:
                        most_common_word, count = Counter(prediction_history).most_common(1)[0]
                        if most_common_word != "Unknown" and count >= 10:
                            hasil_prediksi = str(most_common_word)
                            confidence = float(count / 15.0)
                        else:
                            hasil_prediksi = None
                            confidence = 0.0
                            if most_common_word == "Unknown" and count >= 10:
                                reset_kata = True
                else:
                    hasil_prediksi = None
                    confidence = 0.0

            # Kirim hasil
            await websocket.send_json({
                "hasil": hasil_prediksi, 
                "confidence": confidence,
                "mode": mode_aktif,
                "reset_kata": reset_kata
            })

        except WebSocketDisconnect:
            print("Klien terputus.")
            break
        except Exception as e:
            print(f"Error AI: {e}")