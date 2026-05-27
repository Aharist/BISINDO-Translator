import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
import json
import time
import asyncio
import logging
import os
import sys
from collections import deque, Counter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BISINDO_Backend")

app = FastAPI(title="BISINDO Recognition Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths absolutely to prevent working directory issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from preprocessing import extract_landmarks

# Load model registry once at startup
logger.info("Memuat Model AI ke dalam Server...")

try:
    model_abjad_path = os.path.join(BASE_DIR, 'models', 'bisindo_cnn1d_model.h5')
    encoder_abjad_path = os.path.join(BASE_DIR, 'models', 'cnn1d_label_encoder.pkl')
    model_abjad = load_model(model_abjad_path)
    with open(encoder_abjad_path, 'rb') as f:
        encoder_abjad = pickle.load(f)
    logger.info("[OK] Model Abjad berhasil dimuat.")
except Exception as e:
    logger.error(f"Gagal memuat Model Abjad: {e}")
    model_abjad, encoder_abjad = None, None

try:
    model_angka_path = os.path.join(BASE_DIR, 'models', 'numbers_cnn1d_model.h5')
    encoder_angka_path = os.path.join(BASE_DIR, 'models', 'number_cnn1d_encoder.pkl')
    model_angka = load_model(model_angka_path)
    with open(encoder_angka_path, 'rb') as f:
        encoder_angka = pickle.load(f)
    logger.info("[OK] Model Angka berhasil dimuat.")
except Exception as e:
    logger.error(f"Gagal memuat Model Angka: {e}")
    model_angka, encoder_angka = None, None

try:
    model_kata_path = os.path.join(BASE_DIR, 'models', 'kata_hybrid3_model.h5')
    encoder_kata_path = os.path.join(BASE_DIR, 'models', 'kata_hybrid3_encoder.pkl')
    model_kata = load_model(model_kata_path)
    with open(encoder_kata_path, 'rb') as f:
        encoder_kata = pickle.load(f)
    logger.info("[OK] Model Kata berhasil dimuat.")
except Exception as e:
    logger.error(f"Gagal memuat Model Kata: {e}")
    model_kata, encoder_kata = None, None

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)

logger.info("Server FastAPI Siap Beroperasi! (Ready)")


@app.get("/health")
def health_check():
    """Simple API status endpoint."""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/models/status")
def models_status():
    """Returns load status of ML models."""
    return {
        "model_abjad_loaded": model_abjad is not None,
        "model_angka_loaded": model_angka is not None,
        "model_kata_loaded": model_kata is not None
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Klien terhubung ke WebSocket.")
    
    # Initialize queues local to this WebSocket connection session
    sequence = deque(maxlen=30)         # Word model sequence buffer (sliding window)
    history_abjad = deque(maxlen=5)     # Alphabet temporal smoothing queue (size 5)
    history_angka = deque(maxlen=5)     # Number temporal smoothing queue (size 5)
    history_kata = deque(maxlen=5)      # Word temporal smoothing queue (size 5, early exit at 3)
    
    # Tracking variables for anti-spam (preventing repeated inputs)
    last_appended_char = ""
    last_appended_number = ""
    last_appended_word = ""
    last_append_time = 0.0              # Cooldown timestamp tracker
    
    while True:
        try:
            data = await websocket.receive_text()
            if not data:
                continue
                
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Payload must be valid JSON"})
                continue
                
            mode_raw = payload.get("mode", "KATA")
            image_base64 = payload.get("image", "")
            
            # Map frontend modes safely to internal representations
            mode_aktif = mode_raw.upper()
            if mode_aktif == "ALPHABET":
                mode_aktif = "ABJAD"
            elif mode_aktif == "NUMBER":
                mode_aktif = "ANGKA"
            elif mode_aktif == "WORD":
                mode_aktif = "KATA"
                
            # Handle reset triggers
            if payload.get("reset"):
                sequence.clear()
                history_abjad.clear()
                history_angka.clear()
                history_kata.clear()
                last_appended_char = ""
                last_appended_number = ""
                last_appended_word = ""
                last_append_time = 0.0
                logger.info("Buffer dan history di-reset atas permintaan klien.")
                await websocket.send_json({
                    "hasil": None,
                    "confidence": 0.0,
                    "mode": mode_raw,
                    "status": "State di-reset",
                    "append": False,
                    "reset_kata": True
                })
                continue

            if not image_base64:
                await websocket.send_json({
                    "hasil": None,
                    "confidence": 0.0,
                    "mode": mode_raw,
                    "status": "Payload gambar kosong",
                    "append": False
                })
                continue
                
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
                
            try:
                img_bytes = base64.b64decode(image_base64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.warning(f"Gagal mendecode frame base64: {e}")
                await websocket.send_json({
                    "hasil": None,
                    "confidence": 0.0,
                    "mode": mode_raw,
                    "status": "Gambar rusak",
                    "append": False
                })
                continue
                
            if frame is None:
                await websocket.send_json({
                    "hasil": None,
                    "confidence": 0.0,
                    "mode": mode_raw,
                    "status": "Frame gagal diproses",
                    "append": False
                })
                continue

            # Mirror frame and convert color space for MediaPipe
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            start_time = time.time()
            results = hands.process(frame_rgb)
            
            # Centralized preprocessing
            hands_data = extract_landmarks(results.multi_hand_landmarks)
            hands_detected = results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0
            
            hasil_prediksi = None
            confidence = 0.0
            status_label = ""
            append_to_text = False
            reset_kata = False
            current_time = time.time()
            
            # Execute model predictions asynchronously to prevent event loop blocking
            if mode_aktif == "ABJAD":
                # Clean up unrelated buffers
                sequence.clear()
                history_angka.clear()
                history_kata.clear()
                last_appended_number = ""
                last_appended_word = ""
                
                if model_abjad is None:
                    status_label = "Model Abjad tidak termuat"
                else:
                    if hands_detected:
                        X_input = np.array([hands_data], dtype=np.float32)
                        X_input = np.expand_dims(X_input, axis=2) # Shape: (1, 126, 1)
                        
                        # Background thread execution for TensorFlow
                        preds = await asyncio.to_thread(
                            lambda: model_abjad(X_input, training=False).numpy()[0]
                        )
                        raw_conf = float(np.max(preds))
                        pred_idx = np.argmax(preds)
                        
                        # Confidence minimal 0.75 untuk alfabet
                        if raw_conf >= 0.75:
                            char_pred = str(encoder_abjad.inverse_transform([pred_idx])[0])
                            history_abjad.append(char_pred)
                        else:
                            history_abjad.append("Unknown")
                    else:
                        history_abjad.append("Unknown")
                        
                    # Smooth predictions: valid if same label appears >= 3 times in last 5 predictions
                    if len(history_abjad) == 5:
                        most_common, count = Counter(history_abjad).most_common(1)[0]
                        if most_common != "Unknown" and count >= 3:
                            hasil_prediksi = most_common
                            confidence = raw_conf if hands_detected else 0.0
                            status_label = "Stabil"
                            
                            # 1 second cooldown after successful append
                            if most_common != last_appended_char and (current_time - last_append_time) >= 1.0:
                                append_to_text = True
                                last_appended_char = most_common
                                last_append_time = current_time
                        else:
                            hasil_prediksi = None
                            confidence = 0.0
                            status_label = "Tangan tidak terdeteksi" if not hands_detected else "Tidak yakin"
                            
                            # Release the duplicate lock if hand is lowered
                            if most_common == "Unknown" and count >= 3:
                                last_appended_char = ""
                    else:
                        status_label = f"Mengumpulkan frame... ({len(history_abjad)}/5)"
                        
            elif mode_aktif == "ANGKA":
                # Clean up unrelated buffers
                sequence.clear()
                history_abjad.clear()
                history_kata.clear()
                last_appended_char = ""
                last_appended_word = ""
                
                if model_angka is None:
                    status_label = "Model Angka tidak termuat"
                else:
                    if hands_detected:
                        X_input = np.array([hands_data], dtype=np.float32)
                        X_input = np.expand_dims(X_input, axis=2) # Shape: (1, 126, 1)
                        
                        # Background thread execution for TensorFlow
                        preds = await asyncio.to_thread(
                            lambda: model_angka(X_input, training=False).numpy()[0]
                        )
                        raw_conf = float(np.max(preds))
                        pred_idx = np.argmax(preds)
                        
                        # Confidence minimal 0.75 untuk angka
                        if raw_conf >= 0.75:
                            num_pred = str(encoder_angka.inverse_transform([pred_idx])[0])
                            history_angka.append(num_pred)
                        else:
                            history_angka.append("Unknown")
                    else:
                        history_angka.append("Unknown")
                        
                    # Smooth predictions: valid if same label appears >= 3 times in last 5 predictions
                    if len(history_angka) == 5:
                        most_common, count = Counter(history_angka).most_common(1)[0]
                        if most_common != "Unknown" and count >= 3:
                            hasil_prediksi = most_common
                            confidence = raw_conf if hands_detected else 0.0
                            status_label = "Stabil"
                            
                            # 1 second cooldown after successful append
                            if most_common != last_appended_number and (current_time - last_append_time) >= 1.0:
                                append_to_text = True
                                last_appended_number = most_common
                                last_append_time = current_time
                        else:
                            hasil_prediksi = None
                            confidence = 0.0
                            status_label = "Tangan tidak terdeteksi" if not hands_detected else "Tidak yakin"
                            
                            # Release the duplicate lock
                            if most_common == "Unknown" and count >= 3:
                                last_appended_number = ""
                    else:
                        status_label = f"Mengumpulkan frame... ({len(history_angka)}/5)"
                        
            elif mode_aktif == "KATA":
                # Clean up unrelated buffers
                history_abjad.clear()
                history_angka.clear()
                last_appended_char = ""
                last_appended_number = ""
                
                if model_kata is None:
                    status_label = "Model Kata tidak termuat"
                else:
                    # Always append to sequence (zero-pad if no hands), matching local script behavior
                    if hands_detected:
                        sequence.append(hands_data)
                    else:
                        sequence.append([0.0] * 126)
                        
                    if len(sequence) == 30:
                        seq_array = np.array(list(sequence), dtype=np.float32)
                        X_input = np.expand_dims(seq_array, axis=0) # Shape: (1, 30, 126)
                        
                        # Background thread execution for TensorFlow
                        preds = await asyncio.to_thread(
                            lambda: model_kata(X_input, training=False).numpy()[0]
                        )
                        raw_conf = float(np.max(preds))
                        pred_idx = np.argmax(preds)
                        
                        # Confidence minimal 0.80 untuk kata
                        if raw_conf >= 0.80:
                            word_pred = str(encoder_kata.inverse_transform([pred_idx])[0])
                            history_kata.append(word_pred)
                        else:
                            history_kata.append("Unknown")
                    else:
                        # Buffer belum penuh - JANGAN masukkan apapun ke history_kata
                        # (Bug sebelumnya: memasukkan "Unknown" di sini mempolusi history)
                        status_label = f"Mengumpulkan frame... ({len(sequence)}/30)"
                        
                    # Early consensus check: mulai cek begitu ada >= 3 prediksi
                    hist_len = len(history_kata)
                    if hist_len >= 3:
                        most_common, count = Counter(history_kata).most_common(1)[0]
                        # Perlu >= 70% consensus (3/3, 3/4, 4/5)
                        threshold = max(3, int(hist_len * 0.7))
                        
                        if most_common != "Unknown" and count >= threshold:
                            hasil_prediksi = most_common
                            confidence = raw_conf if len(sequence) == 30 else 0.0
                            status_label = "Stabil"
                            
                            # 1 second cooldown after successful append
                            if most_common != last_appended_word and (current_time - last_append_time) >= 1.0:
                                append_to_text = True
                                last_appended_word = most_common
                                last_append_time = current_time
                        else:
                            hasil_prediksi = None
                            confidence = 0.0
                            if not hands_detected:
                                status_label = "Tangan tidak terdeteksi"
                            elif len(sequence) < 30:
                                status_label = f"Mengumpulkan frame... ({len(sequence)}/30)"
                            else:
                                status_label = "Tidak yakin"
                            
                            # Reset anti-spam lock when consistently Unknown
                            if most_common == "Unknown" and count >= threshold:
                                reset_kata = True
                                last_appended_word = ""
                    elif len(sequence) == 30:
                        # Sudah mulai prediksi tapi history masih < 3
                        status_label = f"Menunggu kestabilan... ({hist_len}/3)"
                    # else: status_label sudah diisi "Mengumpulkan frame..." di atas
            
            elapsed = time.time() - start_time
            
            # Diagnostic logs
            if hands_detected and (hasil_prediksi or status_label == "Stabil"):
                logger.info(
                    f"Mode: {mode_aktif} | "
                    f"Inputs: {126 if mode_aktif != 'KATA' else (30, 126)} | "
                    f"Label: {hasil_prediksi} | "
                    f"Confidence: {confidence:.2f} | "
                    f"Inference: {elapsed*1000:.1f}ms"
                )
            
            await websocket.send_json({
                "hasil": hasil_prediksi,
                "confidence": confidence,
                "mode": mode_raw,
                "status": status_label,
                "append": append_to_text,
                "reset_kata": reset_kata
            })
            
        except WebSocketDisconnect:
            logger.info("Klien terputus dari WebSocket.")
            break
        except Exception as e:
            logger.error(f"Kesalahan pada WebSocket loop: {e}", exc_info=True)
            try:
                await websocket.send_json({
                    "hasil": None,
                    "confidence": 0.0,
                    "mode": "ERROR",
                    "status": f"Kesalahan server: {str(e)}",
                    "append": False
                })
            except Exception:
                pass
            break