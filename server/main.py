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
# Import library standar dan pendukung
from collections import deque, Counter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# Setup structured logging (Konfigurasi pencatatan log/info di terminal agar error mudah terbaca)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BISINDO_Backend")

# Inisialisasi kerangka FastAPI (Ini yang membuat server berjalan)
app = FastAPI(title="BISINDO Recognition Server")

# Enable CORS: Mengizinkan request/komunikasi dari domain atau alamat port lain (Frontend React yang berbeda port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths absolutely to prevent working directory issues
# Memastikan penempatan path direktori akurat misal mencari letak model sekalipun script dijalankan dari luar folder server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Mengimpor modul logika pemrosesan kerangka tangan bawaan
from preprocessing import extract_landmarks

# Load model registry once at startup (Memuat AI satu kali saja saat server baru menyala agar irit memori dan kecepatan tinggi)
logger.info("Memuat Model AI ke dalam Server...")

try:
    # Mengambil jalur path untuk memuat arsitektur Model (.h5) dan File Label Decoder (.pkl)
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
    static_image_mode=False,     # Karena input berbentuk video real-time dinamis berurutan, kita set False
    max_num_hands=2,             # Maksimal memperbolehkan deteksi untuk 2 tangan sekaligus
    min_detection_confidence=0.7 # Toleransi kepercayaan/sensitivitas deteksi minimum batas MediaPipe yaitu 70%
)

logger.info("Server FastAPI Siap Beroperasi! (Ready)")

# Endpoint rute sederhana mengecek apakah server Backend hidup atau tidak
@app.get("/health")
def health_check():
    """Simple API status endpoint."""
    return {"status": "ok", "timestamp": time.time()}

# Endpoint rute mengecek status apakah ke-tiga AI model h5 berhasil di memori server
@app.get("/models/status")
def models_status():
    """Returns load status of ML models."""
    return {
        "model_abjad_loaded": model_abjad is not None,
        "model_angka_loaded": model_angka is not None,
        "model_kata_loaded": model_kata is not None
    }


# Jalur koneksi utama: WebSocket untuk komunikasi streaming
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept() # Menyambut/menerima koneksi dari panggilan masuk antarmuka Frontend
    logger.info("Klien terhubung ke WebSocket.")
    
    # Initialize queues local to this WebSocket connection session
    # Deque membatasi kumpulan array dengan max length, jadi saat over limit yang plaing lama akan antrian otomatis dihapus dari jendela
    sequence = deque(maxlen=30)         # Word model sequence buffer (Penampungan mode "Kata" butuh kumpulan urutan buffer per 30 frame sekaligus)
    history_abjad = deque(maxlen=5)     # Alphabet temporal smoothing queue (Voting konsensus: ambil 5 tebakan terakhir dari frame lalu)
    history_angka = deque(maxlen=5)     # Number temporal smoothing queue (Voting konsensus: ambil 5 tebakan terakhir dari frame lalu)
    history_kata = deque(maxlen=5)      # Word temporal smoothing queue (Khusus Kata, melihat/menyaring stabilitas hasil buffer 30 frame-nya)
    
    # Tracking variables for anti-spam (Mencegah sistem mengetik dobel/berulang kata yang sama terus-terusan)
    last_appended_char = ""
    last_appended_number = ""
    last_appended_word = ""
    last_append_time = 0.0              # Cooldown timestamp tracker (Pencatat waktu terkahir kata ditambah/diketik)
    
    # Looping utama WebSocket, menahan komunikasi terus terbuka agar sistem sanggup melayani data bertubi-tubi
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
                
            mode_raw = payload.get("mode", "KATA") # mode default
            image_base64 = payload.get("image", "") # Gambar dikirim dalam format string Base64 dari frontend
            
            # Map frontend modes safely to internal representations
            mode_aktif = mode_raw.upper()
            if mode_aktif == "ALPHABET":
                mode_aktif = "ABJAD"
            elif mode_aktif == "NUMBER":
                mode_aktif = "ANGKA"
            elif mode_aktif == "WORD":
                mode_aktif = "KATA"
                
            # Handle reset triggers
            # Reset state dikala user menekan Delete/Backspace atau sistem menangkap tangan diturunkan, history di clear agar mulai baru 
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
            
            # Abaikan jika payload gambar di JSON kosong
            if not image_base64:
                await websocket.send_json({
                    "hasil": None,
                    "confidence": 0.0,
                    "mode": mode_raw,
                    "status": "Payload gambar kosong",
                    "append": False
                })
                continue
                
            # Membuang format header base64 
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1] # ambil bagian setelah koma yang merupakan data gambar sebenarnya
                
            try:
                # Mengubah teks sandi raksasa Base64 dikonversi ke gambar Array OpenCV
                img_bytes = base64.b64decode(image_base64) # Decode string Base64 menjadi bytes gambar mentah
                np_arr = np.frombuffer(img_bytes, np.uint8) # Ubah bytes menjadi array NumPy 1D
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Decode array NumPy menjadi format gambar OpenCV (BGR)
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

            # mirror gambar karena tangkapan webcam biasanya terbalik
            frame = cv2.flip(frame, 1)
            # Mengonversi format warna dari BGR (bawaan OpenCV) menjadi RGB (yang dibutuhkan MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            start_time = time.time()
            # Menganalisa gambar RGB ke MediaPipe Hands untuk mencari tulang sendi tangan
            results = hands.process(frame_rgb)
            
            # Centralized preprocessing (Mengubah objek hasil pelacakan tangan MediaPipe menjadi Array angka mentah berisi 126 nilai kordinat dinormalisasi)
            hands_data = extract_landmarks(results.multi_hand_landmarks)
            hands_detected = results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0
            
            # Persiapan wadah menampung hasil tebakan awal
            hasil_prediksi = None
            confidence = 0.0
            status_label = ""
            append_to_text = False
            reset_kata = False
            current_time = time.time()
            
            # Execute model predictions asynchronously to prevent event loop blocking
            # Proses tebak AI Neural Network (TensorFlow) dieksekusi sebagai Threading Latar Belakang (asynchronous) agar tidak membuat WebSocket macet/delay
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
                        # Menyiapkan bentuk (Shape) Matrix untuk CNN 1D yakni berformat ukuran Batch 1x126x1
                        X_input = np.array([hands_data], dtype=np.float32)
                        X_input = np.expand_dims(X_input, axis=2) # Shape: (1, 126, 1)
                        
                        # Background thread execution for TensorFlow (menembak hasil nilai Matrix prediksi mentah dari model)
                        preds = await asyncio.to_thread(
                            lambda: model_abjad(X_input, training=False).numpy()[0] 
                        )
                        # Mencari nilai peluang (Confidence Persentase) tertinggi dari seluru tebakan kategori
                        raw_conf = float(np.max(preds))
                        # Mencari posisi indeks array yang menjadi pemenang kategori tebakan tersebut
                        pred_idx = np.argmax(preds)
                        
                        # Confidence minimal 0.75 untuk alfabet agar sistem tidak mencetak yang asal tebak
                        if raw_conf >= 0.75:
                            # Mentranslasi balik index angka kelas (0-25) menjadi Teks Abjad aslinya ('A', 'B', dsb)
                            char_pred = str(encoder_abjad.inverse_transform([pred_idx])[0])
                            history_abjad.append(char_pred) # Simpan hasil di buku memori antrian jendela waktu
                        else:
                            history_abjad.append("Unknown")
                    else:
                        history_abjad.append("Unknown")
                        
                    # Smooth predictions (Temporal Smoothing): valid if same label appears >= 3 times in last 5 predictions
                    # Smoothing memastikan bahwa model harus melihat simbol yang sama 3x dari 5 tangkapan sebelum mencetaknya
                    if len(history_abjad) == 5:
                        # Cari hasil mana yang paling sering muncul
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
                        raw_conf = float(np.max(preds)) # Confidence mentah tertinggi dari semua kategori tebakan angka
                        pred_idx = np.argmax(preds) # Posisi indeks kategori tebakan angka yang menang
                        
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
                    # Input Mode KATA tidak instan 1x126 nilai saja. 
                    # KATA membutuhkan urutan berkelanjutan gerak (Sequence). Jadi sistem harus merekam dan menjejali array sequence dulu hingga 30 data (frames) baru tebakannya jalan
                    if hands_detected:
                        sequence.append(hands_data)
                    else:
                        sequence.append([0.0] * 126) # Sumpal nilai nol sebagai pertanda jeda waktu istirahat (Tidak Deteksi Tangan)
                        
                    if len(sequence) == 30:
                        # Re-format jejak urutan pergerakan tangan yang sudah cukup 30x126 ini ke dalam Matrix LSTM/Hitungan
                        seq_array = np.array(list(sequence), dtype=np.float32)
                        X_input = np.expand_dims(seq_array, axis=0) # Shape: (1, 30, 126)
                        
                        # Background thread execution for TensorFlow model urutan KATA yang berat
                        preds = await asyncio.to_thread(
                            lambda: model_kata(X_input, training=False).numpy()[0]
                        )
                        raw_conf = float(np.max(preds))
                        pred_idx = np.argmax(preds)
                        
                        # Juru Kunci Threshold: Confidence minimal 0.80 untuk Kata
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
            # Kirim hasil prediksi dan status kembali ke frontend dalam format JSON
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