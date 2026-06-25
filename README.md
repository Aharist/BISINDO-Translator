# BISINDO Translator - Sign Language Recognition System

A real-time sign language recognition system for Indonesian Sign Language (BISINDO) built with **FastAPI** (Backend), **WebSocket**, **TensorFlow/Keras**, **MediaPipe**, and **React** (Frontend).

The system integrates three machine learning models:
1. **Alphabet model** (1D CNN) to recognize spelled letters.
2. **Numbers model** (1D CNN) to recognize digits 0-9.
3. **Word model** (CNN + LSTM hybrid) to translate sequences of movements into words.

---
Datasets:
1. Alphabets :https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo 
2. Numbers: https://github.com/ardamavi/Sign-Language-Digits-Dataset
3. Words: https://www.kaggle.com/datasets/xazhurea/test500


## Fitur Utama & Perbaikan Stabilitas

1. **Preprocessing Terpusat**: Seluruh data koordinat MediaPipe diolah melalui `server/preprocessing.py` agar sinkron dengan data latih (training).
2. **WebSocket Aktif & Capped FPS**: Menggunakan sistem *active feedback loop* yang mengirim frame kamera hanya setelah respons backend diterima, dengan batas maksimal 10 FPS (100ms) untuk mencegah kelambatan jaringan/server.
3. **Koneksi Auto-Reconnect**: Klien secara otomatis menyambungkan kembali koneksi WebSocket jika terputus, dilengkapi indikator status visual (*connected, connecting, disconnected, error*).
4. **Temporal Smoothing**: Prediksi diperhalus menggunakan antrean temporal (sliding window) berukuran 10 frame (untuk abjad/angka) dan 15 frame (untuk kata) sehingga keluaran stabil dan tidak berkedip.
5. **No-Sign & Reject Handling**: State dan buffer prediksi otomatis di-reset ketika tangan diturunkan atau tidak terdeteksi untuk menghindari *false positive*.
6. **Inference Non-Blocking**: Proses prediksi TensorFlow dipindahkan ke thread latar belakang menggunakan `asyncio.to_thread` agar koneksi WebSocket tetap responsif.
7. **Observability & Health Checks**: Endpoint baru `/health` dan `/models/status` ditambahkan untuk memonitor status server dan model, lengkap dengan logs waktu inference per request.
8. **UI/UX Premium**: Webcam control (start/stop), salin teks terjemahan, hapus teks, confidence bar, dan status label disajikan dengan palet warna premium.

---

## Cara Menjalankan Project

### 1. Menjalankan Backend (FastAPI)
Buka terminal di root direktori project, lalu jalankan perintah berikut:

```bash
# Menjalankan server FastAPI menggunakan uvicorn dari virtual environment
env\Scripts\uvicorn.exe server.main:app --port 8000
```
Server akan berjalan di `http://127.0.0.1:8000`. Anda bisa mengecek status model melalui:
- Health check: `http://127.0.0.1:8000/health`
- Status model: `http://127.0.0.1:8000/models/status`

### 2. Menjalankan Frontend (React + Vite)
Buka terminal baru di direktori `client`, lalu jalankan perintah berikut:

```bash
# Masuk ke direktori client
cd client

# Jalankan server development React/Vite
npm run dev
```
Aplikasi frontend dapat diakses di URL yang ditampilkan (biasanya `http://localhost:5173`).

---

## Menjalankan Uji Paritas (Parity Test)
Gunakan script ini untuk membandingkan output preprocessing dan prediksi model antara script lokal lama dengan script API baru:

```bash
env\Scripts\python.exe server/parity_test.py
```
Jika sukses, output akan menampilkan `[OK]` untuk semua komponen dan `ALL PARITY TESTS PASSED SUCCESSFULLY!`.
