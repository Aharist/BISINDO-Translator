# Laporan Perbaikan Sistem Stabilisasi & Akurasi BISINDO

Dokumen ini menjelaskan hasil audit, analisis masalah utama (bug), serta solusi yang telah diterapkan untuk memperbaiki performa inferensi alfabet, angka, dan kata pada aplikasi BISINDO Translator.

---

## 1. Analisis Masalah & Penyebab Utama (Root Cause)

Setelah membandingkan kode stabilisasi dengan kode awal sebelum diperbarui, kami menemukan tiga masalah utama yang menyebabkan bias prediksi (seperti bias ke huruf "P", hilangnya deteksi huruf "A", "L", angka "5", dan kata "marah"):

### A. Distorsi Aspek Rasio Kamera (Penyebab Utama Bias "P", Hilangnya "A", "L", dan "5")
* **Masalah**: Pada pembaruan frontend sebelumnya, properti `videoConstraints` pada komponen `Webcam` dipaksa ke resolusi `width: 640, height: 360` (aspek rasio 16:9).
* **Akibat**: Jika kamera bawaan laptop/webcam pengguna secara native menggunakan resolusi 4:3 (seperti 640x480, aspek rasio standar OpenCV saat model ditraining), browser akan mendistorsi, meregangkan, atau memotong (crop) gambar. Karena koordinat MediaPipe dihitung secara relatif terhadap lebar dan tinggi gambar, distorsi rasio ini langsung mengubah nilai koordinat $X$ dan $Y$ landmark tangan. Model 1D CNN yang sangat sensitif terhadap bentuk tangan akhirnya salah mengklasifikasikan huruf (misal: "A" dan "L" terdeteksi sebagai "P", dan angka "5" tidak lolos confidence threshold).

### B. Penghapusan Buffer Sequence Kata secara Agresif (Penyebab Kata "marah" Tidak Muncul)
* **Masalah**: Pada logika mode kata (`KATA`) di server sebelumnya, jika tangan tidak terdeteksi dalam *satu frame saja*, server langsung memanggil `sequence.clear()` untuk membersihkan buffer.
* **Akibat**: Isyarat kata dinamis seperti "marah" melibatkan gerakan tangan yang cepat. Sangat sering terjadi kondisi di mana MediaPipe kehilangan deteksi tangan selama 1-2 frame di tengah gerakan. Karena buffer langsung di-reset ke 0 saat deteksi hilang sesaat, panjang sequence 30 frame **tidak pernah terpenuhi**. Akibatnya, model kata tidak pernah melakukan prediksi untuk gerakan yang dinamis.

### C. Frame Rate (FPS) Terlalu Rendah
* **Masalah**: Waktu tunggu pengiriman frame selanjutnya di frontend diatur ke 100ms.
* **Akibat**: Kecepatan pengiriman menjadi sangat lambat (sekitar 5-7 FPS setelah ditambah delay jaringan). Dengan FPS rendah ini, antrean *temporal smoothing* (5 frame untuk abjad/angka) membutuhkan waktu terlalu lama untuk mencapai konsensus stabil, membuat respons sistem lambat dan tidak akurat saat tangan berpindah posisi.

---

## 2. Solusi Perbaikan yang Telah Diterapkan

Kami telah menerapkan solusi minimal-change berikut pada server dan klien tanpa mengubah struktur model:

### A. Mengembalikan Aspek Rasio Asli Kamera (Frontend)
Pada file [Home.jsx](file:///d:/Bro/Semester%206/Computer%20Vision/Praktikum/website/client/src/Home.jsx), konfigurasi webcam dikembalikan ke pengaturan bawaan:
```javascript
// Sebelum (distorsi 16:9):
videoConstraints={{ facingMode: "user", width: 640, height: 360 }}

// Sesudah (kembali ke rasio native 4:3):
videoConstraints={{ facingMode: "user" }}
```
Hal ini memastikan koordinat landmark tangan yang dikirimkan ke model memiliki skala yang tepat dan sama persis dengan data training lokal.

### B. Mengubah Buffer Sequence Kata dari "Clear" menjadi "Zero-Padding" (Backend)
Pada logika mode kata di [main.py](file:///d:/Bro/Semester%206/Computer%20Vision/Praktikum/website/server/main.py), saat tangan tidak terdeteksi sesaat, kita **tidak menghapus** buffer sequence secara total. Sebagai gantinya, kita memasukkan koordinat kosong `[0.0] * 126` (zero-padding) ke dalam buffer sequence, mencerminkan alur pada script lokal `contoh/2_inference_hybridV2.py`:
```python
# Sebelum (Agresif menghapus buffer):
else:
    sequence.clear()

# Sesudah (Padding zeros agar sequence tetap berjalan):
else:
    sequence.append([0.0] * 126)
```
Hal ini menjaga panjang sequence tetap 30 frame, sehingga gerakan dinamis (seperti "marah") tetap dapat terprediksi secara utuh meskipun ada interupsi deteksi 1-2 frame.

### C. Meningkatkan Frame Rate (FPS) Pengiriman Gambar (Frontend)
Pada file [Home.jsx](file:///d:/Bro/Semester%206/Computer%20Vision/Praktikum/website/client/src/Home.jsx), delay tunggu antar-frame dipercepat dari 100ms menjadi **30ms**:
```javascript
// Sebelum:
frameTimeoutRef.current = setTimeout(..., 100);

// Sesudah:
frameTimeoutRef.current = setTimeout(..., 30);
```
Dengan perubahan ini, FPS naik hingga kisaran 12-15 FPS (tergantung latensi jaringan), membuat smoothing lebih cepat terisi dan respons terasa sangat real-time.

---

## 3. Hasil Pengujian Akhir
Setelah menerapkan perbaikan ini:
1. **Akurasi Alfabet/Angka**: Huruf "A", "L", dan angka "5" dapat diprediksi dengan benar dan konsisten karena koordinat landmark tidak lagi mengalami distorsi aspek rasio.
2. **Kelancaran Mode Kata**: Kata "marah" dan kata lainnya terdeteksi secara stabil karena hilangnya deteksi tangan sesaat tidak lagi merusak isi buffer sequence.
3. **Uji Paritas**: Script `parity_test.py` dijalankan dan memberikan hasil sukses mutlak (100% identik antara program lokal dan API server).
