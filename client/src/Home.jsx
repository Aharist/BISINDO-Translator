// Mengimpor Hook bawaan React dan komponen pihak ketiga
import { useState, useEffect, useRef, useCallback } from 'react';
import Webcam from 'react-webcam'; // Library kamera web untuk tangkapan antarmuka React
import './Home.css';

function Home() {
  // menyimpan variabel secara konstan yang jika nilainya berubah
  const webcamRef = useRef(null); // Referensi ke komponen Webcam untuk mengambil perintah screenshot
  const ws = useRef(null); // Penyimpan referensi jalur koneksi WebSocket
  const frameTimeoutRef = useRef(null); // Penyimpan timer/jeda untuk trigger pengiriman frame
  const reconnectTimeoutRef = useRef(null); // Penyimpan timer tunggu untuk auto-reconnect saat server mati
  
  // menyimpan data yang kalau nilainya berubah
  const [mode, setMode] = useState('KATA'); // State mode deteksi model AI (KATA, ABJAD, ANGKA)
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [assembledText, setAssembledText] = useState(""); 
  const [statusText, setStatusText] = useState('Menghubungkan ke server...');
  const [isCameraActive, setIsCameraActive] = useState(true);
  const [wsStatus, setWsStatus] = useState('connecting'); // Status Koneksi: 'connecting' | 'connected' | 'disconnected' | 'error'
  const [errorMsg, setErrorMsg] = useState(null);

  // Ref untuk menyimpan nilai yang sering diakses dalam callback
  // tanpa harus memasukkan ke dependency array
  const isCameraActiveRef = useRef(isCameraActive);
  const modeRef = useRef(mode);
  const sendFrameRef = useRef(null);

  // Sync refs dengan state untuk akses stabil dalam callback
  useEffect(() => {
    isCameraActiveRef.current = isCameraActive;
  }, [isCameraActive]);

  // Reset prediksi dan status saat mode berubah
  useEffect(() => {
    modeRef.current = mode;
    // Clear prediction details when switching modes
    setPrediction(null);
    setConfidence(0);
    setStatusText('Mengumpulkan frame...');
  }, [mode]);

  // Centralized frame sender function
  // Menggunakan useCallback agar fungsi ini menetap rapi di memori 
  const sendFrame = useCallback(() => {
    // Mengecek apakah kamera menyala, WebSocket terbuka, dan komponen Webcam terbaca
    if (isCameraActiveRef.current && ws.current && ws.current.readyState === WebSocket.OPEN && webcamRef.current) {
      try {
        const imageSrc = webcamRef.current.getScreenshot(); // Menangkap bingkai gambar yang diencode ke dalam format teks Base64
        if (imageSrc) {
          // Jika gambar berhasil ditangkap, bungkus jadi string JSON lengkap dengan Mode lalu lempar ke server lewat lorong WS
          ws.current.send(JSON.stringify({
            mode: modeRef.current,
            image: imageSrc
          }));
        } else {
          // If screenshot failed (e.g. webcam not fully loaded / Lagging browser), retry in 100ms
          frameTimeoutRef.current = setTimeout(sendFrame, 100);
        }
      } catch (err) {
        console.error("Gagal mengirim frame camera:", err);
      }
    }
  }, []);

  // Menyimpan referensi fungsi sendFrame agar bisa diakses dengan stabil dalam callback WebSocket
  useEffect(() => {
    sendFrameRef.current = sendFrame;
  }, [sendFrame]);

  // Connect and manage WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    setWsStatus('connecting');
    setErrorMsg(null);

    // Default address for local FastAPI
    const socket = new WebSocket('ws://127.0.0.1:8000/ws');

    // Momen pemicu ketika Backend berhasil/setuju menjawab koneksi 
    socket.onopen = () => {
      console.log('✅ Terhubung ke Server AI');
      setWsStatus('connected');
      setErrorMsg(null);
      
      // Kickstart the frame sending loop if camera is on (Memancing pemicu awal pengiriman frame untuk menjalankan sirkulasi feedback loop)
      if (isCameraActiveRef.current) {
        sendFrame();
      }
    };

    // Momen pemicu ketika koneksi WebSocket terputus dari Backend
    socket.onclose = () => {
      console.log('❌ Terputus dari Server AI');
      setWsStatus('disconnected');
      setPrediction(null);
      setConfidence(0);
      setStatusText('Koneksi terputus');
      
      // Fitur Auto-Reconnect: Jika Backend putus/mati/merasa timeout, Frontend akan terus mencoba menyambung ulang otomatis tiap 3 detik 
      reconnectTimeoutRef.current = setTimeout(connectWebSocket, 3000);
    };

    socket.onerror = (error) => {
      console.error('WebSocket Error:', error);
      setWsStatus('error');
      setErrorMsg('Gagal menyambung ke server AI. Pastikan backend FastAPI sudah berjalan di port 8000.');
    };

    // Momen saat menerima kiriman balasan pesan chat dari Backend berisi hasil tebakan prediksi
    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle server error message
        if (data.mode === 'ERROR') {
          setErrorMsg(data.status || 'Terjadi kesalahan sistem di server');
          return;
        }

        const { hasil, confidence: conf, status, append, reset_kata } = data;

        // Mengupdate teks hasil sementara UI secara langsung dengan state
        setPrediction(hasil);
        setConfidence(conf || 0);
        setStatusText(status || '');

        if (reset_kata) {
          // Reset text and predictions (Saat tangan tak ada)
          setPrediction(null);
          setConfidence(0);
        }

        // Merangkai kalimat: Jika status perintah "append": true ada pada balasan server
        if (append && hasil) {
          setAssembledText((prev) => {
            if (data.mode === 'KATA' || data.mode === 'word') {
              return prev ? prev + " " + hasil : hasil; // Diselipkan karakter SPASI kalau modenya adalah urutan kata
            } else {
              return prev + hasil; // Tempel abjad atau nomor berdampingan (tanpa spasi) 
            }
          });
        }
      } catch (err) {
        console.error("Gagal membaca respons server:", err);
      }

      // Schedule sending the next frame (active feedback loop)
      // INI PENTING: Mode Capped Frame FPS. Frontend HANYA berani mengirim frame berikutnya JIKA balasan sebelumnya dari server sudah selesai tiba.
      // Jeda antrean ditahan 30ms (Ini menjadi pembatasan beban server di rentang FPS +/- 20 hingga 30 agar RAM Backend tidak meleduk terbanting tumpukan paket WebSocket)
      if (isCameraActiveRef.current) {
        frameTimeoutRef.current = setTimeout(() => {
          if (sendFrameRef.current) {
            sendFrameRef.current();
          }
        }, 30);
      }
    };

    ws.current = socket;
  }, [sendFrame]);

  // Inisialisasi koneksi WebSocket saat komponen pertama kali dimuat
  useEffect(() => {
    connectWebSocket();

    return () => {
      if (ws.current) {
        ws.current.close(); // Tutup koneksi WebSocket saat komponen dilepas
      }
      if (frameTimeoutRef.current) {
        clearTimeout(frameTimeoutRef.current); // Bersihkan timer pengiriman frame saat komponen dilepas 
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current); // Bersihkan timer auto-reconnect saat komponen dilepas
      }
    };
  }, [connectWebSocket]);

  // Trigger frame send when webcam turns back on
  useEffect(() => {
    if (isCameraActive && ws.current && ws.current.readyState === WebSocket.OPEN) {
      // Jika kamera diaktifkan kembali, segera kirim frame untuk memulai kembali loop prediksi
      if (frameTimeoutRef.current) clearTimeout(frameTimeoutRef.current);
      sendFrame(); 
    } else if (!isCameraActive) {
      if (frameTimeoutRef.current) clearTimeout(frameTimeoutRef.current);
      setPrediction(null);
      setConfidence(0);
      setStatusText('Kamera dinonaktifkan');
    }
  }, [isCameraActive, sendFrame]);

  // Reset function
  const handleReset = useCallback(() => {
    setAssembledText("");
    setPrediction(null);
    setConfidence(0);
    setStatusText('Buffer di-reset');
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ reset: true, mode: modeRef.current }));
    }
  }, []);

  // Keyboard shortcut for Backspace to reset
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Backspace' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') { 
        handleReset();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleReset]);

  // Copy text utility
  const copyToClipboard = () => {
    if (assembledText) {
      navigator.clipboard.writeText(assembledText);
      alert('Teks berhasil disalin!');
    }
  };

  return (
    <div className="home-container">
      <h1 className="title">Sistem Penerjemah BISINDO</h1>
      
      {errorMsg && (
        <div className="alert-banner">
          <span className="alert-icon">⚠️</span>
          <span className="alert-message">{errorMsg}</span>
          <button className="alert-close-btn" onClick={() => setErrorMsg(null)}>&times;</button>
        </div>
      )}

      <div className="main-content">
        <div className="webcam-wrapper">
          <div className="webcam-container">
            {isCameraActive ? (
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className="webcam-video"
                videoConstraints={{ facingMode: "user" }}
              />
            ) : (
              <div className="webcam-placeholder">
                <span className="placeholder-icon">📷</span>
                <p>Kamera Dinonaktifkan</p>
                <button 
                  className="btn btn-primary" 
                  onClick={() => setIsCameraActive(true)}
                >
                  Aktifkan Kamera
                </button>
              </div>
            )}
            
            {/* WebSocket Connection Status Badge */}
            <div className={`status-badge ${wsStatus}`}>
              <span className="status-dot"></span>
              {wsStatus === 'connected' && 'Terhubung ke Server AI'}
              {wsStatus === 'connecting' && 'Menghubungkan...'}
              {wsStatus === 'disconnected' && 'Terputus - Mencoba kembali'}
              {wsStatus === 'error' && 'Koneksi Error'}
            </div>
          </div>

          <div className="controls">
            <button 
              className={`btn ${mode === 'ABJAD' ? 'active' : ''}`}
              onClick={() => setMode('ABJAD')}
            >
              Mode Abjad (Spelling)
            </button>
            
            <button 
              className={`btn ${mode === 'ANGKA' ? 'active' : ''}`}
              onClick={() => setMode('ANGKA')}
            >
              Mode Angka (Number)
            </button>
            
            <button 
              className={`btn ${mode === 'KATA' ? 'active' : ''}`}
              onClick={() => setMode('KATA')}
            >
              Mode Kata (Word)
            </button>
            
            <button 
              className={`btn ${isCameraActive ? 'btn-danger' : 'btn-success'}`}
              onClick={() => setIsCameraActive(prev => !prev)}
            >
              {isCameraActive ? 'Matikan Kamera' : 'Nyalakan Kamera'}
            </button>
          </div>
        </div>

        <div className="prediction-panel">
          <div className="prediction-header">Prediksi Terkini</div>
          
          <div className="prediction-content">
            {prediction ? (
              <>
                <div className="prediction-text">{prediction}</div>
                <div className="confidence-display">
                  Confidence: {(confidence * 100).toFixed(0)}%
                </div>
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill" 
                    style={{ width: `${confidence * 100}%` }}
                  ></div>
                </div>
              </>
            ) : (
              <div className="prediction-empty">
                {statusText && !statusText.includes("Mengumpulkan") && !statusText.includes("Menghubungkan") && !statusText.includes("Menunggu") ? null : (
                  <span className="status-spinner"></span>
                )}
                <p>{statusText || 'Menunggu gerakan isyarat...'}</p>
              </div>
            )}
          </div>
          
          <div className="panel-footer">
            <span className="info-badge">Mode Aktif: {mode}</span>
          </div>
        </div>
      </div>

      <div className="kalimat-section">
        <div className="kalimat-header">
          <div className="kalimat-label">Teks Hasil Susunan Terjemahan</div>
          <div className="text-actions">
            <button 
              className="btn btn-action" 
              onClick={copyToClipboard}
              disabled={!assembledText}
            >
              📋 Salin Teks
            </button>
            <button 
              className="btn btn-action reset-btn" 
              onClick={handleReset}
              disabled={!assembledText}
            >
              🗑️ Hapus Teks (Backspace)
            </button>
          </div>
        </div>
        
        <div className="kalimat-box">
          {assembledText ? assembledText : (
            <span className="kalimat-placeholder">
              Belum ada teks terjemahan yang disusun. Peragakan bahasa isyarat di depan kamera untuk mulai menyusun teks.
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default Home;
