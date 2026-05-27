import { useState, useEffect, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import './Home.css';

function Home() {
  const webcamRef = useRef(null);
  const ws = useRef(null);
  const frameTimeoutRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  
  const [mode, setMode] = useState('KATA'); 
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [assembledText, setAssembledText] = useState("");
  const [statusText, setStatusText] = useState('Menghubungkan ke server...');
  const [isCameraActive, setIsCameraActive] = useState(true);
  const [wsStatus, setWsStatus] = useState('connecting'); // 'connecting' | 'connected' | 'disconnected' | 'error'
  const [errorMsg, setErrorMsg] = useState(null);

  // Keep references to state so the WebSocket message handler can access them
  // without needing to close and reopen the socket connection.
  const isCameraActiveRef = useRef(isCameraActive);
  const modeRef = useRef(mode);
  const sendFrameRef = useRef(null);

  useEffect(() => {
    isCameraActiveRef.current = isCameraActive;
  }, [isCameraActive]);

  useEffect(() => {
    modeRef.current = mode;
    // Clear prediction details when switching modes
    setPrediction(null);
    setConfidence(0);
    setStatusText('Mengumpulkan frame...');
  }, [mode]);

  // Centralized frame sender function
  const sendFrame = useCallback(() => {
    if (isCameraActiveRef.current && ws.current && ws.current.readyState === WebSocket.OPEN && webcamRef.current) {
      try {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          ws.current.send(JSON.stringify({
            mode: modeRef.current,
            image: imageSrc
          }));
        } else {
          // If screenshot failed (e.g. webcam not fully loaded), retry in 100ms
          frameTimeoutRef.current = setTimeout(sendFrame, 100);
        }
      } catch (err) {
        console.error("Gagal mengirim frame camera:", err);
      }
    }
  }, []);

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

    socket.onopen = () => {
      console.log('✅ Terhubung ke Server AI');
      setWsStatus('connected');
      setErrorMsg(null);
      
      // Kickstart the frame sending loop if camera is on
      if (isCameraActiveRef.current) {
        sendFrame();
      }
    };

    socket.onclose = () => {
      console.log('❌ Terputus dari Server AI');
      setWsStatus('disconnected');
      setPrediction(null);
      setConfidence(0);
      setStatusText('Koneksi terputus');
      
      // Auto-reconnect after 3 seconds
      reconnectTimeoutRef.current = setTimeout(connectWebSocket, 3000);
    };

    socket.onerror = (error) => {
      console.error('WebSocket Error:', error);
      setWsStatus('error');
      setErrorMsg('Gagal menyambung ke server AI. Pastikan backend FastAPI sudah berjalan di port 8000.');
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle server error message
        if (data.mode === 'ERROR') {
          setErrorMsg(data.status || 'Terjadi kesalahan sistem di server');
          return;
        }

        const { hasil, confidence: conf, status, append, reset_kata } = data;

        setPrediction(hasil);
        setConfidence(conf || 0);
        setStatusText(status || '');

        if (reset_kata) {
          // Reset text and predictions
          setPrediction(null);
          setConfidence(0);
        }

        // Appending text decided by the backend's smoothing/cooldown logic
        if (append && hasil) {
          setAssembledText((prev) => {
            if (data.mode === 'KATA' || data.mode === 'word') {
              return prev ? prev + " " + hasil : hasil;
            } else {
              return prev + hasil;
            }
          });
        }
      } catch (err) {
        console.error("Gagal membaca respons server:", err);
      }

      // Schedule sending the next frame (active feedback loop)
      // Caps the rate to 10 FPS (100ms interval) to avoid flooding
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

  // Initial connection
  useEffect(() => {
    connectWebSocket();

    return () => {
      if (ws.current) {
        ws.current.close();
      }
      if (frameTimeoutRef.current) {
        clearTimeout(frameTimeoutRef.current);
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connectWebSocket]);

  // Trigger frame send when webcam turns back on
  useEffect(() => {
    if (isCameraActive && ws.current && ws.current.readyState === WebSocket.OPEN) {
      // Clear any existing timeouts first to avoid multiple concurrent loops
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
