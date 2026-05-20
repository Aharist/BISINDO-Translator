import { useState, useEffect, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import './Home.css';

function Home() {
  const webcamRef = useRef(null);
  const ws = useRef(null);
  
  const [mode, setMode] = useState('KATA'); 
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [kalimat, setKalimat] = useState([]);
  const [framingStatus, setFramingStatus] = useState('');
  
  const modeRef = useRef(mode);
  const kataTerakhirRef = useRef("");

  useEffect(() => {
    modeRef.current = mode;
    setPrediction(null);
    setConfidence(0);
    setFramingStatus('');
  }, [mode]);

  useEffect(() => {
    ws.current = new WebSocket('ws://127.0.0.1:8000/ws');

    ws.current.onopen = () => console.log('✅ Terhubung ke Server AI');
    ws.current.onclose = () => console.log('❌ Terputus dari Server AI');

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      const hasil = data.hasil;
      const conf = data.confidence || 0;
      const resetKata = data.reset_kata === true;

      if (resetKata) {
        kataTerakhirRef.current = "";
      }

      if (data.mode === 'KATA') {
        if (hasil) {
          setConfidence(conf);
          setPrediction(hasil);
          setFramingStatus('');
          if (hasil !== kataTerakhirRef.current) {
            setKalimat((prev) => [...prev, hasil]);
            kataTerakhirRef.current = hasil;
          }
        } else {
          setPrediction(null);
          setConfidence(0);
          setFramingStatus('');
        }
      } else if (data.mode === 'ABJAD' || data.mode === 'ANGKA') {
        if (hasil) {
          setPrediction(hasil);
          setConfidence(conf);
          setFramingStatus('');
        } else {
          setFramingStatus('');
          setPrediction(null);
          setConfidence(0);
        }
      }
    };

    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN && webcamRef.current) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          ws.current.send(JSON.stringify({
            mode: modeRef.current,
            image: imageSrc
          }));
        }
      }
    }, 50); 

    return () => clearInterval(interval);
  }, []); 

  const handleReset = useCallback(() => {
    setKalimat([]);
    kataTerakhirRef.current = "";
    setPrediction(null);
    setConfidence(0);
    setFramingStatus('');
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ reset: true }));
    }
  }, []);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Backspace') handleReset();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleReset]);

  return (
    <div className="home-container">
      <h1 className="title">Sistem Penerjemah BISINDO</h1>
      
      <div className="main-content">
        <div className="webcam-wrapper">
          <div className="webcam-container">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              className="webcam-video"
              videoConstraints={{ facingMode: "user" }}
            />
          </div>

          <div className="controls">
            <button 
              className={`btn ${mode === 'ABJAD' ? 'active' : ''}`}
              onClick={() => setMode('ABJAD')}
            >Mode Abjad</button>
            
            <button 
              className={`btn ${mode === 'ANGKA' ? 'active' : ''}`}
              onClick={() => setMode('ANGKA')}
            >Mode Angka</button>
            
            <button 
              className={`btn ${mode === 'KATA' ? 'active' : ''}`}
              onClick={() => {
                setMode('KATA');
                handleReset();
              }}
            >Mode Kata</button>
            
            {mode === 'KATA' && (
              <button className="btn reset-btn" onClick={handleReset}>
                Reset (Backspace)
              </button>
            )}
          </div>
        </div>

        <div className="prediction-panel">
          <div className="prediction-header">Prediksi Terkini</div>
          
          {prediction ? (
            <div className="prediction-content">
              <div className="prediction-text">{prediction}</div>
              <div className="confidence-display">
                Confidence: {(confidence * 100).toFixed(1)}%
              </div>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill" 
                  style={{ width: `${confidence * 100}%` }}
                ></div>
              </div>
            </div>
          ) : (
            <div className="prediction-empty">
              {framingStatus || 'Tidak ada prediksi'}
            </div>
          )}
        </div>
      </div>

      <div className="kalimat-section">
        <div className="kalimat-label">Kalimat</div>
        <div className="kalimat-box">
          {mode === 'KATA'
            ? (kalimat.length > 0 ? kalimat.join(" ") : "Belum ada kalimat. Mulai peragakan isyarat...")
            : "Kalimat hanya tersedia di Mode Kata."}
        </div>
      </div>

      <div className="hasil-box">
        <span className="mode-label">Mode: {mode}</span>
      </div>
    </div>
  );
}

export default Home;
