import { useState, useEffect } from 'react';
import './Information.css';

function Information() {
  const [activeTab, setActiveTab] = useState('huruf');
  const [items, setItems] = useState([]);
  
  // State untuk melacak video mana yang sedang dibuka di pop-up
  const [selectedVideo, setSelectedVideo] = useState(null);

  useEffect(() => {
    loadData(activeTab);
  }, [activeTab]);

  const loadData = (tab) => {
    const mockData = generateMockData(tab);
    setItems(mockData);
  };

  const generateMockData = (tab) => {
    if (tab === 'huruf') {
      return Array.from({ length: 26 }, (_, i) => {
        const letter = String.fromCharCode(65 + i);
        return {
          id: `huruf-${i}`,
          name: letter,
          image: new URL(`./assets/1DATA/huruf/${letter}.jpg`, import.meta.url).href,
          video: null, 
          description: `Isyarat Huruf ${letter}`
        };
      });
    } else if (tab === 'angka') {
      return Array.from({ length: 10 }, (_, i) => {
        return {
          id: `angka-${i}`,
          name: i.toString(),
          image: new URL(`./assets/1DATA/angka/${i}.JPG`, import.meta.url).href,
          video: null, 
          description: `Isyarat Angka ${i}`
        };
      });
    } else if (tab === 'kata') {
      // MASUKKAN NAMA-NAMA FILE MP4 KAMU DI SINI (tanpa .mp4)
      const daftarKata = ["Anak", "Makan", "Saya", "Halo"]; 
      
      return daftarKata.map((word, i) => ({
        id: `kata-${i}`,
        name: word,
        image: null, // Kita set null karena tidak punya file JPG
        video: new URL(`./assets/1DATA/kata/${word}.mp4`, import.meta.url).href,
        description: `Peragakan isyarat kata "${word}"`
      }));
    }
    return [];
  };

  const handleCardClick = (item) => {
    // Buka pop-up jika item punya video
    if (item.video) {
      setSelectedVideo(item.video);
    }
  };

  const handleImageError = (e) => {
    e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect fill="%23f0f0f0" width="200" height="200"/%3E%3Ctext x="50%25" y="50%25" font-family="Arial" font-size="14" fill="%23999" text-anchor="middle" dy=".3em"%3EImage Not Found%3C/text%3E%3C/svg%3E';
  };

  return (
    <div className="information-container">
      <h1 className="info-title">Informasi Bahasa Isyarat BISINDO</h1>
      
      <div className="tabs">
        <button 
          className={`tab-btn ${activeTab === 'huruf' ? 'active' : ''}`}
          onClick={() => setActiveTab('huruf')}
        >
          Huruf
        </button>
        <button 
          className={`tab-btn ${activeTab === 'angka' ? 'active' : ''}`}
          onClick={() => setActiveTab('angka')}
        >
          Angka
        </button>
        <button 
          className={`tab-btn ${activeTab === 'kata' ? 'active' : ''}`}
          onClick={() => setActiveTab('kata')}
        >
          Kata
        </button>
      </div>

      <div className="cards-container">
        {items.length > 0 ? (
          items.map((item) => (
            <div 
              key={item.id} 
              className={`card ${item.video ? 'playable-card' : ''}`}
              onClick={() => handleCardClick(item)}
            >
              <div className="card-image-wrapper">
                
                {/* LOGIKA BARU: Jika ada gambar, tampilkan IMG. Jika tidak, jadikan VIDEO sebagai thumbnail */}
                {item.image ? (
                  <img 
                    src={item.image} 
                    alt={item.description}
                    className="card-image"
                    onError={handleImageError}
                  />
                ) : (
                  <video 
                    src={`${item.video}#t=0.1`} /* #t=0.1 trik mengambil detik ke 0.1 sbg foto sampul */
                    className="card-image"
                    muted
                    playsInline
                  />
                )}

                {item.video && (
                  <div className="play-overlay">
                    <span>▶ Putar Video</span>
                  </div>
                )}
              </div>
              <div className="card-content">
                <div className="card-name">{item.name}</div>
                <div className="card-description">{item.description}</div>
              </div>
            </div>
          ))
        ) : (
          <div className="empty-state">
            <p>Data tidak tersedia untuk tab ini</p>
          </div>
        )}
      </div>

      {/* MODAL POP-UP VIDEO */}
      {selectedVideo && (
        <div className="modal-overlay" onClick={() => setSelectedVideo(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-modal-btn" onClick={() => setSelectedVideo(null)}>
              &times;
            </button>
            <video 
              src={selectedVideo} 
              autoPlay 
              loop 
              muted 
              controls 
              className="modal-video"
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default Information;