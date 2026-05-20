import { useState, useEffect } from 'react';
import './Information.css';

function Information() {
  const [activeTab, setActiveTab] = useState('huruf');
  const [items, setItems] = useState([]);

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
          id: i,
          name: letter,
          // TRIK VITE: Load gambar dinamis dari folder src/assets
          // (Pastikan ekstensi gambarmu benar .jpg atau .png)
          image: new URL(`./assets/1DATA/huruf/${letter}.jpg`, import.meta.url).href,
          description: `Huruf ${letter}`
        };
      });
    } else if (tab === 'angka') {
      return Array.from({ length: 10 }, (_, i) => {
        return {
          id: i,
          name: i.toString(),
          // TRIK VITE: Load gambar dinamis dari folder src/assets
          image: new URL(`./assets/1DATA/angka/${i}.JPG`, import.meta.url).href,
          description: `Angka ${i}`
        };
      });
    } else {
      return [];
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
          disabled
        >
          Kata (Coming Soon)
        </button>
      </div>

      <div className="cards-container">
        {items.length > 0 ? (
          items.map((item) => (
            <div key={item.id} className="card">
              <div className="card-image-wrapper">
                <img 
                  src={item.image} 
                  alt={item.description}
                  className="card-image"
                  onError={handleImageError}
                />
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
    </div>
  );
}

export default Information;