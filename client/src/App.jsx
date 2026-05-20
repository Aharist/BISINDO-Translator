import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './Navbar';
import Home from './Home';
import Information from './Information';
import './App.css';

function App() {
  return (
    <Router>
      <Navbar />
      <main className="main-content-wrapper">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/informasi" element={<Information />} />
        </Routes>
      </main>
    </Router>
  );
}

export default App;