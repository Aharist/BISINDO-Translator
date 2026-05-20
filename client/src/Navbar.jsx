import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

function Navbar() {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand">
          <h2>BISINDO Translator</h2>
        </div>
        <ul className="navbar-menu">
          <li>
            <Link 
              to="/" 
              className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
            >
              Home
            </Link>
          </li>
          <li>
            <Link 
              to="/informasi" 
              className={`nav-link ${location.pathname === '/informasi' ? 'active' : ''}`}
            >
              Informasi
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar;
