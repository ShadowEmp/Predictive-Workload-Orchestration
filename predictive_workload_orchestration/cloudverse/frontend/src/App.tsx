import React from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import Overview from './pages/Overview';
import HowItWorks from './pages/HowItWorks';
import Technology from './pages/Technology';
import Analytics from './pages/Analytics';
import CloudVerseApp from './cloudverse/CloudVerseApp';
import ComparisonView from './cloudverse/comparison/ComparisonView';
import { Activity, LayoutDashboard, Server, BarChart3, Globe } from 'lucide-react';
import './components/Navigation.css';

function Navigation() {
  const location = useLocation();
  const isCloudVerse = location.pathname.startsWith('/cloudverse');
  
  if (isCloudVerse) return null; // Hide navigation inside the full 3D app

  return (
    <nav className="glass-panel main-nav">
      <div className="nav-brand">
        <Globe className="brand-icon" />
        <span className="brand-text">CLOUDVERSE</span>
      </div>
      <div className="nav-links">
        <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>
          <LayoutDashboard size={18} className="nav-icon" /> Overview
        </Link>
        <Link to="/how-it-works" className={`nav-link ${location.pathname === '/how-it-works' ? 'active' : ''}`}>
          <Activity size={18} className="nav-icon" /> How It Works
        </Link>
        <Link to="/technology" className={`nav-link ${location.pathname === '/technology' ? 'active' : ''}`}>
          <Server size={18} className="nav-icon" /> Technology
        </Link>
        <Link to="/analytics" className={`nav-link ${location.pathname === '/analytics' ? 'active' : ''}`}>
          <BarChart3 size={18} className="nav-icon" /> Analytics
        </Link>
        <Link to="/cloudverse" className="btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          Enter Command Center
        </Link>
      </div>
    </nav>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Navigation />
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/how-it-works" element={<HowItWorks />} />
        <Route path="/technology" element={<Technology />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/cloudverse" element={<CloudVerseApp />} />
        <Route path="/cloudverse/comparison" element={<ComparisonView />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
