import React, { useState, useEffect } from 'react';
import { 
  Compass, 
  Navigation, 
  Snowflake, 
  ShieldAlert, 
  BarChart3, 
  Download, 
  Radio, 
  Clock
} from 'lucide-react';

export default function Navbar({ 
  onOpenRoutePlanner, 
  onOpenModelValidation, 
  onOpenIcebergs, 
  onOpenAlerts,
  activeAlertsCount = 3,
  currentRoute = null,
  onExportGeoJSON
}) {
  const [utcTime, setUtcTime] = useState('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toUTCString().replace('GMT', 'UTC'));
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header className="glass-panel" style={{ 
      margin: '10px 14px', 
      padding: '8px 18px', 
      zIndex: 1000, 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'space-between',
      background: 'rgba(255, 255, 255, 0.95)',
      boxShadow: '0 4px 20px -2px rgba(15, 23, 42, 0.08)'
    }}>
      {/* Brand & Organization */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '9px',
            background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 2px 8px rgba(2, 132, 199, 0.25)'
          }}>
            <Compass style={{ color: '#ffffff', width: '22px', height: '22px' }} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 style={{ fontSize: '16px', fontWeight: '800', letterSpacing: '-0.02em', color: '#0f172a' }}>
                POLARIS-AI
              </h1>
              <span className="polar-badge polar-badge-cyan" style={{ fontSize: '10px' }}>
                MoES · NCPOR
              </span>
              <span style={{ fontSize: '14px' }}>🇮🇳</span>
            </div>
            <p style={{ fontSize: '11px', color: '#64748b', fontWeight: '500' }}>
              Antarctic Sea-Ice & Navigation Decision Support Platform
            </p>
          </div>
        </div>

        {/* AI Engine Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginLeft: '10px', paddingLeft: '14px', borderLeft: '1px solid #e2e8f0' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#059669' }}></span>
            <span style={{ fontSize: '11px', color: '#475569', fontWeight: '500' }}>
              ConvLSTM: <strong style={{ color: '#0284c7' }}>ONLINE</strong>
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Radio style={{ width: '13px', height: '13px', color: '#d97706' }} />
            <span style={{ fontSize: '11px', color: '#475569', fontWeight: '500' }}>
              SAR/ERA5: <strong style={{ color: '#d97706' }}>INGESTED</strong>
            </span>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <button 
          onClick={onOpenRoutePlanner} 
          className="btn-primary"
        >
          <Navigation style={{ width: '15px', height: '15px' }} />
          <span>Optimize Route</span>
        </button>

        <button 
          onClick={onOpenModelValidation} 
          className="btn-secondary"
        >
          <BarChart3 style={{ width: '15px', height: '15px', color: '#0284c7' }} />
          <span>Model Benchmark</span>
        </button>

        <button 
          onClick={onOpenIcebergs} 
          className="btn-secondary"
        >
          <Snowflake style={{ width: '15px', height: '15px', color: '#0284c7' }} />
          <span>Tracked Icebergs</span>
        </button>

        <button 
          onClick={onOpenAlerts} 
          className="btn-secondary"
          style={{ position: 'relative' }}
        >
          <ShieldAlert style={{ width: '15px', height: '15px', color: '#dc2626' }} />
          <span>NAVAREA Alerts</span>
          {activeAlertsCount > 0 && (
            <span style={{
              position: 'absolute',
              top: '-4px',
              right: '-4px',
              backgroundColor: '#dc2626',
              color: '#fff',
              fontSize: '10px',
              fontWeight: '700',
              borderRadius: '9999px',
              width: '18px',
              height: '18px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 6px rgba(220, 38, 38, 0.4)'
            }}>
              {activeAlertsCount}
            </span>
          )}
        </button>

        {currentRoute && (
          <button 
            onClick={onExportGeoJSON}
            className="btn-secondary"
            style={{ color: '#059669', borderColor: '#a7f3d0' }}
            title="Download GeoJSON for ship ECDIS"
          >
            <Download style={{ width: '14px', height: '14px' }} />
            <span>Export ECDIS</span>
          </button>
        )}

        {/* Clock */}
        <div style={{ marginLeft: '8px', paddingLeft: '12px', borderLeft: '1px solid #e2e8f0', textAlign: 'right' }}>
          <div style={{ fontSize: '11px', color: '#0f172a', fontWeight: '700', fontFamily: 'var(--font-mono)' }}>
            {utcTime.split(' ').slice(4, 5).join(' ') || 'UTC --:--:--'}
          </div>
          <div style={{ fontSize: '9px', color: '#64748b', fontWeight: '600' }}>
            POLAR MISSION CLOCK
          </div>
        </div>
      </div>
    </header>
  );
}
