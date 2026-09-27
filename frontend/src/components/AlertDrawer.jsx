import React from 'react';
import { 
  X, 
  ShieldAlert
} from 'lucide-react';

export default function AlertDrawer({ isOpen, onClose, alerts }) {
  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      right: 0,
      width: '400px',
      height: '100vh',
      backgroundColor: '#ffffff',
      borderLeft: '1px solid #cbd5e1',
      boxShadow: 'var(--shadow-float)',
      zIndex: 2000,
      display: 'flex',
      flexDirection: 'column',
      animation: 'slideIn 0.25s ease-out'
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 20px',
        borderBottom: '1px solid #e2e8f0',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: '#f8fafc'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <ShieldAlert style={{ width: '20px', height: '20px', color: '#dc2626' }} />
          <div>
            <h3 style={{ fontSize: '15px', fontWeight: '800', color: '#0f172a' }}>
              NAVAREA & MoES Polar Alerts
            </h3>
            <span style={{ fontSize: '11px', color: '#64748b' }}>
              Official Maritime Safety Information (MSI)
            </span>
          </div>
        </div>
        <button
          onClick={onClose}
          style={{ background: 'transparent', border: 'none', color: '#64748b', cursor: 'pointer', padding: '4px' }}
        >
          <X style={{ width: '18px', height: '18px' }} />
        </button>
      </div>

      {/* Alert List */}
      <div style={{ padding: '16px', overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {alerts && alerts.map((alert) => {
          const isCritical = alert.severity === 'CRITICAL';
          return (
            <div
              key={alert.id}
              className="glass-card"
              style={{
                padding: '14px',
                borderColor: isCritical ? '#fecaca' : '#fde68a',
                background: isCritical ? '#fef2f2' : '#fffbeb'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <span className={`polar-badge ${isCritical ? 'polar-badge-danger' : 'polar-badge-gold'}`} style={{ fontSize: '9px' }}>
                  {alert.severity}
                </span>
                <span style={{ fontSize: '10px', color: '#64748b', fontFamily: 'var(--font-mono)' }}>
                  {alert.id}
                </span>
              </div>

              <h4 style={{ fontSize: '13px', fontWeight: '700', color: '#0f172a', marginBottom: '4px' }}>
                {alert.title}
              </h4>

              <p style={{ fontSize: '11px', color: '#475569', lineHeight: '1.4', marginBottom: '8px' }}>
                {alert.description}
              </p>

              <div style={{ padding: '7px 9px', background: '#ffffff', borderRadius: '6px', fontSize: '10px', marginBottom: '6px', border: '1px solid rgba(0,0,0,0.06)' }}>
                <div style={{ color: '#0284c7', fontWeight: '700', marginBottom: '1px' }}>
                  Directive / Action Required:
                </div>
                <div style={{ color: '#1e293b' }}>
                  {alert.recommended_action}
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#64748b', fontFamily: 'var(--font-mono)' }}>
                <span>Source: {alert.source.split(' / ')[0]}</span>
                <span>Lat: {alert.coordinates.lat}°S, {alert.coordinates.lon}°E</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
