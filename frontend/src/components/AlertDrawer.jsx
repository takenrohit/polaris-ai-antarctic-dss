import React from 'react';
import { 
  X, 
  ShieldAlert,
  Info
} from 'lucide-react';

export default function AlertDrawer({ isOpen, onClose, alerts }) {
  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      right: 0,
      width: '420px',
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
          <ShieldAlert style={{ width: '20px', height: '20px', color: '#ea580c' }} />
          <div>
            <h3 style={{ fontSize: '15px', fontWeight: '800', color: '#0f172a' }}>
              Automated Scenario Advisories
            </h3>
            <span style={{ fontSize: '11px', color: '#64748b' }}>
              Rule-triggered alerts (illustrative, not official MSI)
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

      {/* Official Disclaimer Banner */}
      <div style={{ padding: '8px 16px', background: '#fffbeb', borderBottom: '1px solid #fef3c7', display: 'flex', gap: '8px', alignItems: 'center' }}>
        <Info style={{ width: '14px', height: '14px', color: '#d97706', flexShrink: 0 }} />
        <span style={{ fontSize: '10px', color: '#92400e', lineHeight: '1.3' }}>
          These advisories are rule-triggered simulation alerts computed from active environmental telemetry. They are <strong>not official NAVAREA, WMO, or government bulletins</strong>.
        </span>
      </div>

      {/* Alert List */}
      <div style={{ padding: '16px', overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {(!alerts || alerts.length === 0) ? (
          <div style={{
            padding: '36px 16px',
            textAlign: 'center',
            color: '#64748b',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '12px',
            background: '#f8fafc',
            borderRadius: '8px',
            border: '1px dashed #cbd5e1'
          }}>
            <ShieldAlert style={{ width: '32px', height: '32px', color: '#10b981' }} />
            <div>
              <div style={{ fontSize: '14px', fontWeight: '700', color: '#1e293b', marginBottom: '4px' }}>
                All Clear — No Active Hazard Advisories
              </div>
              <div style={{ fontSize: '11px', color: '#64748b', maxWidth: '300px', lineHeight: '1.4' }}>
                All observed wind speeds, sea ice concentrations, and thermal conditions are currently below warning thresholds.
              </div>
            </div>
          </div>
        ) : (
          alerts.map((alert) => {
          const isCritical = alert.severity === 'CRITICAL';
          return (
            <div
              key={alert.id}
              className="glass-card"
              style={{
                padding: '14px',
                borderColor: isCritical ? '#fecaca' : '#fde68a',
                background: isCritical ? '#fef2f2' : '#ffffff'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span className={`polar-badge ${isCritical ? 'polar-badge-danger' : 'polar-badge-gold'}`} style={{ fontSize: '9px' }}>
                    {alert.severity}
                  </span>
                  {alert.is_live ? (
                    <span style={{ fontSize: '9px', background: '#dcfce7', color: '#166534', padding: '1px 5px', borderRadius: '4px', fontWeight: '700' }}>
                      LIVE FEED
                    </span>
                  ) : (
                    <span style={{ fontSize: '9px', background: '#f1f5f9', color: '#475569', padding: '1px 5px', borderRadius: '4px', fontWeight: '600' }}>
                      OFFLINE STORE
                    </span>
                  )}
                </div>
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

              <div style={{ padding: '7px 9px', background: '#f8fafc', borderRadius: '6px', fontSize: '10px', marginBottom: '6px', border: '1px solid rgba(0,0,0,0.06)' }}>
                <div style={{ color: '#0284c7', fontWeight: '700', marginBottom: '1px' }}>
                  Rule-Recommended Action:
                </div>
                <div style={{ color: '#1e293b' }}>
                  {alert.recommended_action}
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', fontSize: '9px', color: '#64748b', fontFamily: 'var(--font-mono)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>{alert.source}</span>
                  {alert.coordinates && <span>{Math.abs(alert.coordinates.lat).toFixed(2)}°S, {Math.abs(alert.coordinates.lon).toFixed(2)}°E</span>}
                </div>
                <div style={{ color: '#94a3b8', fontStyle: 'italic' }}>
                  {alert.disclaimer}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
