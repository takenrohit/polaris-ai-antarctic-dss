import React, { useState } from 'react';
import { 
  X, 
  Snowflake
} from 'lucide-react';

export default function IcebergModal({
  isOpen,
  onClose,
  icebergs,
  icebergTrajectories,
  selectedBergId,
  onSelectBerg
}) {
  const [activeId, setActiveId] = useState(selectedBergId || icebergs?.[0]?.id || 'A-23a');

  if (!isOpen) return null;

  const currentBerg = icebergs?.find(b => b.id === activeId) || icebergs?.[0];
  const traj = icebergTrajectories?.find(t => t.iceberg_id === activeId);

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      backgroundColor: 'rgba(15, 23, 42, 0.45)',
      backdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 2000,
      padding: '20px'
    }}>
      <div className="glass-panel" style={{
        maxWidth: '980px',
        width: '100%',
        maxHeight: '90vh',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        background: '#ffffff',
        border: '1px solid #cbd5e1',
        boxShadow: 'var(--shadow-float)'
      }}>
        {/* Header */}
        <div style={{
          padding: '16px 24px',
          borderBottom: '1px solid #e2e8f0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: '#f8fafc'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: '36px',
              height: '36px',
              borderRadius: '8px',
              background: '#fee2e2',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid #fecaca'
            }}>
              <Snowflake style={{ width: '20px', height: '20px', color: '#dc2626' }} />
            </div>
            <div>
              <h2 style={{ fontSize: '17px', fontWeight: '800', color: '#0f172a' }}>
                Antarctic Iceberg Surveillance & Physics Drift Predictions
              </h2>
              <p style={{ fontSize: '12px', color: '#64748b' }}>
                BYU Iceberg Registry · Sentinel-1 SAR Multi-temporal Tracking · CMEMS Current Advection
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{ background: 'transparent', border: 'none', color: '#64748b', cursor: 'pointer', padding: '6px' }}
          >
            <X style={{ width: '20px', height: '20px' }} />
          </button>
        </div>

        {/* Body */}
        <div style={{ display: 'grid', gridTemplateColumns: '270px 1fr', flex: 1, overflow: 'hidden' }}>
          {/* Sidebar */}
          <div style={{ borderRight: '1px solid #e2e8f0', padding: '16px', overflowY: 'auto', background: '#f8fafc' }}>
            <div style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: '#475569', marginBottom: '10px' }}>
              Tracked Antarctic Megabergs
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {icebergs && icebergs.map((b) => (
                <div
                  key={b.id}
                  onClick={() => {
                    setActiveId(b.id);
                    if (onSelectBerg) onSelectBerg(b);
                  }}
                  className="glass-card"
                  style={{
                    padding: '12px',
                    cursor: 'pointer',
                    borderColor: b.id === activeId ? '#0284c7' : '#e2e8f0',
                    background: b.id === activeId ? '#f0f9ff' : '#ffffff'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: '700', fontSize: '13px', color: '#0f172a' }}>{b.id}</span>
                    <span className={`polar-badge ${b.hazard_level === 'CRITICAL' ? 'polar-badge-danger' : 'polar-badge-gold'}`} style={{ fontSize: '9px' }}>
                      {b.hazard_level}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: '#64748b', marginTop: '3px' }}>
                    {b.name}
                  </div>
                  <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: '#0284c7', marginTop: '4px', fontWeight: '600' }}>
                    Area: {b.area_km2.toLocaleString()} km²
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Dossier */}
          {currentBerg && (
            <div style={{ padding: '24px', overflowY: 'auto' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '18px', fontWeight: '800', color: '#0f172a' }}>
                    {currentBerg.name}
                  </h3>
                  <p style={{ fontSize: '12px', color: '#64748b', marginTop: '2px' }}>
                    Calved from {currentBerg.calving_source} (Origin {currentBerg.origin_year})
                  </p>
                </div>
                <div className={`polar-badge ${currentBerg.hazard_level === 'CRITICAL' ? 'polar-badge-danger' : 'polar-badge-gold'}`}>
                  {currentBerg.hazard_level} THREAT LEVEL
                </div>
              </div>

              {/* Physical Parameters */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '16px' }}>
                <div className="glass-card" style={{ padding: '12px', background: '#f8fafc' }}>
                  <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase', fontWeight: '600' }}>Surface Area</div>
                  <div style={{ fontSize: '18px', fontWeight: '800', color: '#0284c7', marginTop: '2px' }}>
                    {currentBerg.area_km2.toLocaleString()} km²
                  </div>
                </div>
                <div className="glass-card" style={{ padding: '12px', background: '#f8fafc' }}>
                  <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase', fontWeight: '600' }}>Dimensions</div>
                  <div style={{ fontSize: '18px', fontWeight: '800', color: '#0f172a', marginTop: '2px' }}>
                    {currentBerg.length_km} x {currentBerg.width_km} km
                  </div>
                </div>
                <div className="glass-card" style={{ padding: '12px', background: '#f8fafc' }}>
                  <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase', fontWeight: '600' }}>Thickness / Mass</div>
                  <div style={{ fontSize: '18px', fontWeight: '800', color: '#0f172a', marginTop: '2px' }}>
                    {currentBerg.thickness_m}m · {currentBerg.mass_gt} Gt
                  </div>
                </div>
                <div className="glass-card" style={{ padding: '12px', background: '#f8fafc' }}>
                  <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase', fontWeight: '600' }}>Drift Velocity</div>
                  <div style={{ fontSize: '18px', fontWeight: '800', color: '#d97706', marginTop: '2px' }}>
                    {currentBerg.drift_speed_knots} kts @ {currentBerg.drift_bearing_deg}°
                  </div>
                </div>
              </div>

              {/* Environmental Dynamics */}
              <div className="glass-card" style={{ padding: '14px', marginBottom: '16px', background: '#f8fafc' }}>
                <h4 style={{ fontSize: '12px', fontWeight: '700', textTransform: 'uppercase', color: '#475569', marginBottom: '8px' }}>
                  Oceanographic & Physical Forcing
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', fontSize: '11px' }}>
                  <div>
                    <div style={{ color: '#64748b' }}>Status & Drift Corridor:</div>
                    <div style={{ fontWeight: '600', color: '#0f172a', marginTop: '1px' }}>{currentBerg.status}</div>
                  </div>
                  <div>
                    <div style={{ color: '#64748b' }}>Surveillance Ingestion:</div>
                    <div style={{ fontWeight: '600', color: '#0284c7', marginTop: '1px' }}>{currentBerg.surveillance_source}</div>
                  </div>
                  <div>
                    <div style={{ color: '#64748b' }}>Keel vs Sail Draft Ratio:</div>
                    <div style={{ fontWeight: '600', color: '#0f172a', marginTop: '1px' }}>
                      ~87.5% Submerged Keel (~{(currentBerg.thickness_m * 0.875).toFixed(0)}m draft) / 12.5% Sail Freeboard
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#64748b' }}>Coriolis Deflection:</div>
                    <div style={{ fontWeight: '600', color: '#0f172a', marginTop: '1px' }}>
                      Leftward deflection in Southern Ocean relative to ACC westerlies
                    </div>
                  </div>
                </div>
              </div>

              {/* 120h Trajectory */}
              {traj?.trajectory && (
                <div>
                  <h4 style={{ fontSize: '12px', fontWeight: '700', textTransform: 'uppercase', color: '#0f172a', marginBottom: '8px' }}>
                    120-Hour Physics Drift Projections & Uncertainty Cones
                  </h4>
                  <div style={{ overflowX: 'auto', border: '1px solid #e2e8f0', borderRadius: '8px' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', fontFamily: 'var(--font-mono)', textAlign: 'left' }}>
                      <thead>
                        <tr style={{ background: '#f8fafc', color: '#475569', borderBottom: '1px solid #e2e8f0' }}>
                          <th style={{ padding: '8px 12px' }}>Forecast Horizon</th>
                          <th style={{ padding: '8px 12px' }}>Predicted Position</th>
                          <th style={{ padding: '8px 12px' }}>Speed / Bearing</th>
                          <th style={{ padding: '8px 12px' }}>Uncertainty Radius</th>
                          <th style={{ padding: '8px 12px' }}>P10-P90 Spread</th>
                        </tr>
                      </thead>
                      <tbody>
                        {traj.trajectory.filter((_, idx) => idx % 2 === 0).map((pt) => (
                          <tr key={pt.hour} style={{ borderBottom: '1px solid #f1f5f9' }}>
                            <td style={{ padding: '6px 12px', fontWeight: '700', color: '#0f172a' }}>T+{pt.hour} Hours</td>
                            <td style={{ padding: '6px 12px', color: '#0284c7' }}>{pt.lat.toFixed(2)}°S, {pt.lon.toFixed(2)}°E</td>
                            <td style={{ padding: '6px 12px' }}>{pt.speed_knots} kts @ {pt.bearing_deg}°</td>
                            <td style={{ padding: '6px 12px', color: '#d97706', fontWeight: '600' }}>±{pt.uncertainty_radius_km} km</td>
                            <td style={{ padding: '6px 12px', color: '#64748b' }}>
                              [{pt.p10_lat.toFixed(2)}, {pt.p10_lon.toFixed(2)}] → [{pt.p90_lat.toFixed(2)}, {pt.p90_lon.toFixed(2)}]
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
