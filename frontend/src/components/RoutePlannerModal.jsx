import React, { useState } from 'react';
import { 
  X, 
  Navigation, 
  ShieldCheck, 
  Zap, 
  Fuel, 
  Scale, 
  Download
} from 'lucide-react';

export default function RoutePlannerModal({
  isOpen,
  onClose,
  stations,
  onOptimize,
  isLoading,
  currentRouteData,
  selectedMode,
  setSelectedMode,
  onExportGeoJSON
}) {
  const [originKey, setOriginKey] = useState('PORT_CAPE_TOWN');
  const [destKey, setDestKey] = useState('BHARATI_STATION');
  const [iceClass, setIceClass] = useState('PC5');
  const [speed, setSpeed] = useState(13.5);

  if (!isOpen) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
    onOptimize({
      origin_key: originKey,
      dest_key: destKey,
      vessel_ice_class: iceClass,
      cruising_speed_knots: parseFloat(speed)
    });
  };

  const routes = currentRouteData?.routes;

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
        maxWidth: '1060px',
        width: '100%',
        maxHeight: '92vh',
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
              background: '#e0f2fe',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid #bae6fd'
            }}>
              <Navigation style={{ width: '18px', height: '18px', color: '#0284c7' }} />
            </div>
            <div>
              <h2 style={{ fontSize: '17px', fontWeight: '800', color: '#0f172a' }}>
                Antarctic Route Optimizer & POLARIS Decision Engine
              </h2>
              <p style={{ fontSize: '12px', color: '#64748b' }}>
                Multi-objective graph optimization over dynamic ConvLSTM ice predictions & iceberg cones
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

        {/* Form Controls */}
        <div style={{ padding: '20px 24px', overflowY: 'auto' }}>
          <form onSubmit={handleSubmit} style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '20px', background: '#f8fafc', padding: '16px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
            {/* Origin */}
            <div>
              <label style={{ display: 'block', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: '#475569', marginBottom: '6px' }}>
                Departure Gateway
              </label>
              <select
                value={originKey}
                onChange={(e) => setOriginKey(e.target.value)}
                style={{
                  width: '100%',
                  background: '#ffffff',
                  border: '1px solid #cbd5e1',
                  borderRadius: '6px',
                  color: '#0f172a',
                  padding: '7px 10px',
                  fontSize: '12px',
                  outline: 'none'
                }}
              >
                {stations && Object.entries(stations)
                  .filter(([_, s]) => s.type === 'gateway_port' || s.type === 'ocean_waypoint')
                  .map(([key, st]) => (
                    <option key={key} value={key}>{st.name}</option>
                  ))
                }
              </select>
            </div>

            {/* Destination */}
            <div>
              <label style={{ display: 'block', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: '#475569', marginBottom: '6px' }}>
                Destination Station
              </label>
              <select
                value={destKey}
                onChange={(e) => setDestKey(e.target.value)}
                style={{
                  width: '100%',
                  background: '#ffffff',
                  border: '1px solid #cbd5e1',
                  borderRadius: '6px',
                  color: '#0f172a',
                  padding: '7px 10px',
                  fontSize: '12px',
                  outline: 'none'
                }}
              >
                {stations && Object.entries(stations)
                  .filter(([_, s]) => s.type === 'research_station' || s.type === 'ocean_waypoint')
                  .map(([key, st]) => (
                    <option key={key} value={key}>{st.name}</option>
                  ))
                }
              </select>
            </div>

            {/* Vessel Ice Class */}
            <div>
              <label style={{ display: 'block', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: '#475569', marginBottom: '6px' }}>
                IMO Polar Class
              </label>
              <select
                value={iceClass}
                onChange={(e) => setIceClass(e.target.value)}
                style={{
                  width: '100%',
                  background: '#ffffff',
                  border: '1px solid #cbd5e1',
                  borderRadius: '6px',
                  color: '#0f172a',
                  padding: '7px 10px',
                  fontSize: '12px',
                  outline: 'none'
                }}
              >
                <option value="PC1">PC1 - Year-round polar waters</option>
                <option value="PC3">PC3 - Multi-year heavy ice</option>
                <option value="PC5">PC5 - Medium First-Year (NCPOR Charter)</option>
                <option value="PC7">PC7 - Thin First-Year Ice</option>
                <option value="OPEN_WATER">Open Water Only (Non-ice)</option>
              </select>
            </div>

            {/* Speed & Submit */}
            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-end' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#475569', marginBottom: '4px' }}>
                <span>Cruising Speed</span>
                <span className="font-mono-numbers" style={{ color: '#0284c7', fontWeight: '700' }}>{speed} kts</span>
              </div>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <input
                  type="range"
                  min="8.0"
                  max="18.0"
                  step="0.5"
                  value={speed}
                  onChange={(e) => setSpeed(e.target.value)}
                  className="polar-slider"
                  style={{ flex: 1 }}
                />
                <button
                  type="submit"
                  disabled={isLoading}
                  className="btn-primary"
                  style={{ padding: '7px 12px', flexShrink: 0 }}
                >
                  {isLoading ? 'Computing...' : 'Recalculate'}
                </button>
              </div>
            </div>
          </form>

          {/* Pareto Cards */}
          {routes && (
            <div style={{ marginBottom: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <h3 style={{ fontSize: '13px', fontWeight: '700', textTransform: 'uppercase', color: '#475569' }}>
                  Pareto-Optimal Route Corridors
                </h3>
                <span style={{ fontSize: '11px', color: '#0284c7', fontWeight: '600' }}>
                  {currentRouteData.origin_name} → {currentRouteData.dest_name}
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                {Object.entries(routes).map(([modeKey, route]) => {
                  const isSelected = selectedMode === modeKey;
                  const icons = {
                    balanced: <Scale style={{ width: '16px', height: '16px', color: '#0284c7' }} />,
                    safest: <ShieldCheck style={{ width: '16px', height: '16px', color: '#059669' }} />,
                    fastest: <Zap style={{ width: '16px', height: '16px', color: '#d97706' }} />,
                    eco_fuel: <Fuel style={{ width: '16px', height: '16px', color: '#7c3aed' }} />
                  };

                  const borderColors = {
                    balanced: '#0284c7',
                    safest: '#059669',
                    fastest: '#d97706',
                    eco_fuel: '#7c3aed'
                  };

                  return (
                    <div
                      key={modeKey}
                      onClick={() => setSelectedMode(modeKey)}
                      className="glass-card"
                      style={{
                        padding: '14px',
                        cursor: 'pointer',
                        borderColor: isSelected ? borderColors[modeKey] : '#e2e8f0',
                        boxShadow: isSelected ? `0 4px 14px ${borderColors[modeKey]}25` : 'none',
                        background: isSelected ? '#f8fafc' : '#ffffff',
                        position: 'relative'
                      }}
                    >
                      {modeKey === 'balanced' && (
                        <span className="polar-badge polar-badge-cyan" style={{ position: 'absolute', top: '-9px', right: '10px', fontSize: '9px', padding: '2px 6px' }}>
                          NCPOR OPTIMAL
                        </span>
                      )}

                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                        {icons[modeKey]}
                        <h4 style={{ fontSize: '13px', fontWeight: '700', color: '#0f172a', textTransform: 'capitalize' }}>
                          {modeKey.replace('_', ' ')}
                        </h4>
                      </div>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: '5px', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: '#64748b' }}>Distance:</span>
                          <span style={{ fontWeight: '700', color: '#0f172a' }}>{route.total_distance_nm} NM</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: '#64748b' }}>Transit:</span>
                          <span style={{ fontWeight: '700', color: '#0f172a' }}>{route.total_transit_days}d ({route.total_transit_hours}h)</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: '#64748b' }}>Fuel Burn:</span>
                          <span style={{ fontWeight: '700', color: '#0f172a' }}>{route.total_fuel_mt} MT</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: '#64748b' }}>POLARIS:</span>
                          <span style={{ fontWeight: '700', color: route.minimum_polaris_rio >= 0 ? '#059669' : '#dc2626' }}>
                            RIO {route.minimum_polaris_rio}
                          </span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: '#64748b' }}>Safety Index:</span>
                          <span style={{ fontWeight: '700', color: '#0284c7' }}>{route.overall_safety_score}/100</span>
                        </div>
                      </div>

                      <div style={{ marginTop: '10px', paddingTop: '6px', borderTop: '1px solid #f1f5f9', textAlign: 'center' }}>
                        <span style={{ fontSize: '10px', color: isSelected ? borderColors[modeKey] : '#94a3b8', fontWeight: '600' }}>
                          {isSelected ? '✓ ACTIVE ON MAP' : 'Click to select'}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Waypoints Table */}
          {routes && routes[selectedMode] && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <h4 style={{ fontSize: '13px', fontWeight: '700', color: '#0f172a' }}>
                  Waypoint Leg Directives ({routes[selectedMode].mode_name})
                </h4>
                <button
                  onClick={onExportGeoJSON}
                  className="btn-secondary"
                  style={{ fontSize: '11px', padding: '4px 10px' }}
                >
                  <Download style={{ width: '13px', height: '13px' }} />
                  <span>Download GeoJSON for ECDIS</span>
                </button>
              </div>

              <div style={{ overflowX: 'auto', border: '1px solid #e2e8f0', borderRadius: '8px' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', fontFamily: 'var(--font-mono)', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ background: '#f8fafc', color: '#475569', borderBottom: '1px solid #e2e8f0' }}>
                      <th style={{ padding: '8px 12px' }}>Leg #</th>
                      <th style={{ padding: '8px 12px' }}>Coordinates</th>
                      <th style={{ padding: '8px 12px' }}>Distance</th>
                      <th style={{ padding: '8px 12px' }}>Speed</th>
                      <th style={{ padding: '8px 12px' }}>Ice Conc</th>
                      <th style={{ padding: '8px 12px' }}>POLARIS RIO</th>
                      <th style={{ padding: '8px 12px' }}>Nearest Iceberg</th>
                      <th style={{ padding: '8px 12px' }}>Fuel Burn</th>
                    </tr>
                  </thead>
                  <tbody>
                    {routes[selectedMode].waypoints.map((wp) => (
                      <tr key={wp.leg_index} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '7px 12px', fontWeight: '700', color: '#0f172a' }}>#{wp.leg_index}</td>
                        <td style={{ padding: '7px 12px', color: '#0284c7' }}>{wp.lat.toFixed(2)}°S, {wp.lon.toFixed(2)}°E</td>
                        <td style={{ padding: '7px 12px' }}>{wp.leg_dist_nm} NM ({wp.cumulative_dist_nm})</td>
                        <td style={{ padding: '7px 12px' }}>{wp.speed_knots} kts</td>
                        <td style={{ padding: '7px 12px', color: wp.ice_concentration_pct > 50 ? '#dc2626' : '#0f172a' }}>
                          {wp.ice_concentration_pct}%
                        </td>
                        <td style={{ padding: '7px 12px' }}>
                          <span style={{ color: wp.polaris_rio >= 0 ? '#059669' : '#dc2626', fontWeight: '700' }}>
                            RIO {wp.polaris_rio} ({wp.polaris_status})
                          </span>
                        </td>
                        <td style={{ padding: '7px 12px' }}>
                          {wp.nearest_iceberg_dist_nm ? (
                            <span style={{ color: wp.nearest_iceberg_dist_nm < 20 ? '#dc2626' : '#d97706' }}>
                              {wp.nearest_iceberg_dist_nm} NM ({wp.nearest_iceberg_id})
                            </span>
                          ) : 'Clear (>90 NM)'}
                        </td>
                        <td style={{ padding: '7px 12px' }}>{wp.leg_fuel_burn_mt} MT</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
