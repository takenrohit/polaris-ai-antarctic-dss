import React from 'react';
import { 
  Play, 
  Pause, 
  Layers, 
  Calendar
} from 'lucide-react';

export default function ControlPanel({
  forecastDay,
  setForecastDay,
  maxForecastDays = 7,
  isPlaying,
  setIsPlaying,
  layers,
  setLayers,
  cursorInfo,
  selectedRouteMode,
  setSelectedRouteMode,
  availableRoutes
}) {
  return (
    <div style={{
      position: 'absolute',
      bottom: '20px',
      left: '16px',
      zIndex: 999,
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
      maxWidth: '430px',
      width: 'calc(100vw - 32px)'
    }}>
      {/* Time & Forecast Horizon Card */}
      <div className="glass-panel" style={{ padding: '14px 18px', background: 'rgba(255, 255, 255, 0.95)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Calendar style={{ width: '15px', height: '15px', color: '#0284c7' }} />
            <span style={{ fontSize: '12px', fontWeight: '700', textTransform: 'uppercase', color: '#0f172a' }}>
              ConvLSTM Forecast Horizon
            </span>
          </div>
          <span className="polar-badge polar-badge-cyan">
            {forecastDay === 0 ? 'DAY 0 (OBSERVED)' : `DAY +${forecastDay} LEAD`}
          </span>
        </div>

        {/* Time Slider & Playback */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            style={{
              width: '32px',
              height: '32px',
              borderRadius: '50%',
              background: isPlaying ? '#fee2e2' : '#f0f9ff',
              border: `1px solid ${isPlaying ? '#fca5a5' : '#bae6fd'}`,
              color: isPlaying ? '#dc2626' : '#0284c7',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              boxShadow: '0 1px 3px rgba(0,0,0,0.06)'
            }}
            title={isPlaying ? 'Pause auto-play' : 'Play 7-day forecast cycle'}
          >
            {isPlaying ? <Pause style={{ width: '14px', height: '14px' }} /> : <Play style={{ width: '14px', height: '14px', marginLeft: '2px' }} />}
          </button>

          <div style={{ flex: 1 }}>
            <input
              type="range"
              min="0"
              max={maxForecastDays}
              value={forecastDay}
              onChange={(e) => {
                setIsPlaying(false);
                setForecastDay(parseInt(e.target.value));
              }}
              className="polar-slider"
              style={{ width: '100%' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px', fontSize: '10px', color: '#64748b', fontFamily: 'var(--font-mono)' }}>
              <span>Day 0</span>
              <span>Day +2</span>
              <span>Day +4</span>
              <span>Day +6</span>
              <span>Day +7</span>
            </div>
          </div>
        </div>
      </div>

      {/* Layer Controls & Route Mode Selector */}
      <div className="glass-panel" style={{ padding: '12px 16px', background: 'rgba(255, 255, 255, 0.95)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Layers style={{ width: '14px', height: '14px', color: '#0284c7' }} />
            <span style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: '#475569' }}>
              Map Layers & Routes
            </span>
          </div>
          {availableRoutes && (
            <div style={{ display: 'flex', gap: '3px' }}>
              {['balanced', 'safest', 'fastest', 'eco_fuel'].map((mode) => (
                <button
                  key={mode}
                  onClick={() => setSelectedRouteMode(mode)}
                  className={`btn-tab ${selectedRouteMode === mode ? 'active' : ''}`}
                  style={{ textTransform: 'capitalize', fontSize: '10px', padding: '3px 7px' }}
                >
                  {mode.replace('_', ' ')}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Toggles */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px', fontSize: '11px', color: '#334155' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={layers.seaIce}
              onChange={(e) => setLayers({ ...layers, seaIce: e.target.checked })}
              style={{ accentColor: '#0284c7' }}
            />
            <span>Sea Ice Concentration</span>
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={layers.icebergs}
              onChange={(e) => setLayers({ ...layers, icebergs: e.target.checked })}
              style={{ accentColor: '#0284c7' }}
            />
            <span>Icebergs & Drift Cones</span>
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={layers.stations}
              onChange={(e) => setLayers({ ...layers, stations: e.target.checked })}
              style={{ accentColor: '#0284c7' }}
            />
            <span>Stations (Bharati/Maitri)</span>
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={layers.fleet}
              onChange={(e) => setLayers({ ...layers, fleet: e.target.checked })}
              style={{ accentColor: '#0284c7' }}
            />
            <span>NCPOR Fleet AIS</span>
          </label>
        </div>

        {/* Legend */}
        <div style={{ marginTop: '10px', paddingTop: '8px', borderTop: '1px solid #f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '10px', color: '#64748b' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#e2e8f0', border: '1px solid #cbd5e1' }}></span>
            <span>Open (&lt;15%)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#bae6fd' }}></span>
            <span>MIZ (15-60%)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#38bdf8' }}></span>
            <span>Pack (60-90%)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#ffffff', border: '1px solid #94a3b8' }}></span>
            <span>Fast Ice (90%+)</span>
          </div>
        </div>
      </div>

      {/* Telemetry mini HUD */}
      {cursorInfo && (
        <div className="glass-panel" style={{ padding: '6px 14px', background: 'rgba(255, 255, 255, 0.95)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
          <div style={{ color: '#0f172a' }}>
            LAT: <strong>{cursorInfo.lat.toFixed(2)}°</strong> | LON: <strong>{cursorInfo.lon.toFixed(2)}°</strong>
          </div>
          <div style={{ color: cursorInfo.iceConc > 0.6 ? '#dc2626' : (cursorInfo.iceConc > 0.15 ? '#d97706' : '#059669'), fontWeight: '700' }}>
            SIC: {(cursorInfo.iceConc * 100).toFixed(1)}% ({cursorInfo.iceConc < 0.15 ? 'OPEN' : (cursorInfo.iceConc < 0.7 ? 'MIZ' : 'PACK')})
          </div>
        </div>
      )}
    </div>
  );
}
