import React, { useEffect, useRef } from 'react';
import { 
  MapContainer, 
  TileLayer, 
  Marker, 
  Popup, 
  Tooltip,
  Polyline, 
  Polygon, 
  CircleMarker, 
  useMap,
  useMapEvents 
} from 'react-leaflet';
import L from 'leaflet';

// Modern, Minimal Marker Icons (prevents visual clutter and label overlap)
const createStationIcon = (name, isIndian) => {
  const color = isIndian ? '#ea580c' : '#0284c7';
  return L.divIcon({
    className: 'custom-station-pin',
    html: `
      <div style="display:flex;align-items:center;justify-content:center;position:relative;cursor:pointer;">
        <div style="background:#ffffff;border:2.5px solid ${color};width:16px;height:16px;border-radius:50%;box-shadow:0 2px 6px rgba(0,0,0,0.25);display:flex;align-items:center;justify-content:center;">
          <div style="width:5px;height:5px;border-radius:50%;background:${color};"></div>
        </div>
      </div>
    `,
    iconSize: [20, 20],
    iconAnchor: [10, 10]
  });
};

const createIcebergIcon = (name, hazard) => {
  const color = hazard === 'CRITICAL' ? '#dc2626' : '#d97706';
  return L.divIcon({
    className: 'custom-iceberg-pin',
    html: `
      <div style="position:relative;display:flex;align-items:center;justify-content:center;cursor:pointer;">
        <div class="radar-ring" style="border-color:${color};"></div>
        <div style="background:#ffffff;border:2.5px solid ${color};width:16px;height:16px;transform:rotate(45deg);box-shadow:0 2px 6px rgba(0,0,0,0.25);display:flex;align-items:center;justify-content:center;">
          <div style="width:4px;height:4px;background:${color};transform:rotate(45deg);"></div>
        </div>
      </div>
    `,
    iconSize: [20, 20],
    iconAnchor: [10, 10]
  });
};

const createVesselIcon = (heading = 0) => {
  return L.divIcon({
    className: 'custom-vessel-pin',
    html: `
      <div style="display:flex;align-items:center;justify-content:center;cursor:pointer;">
        <div style="transform:rotate(${heading}deg);width:24px;height:24px;background:#ffffff;border:2px solid #0284c7;border-radius:50%;box-shadow:0 2px 8px rgba(2,132,199,0.3);display:flex;align-items:center;justify-content:center;">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="#0284c7" stroke="#0284c7" stroke-width="2">
            <polygon points="12 2 19 21 12 17 5 21 12 2" />
          </svg>
        </div>
      </div>
    `,
    iconSize: [24, 24],
    iconAnchor: [12, 12]
  });
};

// Canvas Layer for Sea Ice Concentration Heatmap
function SeaIceCanvasOverlay({ forecastData, forecastDay, opacity = 0.38, onMouseMoveCoord }) {
  const map = useMap();
  const canvasRef = useRef(null);

  useMapEvents({
    mousemove(e) {
      if (!forecastData) return;
      const lat = e.latlng.lat;
      const lon = e.latlng.lng;
      let sic = 0.0;
      if (lat < -55.0) {
        sic = Math.min(1.0, Math.max(0.0, (-55.0 - lat) / 18.0 * 0.88));
        if (lat < -70.0) sic = Math.min(1.0, sic + 0.15);
      }
      if (onMouseMoveCoord) {
        onMouseMoveCoord({ lat, lon, iceConc: sic });
      }
    }
  });

  useEffect(() => {
    if (!map || !forecastData) return;

    let canvas = canvasRef.current;
    if (!canvas) {
      canvas = L.DomUtil.create('canvas', 'leaflet-sea-ice-canvas');
      canvas.style.position = 'absolute';
      canvas.style.top = '0';
      canvas.style.left = '0';
      canvas.style.pointerEvents = 'none';
      canvas.style.zIndex = '200';
      map.getPanes().overlayPane.appendChild(canvas);
      canvasRef.current = canvas;
    }

    const drawGrid = () => {
      const size = map.getSize();
      canvas.width = size.x;
      canvas.height = size.y;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, size.x, size.y);

      const bounds = map.getBounds();
      const lats = forecastData.latitudes;
      const lons = forecastData.longitudes;
      const dayData = forecastDay === 0 
        ? forecastData.ground_truth_day0 
        : (forecastData.forecast_days[forecastDay - 1]?.model_grid || forecastData.ground_truth_day0);

      if (!dayData || !lats || !lons) return;

      const dLat = Math.abs(lats[1] - lats[0]) || 0.5;
      const dLon = Math.abs(lons[1] - lons[0]) || 1.0;

      // Soft blur for natural satellite data blending without blocking the map
      ctx.filter = 'blur(12px)';

      for (let i = 0; i < lats.length; i += 1) {
        const lat = lats[i];
        if (lat > bounds.getNorth() + 2 || lat < bounds.getSouth() - 2) continue;

        for (let j = 0; j < lons.length; j += 1) {
          const lon = lons[j];
          if (lon < bounds.getWest() - 2 || lon > bounds.getEast() + 2) continue;

          const conc = dayData[i]?.[j] || 0.0;
          if (conc < 0.15) continue; // Skip open water (keep ocean visible)

          const p1 = map.latLngToContainerPoint([lat, lon]);
          const p2 = map.latLngToContainerPoint([lat - dLat * 1.5, lon + dLon * 1.5]);
          const w = Math.abs(p2.x - p1.x);
          const h = Math.abs(p2.y - p1.y);

          // Translucent oceanographic tints: bathymetry and coastlines stay clearly visible
          if (conc < 0.40) {
            ctx.fillStyle = `rgba(14, 165, 233, ${opacity * 0.45})`; // Marginal Ice Zone (light cyan tint)
          } else if (conc < 0.75) {
            ctx.fillStyle = `rgba(2, 132, 199, ${opacity * 0.65})`;  // Pack Ice (medium polar blue)
          } else {
            ctx.fillStyle = `rgba(186, 230, 253, ${opacity * 0.85})`; // Consolidated pack / Fast ice
          }

          ctx.beginPath();
          ctx.ellipse(p1.x, p1.y, Math.max(w * 0.85, 7), Math.max(h * 0.85, 7), 0, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
      ctx.filter = 'none';
    };

    drawGrid();
    map.on('moveend zoomend resize', drawGrid);

    return () => {
      map.off('moveend zoomend resize', drawGrid);
      if (canvas && canvas.parentNode) {
        canvas.parentNode.removeChild(canvas);
        canvasRef.current = null;
      }
    };
  }, [map, forecastData, forecastDay, opacity]);

  return null;
}

export default function PolarMap({
  forecastData,
  forecastDay,
  layers,
  stations,
  icebergs,
  icebergTrajectories,
  vessels,
  routes,
  selectedRouteMode,
  onSelectIceberg,
  onMouseMoveCoord
}) {
  const initialCenter = [-55.0, 45.0];
  const initialZoom = 3;

  // Flexible vertical pan bounds: allows moving smoothly from Antarctica (-85°) to North Hemisphere (75°)
  const mapBounds = [
    [-85.0, -180.0], // South Pole
    [75.0, 180.0]    // Northern transit corridors
  ];

  return (
    <div style={{ position: 'relative', width: '100%', height: 'calc(100vh - 84px)' }}>
      <MapContainer
        center={initialCenter}
        zoom={initialZoom}
        minZoom={2}
        maxZoom={9}
        maxBounds={mapBounds}
        maxBoundsViscosity={0.7}
        worldCopyJump={false}
        scrollWheelZoom={true}
        style={{ width: '100%', height: '100%' }}
      >
        {/* Esri World Ocean Basemap - single true map with noWrap to prevent sideways looping */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
          attribution='&copy; Esri, GEBCO, NOAA | MoES NCPOR'
          maxZoom={13}
          noWrap={true}
          bounds={mapBounds}
        />
        {/* Ocean Reference Labels & Depth Contours */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}"
          maxZoom={13}
          opacity={0.65}
          noWrap={true}
          bounds={mapBounds}
        />

        {/* Sea Ice Concentration Heatmap Overlay */}
        {layers.seaIce && forecastData && (
          <SeaIceCanvasOverlay
            forecastData={forecastData}
            forecastDay={forecastDay}
            onMouseMoveCoord={onMouseMoveCoord}
          />
        )}

        {/* Research Stations & Gateway Ports */}
        {layers.stations && stations && Object.entries(stations).map(([key, st]) => {
          const isIndian = key.includes('BHARATI') || key.includes('MAITRI') || key.includes('DG');
          return (
            <Marker
              key={key}
              position={[st.lat, st.lon]}
              icon={createStationIcon(st.name.split(' (')[0], isIndian)}
            >
              {/* Tooltip on hover (clean & uncluttered) */}
              <Tooltip direction="top" offset={[0, -10]} opacity={0.95}>
                <div style={{ fontSize: '11px', fontWeight: '600', color: isIndian ? '#ea580c' : '#0284c7' }}>
                  {isIndian ? '🇮🇳 ' : ''}{st.name}
                </div>
              </Tooltip>

              <Popup>
                <div style={{ padding: '4px', maxWidth: '220px' }}>
                  <div style={{ fontWeight: '700', fontSize: '13px', color: isIndian ? '#ea580c' : '#0284c7' }}>
                    {isIndian ? '🇮🇳 ' : ''}{st.name}
                  </div>
                  <div style={{ fontSize: '11px', color: '#64748b', marginTop: '3px' }}>
                    {st.region}
                  </div>
                  <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', marginTop: '4px', color: '#0f172a' }}>
                    {st.lat.toFixed(3)}°S, {st.lon.toFixed(3)}°E
                  </div>
                  {st.access_fast_ice_zone && (
                    <div className="polar-badge polar-badge-gold" style={{ marginTop: '8px', fontSize: '10px' }}>
                      Fast-Ice Approach Corridor
                    </div>
                  )}
                </div>
              </Popup>
            </Marker>
          );
        })}

        {/* NCPOR Research Vessels */}
        {layers.fleet && vessels && vessels.map((v) => (
          <Marker
            key={v.id}
            position={[v.current_position.lat, v.current_position.lon]}
            icon={createVesselIcon(v.current_position.heading_deg)}
          >
            <Tooltip direction="top" offset={[0, -12]} opacity={0.95}>
              <div style={{ fontSize: '11px', fontWeight: '600', color: '#0284c7' }}>
                🚢 {v.name.split(' (')[0]}
              </div>
            </Tooltip>

            <Popup>
              <div style={{ padding: '4px', minWidth: '200px' }}>
                <div style={{ fontWeight: '700', fontSize: '13px', color: '#0284c7' }}>
                  🚢 {v.name}
                </div>
                <div style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
                  Polar Class: <strong>{v.ice_class}</strong> | Flag: {v.flag.split(' (')[0]}
                </div>
                <div style={{ marginTop: '6px', fontSize: '11px', fontFamily: 'var(--font-mono)', lineHeight: '1.5' }}>
                  <div>Speed: <strong>{v.current_position.speed_knots} kts</strong> ({v.current_position.heading_deg}°)</div>
                  <div>Fuel Flow: <strong>{v.current_position.fuel_flow_mth} MT/h</strong></div>
                  <div>POLARIS: <span style={{ color: '#059669', fontWeight: '700' }}>+{v.current_position.ice_class_rio} RIO (NORMAL)</span></div>
                </div>
              </div>
            </Popup>
          </Marker>
        ))}

        {/* Active Icebergs & 120h Drift Cones */}
        {layers.icebergs && icebergs && icebergs.map((berg) => {
          const trajData = icebergTrajectories?.find(t => t.iceberg_id === berg.id);
          const p50_points = trajData?.trajectory ? trajData.trajectory.map(p => [p.lat, p.lon]) : [];

          // Uncertainty cone polygon
          const upperCone = trajData?.trajectory ? trajData.trajectory.map(p => [p.p90_lat, p.p90_lon]) : [];
          const lowerCone = trajData?.trajectory ? trajData.trajectory.slice().reverse().map(p => [p.p10_lat, p.p10_lon]) : [];
          const conePolygon = [...upperCone, ...lowerCone];

          const color = berg.hazard_level === 'CRITICAL' ? '#dc2626' : '#d97706';

          return (
            <React.Fragment key={berg.id}>
              {/* Uncertainty Cone Polygon */}
              {conePolygon.length > 3 && (
                <Polygon
                  positions={conePolygon}
                  pathOptions={{
                    color: color,
                    weight: 1,
                    dashArray: '4, 4',
                    fillColor: color,
                    fillOpacity: 0.12
                  }}
                />
              )}

              {/* Central Trajectory Vector */}
              {p50_points.length > 1 && (
                <Polyline
                  positions={p50_points}
                  pathOptions={{
                    color: color,
                    weight: 2.5,
                    dashArray: '5, 5',
                    opacity: 0.85
                  }}
                />
              )}

              {/* Minimal Pin Marker */}
              <Marker
                position={[berg.lat, berg.lon]}
                icon={createIcebergIcon(berg.name, berg.hazard_level)}
                eventHandlers={{
                  click: () => onSelectIceberg && onSelectIceberg(berg)
                }}
              >
                <Tooltip direction="top" offset={[0, -10]} opacity={0.95}>
                  <div style={{ fontSize: '11px', fontWeight: '700', color: color }}>
                    ⚠️ {berg.name} ({berg.area_km2.toLocaleString()} km²)
                  </div>
                </Tooltip>

                <Popup>
                  <div style={{ padding: '4px', minWidth: '210px' }}>
                    <div style={{ fontWeight: '700', fontSize: '13px', color: color }}>
                      ⚠️ {berg.name}
                    </div>
                    <div style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
                      Calved: {berg.calving_source} ({berg.origin_year})
                    </div>
                    <div style={{ marginTop: '6px', fontSize: '11px', fontFamily: 'var(--font-mono)', lineHeight: '1.5' }}>
                      <div>Surface Area: <strong>{berg.area_km2.toLocaleString()} km²</strong></div>
                      <div>Thickness: <strong>{berg.thickness_m}m</strong> ({berg.mass_gt} Gt)</div>
                      <div>Drift: <strong>{berg.drift_speed_knots} kts</strong> @ {berg.drift_bearing_deg}°</div>
                      <div style={{ marginTop: '3px', color: '#0284c7' }}>Sensor: {berg.surveillance_source}</div>
                    </div>
                    <div className={`polar-badge ${berg.hazard_level === 'CRITICAL' ? 'polar-badge-danger' : 'polar-badge-gold'}`} style={{ marginTop: '8px' }}>
                      {berg.hazard_level} COLLISION HAZARD
                    </div>
                  </div>
                </Popup>
              </Marker>
            </React.Fragment>
          );
        })}

        {/* Optimized Navigation Routes */}
        {routes && Object.entries(routes).map(([modeKey, route]) => {
          const isSelected = selectedRouteMode === modeKey;
          const latlngs = route.waypoints.map(wp => [wp.lat, wp.lon]);

          const colorMap = {
            balanced: '#0284c7',
            safest: '#059669',
            fastest: '#d97706',
            eco_fuel: '#7c3aed'
          };

          const color = colorMap[modeKey] || '#0284c7';

          return (
            <React.Fragment key={modeKey}>
              <Polyline
                positions={latlngs}
                pathOptions={{
                  color: color,
                  weight: isSelected ? 4 : 2,
                  opacity: isSelected ? 1.0 : 0.45,
                  dashArray: isSelected ? null : '5, 5'
                }}
              />

              {isSelected && route.waypoints.map((wp, idx) => (
                <CircleMarker
                  key={idx}
                  center={[wp.lat, wp.lon]}
                  radius={idx === 0 || idx === route.waypoints.length - 1 ? 5 : 3.5}
                  pathOptions={{
                    color: '#ffffff',
                    weight: 1.5,
                    fillColor: color,
                    fillOpacity: 1
                  }}
                >
                  <Popup>
                    <div style={{ padding: '4px', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
                      <div style={{ fontWeight: '700', color: color }}>Leg #{wp.leg_index}: {wp.cumulative_dist_nm} NM</div>
                      <div>Speed: {wp.speed_knots} kts</div>
                      <div>Ice Conc: {wp.ice_concentration_pct}%</div>
                      <div>POLARIS: <strong style={{ color: wp.polaris_rio >= 0 ? '#059669' : '#dc2626' }}>RIO {wp.polaris_rio}</strong></div>
                      {wp.nearest_iceberg_dist_nm && (
                        <div>Nearest Berg ({wp.nearest_iceberg_id}): {wp.nearest_iceberg_dist_nm} NM</div>
                      )}
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
            </React.Fragment>
          );
        })}
      </MapContainer>
    </div>
  );
}
