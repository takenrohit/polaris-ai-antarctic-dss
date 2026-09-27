import React, { useState, useEffect, useRef } from 'react';
import Navbar from './components/Navbar';
import PolarMap from './components/PolarMap';
import ControlPanel from './components/ControlPanel';
import RoutePlannerModal from './components/RoutePlannerModal';
import ModelValidationModal from './components/ModelValidationModal';
import IcebergModal from './components/IcebergModal';
import AlertDrawer from './components/AlertDrawer';
import { api } from './services/api';

export default function App() {
  // Application Data States
  const [forecastData, setForecastData] = useState(null);
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [stations, setStations] = useState(null);
  const [polarClasses, setPolarClasses] = useState(null);
  const [vessels, setVessels] = useState(null);
  const [icebergs, setIcebergs] = useState([]);
  const [icebergTrajectories, setIcebergTrajectories] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [currentRouteData, setCurrentRouteData] = useState(null);

  // UI Interactive States
  const [forecastDay, setForecastDay] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedRouteMode, setSelectedRouteMode] = useState('balanced');
  const [cursorInfo, setCursorInfo] = useState(null);
  const [selectedIceberg, setSelectedIceberg] = useState(null);
  const [isLoadingRoute, setIsLoadingRoute] = useState(false);

  // Modal Visibility States
  const [isRouteModalOpen, setIsRouteModalOpen] = useState(false);
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [isIcebergModalOpen, setIsIcebergModalOpen] = useState(false);
  const [isAlertDrawerOpen, setIsAlertDrawerOpen] = useState(false);

  // Map Layer Toggles
  const [layers, setLayers] = useState({
    seaIce: true,
    icebergs: true,
    stations: true,
    fleet: true
  });

  // Initial Data Load
  useEffect(() => {
    async function loadInitialData() {
      try {
        const [
          forecastRes,
          benchmarksRes,
          stationsRes,
          classesRes,
          vesselsRes,
          icebergsRes,
          trajRes,
          alertsRes
        ] = await Promise.all([
          api.getSeaIceForecast(7, 45),
          api.getModelBenchmarks(),
          api.getStations(),
          api.getPolarClasses(),
          api.getVessels(),
          api.getIcebergs(),
          api.getIcebergTrajectories(120),
          api.getAlerts()
        ]);

        setForecastData(forecastRes);
        setBenchmarkData(benchmarksRes);
        setStations(stationsRes);
        setPolarClasses(classesRes);
        setVessels(vesselsRes);
        setIcebergs(icebergsRes);
        setIcebergTrajectories(trajRes);
        setAlerts(alertsRes);

        // Precompute default expedition route: Cape Town -> Bharati Research Station (India)
        const defaultRoute = await api.optimizeRoute({
          origin_key: 'PORT_CAPE_TOWN',
          dest_key: 'BHARATI_STATION',
          vessel_ice_class: 'PC5',
          cruising_speed_knots: 13.5
        });
        setCurrentRouteData(defaultRoute);

      } catch (err) {
        console.error('Failed to load initial polar data:', err);
      }
    }
    loadInitialData();
  }, []);

  // Forecast Auto-Playback Timer
  useEffect(() => {
    let timer = null;
    if (isPlaying) {
      timer = setInterval(() => {
        setForecastDay((prev) => (prev >= 7 ? 0 : prev + 1));
      }, 1800);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isPlaying]);

  // Route Optimization Handler
  const handleOptimizeRoute = async (payload) => {
    setIsLoadingRoute(true);
    try {
      const data = await api.optimizeRoute(payload);
      setCurrentRouteData(data);
    } catch (err) {
      console.error('Route optimization failed:', err);
      alert('Failed to optimize route. Please check backend connection.');
    } finally {
      setIsLoadingRoute(false);
    }
  };

  // Export GeoJSON Handler
  const handleExportGeoJSON = async () => {
    if (!currentRouteData) return;
    try {
      const geojson = await api.exportGeoJSON({
        origin_key: 'PORT_CAPE_TOWN',
        dest_key: 'BHARATI_STATION',
        vessel_ice_class: currentRouteData.vessel_ice_class || 'PC5',
        cruising_speed_knots: currentRouteData.cruising_speed_knots || 13.5
      });
      const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `NCPOR_Polar_Route_${currentRouteData.recommended_mode.toUpperCase()}.geojson`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('GeoJSON export error:', err);
    }
  };

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden', backgroundColor: 'var(--bg-abyss)' }}>
      {/* Top Navigation Bar */}
      <Navbar
        onOpenRoutePlanner={() => setIsRouteModalOpen(true)}
        onOpenModelValidation={() => setIsModelModalOpen(true)}
        onOpenIcebergs={() => setIsIcebergModalOpen(true)}
        onOpenAlerts={() => setIsAlertDrawerOpen(true)}
        activeAlertsCount={alerts.length}
        currentRoute={currentRouteData}
        onExportGeoJSON={handleExportGeoJSON}
      />

      {/* Interactive Polar Map */}
      <PolarMap
        forecastData={forecastData}
        forecastDay={forecastDay}
        layers={layers}
        stations={stations}
        icebergs={icebergs}
        icebergTrajectories={icebergTrajectories}
        vessels={vessels}
        routes={currentRouteData?.routes}
        selectedRouteMode={selectedRouteMode}
        onSelectIceberg={(berg) => {
          setSelectedIceberg(berg);
          setIsIcebergModalOpen(true);
        }}
        onMouseMoveCoord={(coord) => setCursorInfo(coord)}
      />

      {/* Floating Control Panel */}
      <ControlPanel
        forecastDay={forecastDay}
        setForecastDay={setForecastDay}
        maxForecastDays={7}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
        layers={layers}
        setLayers={setLayers}
        cursorInfo={cursorInfo}
        selectedRouteMode={selectedRouteMode}
        setSelectedRouteMode={setSelectedRouteMode}
        availableRoutes={currentRouteData?.routes}
      />

      {/* Modals & Drawers */}
      <RoutePlannerModal
        isOpen={isRouteModalOpen}
        onClose={() => setIsRouteModalOpen(false)}
        stations={stations}
        polarClasses={polarClasses}
        onOptimize={handleOptimizeRoute}
        isLoading={isLoadingRoute}
        currentRouteData={currentRouteData}
        selectedMode={selectedRouteMode}
        setSelectedMode={setSelectedRouteMode}
        onExportGeoJSON={handleExportGeoJSON}
      />

      <ModelValidationModal
        isOpen={isModelModalOpen}
        onClose={() => setIsModelModalOpen(false)}
        benchmarkData={benchmarkData}
      />

      <IcebergModal
        isOpen={isIcebergModalOpen}
        onClose={() => setIsIcebergModalOpen(false)}
        icebergs={icebergs}
        icebergTrajectories={icebergTrajectories}
        selectedBergId={selectedIceberg?.id}
        onSelectBerg={(b) => setSelectedIceberg(b)}
      />

      <AlertDrawer
        isOpen={isAlertDrawerOpen}
        onClose={() => setIsAlertDrawerOpen(false)}
        alerts={alerts}
      />
    </div>
  );
}
