/**
 * API client communicating with FastAPI Decision Support Backend.
 */
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';

export const api = {
  // Sea-ice forecasting
  async getSeaIceForecast(daysAhead = 7, dayOfYear = 45) {
    const res = await fetch(`${API_BASE_URL}/forecast/sea-ice?days_ahead=${daysAhead}&day_of_year=${dayOfYear}`);
    if (!res.ok) throw new Error('Failed to fetch sea-ice forecast');
    return res.json();
  },

  // Model benchmarks (ConvLSTM vs Persistence)
  async getModelBenchmarks() {
    const res = await fetch(`${API_BASE_URL}/forecast/metrics`);
    if (!res.ok) throw new Error('Failed to fetch model benchmarks');
    return res.json();
  },

  // Icebergs
  async getIcebergs() {
    const res = await fetch(`${API_BASE_URL}/icebergs`);
    if (!res.ok) throw new Error('Failed to fetch icebergs');
    return res.json();
  },

  async getIcebergTrajectories(hours = 120) {
    const res = await fetch(`${API_BASE_URL}/icebergs/trajectories?hours=${hours}`);
    if (!res.ok) throw new Error('Failed to fetch iceberg trajectories');
    return res.json();
  },

  // Waypoints & Stations
  async getStations() {
    const res = await fetch(`${API_BASE_URL}/navigation/stations`);
    if (!res.ok) throw new Error('Failed to fetch polar stations');
    return res.json();
  },

  async getPolarClasses() {
    const res = await fetch(`${API_BASE_URL}/navigation/polar-classes`);
    if (!res.ok) throw new Error('Failed to fetch polar classes');
    return res.json();
  },

  async getVessels() {
    const res = await fetch(`${API_BASE_URL}/navigation/vessels`);
    if (!res.ok) throw new Error('Failed to fetch NCPOR vessels');
    return res.json();
  },

  // Route Optimization (Pareto corridors)
  async optimizeRoute(payload) {
    const res = await fetch(`${API_BASE_URL}/navigation/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Failed to compute optimized polar routes');
    return res.json();
  },

  // Export GeoJSON
  async exportGeoJSON(payload) {
    const res = await fetch(`${API_BASE_URL}/navigation/export-geojson`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Failed to export GeoJSON');
    return res.json();
  },

  // Navigational Alerts
  async getAlerts() {
    const res = await fetch(`${API_BASE_URL}/telemetry/alerts`);
    if (!res.ok) throw new Error('Failed to fetch navigational alerts');
    return res.json();
  }
};
