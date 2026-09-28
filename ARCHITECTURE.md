# POLARIS-AI System Architecture and Technical Specifications

**Organization:** Ministry of Earth Sciences (MoES), Government of India  
**Department:** National Centre for Polar and Ocean Research (NCPOR)  
**System:** POLARIS-AI Antarctic Decision Support System  

---

## 1. High-Level Modular Decomposition

```
                           +------------------------------------------------+
                           |          SATELLITE & METOCEAN FEEDS            |
                           |  - NOAA/NSIDC G02135 Daily GeoTIFFs (EPSG:3412)|
                           |  - ECMWF ERA5 Daily Winds (u10, v10) & SST     |
                           |  - CMEMS Surface Ocean Current Vectors         |
                           |  - BYU / USNIC Consolidated Iceberg CSVs       |
                           +-----------------------┬------------------------+
                                                   │
                                                   ▼
                           +------------------------------------------------+
                           |           DATA INGESTION & QUALITY GATE        |
                           |  - NetCDF-4 Observational Store (xarray)       |
                           |  - Antarctic Coastline Mask (Shapely Poly)     |
                           |  - Data Quality & Freshness Gate (validators)  |
                           +-----------------------┬------------------------+
                                                   │
                        ┌──────────────────────────┴──────────────────────────┐
                        ▼                                                     ▼
+-----------------------------------------------+   +-----------------------------------------------+
|       SPATIOTEMPORAL CONVLSTM ENGINE          |   |        2D HYDRODYNAMIC DRIFT ENGINE           |
|  - PyTorch 2-layer Recurrent ConvLSTM         |   |  - Lagrangian Momentum Model (Sail + Keel)    |
|  - 5 Physical Channels [SIC, SST, U, V, Curr] |   |  - Coriolis, Wind & Water Form Drag           |
|  - Multi-step 1-to-7 day gridded SIC S(x,y,t) |   |  - 120-hour Uncertainty Cones (p10, p50, p90) |
+-----------------------┬-----------------------+   +-----------------------┬-----------------------+
                        │                                                   │
                        └──────────────────────────┬────────────────────────┘
                                                   │ Dynamic 4D spatiotemporal hazards
                                                   ▼
                           +------------------------------------------------+
                           |         UNIFIED 4D A* ROUTE OPTIMIZER          |
                           |  - Priority Queue (heapq) across (lat, lon, t) |
                           |  - Samples ConvLSTM forecast at ETA t          |
                           |  - Evaluates dynamic 120h iceberg hazard cones |
                           |  - Lindqvist (1989) continuous ice resistance  |
                           |  - IMO MSC.1/Circ.1519 multi-ice-type RIO      |
                           |  - Produces 4 Pareto corridors                 |
                           |  - Generates unconstrained rejection analysis  |
                           +-----------------------┬------------------------+
                                                   │
                        ┌──────────────────────────┴──────────────────────────┐
                        ▼                                                     ▼
+-----------------------------------------------+   +-----------------------------------------------+
|             FASTAPI BACKEND SERVICES          |   |         OPERATOR DASHBOARD (React + Vite)     |
|  - REST endpoints: /navigation, /icebergs     |   |  - Leaflet polar cartographic projection      |
|  - Standard GeoJSON ECDIS export (/export)    |   |  - 7-day sea ice heatmap with lead-time slider|
|  - AIS telemetry simulation WebSocket channel |   |  - Real-time iceberg radar markers & vectors  |
+-----------------------------------------------+   +-----------------------------------------------+
```

---

## 2. Component Directory Layout

```
sih59/
├── backend/
│   ├── app/
│   │   ├── api/                     # FastAPI route controllers
│   │   │   ├── routes_forecast.py   # ConvLSTM inference & benchmark endpoints
│   │   │   ├── routes_icebergs.py   # Iceberg trajectories & BYU surveillance
│   │   │   ├── routes_navigation.py # 4D A* optimization & GeoJSON export
│   │   │   └── routes_telemetry.py  # WebSocket AIS telemetry
│   │   ├── core/                    # Quality assurance & polar geodesics
│   │   │   ├── geodesics.py         # Haversine, bearing, antimeridian wrapping
│   │   │   └── validators.py        # Vessel engineering validation & data quality
│   │   ├── data/                    # Reference NetCDF datastore & weights
│   │   │   ├── antarctic_metocean_reference.nc # CF-1.8 NetCDF4 store
│   │   │   ├── byu_icebergs/        # 649 BYU/USNIC historical iceberg records
│   │   │   ├── raw_nsidc/           # 21 daily NOAA/NSIDC GeoTIFFs
│   │   │   └── weights/             # Trained ConvLSTM PyTorch weights
│   │   ├── models/                  # Core scientific engines
│   │   │   ├── iceberg_drift.py     # 2D Lagrangian momentum model
│   │   │   ├── ice_resistance.py    # Lindqvist & Riska fuel physics
│   │   │   ├── polaris_imo.py       # IMO MSC.1/Circ.1519 Table 1.3 RIO engine
│   │   │   ├── route_optimizer.py   # 4D Spatiotemporal A* graph search
│   │   │   └── sea_ice_convlstm.py  # PyTorch Spatiotemporal ConvLSTM
│   │   ├── services/                # Business logic & vessel registries
│   │   └── config.py                # Waypoints, IMO Polar Classes, initial assets
│   ├── scripts/                     # Dataset generators and model training
│   │   ├── build_real_antarctic_dataset.py
│   │   └── train_convlstm.py
│   └── tests/                       # Automated pytest verification suites
│       ├── test_models.py           # Core scientific & API tests (17 tests)
│       ├── test_polar_geodesics.py  # Geodesic & coordinate tests (6 tests)
│       └── test_robustness_and_safety.py # Quality gate & safety tests (8 tests)
├── evaluation/                      # Model benchmarking & ablation suite
│   ├── run_evaluation.py            # Standalone scientific evaluation script
│   └── results/                     # Evaluation reports and JSON metrics
├── frontend/                        # React + Vite operator dashboard
│   ├── src/                         # Components, map layers, telemetry
│   └── package.json
├── docker-compose.yml               # Multi-container operational deployment
├── DATA_SOURCES.md
├── MODEL_CARD.md
├── EVALUATION.md
├── LIMITATIONS.md
├── SAFETY_CASE.md
└── ARCHITECTURE.md
```

---

## 3. API Contracts and Data Flow

### `/api/navigation/optimize` (POST)
- **Request:**
  ```json
  {
    "origin_key": "PORT_CAPE_TOWN",
    "dest_key": "BHARATI_STATION",
    "vessel_ice_class": "PC5",
    "cruising_speed_knots": 13.5
  }
  ```
- **Response:**
  - `origin`, `destination`, `direct_distance_nm`
  - `vessel_particulars`: LOA, beam, draft, displacement, engine power
  - `data_quality`: quality status, freshness hours, timestamp
  - `rejection_analysis`: unconstrained great-circle failure points (land, RIO, icebergs)
  - `routes`: 4 distinct Pareto corridors (`balanced`, `safest`, `fastest`, `eco_fuel`)
  - Each waypoint: lat, lon, arrival hours, SIC %, wind knots, confidence %, risk band, POLARIS RIO, iceberg distance, fuel burn MT.
