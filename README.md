# POLARIS-AI: Antarctic Sea-Ice, Iceberg Trajectory and Navigation Decision Support System

**Organization:** Ministry of Earth Sciences (MoES), Government of India  
**Department:** National Centre for Polar and Ocean Research (NCPOR)  
**Classification:** Maritime Navigation Support Software  
**Operational Theater:** Southern Ocean and Antarctic Waters (50°S to 82°S, 180°W to 180°E)  

---

## Executive Summary

POLARIS-AI is an operational decision support platform engineered for polar expedition logistics supporting the Indian Antarctic Program at Maitri and Bharati research stations. The platform couples spatiotemporal deep learning, hydrodynamic ice-resistance physics, dynamic Lagrangian drift modeling, and constrained 4D graph search into an integrated bridge decision pipeline.

Key technical capabilities include:

1. **CF-1.8 NetCDF-4 Data Ingestion Layer**: Ingests satellite sea-ice concentration compiled from NOAA/NSIDC G02135 daily polar stereographic GeoTIFFs (reprojected from EPSG:3412 to WGS84), ECMWF ERA5 daily atmospheric 10-meter wind reanalysis ($u_{10}, v_{10}$), Copernicus Marine (CMEMS) surface current vectors ($u_{curr}, v_{curr}$), and Sea Surface Temperature (SST). Bounded by Antarctic Digital Database (ADD) continental coastline and permanent ice shelf exclusion masks (`shapely`).
2. **Spatiotemporal Sea-Ice Concentration Forecasting**: A PyTorch-based Convolutional Long Short-Term Memory (ConvLSTM) neural network with serialized trained weights (`convlstm_antarctic.pt`) generating multi-step autoregressive 1-to-7-day sea-ice concentration forecasts. Model training is executed on the historical period (Days 1–14), and quantitative benchmarking against the persistence baseline using Root Mean Square Error (RMSE) and Integrated Ice Edge Error (IIEE) is conducted strictly on a held-out evaluation window (Days 15–21).
3. **Hydrodynamic Iceberg Drift and Uncertainty Cones**: A two-dimensional Lagrangian momentum engine integrating quadratic atmospheric wind drag, hydrodynamic ocean skin and form drag, latitude-dependent Coriolis acceleration, and ensemble perturbations. Simulates 120-hour drift trajectories and probabilistic uncertainty envelopes ($p_{10}$, $p_{50}$, $p_{90}$) for major tracked tabular icebergs (A-23a, A-76a, D-28, B-15ab, C-39) with ingestion of real satellite scatterometer observations from the BYU / US National Ice Center (NIC) consolidated archive.
4. **Unified 4D Spatiotemporal Polar Route Optimization**: Multi-objective 4D A* graph search over the polar risk mesh that continuously samples the ConvLSTM forecasted ice field at vessel estimated time of arrival (ETA) and enforces time-dependent standoff buffers against dynamic iceberg drift envelopes. Incorporates:
   - IMO Circular MSC.1/Circ.1519 POLARIS Risk Index Outcome (RIO) evaluation across Polar Classes (PC1 through PC7, and Non-Ice Strengthened vessels) using multi-ice-type summation ($\sum C_i \times RIV_i + C_{ow} \times RIV_{ow}$) and official operational criteria.
   - The Lindqvist (1989) and Riska (1997) ice-resistance formulation to quantify crushing, bending, submersion, and velocity-dependent resistance components for Marine Gas Oil (MGO) fuel burn calculations.
   - Generates four distinct Pareto corridors: Balanced, Maximum Safety, Fastest Transit, and Eco-Polar.
5. **ECDIS Compliance and Data Interoperability**: Exports standard GeoJSON LineString geometry and waypoint attribute tables for onboard Electronic Chart Display and Information Systems (ECDIS) and OpenCPN marine chart plotters via `/api/navigation/export` and `/api/navigation/export-geojson`.
6. **Bridge Telemetry Simulation Interface**: WebSocket broadcast channel providing simulated Automatic Identification System (AIS) telemetry updates and instantaneous bridge risk advisories for dashboard integration.

---

## System Architecture

```
+------------------------------------------------------------------------+
|                        DATA INGESTION LAYER                            |
|  - CF-1.8 NetCDF-4 Metocean Store (antarctic_metocean_reference.nc)    |
|  - NOAA/NSIDC G02135 Daily Satellite SIC (EPSG:3412 -> WGS84)          |
|  - Daily Reanalysis Fields (ECMWF ERA5 Winds, CMEMS Currents, SST)     |
|  - High-Resolution Coastline and Ice-Shelf Mask (ADD, Shapely)         |
|  - BYU / US National Ice Center Consolidated Iceberg Database (CSVs)   |
+-----------------------------------┬------------------------------------+
                                    │ Gridded environmental slices
+-----------------------------------▼------------------------------------+
|                    PREDICTIVE ML AND DRIFT ENGINES                     |
|  1. PyTorch Spatiotemporal ConvLSTM -> 7-Day Ice Field S(x, y, t)       |
|  2. 2D Hydrodynamic Momentum Model -> 120-Hour Drift Envelopes B(x,y,t)|
+-----------------------------------┬------------------------------------+
                                    │ Dynamic spatiotemporal hazard states
+-----------------------------------▼------------------------------------+
|               UNIFIED 4D MULTI-OBJECTIVE ROUTE OPTIMIZER               |
|  - 4D A* graph search over spatiotemporal maritime grid                |
|  - Direct queries of ConvLSTM forecast S(t) at waypoint arrival ETA    |
|  - Time-resolved geometric clearance from iceberg uncertainty cones    |
|  - IMO MSC.1/Circ.1519 POLARIS Risk Index Outcome (RIO) evaluation     |
|  - Lindqvist (1989) ice resistance and fuel consumption modeling       |
|  - Computes four Pareto corridors: Balanced, Safe, Fast, Eco-Fuel      |
|  - Dynamic bridge decision brief generation                            |
+-----------------------------------┬------------------------------------+
                                    │ REST Endpoints + WebSocket Stream
+-----------------------------------▼------------------------------------+
|                     BACKEND SERVICES (FastAPI)                         |
|  /api/forecast/sea-ice     - 7-day spatiotemporal grid and IIEE metrics|
|  /api/forecast/metrics     - Lead-time validation statistics           |
|  /api/icebergs             - Active iceberg registry and 120h cones    |
|  /api/navigation/optimize  - 4 Pareto navigation corridors             |
|  /api/navigation/export    - GeoJSON route export for ECDIS / OpenCPN  |
|  /api/telemetry/ws         - Simulated AIS telemetry demonstration     |
+-----------------------------------┬------------------------------------+
                                    │
+-----------------------------------▼------------------------------------+
|                     OPERATOR INTERFACE (React + Vite)                  |
|  - High-Contrast Polar Cartographic Projection Dashboard               |
|  - Interactive Sea-Ice Concentration Heatmap with Lead-Time Slider     |
|  - Iceberg Radar Markers with Probabilistic Uncertainty Cones          |
|  - Indian Antarctic Expedition Fleet Tracking and Station Overlays     |
|  - NAVAREA VI, X, and XIV Maritime Safety Information Bulletins        |
+------------------------------------------------------------------------+
```

---

## Scientific and Mathematical Formulations

### 1. IMO Polar Code / POLARIS Risk Assessment (MSC.1/Circ.1519)

Vessel operability in ice regimes is evaluated following the Polar Operational Limit Assessment Risk Indexing System (POLARIS) set forth in IMO Circular MSC.1/Circ.1519 ("Guidance on Methodologies for Assessing Operational Capabilities and Limitations in Ice"). The Risk Index Outcome (RIO) is computed across each distinct ice type present plus open water:

$$RIO = \sum_{i} (C_i \times RIV_i) + C_{ow} \times RIV_{ow}$$

Where:
- $C_i$: Ice concentration in tenths ($0$ to $10$) for ice type $i$.
- $C_{ow}$: Open water concentration in tenths ($10 - \sum C_i$).
- $RIV_i$: Risk Index Value corresponding to the ship's Polar Class (PC1 through PC7, or Non-Ice-Strengthened / Open Water) obtained from Table 1.3 of MSC.1/Circ.1519 across WMO ice types:
  - Multi-Year Ice
  - Second-Year Ice
  - Thick First-Year Ice ($> 1.2\text{ m}$)
  - Medium First-Year Ice ($0.7 - 1.2\text{ m}$)
  - Thin First-Year Ice Stage 2 ($0.5 - 0.7\text{ m}$)
  - Thin First-Year Ice Stage 1 ($0.3 - 0.5\text{ m}$)
  - Grey-White Ice ($0.15 - 0.3\text{ m}$)
  - Grey Ice ($0.1 - 0.15\text{ m}$)
  - New Ice ($< 0.1\text{ m}$)
  - Open Water / Bergy Water

**Official MSC.1/Circ.1519 Section 2.2 Operational Criteria:**
- $RIO \ge 0$: **Operation Permitted**. Standard navigation permitted without icebreaker assistance.
- $-10 \le RIO < 0$: **Subject to Special Consideration (Escort Required)**. Operation requires icebreaker escort or specific operational risk mitigation.
- $RIO < -10$: **Subject to Special Consideration (High Risk)**. Ice regime exceeds standard design capability; operations require dedicated icebreaker support or rerouting.

The route optimizer evaluates all waypoints along candidate corridors and provides compliance notifications with exact RIO values and official MSC.1/Circ.1519 status descriptions.

### 2. Lindqvist (1989) Ice Resistance and Propulsion Fuel Physics

Propulsion resistance and Marine Gas Oil (MGO) consumption are evaluated using the semi-empirical continuous ice resistance formulation developed by Lindqvist (1989) and validated by Riska et al. (1997):

$$R_{total} = R_{open\_water} + C_{ice}^{1.6} \cdot (R_c + R_b + R_s)$$

#### Crushing Resistance ($R_c$)
Represents continuous crushing along the ship stem:

$$R_c = 0.5 \cdot \sigma_b \cdot h_{ice}^2 \cdot \frac{\tan\phi + \frac{\mu \cos\phi}{\cos\psi}}{1 - \frac{\mu \sin\phi}{\cos\psi}}$$

Where:
- $\sigma_b$: Ice flexural strength ($550\text{ kPa}$ nominal for summer/autumn Antarctic ice).
- $h_{ice}$: Effective level ice thickness ($m$).
- $\phi$: Stem angle relative to vertical.
- $\alpha$: Waterline entrance half-angle.
- $\psi = \arctan(\tan\phi / \sin\alpha)$: Flare angle at the waterline.
- $\mu$: Kinetic friction coefficient between hull steel and ice ($\mu = 0.10$).

#### Bending Resistance ($R_b$)
Accounts for bending failure and breaking of the ice sheet:

$$R_b = \frac{27}{64} \cdot \sigma_b \cdot B \cdot h_{ice}^{1.5} \cdot \sqrt{\frac{\rho_w g}{E}} \cdot \left(1 + \frac{9.22 l}{L}\right) \cdot \left(\frac{\tan\psi}{\cos\phi} + \mu \cos\psi \sin\phi\right)$$

Where $B$ is vessel beam, $L$ is length between perpendiculars, $E$ is Young's modulus of sea ice ($2.0\times 10^9\text{ Pa}$), and $\rho_w$ is seawater density ($1028\text{ kg/m}^3$).

#### Submersion Resistance ($R_s$)
Calculates the hydrostatic clearing and displacement of broken ice blocks along the bottom and hull sides:

$$R_s = (\rho_w - \rho_i) \cdot g \cdot h_{ice} \cdot B \cdot T \cdot \frac{B + T}{B + 2T} \cdot \left(1 + 2\mu \frac{T}{B}\right) + R_v$$

Where $T$ is ship draft, $\rho_i$ is ice density ($900\text{ kg/m}^3$), and $R_v$ is the velocity-dependent submersion component:

$$R_v = 0.063 \cdot (\rho_w - \rho_i) \cdot g \cdot h_{ice} \cdot B \cdot L \cdot \left(\frac{V}{\sqrt{g h_{ice}}}\right)$$

#### Propulsion Power and Fuel Rate
Total delivered power at the propeller shaft $P_D$ and fuel consumption:

$$P_D = \frac{R_{total} \cdot V}{\eta_D} + P_{aux}$$

$$\text{Fuel Consumption (MT)} = \frac{P_D \cdot \Delta t \cdot \text{SFOC}}{10^6}$$

Where $\eta_D \approx 0.65$ represents net propulsive efficiency, $P_{aux} = 800\text{ kW}$ is ship hotel and auxiliary load, and $\text{SFOC} = 185\text{ g/kWh}$ represents Specific Fuel Oil Consumption for four-stroke medium-speed marine diesel engines operating on low-sulfur MGO.

### 3. Spatiotemporal ConvLSTM Sea-Ice Architecture

The forecasting subsystem implements an autoregressive Convolutional LSTM recurrent neural network:

$$\mathcal{X} \in \mathbb{R}^{B \times T_{in} \times C \times H \times W}$$

Where:
- $T_{in} = 5$: Input historical lead sequence.
- $C = 5$: Physical channels comprising $[\text{SIC}, \text{SST}, U_{10}, V_{10}, \text{Current Speed}]$.
- Output: Autoregressively predicted sea-ice concentration grids for lead times $t \in [1, 7]$ days.

**Strictly Held-Out Validation Protocol:**
- Training dataset: Days 1–14 (January 1–14, 2026).
- Strictly held-out test dataset: Days 15–21 (January 15–21, 2026).
- The ConvLSTM model evaluates multi-day projections against true satellite observations on the held-out window, ensuring that benchmarking against the persistence baseline is rigorous and avoids data leakage.
- Serialized weights are loaded into memory from `backend/app/data/weights/convlstm_antarctic.pt`.

### 4. 2D Hydrodynamic Iceberg Momentum Equation

Drift trajectories for tabular icebergs are governed by the two-dimensional momentum equation per unit mass:

$$m \left(\frac{d\vec{v}}{dt} + 2\vec{\Omega}\sin\phi \times \vec{v}\right) = \vec{F}_{air} + \vec{F}_{water} + \vec{F}_{ice} + \vec{F}_{wave}$$

- **Atmospheric Wind Drag**: $\vec{F}_{air} = \frac{1}{2} \rho_a C_a A_a |\vec{v}_{wind} - \vec{v}| (\vec{v}_{wind} - \vec{v})$
- **Ocean Current Drag**: $\vec{F}_{water} = \frac{1}{2} \rho_w C_w A_w |\vec{v}_{curr} - \vec{v}| (\vec{v}_{curr} - \vec{v})$
- **Coriolis Parameter**: $f = 2\Omega\sin\phi$, negative in the Southern Hemisphere.
- **Sail and Keel Geometry**: Computed via hydrostatic Archimedes balance where keel draft accounts for $\sim 87.5\%$ of total iceberg thickness ($\rho_i / \rho_w$).

---

## Pareto Routing Corridors

The 4D A* route optimization engine produces four corridors with quantifiable trade-offs:

| Route Mode | Primary Objective | Ice Strategy | Standoff Margin | Operational Application |
|---|---|---|---|---|
| **Balanced** | Minimum expedition risk and fuel balance | Controlled transit through permissible pack ice ($SIC \le 0.45$) | $20\text{ NM}$ buffer | Standard NCPOR recommended corridor |
| **Maximum Safety** | Complete ice and iceberg collision risk aversion | Skirts the Marginal Ice Zone, stays north in open water until longitude alignment | $35\text{ NM}$ buffer | Unescorted or non-ice-strengthened vessels |
| **Fastest Transit** | Minimum total transit duration | Direct great-circle corridor utilizing icebreaker propulsion capacity | $10\text{ NM}$ buffer | Emergency medical evacuation or rapid crew relief |
| **Eco-Polar** | Minimal MGO fuel consumption and emissions | Throttles engine load, navigates lower ice resistance leads | $18\text{ NM}$ buffer | Scheduled seasonal resupply with fuel conservation priority |

---

## Installation and Execution Guide

### System Requirements
- Python 3.10 or higher
- Node.js 18.0 or higher with npm
- Modern Chromium or WebKit-based browser

### Backend Service Setup
```bash
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python run_backend.py
```
The FastAPI application initializes at `http://127.0.0.1:8000`. Swagger OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### Automated Verification Test Suite
Execute the pytest suite covering the data layer, Lindqvist resistance, POLARIS risk index outcome tables, ConvLSTM inference, and API endpoints:
```bash
pytest backend/tests/test_models.py -v
```

### Frontend User Interface Setup
```bash
cd frontend
npm install
npm run dev
```
Access the operator interface at `http://127.0.0.1:5173`.

---

## NCPOR Indian Antarctic Mission Profiles

### Research Stations and Operating Bases
- **Bharati Station**: 69°24′S, 76°11′E (Larsemann Hills, Prydz Bay sector)
- **Maitri Station**: 70°46′S, 11°44′E (Schirmacher Oasis, Queen Maud Land)
- **Dakshin Gangotri Site**: 70°05′S, 12°00′E (Historic first base, ice shelf depot)

### Primary Maritime Gateways
- **Port of Cape Town, South Africa**: Principal departure port for Atlantic and Dronning Maud Land missions (33°55′S, 18°25′E).
- **Mormugao Port (Goa HQ), India**: NCPOR operational headquarters and research vessel maintenance base (15°24′N, 73°48′E).
- **Port of Punta Arenas, Chile**: Gateway for Antarctic Peninsula and Bellingshausen Sea operations (53°10′S, 70°55′W).
- **Port of Hobart, Australia**: Logistics base for East Antarctic and Prydz Bay missions (42°53′S, 147°20′E).

### Expedition Fleet Particulars
- **MV Vasiliy Golovnin**: Polar Class 5 chartered expedition cargo/passenger vessel (163.0 m LOA, 22.4 m beam, 9.0 m draft, 12,800 kW engine power, helicopter deck).
- **ORV Sagar Nidhi**: Ice-strengthened oceanographic research vessel operated by MoES/NIOT (104.0 m LOA, 18.0 m beam, 6.8 m draft, 8,400 kW power).
- **SA Agulhas II**: Polar Class 5 deep-sea research and supply icebreaker operated in collaboration with South Africa (134.2 m LOA, 21.7 m beam, 7.7 m draft, 12,000 kW power).

---

## License and Academic Citation

Developed under the National Centre for Polar and Ocean Research (NCPOR), Ministry of Earth Sciences (MoES), Government of India.

```bibtex
@misc{polaris_ai_2026,
  title={POLARIS-AI: Antarctic Sea-Ice, Iceberg Trajectory and Navigation Decision Support System},
  author={{National Centre for Polar and Ocean Research (NCPOR)}},
  year={2026},
  publisher={Ministry of Earth Sciences, Government of India},
  howpublished={\url{https://github.com/takenrohit/polaris-ai-antarctic-dss}}
}
```
