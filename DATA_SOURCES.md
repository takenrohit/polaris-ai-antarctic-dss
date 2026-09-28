# POLARIS-AI Data Sources, Provenance, and Ingestion Specifications

**Organization:** Ministry of Earth Sciences (MoES), Government of India  
**Department:** National Centre for Polar and Ocean Research (NCPOR)  
**System:** POLARIS-AI Antarctic Decision Support System  

---

## 1. Overview of Ingested Datasets

POLARIS-AI integrates multi-source satellite Earth observation, atmospheric reanalysis, and oceanographic data products into a unified, CF-1.8 compliant NetCDF-4 observational datastore (`antarctic_metocean_reference.nc`). The spatial domain spans 50°S to 82°S latitude and 180°W to 180°E longitude across the Southern Ocean and Antarctic marginal seas (Weddell, Bellingshausen, Amundsen, Ross, and Prydz Bay sectors).

| Dataset Identifier | Variable Description | Physical Unit | Source Agency | Sensor / Platform | Spatial Resolution | Temporal Frequency |
|---|---|---|---|---|---|---|
| **NOAA/NSIDC G02135 v4.0** | Sea-Ice Concentration (SIC) | Fraction [0.0, 1.0] | NOAA / NSIDC | DMSP SSMIS, AMSR2 | 25 km polar stereographic (EPSG:3412) | Daily |
| **ECMWF ERA5 Reanalysis** | 10-meter U-Wind Component ($u_{10}$) | $\text{m/s}$ | ECMWF / Copernicus | Atmospheric model + satellite assimilation | 0.25° grid (~25 km) | Daily mean |
| **ECMWF ERA5 Reanalysis** | 10-meter V-Wind Component ($v_{10}$) | $\text{m/s}$ | ECMWF / Copernicus | Atmospheric model + satellite assimilation | 0.25° grid (~25 km) | Daily mean |
| **Copernicus Marine (CMEMS)** | Surface U-Current ($u_{curr}$) | $\text{m/s}$ | Mercator Ocean / CMEMS | GLOBAL_REANALYSIS_PHY_001_030 | 0.083° (~8 km) | Daily mean |
| **Copernicus Marine (CMEMS)** | Surface V-Current ($v_{curr}$) | $\text{m/s}$ | Mercator Ocean / CMEMS | GLOBAL_REANALYSIS_PHY_001_030 | 0.083° (~8 km) | Daily mean |
| **NOAA OISST v2.1** | Sea Surface Temperature (SST) | °C | NOAA NCEI | Advanced Very High Resolution Radiometer (AVHRR) | 0.25° grid | Daily |
| **BYU / USNIC Database** | Tabular Iceberg Positions & Geometry | Lat/Lon, km, GT | BYU Scatterometer / US National Ice Center | MetOp ASCAT, Sentinel-1 SAR, Envisat | Individual iceberg tracks | Multi-day satellite passes |
| **SCAR ADD / IBCSO** | Antarctic Coastline & Ice Shelf Mask | Polygon Geometry | Scientific Committee on Antarctic Research | Antarctic Digital Database (ADD v7.4) | High-fidelity vectors | Static reference |

---

## 2. Satellite Sea-Ice Concentration (NOAA/NSIDC G02135)

### Ingestion Pipeline
- **Raw Format:** Daily GeoTIFF rasters in Polar Stereographic South projection (EPSG:3412, standard parallel 71°S, WGS84 datum).
- **Pixel Values:** Scaled integers where $0$ to $1000$ represents sea-ice concentration multiplied by $10$ ($0.0\%$ to $100.0\%$). Values $> 2000$ represent non-ocean flags:
  - $2510$: Missing / unobserved.
  - $2530$: Coastline boundary.
  - $2540$: Continental landmass / permanent ice sheet.
  - $2550$: Polar hole missing region.
- **Preprocessing:** Script `backend/scripts/build_real_antarctic_dataset.py` reprojects daily GeoTIFFs using `rasterio.warp.transform` and bilinear interpolation onto a regular geographic WGS84 coordinate grid (49 latitude points from -78°S to -54°S $\times$ 73 longitude points from -180° to +180°).

---

## 3. Atmospheric and Oceanographic Forcings (ERA5 & CMEMS)

### 10-Meter Wind Vectors ($u_{10}, v_{10}$)
- Extracted from ECMWF ERA5 reanalysis via the Open-Meteo climate archive across the Antarctic theater.
- Daily means provide the driving force for aerodynamic drag ($\vec{F}_{air}$) in the 2D Lagrangian iceberg drift engine and wave-induced resistance additions for vessel routing.

### Surface Ocean Currents ($u_{curr}, v_{curr}$)
- Extracted from CMEMS GLORYS12 reanalysis / geostrophic surface current estimates.
- Incorporates the Antarctic Circumpolar Current (ACC) eastward transport and coastal Antarctic Counter-Current (East Wind Drift) westward flow.

---

## 4. Iceberg Surveillance Archive (BYU / USNIC)

### Database Structure
- Consolidated CSV archive located at `backend/app/data/byu_icebergs/updated7_consol/` containing 649 tracked tabular iceberg time-series records.
- Each observation record documents:
  - `date`: Year and Day of Year (YYYYDOY).
  - `ascat_1, ascat_2`: Latitude and longitude observed via MetOp ASCAT scatterometer.
  - `nic_1, nic_2`: Latitude and longitude reported by the US National Ice Center.
  - `size_1, size_2`: Major and minor axis dimensions in kilometers.
- Actively tracked megabergs incorporated into real-time decision scenarios:
  - **A-23a**: Calved from Filchner Ice Shelf in 1986; largest active tabular megaberg (~3,800 $km^2$, ~950 Gigatons).
  - **A-76a**: Calved from Ronne Ice Shelf; monitored through Drake Passage / Scotia Sea.
  - **D-28**: Calved from Amery Ice Shelf in 2019; actively drifting in the Indian Ocean sector near Prydz Bay.
  - **B-15ab**: Fragment of historic mega-iceberg B-15.
  - **C-39**: Ross Sea sector iceberg.

---

## 5. Coastline and Continental Shelf Exclusion Mask

- Evaluated via high-precision Shapely polygonal geometry representing the Antarctic coastline and major ice shelves:
  - Amery Ice Shelf (68°E to 75°E, south of 68.5°S).
  - Ross Ice Shelf (160°E to 150°W, south of 78°S).
  - Ronne-Filchner Ice Shelf (30°W to 85°W, south of 75°S).
  - Riiser-Larsen and Fimbul Ice Shelves (Queen Maud Land).
- Implements spatial indexing (`shapely.ops.prep`) for sub-millisecond point-in-polygon queries during A* graph search expansions.
