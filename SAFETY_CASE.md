# POLARIS-AI Maritime Safety Case and Operational Assurance

**Classification:** Safety-Critical Decision Support System (SCDSS)  
**Standard Compliance:** IMO Polar Code (MEPC.264(68) / MSC.385(94)) & IMO MSC.1/Circ.1519  

---

## 1. Safety Objectives and Hazard Mitigation

The primary purpose of POLARIS-AI is to minimize expedition hazards during Antarctic transit, specifically:
1. **Hull Structural Overload:** Preventing non-ice-strengthened or low polar-class vessels from entering heavy ice regimes exceeding their design capabilities.
2. **Iceberg Collision Risk:** Enforcing dynamic spatiotemporal standoff margins from moving tabular megabergs and their surrounding bergy bit dispersal fields.
3. **Bespoke Fuel Depletion:** Preventing excessive fuel burn from severe ice resistance, which could imperil Antarctic return passages.
4. **Vessel Besetment:** Alerting bridge personnel to converging sea-ice packs and compressive ice conditions driven by strong onshore winds.

---

## 2. IMO Polar Code / POLARIS Compliance Framework

POLARIS-AI strictly follows the operational criteria set forth in **IMO Circular MSC.1/Circ.1519**:

$$RIO = \sum_{i} (C_i \times RIV_i) + C_{ow} \times RIV_{ow}$$

### Operational Risk Classification
- **RIO $\ge 0$ (Normal Operation / Operation Permitted):** Vessel design capacity is fully compatible with local ice regime. Transit authorized without icebreaker assistance.
- **$-10 \le RIO < 0$ (Subject to Special Consideration - Escort Required):** Vessel structural limits approached. Safe transit requires dedicated icebreaker escort, reduced operating speeds, or daytime-only navigation.
- **$RIO < -10$ (Subject to Special Consideration - High Risk):** Severe risk regime exceeding vessel design capability. Routing through these waters is flagged with critical bridge warnings and alternative avoidance corridors are enforced by the optimizer.

---

## 3. Fail-Safe and Degraded Operational States

To maintain maritime safety integrity, POLARIS-AI incorporates an automated Quality Assurance Gate (`backend/app/core/validators.py`):

| System State | Trigger Condition | Operational Action |
|---|---|---|
| **OPERATIONAL** | Fresh satellite SIC (< 48h latency), valid ERA5 winds, valid ocean currents | Full 4D A* optimization and multi-corridor Pareto outputs active |
| **DEGRADED** | Observation latency 48h to 72h, or missing wind reanalysis | System continues routing with increased safety buffers (+50% standoff) and alerts the bridge of data aging |
| **DO NOT USE FOR NAVIGATION** | Satellite SIC feed absent, corrupted NetCDF datastore, or invalid vessel inputs | Route generation is locked; prominent bridge warning displayed: *"CRITICAL DATA FAULT: DO NOT USE FOR POLAR NAVIGATION"* |

---

## 4. Human-in-the-Loop Operational Doctrine

1. **Master's Authority:** Under SOLAS Chapter V Regulation 34 and the IMO Polar Code, the Master of the vessel retains ultimate responsibility for navigational safety. POLARIS-AI acts exclusively in an advisory capacity.
2. **Visual Lookout:** Automated iceberg drift predictions do not replace continuous radar watches and 24-hour visual lookouts required by COLREGS Rule 5.
3. **ECDIS Cross-Verification:** Exported GeoJSON route plans must be loaded into type-approved bridge ECDIS and verified against official nautical charts prior to voyage execution.
