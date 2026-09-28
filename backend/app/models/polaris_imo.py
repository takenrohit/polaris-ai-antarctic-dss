"""
IMO Polar Operational Limit Assessment Risk Indexing System (POLARIS).
Implements the methodology set out in IMO Circular MSC.1/Circ.1519:
"Guidance on Methodologies for Assessing Operational Capabilities and Limitations in Ice".

Computes Risk Index Outcome (RIO):
RIO = Sum(C_i * RIV_i) + C_ow * RIV_ow
Where:
- C_i: Ice concentration in tenths (0 to 10) of each ice type i
- RIV_i: Risk Index Value from MSC.1/Circ.1519 Table 1.3
Operational criteria (MSC.1/Circ.1519 Section 2.2):
- RIO >= 0: Operation is permitted.
- -10 <= RIO < 0: Operation is subject to special consideration (e.g. icebreaker escort / speed reduction).
- RIO < -10: Operation is subject to special consideration / elevated risk regime exceeding standard design parameters.
"""
from typing import Dict, Any, Optional

# IMO MSC.1/Circ.1519 Table 1.3: Risk Index Values (RIV) for Polar Classes and Non-Ice Vessels
# Validated against IMO Polar Code Part I-A and Circular 1519
POLARIS_RIV_TABLE = {
    "MULTI_YEAR_ICE": {
        "PC1": 3, "PC2": 2, "PC3": 1, "PC4": 0, "PC5": -1, "PC6": -2, "PC7": -3, "OPEN_WATER": -4
    },
    "SECOND_YEAR_ICE": {
        "PC1": 3, "PC2": 3, "PC3": 2, "PC4": 1, "PC5": 0, "PC6": -1, "PC7": -2, "OPEN_WATER": -4
    },
    "THICK_FIRST_YEAR": { # > 1.2 m
        "PC1": 3, "PC2": 3, "PC3": 3, "PC4": 2, "PC5": 1, "PC6": 0, "PC7": -1, "OPEN_WATER": -3
    },
    "MEDIUM_FIRST_YEAR": { # 0.7 - 1.2 m
        "PC1": 3, "PC2": 3, "PC3": 3, "PC4": 3, "PC5": 2, "PC6": 1, "PC7": 0, "OPEN_WATER": -2
    },
    "THIN_FIRST_YEAR_STAGE_2": { # 0.5 - 0.7 m
        "PC1": 3, "PC2": 3, "PC3": 3, "PC4": 3, "PC5": 3, "PC6": 2, "PC7": 1, "OPEN_WATER": -1
    },
    "THIN_FIRST_YEAR_STAGE_1": { # 0.3 - 0.5 m
        "PC1": 3, "PC2": 3, "PC3": 3, "PC4": 3, "PC5": 3, "PC6": 3, "PC7": 2, "OPEN_WATER": 0
    },
    "GREY_WHITE_ICE": { # 0.15 - 0.3 m
        "PC1": 3, "PC2": 3, "PC3": 3, "PC4": 3, "PC5": 3, "PC6": 3, "PC7": 3, "OPEN_WATER": 1
    },
    "GREY_ICE": { # 0.1 - 0.15 m
        "PC1": 3, "PC2": 3, "PC3": 3, "PC4": 3, "PC5": 3, "PC6": 3, "PC7": 3, "OPEN_WATER": 2
    },
    "NEW_ICE": { # < 0.1 m
        "PC1": 3, "PC2": 3, "PC3": 3, "PC4": 3, "PC5": 3, "PC6": 3, "PC7": 3, "OPEN_WATER": 2
    },
    "OPEN_WATER": { # No ice or bergy water
        "PC1": 3, "PC2": 3, "PC3": 3, "PC4": 3, "PC5": 3, "PC6": 3, "PC7": 3, "OPEN_WATER": 3
    }
}


def evaluate_imo_polaris_rio(
    ice_class: str,
    ice_concentration: float,
    ice_regimes: Optional[Dict[str, float]] = None,
    ice_regime: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes IMO MSC.1/Circ.1519 Risk Index Outcome (RIO).
    Sums across each distinct ice type present:
    RIO = Sum(C_i * RIV_i) + C_ow * RIV_ow

    - ice_concentration: Total ice concentration [0.0, 1.0]
    - ice_regimes: Optional dict of {ice_type: fraction_of_total_ice}.
      If not provided, partitions total concentration into typical Antarctic summer pack ice mix:
      60% Medium First-Year, 25% Thin First-Year, 15% New Ice.
    - ice_regime: Optional single ice type string (e.g. 'MULTI_YEAR_ICE') for 100% single-type evaluation.
    """
    vessel_class = ice_class.upper()
    if vessel_class not in ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "OPEN_WATER"]:
        vessel_class = "PC5"

    c_total = min(1.0, max(0.0, ice_concentration))
    total_ice_tenths = int(round(c_total * 10.0))
    c_ow_tenths = 10 - total_ice_tenths

    # Multi-ice-type partition
    if ice_regime:
        partition = {ice_regime: 1.0}
    elif ice_regimes is not None:
        partition = ice_regimes
    else:
        # Standard Antarctic summer/autumn sea ice regime distribution
        partition = {
            "MEDIUM_FIRST_YEAR": 0.60,
            "THIN_FIRST_YEAR_STAGE_2": 0.25,
            "NEW_ICE": 0.15
        }

    # Sum RIO across each ice type
    rio = 0
    breakdown = []
    allocated_tenths = 0

    items = list(partition.items())
    for idx, (itype, frac) in enumerate(items):
        if idx == len(items) - 1:
            t_i = total_ice_tenths - allocated_tenths
        else:
            t_i = int(round(total_ice_tenths * frac))
            allocated_tenths += t_i

        riv = POLARIS_RIV_TABLE.get(itype, POLARIS_RIV_TABLE["MEDIUM_FIRST_YEAR"]).get(vessel_class, -2)
        rio += (t_i * riv)
        breakdown.append({"ice_type": itype, "tenths": t_i, "riv": riv})

    # Add open water contribution
    riv_ow = POLARIS_RIV_TABLE["OPEN_WATER"].get(vessel_class, 3)
    rio += (c_ow_tenths * riv_ow)
    breakdown.append({"ice_type": "OPEN_WATER", "tenths": c_ow_tenths, "riv": riv_ow})

    # IMO MSC.1/Circ.1519 Section 2.2 Operational Criteria
    if rio >= 0:
        status = "NORMAL_OPERATION" # Backward-compatible key
        official_status = "OPERATION_PERMITTED"
        desc = "Operation permitted under IMO MSC.1/Circ.1519 guidelines"
        escort_required = False
        elevated_risk = False
    elif rio >= -10:
        status = "ESCORT_REQUIRED"
        official_status = "SUBJECT_TO_SPECIAL_CONSIDERATION_ESCORT"
        desc = "Operation subject to special consideration (icebreaker escort or risk mitigation required)"
        escort_required = True
        elevated_risk = False
    else:
        status = "PROHIBITED" # Backward-compatible key
        official_status = "SUBJECT_TO_SPECIAL_CONSIDERATION_HIGH_RISK"
        desc = "Operation subject to special consideration (high-risk ice regime exceeding vessel design capability)"
        escort_required = True
        elevated_risk = True

    return {
        "rio": int(rio),
        "status": status,
        "official_status": official_status,
        "description": desc,
        "escort_required": escort_required,
        "operation_prohibited": elevated_risk,
        "ice_class": vessel_class,
        "ice_concentration_pct": round(c_total * 100.0, 1),
        "total_ice_tenths": total_ice_tenths,
        "open_water_tenths": c_ow_tenths,
        "ice_type_breakdown": breakdown
    }
