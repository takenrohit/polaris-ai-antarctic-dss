"""
IMO Polar Operational Limit Assessment Risk Indexing System (POLARIS).
According to IMO MSC.1/Circ.1519 Guidelines for Ships Operating in Polar Waters.

Computes Risk Index Outcome (RIO):
RIO = Sum(C_i * RIV_i)
Where:
- C_i: Ice concentration in tenths (0 to 10) of ice type i
- RIV_i: Risk Index Value from MSC.1/Circ.1519 Table 1.3
Operational criteria:
- RIO >= 0: Normal Operation
- -10 <= RIO < 0: Operation Subject to Special Conditions (Icebreaker Escort Required)
- RIO < -10: Operation Prohibited
"""
from typing import Dict, Any

# IMO MSC.1/Circ.1519 Table 1.3: Risk Index Values (RIV) for Polar Classes and Non-Ice Vessels
# Columns correspond to vessel ice classes: PC1, PC2, PC3, PC4, PC5, PC6, PC7, Category B (Non-Ice)
POLARIS_RIV_TABLE = {
    # Ice Type: (PC1, PC2, PC3, PC4, PC5, PC6, PC7, OPEN_WATER)
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
    "OPEN_WATER": { # No ice or < 1/10
        "PC1": 3, "PC2": 3, "PC3": 3, "PC4": 3, "PC5": 3, "PC6": 3, "PC7": 3, "OPEN_WATER": 3
    }
}


def evaluate_imo_polaris_rio(
    ice_class: str,
    ice_concentration: float,
    ice_regime: str = "MEDIUM_FIRST_YEAR"
) -> Dict[str, Any]:
    """
    Computes exact IMO MSC.1/Circ.1519 Risk Index Outcome (RIO).
    - ice_concentration: float [0.0, 1.0] (converted to tenths C_i in [0, 10])
    - ice_regime: primary ice type (defaults to Medium First-Year Ice typical of Antarctic summer/autumn)
    """
    vessel_class = ice_class.upper()
    if vessel_class not in ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "OPEN_WATER"]:
        vessel_class = "PC5"

    c_total = min(1.0, max(0.0, ice_concentration))
    c_ice_tenths = round(c_total * 10.0)
    c_ow_tenths = 10 - c_ice_tenths

    # Look up Risk Index Values
    riv_ice = POLARIS_RIV_TABLE.get(ice_regime, POLARIS_RIV_TABLE["MEDIUM_FIRST_YEAR"]).get(vessel_class, -2)
    riv_ow = POLARIS_RIV_TABLE["OPEN_WATER"].get(vessel_class, 3)

    # RIO = (C_ice * RIV_ice) + (C_ow * RIV_ow)
    rio = (c_ice_tenths * riv_ice) + (c_ow_tenths * riv_ow)

    if rio >= 0:
        status = "NORMAL_OPERATION"
        desc = "Normal navigation permitted under IMO Polar Code Part I-A"
        escort_required = False
        prohibited = False
    elif rio >= -10:
        status = "ESCORT_REQUIRED"
        desc = "Operation Subject to Special Conditions (Icebreaker Escort Recommended / Required)"
        escort_required = True
        prohibited = False
    else:
        status = "PROHIBITED"
        desc = "Operation Prohibited (Ice conditions exceed vessel structural design capability)"
        escort_required = True
        prohibited = True

    return {
        "rio": int(rio),
        "status": status,
        "description": desc,
        "escort_required": escort_required,
        "operation_prohibited": prohibited,
        "ice_class": vessel_class,
        "ice_regime": ice_regime,
        "ice_concentration_pct": round(c_total * 100.0, 1),
        "ice_tenths": c_ice_tenths,
        "open_water_tenths": c_ow_tenths,
        "riv_ice": riv_ice,
        "riv_open_water": riv_ow
    }
