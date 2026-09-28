"""
Data Layer for POLARIS-AI Antarctic Decision Support System.
Provides NetCDF, GeoTIFF, and observational dataset ingestion.
"""
from .ingestion import (
    NetCDFDatasetReader,
    GeoTIFFReader,
    AntarcticCoastlineMask,
    EnvironmentalDataProvider,
    environmental_data_provider
)

__all__ = [
    "NetCDFDatasetReader",
    "GeoTIFFReader",
    "AntarcticCoastlineMask",
    "EnvironmentalDataProvider",
    "environmental_data_provider"
]
