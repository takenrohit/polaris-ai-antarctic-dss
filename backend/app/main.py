"""
Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System (DSS)
Organization: Ministry of Earth Sciences (MoES) / National Centre for Polar and Ocean Research (NCPOR)
FastAPI Main Application Entry Point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .api.routes_forecast import router as forecast_router
from .api.routes_icebergs import router as icebergs_router
from .api.routes_navigation import router as navigation_router
from .api.routes_telemetry import router as telemetry_router

app = FastAPI(
    title=settings.APP_NAME,
    description="AI/ML-Enabled Decision Support Platform for Antarctic Sea-Ice Forecasting, "
                "Iceberg Drift Prediction, and Polar Vessel Route Optimization (IMO POLARIS / MoES NCPOR).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for modern React / Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(forecast_router, prefix=settings.API_V1_PREFIX)
app.include_router(icebergs_router, prefix=settings.API_V1_PREFIX)
app.include_router(navigation_router, prefix=settings.API_V1_PREFIX)
app.include_router(telemetry_router, prefix=settings.API_V1_PREFIX)

@app.get("/")
def root():
    return {
        "system": settings.APP_NAME,
        "organization": "National Centre for Polar and Ocean Research (NCPOR)",
        "ministry": "Ministry of Earth Sciences (MoES), Government of India",
        "status": "OPERATIONAL",
        "documentation": "/docs",
        "modules": {
            "sea_ice_forecasting": f"{settings.API_V1_PREFIX}/forecast/sea-ice",
            "iceberg_tracking": f"{settings.API_V1_PREFIX}/icebergs",
            "route_optimization": f"{settings.API_V1_PREFIX}/navigation/optimize",
            "polar_stations": f"{settings.API_V1_PREFIX}/navigation/stations",
            "telemetry_stream": f"{settings.API_V1_PREFIX}/telemetry/ws"
        }
    }

@app.get("/health")
@app.get(f"{settings.API_V1_PREFIX}/health")
def health_check():
    return {"status": "healthy", "engine": "PyTorch + FastAPI Polar DSS"}
