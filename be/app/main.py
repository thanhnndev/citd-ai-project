"""
Main FastAPI Application Entrypoint
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from be.app.api.endpoints import router as api_router
from be.app.services.model_service import ModelService
from be.app.services.data_service import DataService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload model and dataset
    print("[BACKEND] Preloading CatBoost Model and Holdout Data...")
    _ = ModelService.get_instance()
    _ = DataService.get_instance()
    print("[BACKEND] Ready to serve inference and backtest APIs!")
    yield
    print("[BACKEND] Shutting down...")

app = FastAPI(
    title="CITD ML LAB: Meta-Labeling CatBoost API",
    description="Backend API serving ML inference, Triple-Barrier labeling simulation, and Data Leakage evaluation.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Frontend (Vite on 5173 or other local ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {
        "project": "CITD ML Meta-Labeling CatBoost",
        "docs": "/docs",
        "status": "active"
    }
