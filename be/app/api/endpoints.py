"""
API Endpoints for CITD ML Meta-Labeling
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from be.app.services.model_service import ModelService
from be.app.services.data_service import DataService

router = APIRouter()

class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(default_factory=dict, description="Dictionary of 23 input features")
    threshold: Optional[float] = Field(0.50, description="Decision threshold, default 0.50")

@router.get("/health")
def get_health():
    model_service = ModelService.get_instance()
    data_service = DataService.get_instance()
    holdout_count = len(data_service.universe_df) if data_service.universe_df is not None else 0
    return {
        "status": "online",
        "model": "CatBoostClassifier (Run 1)",
        "features_count": len(model_service.feature_names),
        "holdout_samples": holdout_count,
        "framework": "FastAPI + Uvicorn"
    }

@router.get("/market/sample")
def get_market_sample(limit: int = Query(120, ge=30, le=300)):
    data_service = DataService.get_instance()
    return data_service.get_sample_candles_and_signals(limit=limit)

@router.post("/predict")
def predict_signal(req: PredictRequest):
    model_service = ModelService.get_instance()
    return model_service.predict_one(req.features, threshold=req.threshold or 0.50)

@router.get("/features/importances")
def get_feature_importances():
    model_service = ModelService.get_instance()
    return model_service.get_global_importances()

@router.get("/backtest/summary")
def get_backtest_summary():
    data_service = DataService.get_instance()
    return data_service.get_backtest_summary()

@router.get("/backtest/equity-curve")
def get_equity_curve(keep_pct: int = Query(50, ge=10, le=100)):
    data_service = DataService.get_instance()
    return data_service.get_equity_curve_data(keep_pct=keep_pct)

@router.get("/splits/comparison")
def get_splits_comparison():
    data_service = DataService.get_instance()
    return data_service.get_splits_comparison()
