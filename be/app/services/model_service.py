"""
Model Service: Loads CatBoost Meta-Model and handles inference & explainability
"""
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "outputs/holdout/stage2/catboost_final_holdout_run1.cbm"

FEATURES = [
    "breakeven_R",
    "atr14_pct",
    "atr_ratio_14_90",
    "vol20",
    "vol200",
    "vol_ratio_20_200",
    "range_pct",
    "dist_ema20_atr",
    "dist_ema50_atr",
    "dist_ema200_atr",
    "ret20_atr",
    "ret50_atr",
    "pos_in_range50",
    "dist_hh20_atr",
    "mom14",
    "mom_diff",
    "vwap_dist_atr",
    "vwap_slope_atr",
    "rsi14",
    "vol_ratio_volume",
    "gap_open_atr",
    "hour",
    "dow"
]

class ModelService:
    _instance = None

    def __init__(self):
        self.model = None
        self.feature_names = FEATURES
        self.feature_importances = {}
        self._load_model()

    def _load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        
        self.model = CatBoostClassifier()
        self.model.load_model(str(MODEL_PATH))
        
        # Calculate feature importances
        raw_fi = self.model.get_feature_importance()
        total_fi = sum(raw_fi) if sum(raw_fi) > 0 else 1.0
        normalized_fi = {feat: float(raw_fi[i] / total_fi * 100) for i, feat in enumerate(self.feature_names)}
        # Sort descending
        self.feature_importances = dict(sorted(normalized_fi.items(), key=lambda x: x[1], reverse=True))

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict_one(self, feature_dict: dict, threshold: float = 0.50):
        # Build feature vector
        vector = []
        for feat in self.feature_names:
            val = feature_dict.get(feat, 0.0)
            vector.append(float(val))
            
        df = pd.DataFrame([vector], columns=self.feature_names)
        prob = float(self.model.predict_proba(df)[0, 1])
        decision = "PASS" if prob >= threshold else "SKIP"
        
        # Top 5 most influential features for this model
        top5_features = [{"feature": k, "importance_pct": round(v, 2), "value": round(feature_dict.get(k, 0.0), 4)} 
                         for k, v in list(self.feature_importances.items())[:5]]

        return {
            "probability": round(prob, 4),
            "threshold": threshold,
            "decision": decision,
            "pass": prob >= threshold,
            "top5_features": top5_features
        }

    def get_global_importances(self):
        return [
            {"feature": feat, "importance_pct": round(pct, 2)}
            for feat, pct in self.feature_importances.items()
        ]
