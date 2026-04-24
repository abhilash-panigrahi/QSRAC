# ml_module.py
import os
import numpy as np
import pandas as pd
import joblib
import logging
import config
import numpy as np

log = logging.getLogger("qsrac.ml")

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
FORCE_RISK = os.getenv("FORCE_RISK")

FEATURE_COLUMNS = [
    "hour_of_day",
    "request_rate",
    "failed_attempts",
    "geo_risk_score",
    "device_trust_score",
    "sensitivity_level",
    "is_vpn",
    "is_tor"
]

class HybridRiskModel:
    def __init__(self, lgbm, iforest, scaler, if_scaler, platt_scaler, risk_scaler):
        self.lgbm = lgbm
        self.iforest = iforest
        self.scaler = scaler
        self.if_scaler = if_scaler
        self.platt_scaler = platt_scaler
        self.risk_scaler = risk_scaler

    def predict_risk(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame with correct feature names.")
        
        if set(X.columns) != set(FEATURE_COLUMNS):
            raise ValueError("Feature mismatch")
        X = X[FEATURE_COLUMNS]

        if self.scaler:
            X_scaled_array = self.scaler.transform(X)
        else:
            log.warning("Scaler missing — using raw features")
            X_scaled_array = X.values
        X_scaled_df = pd.DataFrame(X_scaled_array, columns=FEATURE_COLUMNS)

        lgbm_prob = self.lgbm.predict_proba(X_scaled_df)[:, 1]
        
        if_raw = -self.iforest.score_samples(X_scaled_df)
        if_norm = self.if_scaler.transform(if_raw.reshape(-1, 1)).flatten() if self.if_scaler else if_raw
        if_norm = np.clip(if_norm, 0, 1)
        
        raw_risk = (0.7 * lgbm_prob) + (0.3 * if_norm)
        
        
        if np.mean(raw_risk) > 0.9:
            logging.getLogger("qsrac.ml").warning("Risk saturation detected in inference")
        
        calibrated_risk = self.platt_scaler.predict_proba(raw_risk.reshape(-1, 1))[:, 1]

        # 🔧 desaturate using logit transform
        eps = 1e-6
        p = np.clip(calibrated_risk, eps, 1 - eps)
        logit = np.log(p / (1 - p))

        # compress extremes
        logit = logit / 1.5

        calibrated_risk = 1 / (1 + np.exp(-logit))

        return calibrated_risk

_model = None

def init_model():
    global _model
    if _model is not None:
        return

    try:
        _model = joblib.load(MODEL_PATH)
        if not hasattr(_model, "predict_risk"):
            raise RuntimeError("Invalid model: missing predict_risk")
        log.info("ML model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"CRITICAL: Model load failed: {e}")

def _extract_features(context_dict: dict) -> pd.DataFrame:
    if not context_dict:
        raise ValueError("Empty context provided to ML module.")
        
    for key in config.REQUIRED_CONTEXT_KEYS:
        if key not in context_dict:
            raise ValueError(f"Missing required context key: {key}")

    
    features = {
        "hour_of_day": float(context_dict["hour_of_day"]),
        "request_rate": float(context_dict["request_rate"]),
        "failed_attempts": float(context_dict["failed_attempts"]),  
        "geo_risk_score": float(context_dict["geo_risk_score"]),
        "device_trust_score": float(context_dict["device_trust_score"]),
        "sensitivity_level": float(context_dict["sensitivity_level"]),
        "is_vpn": float(context_dict["is_vpn"]),
        "is_tor": float(context_dict["is_tor"])
    }
    
    df = pd.DataFrame([features])
    return df[FEATURE_COLUMNS]

_THRESH = {
    "low_medium": 0.32,
    "medium_high": 0.6,
    "high_critical": 0.94
}

def _map_risk(score: float) -> str:
    s = float(np.clip(score, 0.0, 1.0))

    if s < _THRESH["low_medium"]:
        return "Low"
    elif s < _THRESH["medium_high"]:
        return "Medium"
    elif s < _THRESH["high_critical"]:
        return "High"
    else:
        return "Critical"

def get_risk_score(context_dict: dict) -> str:
    global _model

    features_df = _extract_features(context_dict)

    if _model is None:
        init_model()

    risk_score = float(_model.predict_risk(features_df)[0])

    label = _map_risk(risk_score)

    print("RAW RISK SCORE =", risk_score)
    return label




