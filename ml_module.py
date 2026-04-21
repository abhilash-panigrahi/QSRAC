# ml_module.py
import os
import numpy as np
import pandas as pd
import joblib
import json
import logging
import config

log = logging.getLogger("qsrac.ml")

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "thresholds.json")

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
        
        if list(X.columns) != FEATURE_COLUMNS:
            raise ValueError("Input feature columns do not match expected training columns.")

        X_scaled_array = self.scaler.transform(X) if self.scaler else X.values
        X_scaled_df = pd.DataFrame(X_scaled_array, columns=FEATURE_COLUMNS)

        lgbm_prob = self.lgbm.predict_proba(X_scaled_df)[:, 1]
        
        if_raw = -self.iforest.score_samples(X_scaled_df)
        if_norm = self.if_scaler.transform(if_raw.reshape(-1, 1)).flatten() if self.if_scaler else if_raw
        if_norm = np.clip(if_norm, 0, 1)
        
        raw_risk = (0.3 * lgbm_prob) + (0.7 * if_norm)
        
        
        if np.mean(raw_risk) > 0.9:
            pass
        
        calibrated_risk = self.platt_scaler.predict_proba(raw_risk.reshape(-1, 1))[:, 1]
        return calibrated_risk

_model = None
_THRESH = None

def _load_thresholds():
    global _THRESH
    if _THRESH is None:
        if config.USE_DYNAMIC_THRESHOLDS:
            try:
                with open(THRESHOLD_PATH) as f:
                    _THRESH = json.load(f)
            except Exception:
                _THRESH = {
                    "low": config.RISK_THRESHOLD_LOW,
                    "medium": config.RISK_THRESHOLD_MEDIUM,
                    "high": config.RISK_THRESHOLD_HIGH,
                }
        else:
            _THRESH = {
                "low": config.RISK_THRESHOLD_LOW,
                "medium": config.RISK_THRESHOLD_MEDIUM,
                "high": config.RISK_THRESHOLD_HIGH,
            }

def _load_model():
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            log.error(f"CRITICAL: Failed to load models from {MODEL_PATH}: {e}")
            raise

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

def _map_score_to_risk(trust_score: float) -> str:
    _load_thresholds()
    
    if trust_score >= _THRESH["low"]:
        return "Low"
    elif trust_score >= _THRESH["medium"]:
        return "Medium"
    elif trust_score >= _THRESH["high"]:
        return "High"
    else:
        return "Critical"

def get_risk_score(context_dict: dict) -> str:
    try:
        _load_model()
        features_df = _extract_features(context_dict)

        risk_score = float(_model.predict_risk(features_df)[0])
        trust_score = 1.0 - risk_score 

        log.debug(f"[ML_INFERENCE] RISK={risk_score:.4f}, TRUST={trust_score:.4f}")

        return _map_score_to_risk(trust_score)

    except Exception as e:
        log.error(f"ML Inference Error: {e}. Defaulting to 'Medium' risk fail-safe.")
        return "Medium"