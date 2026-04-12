import os
import numpy as np
import joblib
import config
import json

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "thresholds.json")

_model = None
_scaler = None
_THRESH = None

def _load_thresholds():
    global _THRESH
    if _THRESH is None:
        try:
            with open(THRESHOLD_PATH) as f:
                _THRESH = json.load(f)
        except Exception:
            import config
            _THRESH = {
                "low": config.RISK_THRESHOLD_LOW,
                "medium": config.RISK_THRESHOLD_MEDIUM,
                "high": config.RISK_THRESHOLD_HIGH,
            }

def _load_model():
    global _model, _scaler
    if _model is None:
        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict):
            _model = artifact["model"]
            _scaler = artifact.get("scaler", None)
        else:
            _model = artifact
            _scaler = None


def _extract_features(context_dict: dict) -> np.ndarray:
    hour = float(context_dict.get("hour_of_day", 12))
    req = float(context_dict.get("request_rate", 1.0))
    fail = float(context_dict.get("failed_attempts", 0))
    geo = float(context_dict.get("geo_risk_score", 0.0))
    trust = float(context_dict.get("device_trust_score", 1.0))
    sens = float(context_dict.get("sensitivity_level", 1.0))
    vpn = float(context_dict.get("is_vpn", 0))
    tor = float(context_dict.get("is_tor", 0))


    features = [
    hour, req, fail, geo, trust, sens, vpn, tor
    ]

    return np.array(features).reshape(1, -1)


def _map_score_to_risk(score: float) -> str:
    _load_thresholds()

    if score >= _THRESH["low"]:
        return "Low"
    elif score >= _THRESH["medium"]:
        return "Medium"
    elif score >= _THRESH["high"]:
        return "High"
    else:
        return "Critical"


def get_risk_score(context_dict: dict) -> str:
    _load_model()

    features = _extract_features(context_dict)

    if _scaler is not None:
        features = _scaler.transform(features)

    score = _model.score_samples(features)[0]

    return _map_score_to_risk(score)