import os
import numpy as np
import joblib
import config
import json
import logging

# Centralized logging for QSRAC ML
log = logging.getLogger("qsrac.ml")

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "thresholds.json")

# Global model state
_lgbm = None
_iforest = None
_scaler = None
_if_scaler = None
_THRESH = None

def _load_thresholds():
    global _THRESH
    if _THRESH is None:
        if config.USE_DYNAMIC_THRESHOLDS:
            try:
                with open(THRESHOLD_PATH) as f:
                    _THRESH = json.load(f)
            except Exception:
                # Fallback if dynamic is True but file is missing
                _THRESH = {
                    "low": config.RISK_THRESHOLD_LOW,
                    "medium": config.RISK_THRESHOLD_MEDIUM,
                    "high": config.RISK_THRESHOLD_HIGH,
                }
        else:
            # Strictly use static config values
            _THRESH = {
                "low": config.RISK_THRESHOLD_LOW,
                "medium": config.RISK_THRESHOLD_MEDIUM,
                "high": config.RISK_THRESHOLD_HIGH,
            }

def _load_model():
    global _lgbm, _iforest, _scaler
    if _lgbm is None:
        try:
            artifact = joblib.load(MODEL_PATH)
            _lgbm = artifact["lgbm"]
            _iforest = artifact["iforest"]
            _scaler = artifact.get("scaler", None)
            _if_scaler = artifact.get("if_scaler", None)
            log.info("Hybrid ML models (LGBM + IForest) loaded successfully.")
        except Exception as e:
            log.error(f"CRITICAL: Failed to load models from {MODEL_PATH}: {e}")
            raise

def _extract_features(context_dict: dict) -> np.ndarray:
    """Extracts 8-dimensional feature vector from request context."""
    features = [
        float(context_dict.get("hour_of_day", 12)),
        float(context_dict.get("request_rate", 1.0)),
        float(context_dict.get("failed_attempts", 0)),
        float(context_dict.get("geo_risk_score", 0.0)),
        float(context_dict.get("device_trust_score", 1.0)),
        float(context_dict.get("sensitivity_level", 1.0)),
        float(context_dict.get("is_vpn", 0)),
        float(context_dict.get("is_tor", 0))
    ]
    return np.array(features).reshape(1, -1)

def _map_score_to_risk(score: float) -> str:
    """Maps continuous [0,1] hybrid score to categorical risk levels."""
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
    """
    Computes hybrid risk score using Max-Fusion of LightGBM and Isolation Forest.
    Includes robust statistical normalization and fail-safe error handling.
    """
    try:
        _load_model()
        features = _extract_features(context_dict)

        # 1. Safety Clipping: Prevents extreme outliers from skewing normalization
        features = np.clip(features, -10, 10)

        if _scaler is not None:
            features = _scaler.transform(features)

        # 2. Supervised Score: LightGBM attack probability [0, 1]
        lgbm_prob = _lgbm.predict_proba(features)[0][1]
        lgbm_prob = max(min(lgbm_prob, 1.0), 0.0) # Redundant, but safe

        # 3. Unsupervised Score: Isolation Forest anomaly detection
        # score_samples returns negative values (lower = more anomalous)
        if_score_raw = -_iforest.score_samples(features)[0]

        # 4. Normalization: Use the ruler from training (MinMaxScaler)
        if _if_scaler is not None:
            # Matches the exact linear scaling used in train_model.py
            if_score_norm = _if_scaler.transform([[if_score_raw]])[0][0]
        else:
            # Fail-safe only
            if_score_norm = (np.tanh(if_score_raw) + 1) / 2

        # 5. Hybrid Fusion: Weighted Average (0.7 / 0.3)
        risk_score = (0.7 * lgbm_prob) + (0.3 * if_score_norm)

        # 6. Trust Inversion: Risk (High=Bad) -> Trust (High=Safe)
        trust_score = 1.0 - risk_score

        
        # 5. Hybrid Fusion: Weighted Average 
        risk_score = (0.7 * lgbm_prob) + (0.3 * if_score_norm)

        # 6. Trust Inversion: Risk (High=Bad) -> Trust (High=Safe)
        # This aligns the ML output with the policy engine's trust-based thresholds
        trust_score = 1.0 - risk_score 

        log.debug(
            f"[ML_INFERENCE] LGBM={lgbm_prob:.4f}, IF={if_score_norm:.4f}, "
            f"RISK={risk_score:.4f}, TRUST={trust_score:.4f}"
        )

        return _map_score_to_risk(trust_score)

    except Exception as e:
        log.error(f"ML Inference Error: {e}. Defaulting to 'Medium' risk fail-safe.")
        # Fail-safe: "Medium" risk ensures continuity without granting full trust
        return "Medium"