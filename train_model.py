# train_model.py
import pandas as pd
import numpy as np
import joblib
import logging
import json
import os
import sys
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from ml_module import HybridRiskModel

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
CSV_FILES = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]
MODEL_PATH = "model.joblib"
THRESHOLDS_PATH = "thresholds.json"
RANDOM_STATE = 42

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("qsrac.trainer")

# ─── FEATURE MAPPING (UNSW-NB15 -> 8 QSRAC Features) ────────────────────────
def map_unsw_to_qsrac(df: pd.DataFrame) -> pd.DataFrame:
    mapped = pd.DataFrame(index=df.index)

    mapped["hour_of_day"] = 12.0

    rate = pd.to_numeric(df.get("rate", 0), errors="coerce").fillna(0)
    mapped["request_rate"] = rate
    dloss = pd.to_numeric(df.get("dloss", 0), errors="coerce").fillna(0)
    spkts = pd.to_numeric(df.get("spkts", 1), errors="coerce").fillna(1)
    mapped["failed_attempts"] = dloss / (spkts + 1)

    sttl = pd.to_numeric(df.get("sttl", 0), errors="coerce").fillna(0)
    mapped["geo_risk_score"] = sttl / 255.0

    mapped["device_trust_score"] = 1.0 / (1.0 + mapped["failed_attempts"])

    proto = df.get("proto", "").astype(str).str.lower()
    service = df.get("service", "").astype(str).str.lower()
    is_tcp = proto.isin(["tcp"]).astype(float)
    is_sensitive = service.isin(["ssh", "http", "https", "ftp"]).astype(float)
    mapped["sensitivity_level"] = is_tcp * 2 + is_sensitive * 2 + 1.0

    mapped["is_vpn"] = 0.0
    mapped["is_tor"] = 0.0

    return mapped[FEATURE_COLUMNS]

def load_data(file_paths):
    dfs = []
    for fp in file_paths:
        if not os.path.exists(fp):
            continue
        log.info(f"Loading {fp}...")
        df = pd.read_csv(fp)
        mapped = map_unsw_to_qsrac(df)
        
        if "label" in df.columns:
            mapped["label"] = df["label"].astype(int)
        else:
            mapped["label"] = 0
            
        dfs.append(mapped)
        
    if not dfs:
        raise FileNotFoundError("No valid CSV files found for training.")
        
    full_df = pd.concat(dfs, ignore_index=True).dropna()
    return full_df[FEATURE_COLUMNS], full_df["label"]

# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    X, y = load_data(CSV_FILES)
    log.info(f"Dataset loaded. Shape: {X.shape}, Attacks: {y.sum()}")

    X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_train, X_calib, y_train, y_calib = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_full)

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_calib = X_calib.reset_index(drop=True)
    y_calib = y_calib.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    log.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLUMNS)
    X_calib_scaled = pd.DataFrame(scaler.transform(X_calib), columns=FEATURE_COLUMNS)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=FEATURE_COLUMNS)

    log.info("Training base LightGBM model...")
    lgbm = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,random_state=RANDOM_STATE, verbose=-1)
    lgbm.fit(X_train_scaled, y_train)

    log.info("Training Isolation Forest...")
    iforest = IsolationForest(n_estimators=100, contamination=0.05,random_state=RANDOM_STATE, n_jobs=-1)
    iforest.fit(X_train_scaled[(y_train == 0).values])

    log.info("Computing raw hybrid scores on calibration set...")
    lgbm_probs_calib = lgbm.predict_proba(X_calib_scaled)[:, 1]
    
    if_raw_train = -iforest.score_samples(X_train_scaled)
    if_raw_calib = -iforest.score_samples(X_calib_scaled)
    
    if_scaler = MinMaxScaler().fit(if_raw_train.reshape(-1, 1))
    if_norm_calib = if_scaler.transform(if_raw_calib.reshape(-1, 1)).flatten()
    if_norm_calib = np.clip(if_norm_calib, 0, 1)
    
    raw_risk_calib = (0.3 * lgbm_probs_calib) + (0.7 * if_norm_calib)

    risk_scaler = None

    print("Raw risk stats:",
          raw_risk_calib.min(),
          raw_risk_calib.mean(),
          raw_risk_calib.max())

    log.info("Fitting Platt Scaler (Logistic Regression) on hybrid risk...")
    platt_scaler = LogisticRegression(max_iter=1000,C=0.5,random_state=RANDOM_STATE)
    platt_scaler.fit(raw_risk_calib.reshape(-1, 1), y_calib)


    log.info("Creating and validating final hybrid model artifact...")
    model = HybridRiskModel(lgbm, iforest, scaler, if_scaler, platt_scaler, risk_scaler)
    
    calibrated_risk_val = model.predict_risk(X_val)

    print("Calibrated risk stats:",
          calibrated_risk_val.min(),
          calibrated_risk_val.mean(),
          calibrated_risk_val.max())

    hybrid_trust_val = 1.0 - calibrated_risk_val

    best_f1 = 0
    best_trust_thresh = 0.5
    trust_candidates = np.linspace(0.1, 0.9, 100)
    
    for t in trust_candidates:
        preds = (hybrid_trust_val <= t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_trust_thresh = t

    log.info(f"Optimal Trust Threshold: {best_trust_thresh:.4f} (F1: {best_f1:.4f})")

    log.info("Saving unified model artifact...")
    joblib.dump(model, MODEL_PATH)

    benign_trust = hybrid_trust_val[(y_val == 0).values]
    
    med_t = best_trust_thresh
    
    low_candidates = benign_trust[benign_trust > med_t]
    if len(low_candidates) > 0:
        low_t = float(np.percentile(low_candidates, 50))
    else:
        low_t = med_t + 0.1
        
    high_candidates = benign_trust[benign_trust < med_t]
    if len(high_candidates) > 0:
        high_t = float(np.percentile(high_candidates, 10))
    else:
        high_t = max(0.01, med_t - 0.1)

    if med_t >= low_t:
        med_t = low_t - 0.01
    if high_t >= med_t:
        high_t = med_t - 0.01

    risk_thresholds = {
        "low": float(max(low_t, 0.02)),
        "medium": float(max(med_t, 0.01)),
        "high": float(max(high_t, 0.00))
    }

    with open(THRESHOLDS_PATH, "w") as f: 
        json.dump(risk_thresholds, f, indent=4)

    log.info(f"Process Complete. Thresholds mapped: {risk_thresholds}")

if __name__ == "__main__":
    if len(sys.argv) > 1: CSV_FILES = sys.argv[1:]
    main()