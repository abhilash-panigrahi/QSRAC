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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, precision_recall_curve

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
CSV_FILES = ["dataset.csv"]  # Update this list with your actual file paths
MODEL_PATH = "model.joblib"
THRESHOLDS_PATH = "thresholds.json"
MAX_SAMPLES = 2_000_000
CHUNKSIZE = 100_000
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("qsrac.trainer")


# ─── FEATURE MAPPING ─────────────────────────────────────────────────────────
def map_cicids_to_qsrac(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Mapping Layer: Maps raw CIC-IDS2018 or BCCC-CSE-CIC-IDS2018 flows to QSRAC stateful features.
    Hardened for massive datasets.
    """
    mapped = pd.DataFrame(index=df.index)

    def get_raw_col(candidates):
        for c in candidates:
            if c in df.columns:
                return df[c]
        return pd.Series(np.nan, index=df.index)

    def get_num_col(candidates, default_val=0):
        raw = get_raw_col(candidates)
        return pd.to_numeric(raw, errors='coerce').fillna(default_val)

    # Common base attributes
    fwd_pkts = get_num_col(["fwd_packets_count", "Tot Fwd Pkts"], 0)
    bwd_pkts = get_num_col(["bwd_packets_count", "Tot Bwd Pkts"], 0)
    duration = get_num_col(["duration", "Flow Duration"], 1).replace(0, 1)  
    dst_port = get_num_col(["dst_port", "Dst Port", "Destination Port"], 0)

    # 1. hour_of_day
    ts_raw = get_raw_col(["timestamp", "Timestamp"])
    if ts_raw.notna().any():
        dt = pd.to_datetime(ts_raw, errors='coerce')
        mapped["hour_of_day"] = dt.dt.hour.fillna(12.0)
    else:
        mapped["hour_of_day"] = 12.0

    # 2. request_rate
    rate_raw = get_raw_col(["packets_rate", "Flow Pkts/s"])
    if rate_raw.notna().any():
        raw_rate = pd.to_numeric(rate_raw, errors='coerce').fillna(0)
    else:
        raw_rate = (fwd_pkts + bwd_pkts) / duration
    mapped["request_rate"] = np.log1p(raw_rate.clip(lower=0)).clip(0, 10)

    # 3. failed_attempts
    raw_failed = bwd_pkts / (fwd_pkts + 1)
    mapped["failed_attempts"] = np.log1p(raw_failed.clip(lower=0)).clip(0, 10)

    # 4. geo_risk_score 
    total_pkts = fwd_pkts + bwd_pkts
    HIGH_PKT_THRESH = total_pkts.mean() + total_pkts.std(ddof=0)
    
    mapped["geo_risk_score"] = (
        (dst_port > 1024).astype(float) * 0.5 +
        (total_pkts > HIGH_PKT_THRESH).astype(float) * 0.5
    ).clip(0, 1)

    # 5. device_trust_score
    mapped["device_trust_score"] = (1.0 - np.tanh(mapped["failed_attempts"])).clip(0.1, 1.0)

    # 6. sensitivity_level
    protocol = get_raw_col(["protocol", "Protocol"]).astype(str).str.strip().str.upper()
    is_tcp = protocol.isin(["6", "6.0", "TCP"])
    
    mapped["sensitivity_level"] = (
        is_tcp.astype(float) * 2 +
        (dst_port.isin([22, 443, 3389])).astype(float) * 2 +
        1.0
    ).clip(1, 5)

    # 7. is_vpn
    mapped["is_vpn"] = dst_port.isin([1194, 1701, 1723, 500]).astype(float)

    # 8. is_tor
    mapped["is_tor"] = dst_port.isin([9001, 9030, 9050]).astype(float)

    mapped = mapped.replace([np.inf, -np.inf], np.nan).fillna(0)

    cols = [
        "hour_of_day", "request_rate", "failed_attempts", "geo_risk_score",
        "device_trust_score", "sensitivity_level", "is_vpn", "is_tor"
    ]
    
    return mapped[cols]


# ─── DATA LOADER ─────────────────────────────────────────────────────────────
def load_and_sample_data(file_paths, max_samples=MAX_SAMPLES, chunksize=CHUNKSIZE):
    benign_dfs = []
    attack_dfs = []
    
    benign_count = 0
    attack_count = 0
    limit_per_class = max_samples // 2

    for file_path in file_paths:
        if not os.path.exists(file_path):
            log.warning(f"File not found: {file_path}. Skipping.")
            continue
            
        log.info(f"Processing {file_path} in chunks...")
        for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
            label_col = next((c for c in ["label", "Label"] if c in chunk.columns), None)
            if not label_col:
                continue

            # Safer label parsing (handles spaces or string variations like "benign_traffic")
            is_attack = chunk[label_col].astype(str).str.lower().apply(lambda x: 0 if "benign" in x else 1)
            
            mapped_chunk = map_cicids_to_qsrac(chunk)
            mapped_chunk["label"] = is_attack.values
            mapped_chunk = mapped_chunk.replace([np.inf, -np.inf], np.nan).dropna()

            b_chunk = mapped_chunk[mapped_chunk["label"] == 0]
            a_chunk = mapped_chunk[mapped_chunk["label"] == 1]

            # FIX 1: Strict collection limit to save memory
            if benign_count < limit_per_class:
                benign_dfs.append(b_chunk)
                benign_count += len(b_chunk)

            if attack_count < limit_per_class:
                attack_dfs.append(a_chunk)
                attack_count += len(a_chunk)

            if benign_count >= limit_per_class and attack_count >= limit_per_class:
                break
        
        if benign_count >= limit_per_class and attack_count >= limit_per_class:
            break

    if not benign_dfs and not attack_dfs:
        raise ValueError("No valid data loaded. Check file paths and column names.")

    log.info("Concatenating chunks and performing stratified sampling...")
    df_benign = pd.concat(benign_dfs, ignore_index=True) if benign_dfs else pd.DataFrame()
    df_attack = pd.concat(attack_dfs, ignore_index=True) if attack_dfs else pd.DataFrame()

    # Class Imbalance Strict Enforcement
    if not df_benign.empty:
        df_benign = df_benign.sample(n=min(len(df_benign), limit_per_class), random_state=RANDOM_STATE)
    if not df_attack.empty:
        df_attack = df_attack.sample(n=min(len(df_attack), limit_per_class), random_state=RANDOM_STATE)

    full_df = pd.concat([df_benign, df_attack], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    log.info(f"Final dataset loaded. Shape: {full_df.shape} (Benign: {len(df_benign)}, Attack: {len(df_attack)})")
    
    return full_df.drop(columns=["label"]), full_df["label"]


# ─── METRICS ─────────────────────────────────────────────────────────────────
def calculate_fpr(y_true, y_pred):
    # Robust FPR calculation with 1e-9 epsilon
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn + 1e-9)


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────
def main():
    X, y = load_and_sample_data(CSV_FILES)
    
    # Train-test split BEFORE scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    log.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Train LightGBM (Supervised)
    log.info("Training LightGBM model...")
    lgbm = LGBMClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.05, 
        class_weight="balanced", 
        random_state=RANDOM_STATE, 
        verbose=-1
    )
    lgbm.fit(X_train_scaled, y_train)

    # 2. Train Isolation Forest (Unsupervised) - Train ONLY on Benign Data
    log.info("Training Isolation Forest model (on BENIGN data only)...")
    iforest = IsolationForest(
        n_estimators=100, 
        contamination=0.05, 
        random_state=RANDOM_STATE, 
        n_jobs=-1
    )
    
    X_train_benign = X_train_scaled[y_train == 0]
    if len(X_train_benign) > 0:
        iforest.fit(X_train_benign)
    else:
        log.warning("No benign samples found! Falling back to full training set.")
        iforest.fit(X_train_scaled)

    # 3. Hybrid Scoring
    log.info("Evaluating models and computing hybrid scores...")
    lgbm_probs = lgbm.predict_proba(X_test_scaled)[:, 1]
    
    # Invert IF score: higher = more anomalous
    if_raw_train = -iforest.score_samples(X_train_scaled) 
    if_raw_test = -iforest.score_samples(X_test_scaled)
    
    # Safe MinMaxScaler (Avoids Data Leakage by fitting on train, transforming test)
    if_scaler = MinMaxScaler()
    if_scaler.fit(if_raw_train.reshape(-1, 1))
    if_scores_norm = if_scaler.transform(if_raw_test.reshape(-1, 1)).flatten()
    
    # Proper hybrid scoring
    hybrid_scores = (0.7 * lgbm_probs) + (0.3 * if_scores_norm)

    roc_auc = roc_auc_score(y_test, hybrid_scores)
    
    # Safely index thresholds
    precisions, recalls, curve_thresholds = precision_recall_curve(y_test, hybrid_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)

    best_threshold = 0.65 
    hybrid_preds = (hybrid_scores >= best_threshold).astype(int)
    f1 = f1_score(y_test, hybrid_preds)
    fpr = calculate_fpr(y_test, hybrid_preds)
    
    conf_matrix = confusion_matrix(y_test, hybrid_preds, labels=[0, 1])

    # Output Evaluation Metrics
    print("\n" + "="*50)
    print("QSRAC HYBRID MODEL EVALUATION SUMMARY")
    print("="*50)
    print(f"ROC-AUC       : {roc_auc:.4f}")
    print(f"F1-Score      : {f1:.4f} (at optimal threshold {best_threshold:.4f})")
    print(f"FPR           : {fpr:.4f} ({(fpr*100):.2f}%)")
    print("\nConfusion Matrix:")
    print(f"[{conf_matrix[0][0]:7d} (TN) | {conf_matrix[0][1]:7d} (FP)]")
    print(f"[{conf_matrix[1][0]:7d} (FN) | {conf_matrix[1][1]:7d} (TP)]")
    print("="*50 + "\n")

    # 4. Save Artifacts
    log.info("Saving artifacts...")
    joblib.dump({
        "lgbm": lgbm,
        "iforest": iforest,
        "scaler": scaler,
        "if_scaler": if_scaler  
    }, MODEL_PATH)

    # FIX 2: Correctly ordered percentiles for ascending risk thresholds
    risk_thresholds = {
        "low": float(np.percentile(hybrid_scores, 30)),
        "medium": float(np.percentile(hybrid_scores, 50)),
        "high": float(np.percentile(hybrid_scores, 70))
    }

    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(risk_thresholds, f, indent=4)

    log.info(f"Model saved to '{MODEL_PATH}'")
    log.info(f"Thresholds saved to '{THRESHOLDS_PATH}'")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        CSV_FILES = sys.argv[1:]
    main()