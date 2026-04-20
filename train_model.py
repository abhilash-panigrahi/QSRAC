import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import logging
import json
import os
import sys
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve 

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
CSV_FILES = ["dataset.csv"] 
MODEL_PATH = "model.joblib"
THRESHOLDS_PATH = "thresholds.json"
MAX_SAMPLES = 2_000_000
CHUNKSIZE = 100_000
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("qsrac.trainer")

# ─── FEATURE MAPPING ─────────────────────────────────────────────────────────
def map_cicids_to_qsrac(df: pd.DataFrame) -> pd.DataFrame:
    mapped = pd.DataFrame(index=df.index)

    def get_raw_col(candidates):
        for c in candidates:
            if c in df.columns: return df[c]
        return pd.Series(np.nan, index=df.index)

    def get_num_col(candidates, default_val=0):
        raw = get_raw_col(candidates)
        return pd.to_numeric(raw, errors='coerce').fillna(default_val)

    fwd_pkts = get_num_col(["fwd_packets_count", "Tot Fwd Pkts"], 0)
    bwd_pkts = get_num_col(["bwd_packets_count", "Tot Bwd Pkts"], 0)
    duration = get_num_col(["duration", "Flow Duration"], 1).replace(0, 1)  
    dst_port = get_num_col(["dst_port", "Dst Port", "Destination Port"], 0)

    ts_raw = get_raw_col(["timestamp", "Timestamp"])
    if ts_raw.notna().any():
        dt = pd.to_datetime(ts_raw, errors='coerce')
        mapped["hour_of_day"] = dt.dt.hour.fillna(12.0)
    else:
        mapped["hour_of_day"] = 12.0

    rate_raw = get_raw_col(["packets_rate", "Flow Pkts/s"])
    if rate_raw.notna().any():
        raw_rate = pd.to_numeric(rate_raw, errors='coerce').fillna(0)
    else:
        raw_rate = (fwd_pkts + bwd_pkts) / duration
    mapped["request_rate"] = np.log1p(raw_rate.clip(lower=0)).clip(0, 10)

    raw_failed = bwd_pkts / (fwd_pkts + 1)
    mapped["failed_attempts"] = np.log1p(raw_failed.clip(lower=0)).clip(0, 10)

    total_pkts = fwd_pkts + bwd_pkts
    HIGH_PKT_THRESH = total_pkts.mean() + total_pkts.std(ddof=0)
    
    mapped["geo_risk_score"] = (
        (dst_port > 1024).astype(float) * 0.5 +
        (total_pkts > HIGH_PKT_THRESH).astype(float) * 0.5
    ).clip(0, 1)

    mapped["device_trust_score"] = (1.0 - np.tanh(mapped["failed_attempts"])).clip(0.1, 1.0)

    protocol = get_raw_col(["protocol", "Protocol"]).astype(str).str.strip().str.upper()
    is_tcp = protocol.isin(["6", "6.0", "TCP"])
    mapped["sensitivity_level"] = (is_tcp.astype(float) * 2 + (dst_port.isin([22, 443, 3389])).astype(float) * 2 + 1.0).clip(1, 5)
    mapped["is_vpn"] = dst_port.isin([1194, 1701, 1723, 500]).astype(float)
    mapped["is_tor"] = dst_port.isin([9001, 9030, 9050]).astype(float)

    mapped = mapped.replace([np.inf, -np.inf], np.nan).fillna(0)
    cols = ["hour_of_day", "request_rate", "failed_attempts", "geo_risk_score", "device_trust_score", "sensitivity_level", "is_vpn", "is_tor"]
    return mapped[cols]

def load_and_sample_data(file_paths, max_samples=MAX_SAMPLES, chunksize=CHUNKSIZE):
    benign_dfs, attack_dfs = [], []
    benign_count, attack_count = 0, 0
    limit_per_class = max_samples // 2

    for file_path in file_paths:
        if not os.path.exists(file_path): continue
        log.info(f"Processing {file_path} in chunks...")
        for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
            label_col = next((c for c in ["label", "Label"] if c in chunk.columns), None)
            if not label_col: continue
            is_attack = chunk[label_col].astype(str).str.lower().apply(lambda x: 0 if "benign" in x else 1)
            mapped_chunk = map_cicids_to_qsrac(chunk)
            mapped_chunk["label"] = is_attack.values
            mapped_chunk = mapped_chunk.replace([np.inf, -np.inf], np.nan).dropna()
            if benign_count < limit_per_class:
                b = mapped_chunk[mapped_chunk["label"] == 0]
                benign_dfs.append(b); benign_count += len(b)
            if attack_count < limit_per_class:
                a = mapped_chunk[mapped_chunk["label"] == 1]
                attack_dfs.append(a); attack_count += len(a)
            if benign_count >= limit_per_class and attack_count >= limit_per_class: break
        if benign_count >= limit_per_class and attack_count >= limit_per_class: break

    df_benign = pd.concat(benign_dfs, ignore_index=True).sample(n=min(benign_count, limit_per_class), random_state=RANDOM_STATE)
    df_attack = pd.concat(attack_dfs, ignore_index=True).sample(n=min(attack_count, limit_per_class), random_state=RANDOM_STATE)
    full_df = pd.concat([df_benign, df_attack], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return full_df.drop(columns=["label"]), full_df["label"]

def calculate_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn + 1e-9)

def generate_evaluation_outputs(y_test, hybrid_trust, risk_scores, hybrid_preds, trust_threshold):
    tn, fp, fn, tp = confusion_matrix(y_test, hybrid_preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn + 1e-9)
    tpr = tp / (tp + fn + 1e-9)
    precision_val = tp / (tp + fp + 1e-9)
    roc_auc = roc_auc_score(y_test, risk_scores)
    f1 = f1_score(y_test, hybrid_preds)
    
    metrics_dict = {
        "Trust_Threshold": float(trust_threshold),
        "Risk_Threshold": float(1.0 - trust_threshold),
        "ROC_AUC": float(roc_auc),
        "F1_Score": float(f1),
        "Precision": float(precision_val),
        "Recall": float(tpr),
        "False_Positive_Rate": float(fpr),
        "Confusion_Matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }
    with open("evaluation.json", "w") as f: json.dump(metrics_dict, f, indent=4)

    # ROC Plot (Standard ML orientation uses Risk)
    fpr_curve, tpr_curve, _ = roc_curve(y_test, risk_scores)
    plt.figure(); plt.plot(fpr_curve, tpr_curve, label=f"AUC = {roc_auc:.4f}"); plt.grid(True)
    plt.savefig("roc_curve.png", dpi=300); plt.close()

    # Distribution Plot (Showing Trust Scores)
    plt.figure()
    plt.hist(hybrid_trust[y_test == 0], bins=50, alpha=0.5, label="Benign (Trustworthy)", density=True)
    plt.hist(hybrid_trust[y_test == 1], bins=50, alpha=0.5, label="Attack (Untrustworthy)", density=True)
    plt.legend(); plt.savefig("score_distribution.png", dpi=300); plt.close()

# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    X, y = load_and_sample_data(CSV_FILES)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    log.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    log.info("Training LightGBM model...")
    lgbm = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)
    lgbm.fit(X_train_scaled, y_train)

    log.info("Training Isolation Forest (Contamination=0.02)...")
    iforest = IsolationForest(n_estimators=100, contamination=0.02, random_state=RANDOM_STATE, n_jobs=-1)
    iforest.fit(X_train_scaled[y_train == 0])

    log.info("Computing hybrid scores...")
    lgbm_probs = lgbm.predict_proba(X_test_scaled)[:, 1]
    if_raw_train = -iforest.score_samples(X_train_scaled)
    if_raw_test = -iforest.score_samples(X_test_scaled)
    if_scaler = MinMaxScaler().fit(if_raw_train.reshape(-1, 1))
    if_scores_norm = if_scaler.transform(if_raw_test.reshape(-1, 1)).flatten()
    
    # Semantic Alignment Fix: Risk (High=Attack) to Trust (High=Safe)
    risk_scores = (0.7 * lgbm_probs) + (0.3 * if_scores_norm)
    hybrid_trust = 1.0 - risk_scores

    risk_threshold = 0.6275 
    trust_threshold = 1.0 - risk_threshold
    
    # Attack occurs when Trust is BELOW the threshold
    hybrid_preds = (hybrid_trust <= trust_threshold).astype(int)

    joblib.dump({"lgbm": lgbm, "iforest": iforest, "scaler": scaler, "if_scaler": if_scaler}, MODEL_PATH)

    benign_trust = hybrid_trust[y_test == 0]
    low = float(np.percentile(benign_trust, 10))
    medium = float((low + trust_threshold) / 2)
    risk_thresholds = {
        "low": low,
        "medium": medium,
        "high": float(trust_threshold),
    }

    with open(THRESHOLDS_PATH, "w") as f: json.dump(risk_thresholds, f, indent=4)

    generate_evaluation_outputs(y_test, hybrid_trust, risk_scores, hybrid_preds, trust_threshold)
    log.info(f"Process Complete. Trust Thresholds: {risk_thresholds}")

if __name__ == "__main__":
    if len(sys.argv) > 1: CSV_FILES = sys.argv[1:]
    main()
#to run use python3 train_model.py /mnt/e/Dataset/*.csv