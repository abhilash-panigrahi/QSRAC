import os
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score
)

# ===================== CONFIG ===================== #
TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE  = "UNSW_NB15_testing-set.csv"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "hour_of_day", "request_rate", "failed_attempts",
    "geo_risk_score", "device_trust_score",
    "sensitivity_level", "is_vpn", "is_tor"
]

# ===================== FEATURE MAPPING ===================== #
def map_unsw_to_qsrac(df: pd.DataFrame) -> pd.DataFrame:
    mapped = pd.DataFrame(index=df.index)

    mapped["hour_of_day"] = 12.0

    rate = pd.to_numeric(df.get("rate", 0), errors="coerce").fillna(0)
    mapped["request_rate"] = np.log1p(rate.clip(lower=0)).clip(0, 10)

    dloss = pd.to_numeric(df.get("dloss", 0), errors="coerce").fillna(0)
    spkts = pd.to_numeric(df.get("spkts", 1), errors="coerce").fillna(1)
    mapped["failed_attempts"] = np.log1p(dloss / (spkts + 1)).clip(0, 10)

    sttl = pd.to_numeric(df.get("sttl", 0), errors="coerce").fillna(0)
    mapped["geo_risk_score"] = (sttl > 64).astype(float) * 0.5

    mapped["device_trust_score"] = (1.0 - np.tanh(mapped["failed_attempts"])).clip(0.1, 1.0)

    proto = df.get("proto", "").astype(str).str.lower()
    service = df.get("service", "").astype(str).str.lower()
    is_tcp = proto.isin(["tcp"]).astype(float)
    is_sensitive = service.isin(["ssh", "http", "https", "ftp"]).astype(float)

    mapped["sensitivity_level"] = (is_tcp * 2 + is_sensitive * 2 + 1.0).clip(1, 5)

    mapped["is_vpn"] = 0.0
    mapped["is_tor"] = 0.0

    return mapped[FEATURE_COLUMNS]

# ===================== LOAD DATA ===================== #
def load_train():
    df = pd.read_csv(TRAIN_FILE)
    X = map_unsw_to_qsrac(df)
    y = df["label"].astype(int)
    return X, y

def load_test():
    df = pd.read_csv(TEST_FILE)
    X = map_unsw_to_qsrac(df)
    y = df["label"].astype(int)
    return X, y

# ===================== MAIN ===================== #
def main():
    # -------- Train / Validation split ONLY from training set -------- #
    X, y = load_train()

    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    X_val, X_unused, y_val, y_unused = train_test_split(
        X_calib, y_calib, test_size=0.5, random_state=RANDOM_STATE, stratify=y_calib
    )

    # -------- Scaling -------- #
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLUMNS)
    X_calib_s = pd.DataFrame(scaler.transform(X_calib), columns=FEATURE_COLUMNS)
    X_val_s   = pd.DataFrame(scaler.transform(X_val), columns=FEATURE_COLUMNS)

    # -------- Model -------- #
    lgbm = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1
    )
    lgbm.fit(X_train_s, y_train)

    iforest = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    iforest.fit(X_train_s[(y_train == 0).values])

    # -------- Calibration -------- #
    lgbm_calib = lgbm.predict_proba(X_calib_s)[:, 1]

    if_raw_train = -iforest.score_samples(X_train_s)
    if_raw_calib = -iforest.score_samples(X_calib_s)

    if_scaler = MinMaxScaler().fit(if_raw_train.reshape(-1, 1))
    if_norm_calib = np.clip(if_scaler.transform(if_raw_calib.reshape(-1, 1)).flatten(), 0, 1)

    raw_risk_calib = 0.7 * lgbm_calib + 0.3 * if_norm_calib

    # Balanced Platt Scaling
    pos = raw_risk_calib[y_calib == 1]
    neg = raw_risk_calib[y_calib == 0]
    n = min(len(pos), len(neg))

    rng = np.random.default_rng(RANDOM_STATE)
    X_bal = np.concatenate([
        pos[rng.choice(len(pos), n, replace=False)],
        neg[rng.choice(len(neg), n, replace=False)]
    ])
    y_bal = np.concatenate([np.ones(n), np.zeros(n)])

    platt = LogisticRegression(class_weight="balanced", max_iter=1000)
    platt.fit(X_bal.reshape(-1, 1), y_bal)

    # -------- Threshold tuning (validation only) -------- #
    lgbm_val = lgbm.predict_proba(X_val_s)[:, 1]
    if_raw_val = -iforest.score_samples(X_val_s)
    if_norm_val = np.clip(if_scaler.transform(if_raw_val.reshape(-1, 1)).flatten(), 0, 1)

    risk_val = platt.predict_proba((0.7 * lgbm_val + 0.3 * if_norm_val).reshape(-1, 1))[:, 1]

    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.1, 0.9, 100):
        pred = (risk_val >= t).astype(int)
        f1 = f1_score(y_val, pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    # -------- FINAL TEST (STRICTLY UNSEEN) -------- #
    X_test, y_test = load_test()
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLUMNS)

    lgbm_test = lgbm.predict_proba(X_test_s)[:, 1]
    if_raw_test = -iforest.score_samples(X_test_s)
    if_norm_test = np.clip(if_scaler.transform(if_raw_test.reshape(-1, 1)).flatten(), 0, 1)

    risk_test = platt.predict_proba((0.7 * lgbm_test + 0.3 * if_norm_test).reshape(-1, 1))[:, 1]
    trust_test = 1 - risk_test

    y_pred = (trust_test <= best_t).astype(int)

    # -------- METRICS -------- #
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n===== IEEE METRICS =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("Specificity:", tn / (tn + fp))
    print("FPR:", fp / (fp + tn))
    print("Balanced Acc:", balanced_accuracy_score(y_test, y_pred))
    print("MCC:", matthews_corrcoef(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, risk_test))
    print("PR-AUC:", average_precision_score(y_test, risk_test))
    print("Threshold:", best_t)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("========================\n")

if __name__ == "__main__":
    main()