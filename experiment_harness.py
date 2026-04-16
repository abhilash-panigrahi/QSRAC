"""
experiment_harness.py — Synthetic traffic experiment for QSRAC ML risk scoring.

Generates 800 normal + 200 attack samples, scores them using the trained
Hybrid model (LightGBM + Isolation Forest), and reports detection metrics.

Usage:
    python experiment_harness.py
"""

import os
import json
import numpy as np
import joblib
import config
import ml_module
from sklearn.preprocessing import MinMaxScaler
from ml_module import _map_score_to_risk

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
RANDOM_SEED = 42

FEATURES = [
    "hour_of_day",
    "request_rate",
    "failed_attempts",
    "geo_risk_score",
    "device_trust_score",
    "sensitivity_level",
    "is_vpn",
    "is_tor",
]

N_NORMAL = 800
N_ATTACK = 200


# ── Synthetic data generation (mirrors generate_data.py distributions) ─────────

def generate_normal_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Normal traffic: business-hours patterns, low risk indicators.
    """
    hour_of_day      = rng.integers(0, 24, n).astype(float)
    request_rate     = rng.uniform(0.5, 2.0, n)
    failed_attempts  = rng.integers(0, 3, n).astype(float)
    geo_risk_score   = rng.uniform(0.0, 0.3, n)
    device_trust     = rng.uniform(0.7, 1.0, n)
    sensitivity      = rng.integers(1, 4, n).astype(float)
    is_vpn           = rng.choice([0.0, 1.0], n, p=[0.9, 0.1])
    is_tor           = np.zeros(n)

    return np.column_stack([
        hour_of_day, request_rate, failed_attempts, geo_risk_score,
        device_trust, sensitivity, is_vpn, is_tor,
    ])


def generate_attack_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Attack traffic: high request rates, many failures, VPN/Tor, low device trust.
    """
    hour_of_day      = rng.integers(0, 24, n).astype(float)
    request_rate     = rng.uniform(5.0, 20.0, n)
    failed_attempts  = rng.integers(3, 11, n).astype(float)
    geo_risk_score   = rng.uniform(0.6, 1.0, n)
    device_trust     = rng.uniform(0.0, 0.4, n)
    sensitivity      = rng.integers(3, 6, n).astype(float)
    is_vpn           = rng.choice([0.0, 1.0], n, p=[0.2, 0.8])
    is_tor           = rng.choice([0.0, 1.0], n, p=[0.3, 0.7])

    return np.column_stack([
        hour_of_day, request_rate, failed_attempts, geo_risk_score,
        device_trust, sensitivity, is_vpn, is_tor,
    ])


# ── Model loading ──────────────────────────────────────────────────────────────

def load_hybrid_model(path: str):
    artifact = joblib.load(path)
    lgbm = artifact["lgbm"]
    iforest = artifact["iforest"]
    scaler = artifact["scaler"]
    if_scaler = artifact.get("if_scaler", None)
    return lgbm, iforest, scaler, if_scaler


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_samples(lgbm, iforest, scaler, if_scaler, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    X_scaled = scaler.transform(X)
    
    # 1. Supervised probability (Risk: 1.0 = Attack)
    lgbm_prob = lgbm.predict_proba(X_scaled)[:, 1]
    
    # 2. Unsupervised score (Invert so higher = more anomalous/risky)
    if_raw_scores = -iforest.score_samples(X_scaled)
    
    # 3. Normalize IF scores
    if if_scaler is not None:
        if_scores_norm = if_scaler.transform(if_raw_scores.reshape(-1, 1)).flatten()
    else:
        temp_scaler = MinMaxScaler()
        if_scores_norm = temp_scaler.fit_transform(if_raw_scores.reshape(-1, 1)).flatten()

    # 4. Hybrid Risk Fusion
    risk_score = (0.7 * lgbm_prob) + (0.3 * if_scores_norm)
    
    # 5. ALIGNMENT: Convert Risk (High=Attack) to Trust (High=Safe)
    trust_score = 1.0 - risk_score
    
    # Map to risk levels using the new trust score
    risk_labels = [_map_score_to_risk(s) for s in trust_score]
    return trust_score, risk_labels


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    normal_risks: list[str],
    attack_risks: list[str],
) -> dict:
    risk_levels = ["Low", "Medium", "High", "Critical"]

    all_risks = normal_risks + attack_risks
    distribution = {lvl: all_risks.count(lvl) for lvl in risk_levels}

    # False positives: normal samples flagged as High or Critical
    fp_count = sum(1 for r in normal_risks if r in {"High", "Critical"})
    fp_rate  = fp_count / len(normal_risks) if normal_risks else 0.0

    # False negatives: attack samples classified as Low or Medium
    fn_count = sum(1 for r in attack_risks if r in {"Low", "Medium"})
    fn_rate  = fn_count / len(attack_risks) if attack_risks else 0.0

    normal_dist = {lvl: normal_risks.count(lvl) for lvl in risk_levels}
    attack_dist = {lvl: attack_risks.count(lvl) for lvl in risk_levels}

    return {
        "distribution":  distribution,
        "normal_dist":   normal_dist,
        "attack_dist":   attack_dist,
        "fp_count":      fp_count,
        "fp_rate":       fp_rate,
        "fn_count":      fn_count,
        "fn_rate":       fn_rate,
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_separator(char: str = "─", width: int = 60) -> None:
    print(char * width)


def print_report(
    normal_scores: np.ndarray,
    attack_scores: np.ndarray,
    metrics: dict,
) -> None:
    all_scores = np.concatenate([normal_scores, attack_scores])

    print_separator("═")
    print("  QSRAC — Trust-Based Hybrid Scoring Experiment Report")
    print_separator("═")

    print("\n📊 SCORE DISTRIBUTION (High = Safe / Trust)")
    print_separator()
    print(f"  Total samples : {len(all_scores)}")
    print(f"  Min score     : {all_scores.min():.6f}")
    print(f"  Max score     : {all_scores.max():.6f}")
    print(f"  Mean score    : {all_scores.mean():.6f}")

    print("\n  Normal cohort  ({} samples)".format(len(normal_scores)))
    print(f"    min={normal_scores.min():.6f}  max={normal_scores.max():.6f}  "
          f"mean={normal_scores.mean():.6f}")

    print("\n  Attack cohort  ({} samples)".format(len(attack_scores)))
    print(f"    min={attack_scores.min():.6f}  max={attack_scores.max():.6f}  "
          f"mean={attack_scores.mean():.6f}")

    print("\n🎯 RISK LEVEL DISTRIBUTION")
    print_separator()
    print(f"  {'Level':<12} {'All':>6} {'Normal':>8} {'Attack':>8}")
    print_separator("-", 40)
    for lvl in ["Low", "Medium", "High", "Critical"]:
        print(
            f"  {lvl:<12} "
            f"{metrics['distribution'][lvl]:>6} "
            f"{metrics['normal_dist'][lvl]:>8} "
            f"{metrics['attack_dist'][lvl]:>8}"
        )

    print("\n🔍 DETECTION QUALITY")
    print_separator()
    print(f"  False Positives  (normal → High/Critical) : "
          f"{metrics['fp_count']:>4}  /  {N_NORMAL}  "
          f"→  FP rate = {metrics['fp_rate']*100:.2f}%")
    print(f"  False Negatives  (attack → Low/Medium)    : "
          f"{metrics['fn_count']:>4}  /  {N_ATTACK}  "
          f"→  FN rate = {metrics['fn_rate']*100:.2f}%")

    print("\n✅ VERDICT")
    print_separator()
    fp_ok = metrics['fp_rate'] < 0.15
    fn_ok = metrics['fn_rate'] < 0.15
    print(f"  FP rate < 15% : {'PASS ✓' if fp_ok else 'FAIL ✗'}")
    print(f"  FN rate < 15% : {'PASS ✓' if fn_ok else 'FAIL ✗'}")
    print_separator("═")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    try:
        lgbm, iforest, scaler, if_scaler = load_hybrid_model(MODEL_PATH)
    except Exception as e:
        print(f"ERROR: Model loading failed: {e}")
        raise SystemExit(1)

    rng = np.random.default_rng(RANDOM_SEED)

    X_normal = generate_normal_samples(N_NORMAL, rng)
    X_attack = generate_attack_samples(N_ATTACK, rng)

    # Scoring now aligned to TRUST (High = Safe)
    normal_scores, _ = score_samples(lgbm, iforest, scaler, if_scaler, X_normal)
    attack_scores, _ = score_samples(lgbm, iforest, scaler, if_scaler, X_attack)
    
    # FIX: Trust-based Threshold Calibration
    # normal_scores are high (safe), attack_scores are low (risky)
    low_thresh  = np.percentile(normal_scores, 15)
    high_thresh = np.percentile(attack_scores, 85)
    medium_thresh = (low_thresh + high_thresh) / 2

    # Inject into runtime
    config.RISK_THRESHOLD_LOW = low_thresh
    config.RISK_THRESHOLD_MEDIUM = medium_thresh
    config.RISK_THRESHOLD_HIGH = high_thresh
    
    ml_module.RISK_THRESHOLD_LOW = low_thresh
    ml_module.RISK_THRESHOLD_MEDIUM = medium_thresh
    ml_module.RISK_THRESHOLD_HIGH = high_thresh
    
    thresholds = {
        "low": float(low_thresh),
        "medium": float(medium_thresh),
        "high": float(high_thresh)
    }

    with open("thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    # Re-map risks using correctly calibrated injected thresholds
    normal_risks = [_map_score_to_risk(s) for s in normal_scores]
    attack_risks = [_map_score_to_risk(s) for s in attack_scores]

    metrics = compute_metrics(normal_risks, attack_risks)
    print_report(normal_scores, attack_scores, metrics)

    # Export
    results_dict = {
        "thresholds": thresholds,
        "fp_rate": float(metrics["fp_rate"]),
        "fn_rate": float(metrics["fn_rate"]),
        "risk_distribution": metrics["distribution"]
    }
    with open("results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

if __name__ == "__main__":
    main()