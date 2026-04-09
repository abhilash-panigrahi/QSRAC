"""
experiment_harness.py — Synthetic traffic experiment for QSRAC ML risk scoring.

Generates 800 normal + 200 attack samples, scores them using the trained
IsolationForest model, and reports detection metrics.

Usage:
    python experiment_harness.py
"""

import os
import json
import numpy as np
import joblib
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
    Mirrors generate_data.py generate_normal_samples().
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
    Mirrors generate_data.py generate_anomalous_samples().
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

def load_model(path: str):
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        model  = artifact["model"]
        scaler = artifact.get("scaler", None)
    else:
        model  = artifact
        scaler = None
    return model, scaler


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_samples(model, scaler, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    X_input = scaler.transform(X) if scaler is not None else X
    raw_scores = model.score_samples(X_input)
    # Using the imported mapping from ml_module
    risk_labels = [_map_score_to_risk(s) for s in raw_scores]
    return raw_scores, risk_labels


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    normal_risks: list[str],
    attack_risks: list[str],
) -> dict:
    risk_levels = ["Low", "Medium", "High", "Critical"]

    # Overall distribution across all samples
    all_risks = normal_risks + attack_risks
    distribution = {lvl: all_risks.count(lvl) for lvl in risk_levels}

    # False positives: normal samples flagged as High or Critical
    fp_count = sum(1 for r in normal_risks if r in {"High", "Critical"})
    fp_rate  = fp_count / len(normal_risks) if normal_risks else 0.0

    # False negatives: attack samples classified as Low or Medium
    fn_count = sum(1 for r in attack_risks if r in {"Low", "Medium"})
    fn_rate  = fn_count / len(attack_risks) if attack_risks else 0.0

    # Per-label breakdown for normal and attack cohorts
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
    print("  QSRAC — ML Risk Scoring Experiment Report")
    print_separator("═")

    # ── Score distribution ─────────────────────────────────────────────────────
    print("\n📊 SCORE DISTRIBUTION (all samples)")
    print_separator()
    print(f"  Total samples : {len(all_scores)}")
    print(f"  Min score     : {all_scores.min():.6f}")
    print(f"  Max score     : {all_scores.max():.6f}")
    print(f"  Mean score    : {all_scores.mean():.6f}")
    print(f"  Std  score    : {all_scores.std():.6f}")

    print("\n  Normal cohort  ({} samples)".format(len(normal_scores)))
    print(f"    min={normal_scores.min():.6f}  max={normal_scores.max():.6f}  "
          f"mean={normal_scores.mean():.6f}")

    print("\n  Attack cohort  ({} samples)".format(len(attack_scores)))
    print(f"    min={attack_scores.min():.6f}  max={attack_scores.max():.6f}  "
          f"mean={attack_scores.mean():.6f}")

    # ── Risk distribution ──────────────────────────────────────────────────────
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

    # ── Detection quality ──────────────────────────────────────────────────────
    print("\n🔍 DETECTION QUALITY")
    print_separator()
    print(f"  False Positives  (normal → High/Critical) : "
          f"{metrics['fp_count']:>4}  /  {N_NORMAL}  "
          f"→  FP rate = {metrics['fp_rate']*100:.2f}%")
    print(f"  False Negatives  (attack → Low/Medium)    : "
          f"{metrics['fn_count']:>4}  /  {N_ATTACK}  "
          f"→  FN rate = {metrics['fn_rate']*100:.2f}%")

    # ── Quick verdict ──────────────────────────────────────────────────────────
    print("\n✅ VERDICT")
    print_separator()
    fp_ok = metrics['fp_rate'] < 0.10
    fn_ok = metrics['fn_rate'] < 0.20
    print(f"  FP rate < 10% : {'PASS ✓' if fp_ok else 'FAIL ✗'}")
    print(f"  FN rate < 20% : {'PASS ✓' if fn_ok else 'FAIL ✗'}")

    if fp_ok and fn_ok:
        print("\n  Model thresholds are production-ready.")
    else:
        print("\n  ⚠  Thresholds need recalibration before publication.")
        if not fp_ok:
            print("     → Too many normal requests flagged; raise the Low boundary.")
        if not fn_ok:
            print("     → Too many attacks missed; lower the Medium/High boundary.")

    print_separator("═")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model, scaler = load_model(MODEL_PATH)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{MODEL_PATH}'.")
        print("Run train_model.py first (after generate_data.py) to produce it.")
        raise SystemExit(1)

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Generating {N_NORMAL} normal samples and {N_ATTACK} attack samples …")
    X_normal = generate_normal_samples(N_NORMAL, rng)
    X_attack = generate_attack_samples(N_ATTACK, rng)

    normal_scores, normal_risks = score_samples(model, scaler, X_normal)
    attack_scores, attack_risks = score_samples(model, scaler, X_attack)
    low_thresh = np.percentile(normal_scores, 70)
    high_thresh = np.percentile(attack_scores, 30)
    medium_thresh = (low_thresh + high_thresh) / 2
    # ── Persist calibrated thresholds ──
    thresholds = {
        "low": float(low_thresh),
        "medium": float(medium_thresh),
        "high": float(high_thresh)
    }

    with open("thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    print("\n✅ thresholds.json saved")

    print("CALIBRATED THRESHOLDS:")
    print(f"LOW     >= {low_thresh:.4f}")
    print(f"MEDIUM  >= {medium_thresh:.4f}")
    print(f"HIGH    >= {high_thresh:.4f}")
    print(f"CRITICAL <  {high_thresh:.4f}")
    all_scores = np.concatenate([normal_scores, attack_scores])
    mean_score = float(all_scores.mean())
    std_score  = float(all_scores.std())   

    metrics = compute_metrics(normal_risks, attack_risks)

    print_report(normal_scores, attack_scores, metrics)

    # ── Export Results to JSON ─────────────────────────────────────────────────

    print("\n📉 DRIFT CHECK")
    print(f"Mean: {mean_score:.6f}")
    print(f"Std : {std_score:.6f}")
    
    results_dict = {
        "thresholds": thresholds,
        "score_stats": {
            "min": float(all_scores.min()),
            "max": float(all_scores.max()),
            "mean": float(all_scores.mean()),
            "std": float(all_scores.std())
        },
        "normal_mean": float(normal_scores.mean()),
        "attack_mean": float(attack_scores.mean()),
        "fp_rate": float(metrics["fp_rate"]),
        "fn_rate": float(metrics["fn_rate"]),
        "risk_distribution": metrics["distribution"]
    }

    with open("results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    # ── Clean Summary Table ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  EXECUTIVE SUMMARY TABLE")
    print("═" * 60)
    print(f"  False Positive Rate (FP %) : {metrics['fp_rate'] * 100:.2f}%")
    print(f"  False Negative Rate (FN %) : {metrics['fn_rate'] * 100:.2f}%")
    print(f"  Normal Mean Score          : {normal_scores.mean():.6f}")
    print(f"  Attack Mean Score          : {attack_scores.mean():.6f}")
    print("═" * 60)
    print("  Results structured and saved to results.json\n")


if __name__ == "__main__":
    main()