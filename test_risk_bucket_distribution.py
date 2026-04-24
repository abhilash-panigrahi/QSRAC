#!/usr/bin/env python3
"""
stress_test_risk_scores.py

Generates 1000–5000 synthetic feature rows in the model's 8-feature space,
scores them using HybridRiskModel, and checks:
- spread
- clustering
- extreme outputs
- bucket occupancy

Aligned with ml_module.py (production logic).
"""

from __future__ import annotations

import argparse
import logging
import joblib

import numpy as np
import pandas as pd

from ml_module import FEATURE_COLUMNS, _map_risk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def generate_synthetic_inputs(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Mixture to cover normal, suspicious, and extreme demo cases
    regime = rng.choice([0, 1, 2], size=n, p=[0.55, 0.30, 0.15])

    hour = np.where(
        regime == 0,
        rng.integers(7, 19, size=n),
        rng.integers(0, 24, size=n),
    ).astype(float)

    request_rate = np.where(
        regime == 0,
        rng.lognormal(mean=1.5, sigma=0.5, size=n),
        np.where(
            regime == 1,
            rng.lognormal(mean=2.2, sigma=0.7, size=n),
            rng.lognormal(mean=2.8, sigma=0.9, size=n),
        ),
    )
    request_rate = _clip01(request_rate / np.percentile(request_rate, 99) * 10.0)

    failed_attempts = np.where(
        regime == 0,
        rng.poisson(0.2, size=n),
        np.where(
            regime == 1,
            rng.poisson(1.5, size=n),
            rng.poisson(4.0, size=n),
        ),
    ).astype(float)
    failed_attempts = _clip01(failed_attempts)

    geo_risk = np.where(
        regime == 0,
        rng.beta(1.5, 8.0, size=n) * 0.3,
        np.where(
            regime == 1,
            rng.beta(2.5, 4.5, size=n) * 0.7,
            rng.beta(4.0, 1.8, size=n),
        ),
    )
    geo_risk = _clip01(geo_risk)

    device_trust = np.where(
        regime == 0,
        rng.beta(8.0, 2.0, size=n),
        np.where(
            regime == 1,
            rng.beta(4.5, 3.5, size=n),
            rng.beta(1.8, 5.5, size=n),
        ),
    )
    device_trust = _clip01(device_trust)

    sensitivity = np.where(
        regime == 0,
        rng.choice([1, 2, 3], size=n, p=[0.35, 0.45, 0.20]),
        np.where(
            regime == 1,
            rng.choice([2, 3, 4], size=n, p=[0.20, 0.50, 0.30]),
            rng.choice([3, 4, 5], size=n, p=[0.20, 0.35, 0.45]),
        ),
    ).astype(float)

    is_vpn = np.where(
        regime == 2,
        rng.binomial(1, 0.35, size=n),
        rng.binomial(1, 0.12, size=n),
    ).astype(float)

    is_tor = np.where(
        regime == 2,
        rng.binomial(1, 0.15, size=n),
        rng.binomial(1, 0.02, size=n),
    ).astype(float)

    # Correlation adjustments
    device_trust = _clip01(device_trust - 0.06 * failed_attempts + 0.05 * (1 - is_vpn))
    geo_risk = _clip01(geo_risk + 0.08 * is_tor + 0.04 * is_vpn)

    df = pd.DataFrame(
        {
            "hour_of_day": hour,
            "request_rate": request_rate,
            "failed_attempts": failed_attempts,
            "geo_risk_score": geo_risk,
            "device_trust_score": device_trust,
            "sensitivity_level": sensitivity,
            "is_vpn": is_vpn,
            "is_tor": is_tor,
        }
    )

    return df


def summarize_scores(scores: np.ndarray) -> None:
    print("\n=== Score summary ===")
    print(f"min : {scores.min():.4f}")
    print(f"max : {scores.max():.4f}")
    print(f"mean: {scores.mean():.4f}")
    print(f"std : {scores.std():.4f}")

    labels = np.array([_map_risk(s) for s in scores])

    unique, counts = np.unique(labels, return_counts=True)

    print("\n=== Bucket distribution ===")
    for u, c in zip(unique, counts):
        print(f"{u:>9}: {c:>5} ({c / len(scores):.2%})")

    print("\n=== Extreme outputs ===")
    print(f"score <= 0.05 : {np.mean(scores <= 0.05):.2%}")
    print(f"score >= 0.95 : {np.mean(scores >= 0.95):.2%}")
    print(f"score <= 0.10 : {np.mean(scores <= 0.10):.2%}")
    print(f"score >= 0.90 : {np.mean(scores >= 0.90):.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.joblib", help="Saved HybridRiskModel artifact")
    parser.add_argument("--n", type=int, default=3000, help="Number of synthetic inputs (1000–5000)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    n = int(np.clip(args.n, 1000, 5000))

    df = generate_synthetic_inputs(n=n, seed=args.seed)
    print(f"Generated synthetic feature frame: shape={df.shape}")

    # Ensure correct feature order
    df = df[FEATURE_COLUMNS]

    if list(df.columns) != FEATURE_COLUMNS:
        raise ValueError("Feature order mismatch with model")

    # Load model
    model = joblib.load(args.model)

    if not hasattr(model, "predict_risk"):
        raise RuntimeError("Invalid model: missing predict_risk")

    # Predict
    scores = np.asarray(model.predict_risk(df), dtype=float)
    scores = np.clip(scores, 0.0, 1.0)

    summarize_scores(scores)


if __name__ == "__main__":
    main()