#!/usr/bin/env python3
"""
risk_thresholds.py

Shared utilities for:
- mapping UNSW-NB15 rows to the 8-feature model space
- scoring with the saved HybridRiskModel artifact
- analysing score distributions
- deriving robust 4-bucket thresholds
- mapping score -> Low / Medium / High / Critical

Design goals:
- no retraining
- no percentile-only thresholding
- stable for dataset and demo/random inputs
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

LOG = logging.getLogger("risk_thresholds")

FEATURE_COLUMNS = [
    "hour_of_day",
    "request_rate",
    "failed_attempts",
    "geo_risk_score",
    "device_trust_score",
    "sensitivity_level",
    "is_vpn",
    "is_tor",
]


# ---------------------------- Data mapping ----------------------------

def map_unsw_to_qsrac(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates the feature mapping used in the current trainer.
    This keeps the analysis and thresholding aligned with the deployed model.
    """
    mapped = pd.DataFrame(index=df.index)

    mapped["hour_of_day"] = 0.0

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


def load_unsw_csvs(paths: Sequence[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing CSV: {p}")
        frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True)


# ---------------------------- Model loading ----------------------------

def load_model(model_path: str):
    obj = joblib.load(model_path)
    if not hasattr(obj, "predict_risk"):
        raise TypeError(f"Loaded object from {model_path} does not expose predict_risk()")
    return obj


def score_dataframe(df: pd.DataFrame, model_path: str = "model.joblib") -> np.ndarray:
    model = load_model(model_path)
    X = map_unsw_to_qsrac(df)
    scores = model.predict_risk(X)
    scores = np.asarray(scores, dtype=float)
    return np.clip(scores, 0.0, 1.0)


# ---------------------------- Distribution stats ----------------------------

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def describe_scores(scores: Sequence[float], bins: int = 30) -> Dict[str, object]:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("No finite scores supplied.")

    hist_counts, hist_edges = np.histogram(s, bins=bins, range=(0.0, 1.0))
    q = np.quantile(s, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

    mu = float(np.mean(s))
    sigma = float(np.std(s, ddof=1)) if s.size > 1 else 0.0
    m2 = float(np.mean((s - mu) ** 2))
    if m2 == 0:
        skew = 0.0
        kurt = 0.0
    else:
        m3 = float(np.mean((s - mu) ** 3))
        m4 = float(np.mean((s - mu) ** 4))
        skew = m3 / (m2 ** 1.5)
        kurt = m4 / (m2 ** 2)  # Pearson kurtosis
    bc = ((skew ** 2) + 1.0) / kurt if kurt > 0 else float("inf")

    # Detect heavy edge concentration and exact-value clustering
    unique_vals, counts = np.unique(np.round(s, 10), return_counts=True)
    top_idx = np.argsort(counts)[::-1][:10]
    top_repeats = [
        {"value": float(unique_vals[i]), "count": int(counts[i]), "share": float(counts[i] / s.size)}
        for i in top_idx
    ]

    # Simple peak detection on smoothed histogram
    smoothed = np.convolve(hist_counts.astype(float), np.ones(3) / 3.0, mode="same")
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] >= smoothed[i - 1] and smoothed[i] > smoothed[i + 1] and smoothed[i] > 0:
            peaks.append(i)
    peak_info = []
    for i in peaks:
        left = max(0, i - 1)
        right = min(len(hist_counts) - 1, i + 1)
        peak_info.append(
            {
                "bin": int(i),
                "center": float((hist_edges[i] + hist_edges[i + 1]) / 2.0),
                "count": int(hist_counts[i]),
                "smoothed": float(smoothed[i]),
                "local_span": [int(left), int(right)],
            }
        )

    return {
        "n": int(s.size),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "mean": mu,
        "std": sigma,
        "unique_values": int(unique_vals.size),
        "quantiles": {
            "p10": _safe_float(q[0]),
            "p25": _safe_float(q[1]),
            "p50": _safe_float(q[2]),
            "p75": _safe_float(q[3]),
            "p90": _safe_float(q[4]),
            "p95": _safe_float(q[5]),
            "p99": _safe_float(q[6]),
        },
        "skewness": float(skew),
        "pearson_kurtosis": float(kurt),
        "bimodality_coefficient": float(bc),
        "histogram": {
            "edges": hist_edges.tolist(),
            "counts": hist_counts.astype(int).tolist(),
        },
        "peak_count": int(len(peak_info)),
        "peaks": peak_info,
        "top_repeated_scores": top_repeats,
        "edge_mass": {
            "<=0.05": float(np.mean(s <= 0.05)),
            ">=0.95": float(np.mean(s >= 0.95)),
            "<=0.10": float(np.mean(s <= 0.10)),
            ">=0.90": float(np.mean(s >= 0.90)),
        },
    }


def print_score_report(scores: Sequence[float], title: str = "Score distribution") -> Dict[str, object]:
    rep = describe_scores(scores)
    print(f"\n=== {title} ===")
    print(
        f"n={rep['n']}  min={rep['min']:.6f}  max={rep['max']:.6f}  "
        f"mean={rep['mean']:.6f}  std={rep['std']:.6f}  unique={rep['unique_values']}"
    )
    q = rep["quantiles"]
    print(
        "quantiles: "
        f"p10={q['p10']:.6f}  p25={q['p25']:.6f}  p50={q['p50']:.6f}  "
        f"p75={q['p75']:.6f}  p90={q['p90']:.6f}  p95={q['p95']:.6f}  p99={q['p99']:.6f}"
    )
    print(
        f"skewness={rep['skewness']:.4f}  "
        f"kurtosis={rep['pearson_kurtosis']:.4f}  "
        f"bimodality_coefficient={rep['bimodality_coefficient']:.4f}  "
        f"peaks={rep['peak_count']}"
    )
    print("edge mass:", rep["edge_mass"])
    print("top repeated score values:")
    for item in rep["top_repeated_scores"][:8]:
        print(f"  value={item['value']:.6f}  count={item['count']}  share={item['share']:.3%}")
    if rep["peaks"]:
        print("detected peaks:")
        for p in rep["peaks"][:8]:
            print(
                f"  bin={p['bin']:>3} center={p['center']:.6f} "
                f"count={p['count']} smoothed={p['smoothed']:.2f}"
            )
    print("histogram bin counts:")
    print(rep["histogram"]["counts"])
    return rep


# ---------------------------- Robust thresholding ----------------------------

@dataclass
class ThresholdResult:
    thresholds: Dict[str, float]
    method: str
    diagnostics: Dict[str, object]


def _weighted_jenks_thresholds(scores: np.ndarray, n_classes: int = 4, n_bins: int = 256) -> Tuple[List[float], Dict[str, object]]:
    """
    Weighted Fisher-Jenks on score histogram counts.
    This is data-driven, not percentile-based, and works well when scores are skewed or clustered.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("No finite scores supplied.")

    counts, edges = np.histogram(s, bins=n_bins, range=(0.0, 1.0))
    active = np.where(counts > 0)[0]
    if active.size < n_classes:
        raise ValueError("Not enough occupied histogram bins for Jenks segmentation.")

    centers = (edges[:-1] + edges[1:]) / 2.0
    x = centers[active]
    w = counts[active].astype(float)
    m = x.size

    # Prefix sums for weighted SSE over segments.
    pw = np.zeros(m + 1)
    ps = np.zeros(m + 1)
    pss = np.zeros(m + 1)
    pw[1:] = np.cumsum(w)
    ps[1:] = np.cumsum(w * x)
    pss[1:] = np.cumsum(w * x * x)

    def seg_sse(i: int, j: int) -> float:
        """1-based inclusive indices into x/w."""
        W = pw[j] - pw[i - 1]
        if W <= 0:
            return 0.0
        S = ps[j] - ps[i - 1]
        SS = pss[j] - pss[i - 1]
        return float(max(0.0, SS - (S * S) / W))

    dp = np.full((n_classes + 1, m + 1), np.inf, dtype=float)
    back = np.zeros((n_classes + 1, m + 1), dtype=int)

    # Base case: one class.
    for j in range(1, m + 1):
        dp[1, j] = seg_sse(1, j)
        back[1, j] = 0

    # Dynamic programming.
    for k in range(2, n_classes + 1):
        for j in range(k, m + 1):
            best_val = np.inf
            best_i = k - 1
            # i is the start index of the last segment.
            for i in range(k - 1, j):
                val = dp[k - 1, i] + seg_sse(i + 1, j)
                if val < best_val:
                    best_val = val
                    best_i = i
            dp[k, j] = best_val
            back[k, j] = best_i

    # Reconstruct segment boundaries.
    cuts = []
    j = m
    for k in range(n_classes, 1, -1):
        i = back[k, j]
        cuts.append(i)
        j = i
    cuts = sorted(cuts)

    # Convert segment boundaries to score thresholds.
    # boundary between active bins at positions i and i+1 (1-based active index).
    thresholds = []
    for cut in cuts:
        left_idx = cut - 1
        right_idx = cut
        if left_idx < 0:
            thr = float(x[0])
        elif right_idx >= m:
            thr = float(x[-1])
        else:
            thr = float((x[left_idx] + x[right_idx]) / 2.0)
        thresholds.append(float(np.clip(thr, 0.0, 1.0)))

    diagnostics = {
        "occupied_bins": int(m),
        "hist_bins": int(n_bins),
        "active_bin_centers": x.tolist(),
        "active_bin_weights": w.astype(int).tolist(),
        "segment_cost": float(dp[n_classes, m]),
        "raw_cuts": [int(c) for c in cuts],
    }
    return thresholds, diagnostics


def _largest_gap_thresholds(scores: np.ndarray, n_classes: int = 4, min_gap: float = 1e-6) -> Tuple[List[float], Dict[str, object]]:
    """
    Fallback: choose thresholds at the largest gaps between consecutive unique scores.
    Useful when distribution is extremely collapsed and Jenks becomes unstable.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    uniq = np.unique(np.round(s, 10))
    if uniq.size < n_classes:
        # Degenerate but we can still create monotone thresholds by spacing around the median.
        med = float(np.median(s))
        eps = max(1e-4, 0.01 * max(1e-6, float(np.std(s))))
        th = [max(0.0, med - eps), med, min(1.0, med + eps)]
        th = sorted(set(th))
        while len(th) < 3:
            th = sorted(set(th + [min(1.0, th[-1] + eps)]))
        return th[:3], {"method": "median_spread_fallback", "unique_values": int(uniq.size)}

    diffs = np.diff(uniq)
    order = np.argsort(diffs)[::-1]
    chosen = []
    for idx in order:
        if diffs[idx] < min_gap:
            break
        thr = float((uniq[idx] + uniq[idx + 1]) / 2.0)
        # Keep thresholds sufficiently separated.
        if all(abs(thr - c) > 1e-4 for c in chosen):
            chosen.append(thr)
        if len(chosen) == 3:
            break

    if len(chosen) < 3:
        # Fill remaining thresholds with evenly spaced cut points between min/max, but not as the primary strategy.
        lo, hi = float(np.min(s)), float(np.max(s))
        if hi <= lo:
            chosen = [lo, lo, lo]
        else:
            candidates = np.linspace(lo, hi, 5)[1:-1].tolist()  # 3 interior points
            for c in candidates:
                if len(chosen) >= 3:
                    break
                if all(abs(c - c0) > 1e-4 for c0 in chosen):
                    chosen.append(float(c))

    chosen = sorted(chosen)[:3]
    while len(chosen) < 3:
        chosen.append(chosen[-1] if chosen else 0.5)

    return chosen, {"method": "largest_gap_fallback", "unique_values": int(uniq.size)}


def derive_thresholds(
    scores: Sequence[float],
    n_classes: int = 4,
    min_bucket_fraction: float = 0.05,
) -> ThresholdResult:
    """
    Returns 3 thresholds for Low/Medium/High/Critical.

    Primary method:
      - weighted Jenks natural breaks on the score histogram

    Safeguards:
      - enforces monotonic thresholds
      - checks minimum bucket occupancy
      - falls back to largest-gap thresholds if needed
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 20:
        raise ValueError("Need at least 20 finite scores to derive stable thresholds.")

    report = describe_scores(s)

    method = "weighted_jenks_histogram"
    try:
        thr, extra = _weighted_jenks_thresholds(s, n_classes=n_classes)
        diagnostics = {"analysis": report, **extra}
    except Exception as e:
        method = "largest_gap_fallback"
        thr, extra = _largest_gap_thresholds(s, n_classes=n_classes)
        diagnostics = {"analysis": report, "jenks_error": str(e), **extra}

    thr = sorted(float(np.clip(t, 0.0, 1.0)) for t in thr[:3])

    # Repair non-monotone / too-close thresholds.
    min_gap = max(1e-4, 0.005 * max(1e-6, float(np.std(s))))
    for i in range(1, len(thr)):
        if thr[i] <= thr[i - 1] + min_gap:
            thr[i] = min(1.0, thr[i - 1] + min_gap)

    # Check bucket occupancy.
    def bucket_counts(t: Sequence[float]) -> List[int]:
        a, b, c = t
        return [
            int(np.sum(s < a)),
            int(np.sum((s >= a) & (s < b))),
            int(np.sum((s >= b) & (s < c))),
            int(np.sum(s >= c)),
        ]

    counts = bucket_counts(thr)
    diagnostics["bucket_counts"] = counts
    diagnostics["bucket_fractions"] = [c / s.size for c in counts]
    diagnostics["thresholds_initial"] = thr.copy()

    tiny = any(c < max(5, int(math.ceil(min_bucket_fraction * s.size))) for c in counts)
    collapsed = any((thr[i] - thr[i - 1]) < min_gap for i in range(1, 3))

    if tiny or collapsed:
        # Try the largest-gap fallback, which is often better for collapsed, clustered scores.
        fallback_thr, extra = _largest_gap_thresholds(s, n_classes=n_classes)
        fallback_thr = sorted(float(np.clip(t, 0.0, 1.0)) for t in fallback_thr)
        for i in range(1, len(fallback_thr)):
            if fallback_thr[i] <= fallback_thr[i - 1] + min_gap:
                fallback_thr[i] = min(1.0, fallback_thr[i - 1] + min_gap)
        fb_counts = bucket_counts(fallback_thr)
        diagnostics["fallback_bucket_counts"] = fb_counts
        diagnostics["fallback_bucket_fractions"] = [c / s.size for c in fb_counts]
        diagnostics["fallback_thresholds"] = fallback_thr.copy()

        # Prefer fallback only if it is strictly better wrt occupancy.
        if min(fb_counts) >= min(counts):
            thr = fallback_thr
            method = extra.get("method", "largest_gap_fallback")
            diagnostics["selected_fallback"] = True
        else:
            diagnostics["selected_fallback"] = False

    final_counts = bucket_counts(thr)
    diagnostics["final_bucket_counts"] = final_counts
    diagnostics["final_bucket_fractions"] = [c / s.size for c in final_counts]

    threshold_map = {
        "low_medium": float(thr[0]),
        "medium_high": float(thr[1]),
        "high_critical": float(thr[2]),
    }

    return ThresholdResult(
        thresholds=threshold_map,
        method=method,
        diagnostics=diagnostics,
    )


def classify_score(score: float, thresholds: Dict[str, float]) -> str:
    s = float(score)
    if not np.isfinite(s):
        return "Critical"

    s = float(np.clip(s, 0.0, 1.0))
    t1 = float(thresholds.get("low_medium", 0.25))
    t2 = float(thresholds.get("medium_high", 0.50))
    t3 = float(thresholds.get("high_critical", 0.75))

    # Enforce monotonic ordering defensively.
    t1, t2, t3 = sorted([t1, t2, t3])

    if s < t1:
        return "Low"
    if s < t2:
        return "Medium"
    if s < t3:
        return "High"
    return "Critical"


def save_thresholds(path: str, result: ThresholdResult) -> None:
    payload = {
        **result.thresholds,
        "method": result.method,
        "diagnostics": result.diagnostics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_thresholds(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and all(k in data for k in ("low_medium", "medium_high", "high_critical")):
        return {
            "low_medium": float(data["low_medium"]),
            "medium_high": float(data["medium_high"]),
            "high_critical": float(data["high_critical"]),
        }
    # backward compatible with older files
    if isinstance(data, dict) and all(k in data for k in ("low", "medium", "high")):
        return {
            "low_medium": float(data["low"]),
            "medium_high": float(data["medium"]),
            "high_critical": float(data["high"]),
        }
    raise ValueError(f"Unrecognized threshold file format: {path}")


def print_threshold_report(result: ThresholdResult) -> None:
    t = result.thresholds
    print("\n=== Robust Thresholds ===")
    print(f"method: {result.method}")
    print(f"low / medium   = {t['low_medium']:.6f}")
    print(f"medium / high  = {t['medium_high']:.6f}")
    print(f"high / critical= {t['high_critical']:.6f}")
    print("bucket counts  =", result.diagnostics.get("final_bucket_counts"))
    print("bucket fractions=", [f"{x:.3%}" for x in result.diagnostics.get("final_bucket_fractions", [])])


# ---------------------------- CLI helpers ----------------------------

def _load_scores_from_csvs(csv_paths: Sequence[str], model_path: str) -> np.ndarray:
    df = load_unsw_csvs(csv_paths)
    return score_dataframe(df, model_path=model_path)


def main():
    parser = argparse.ArgumentParser(description="Derive robust score thresholds from UNSW-NB15 datasets.")
    parser.add_argument("--model", default="model.joblib", help="Path to saved HybridRiskModel artifact")
    parser.add_argument("--csv", nargs="+", required=True, help="CSV file(s) to score")
    parser.add_argument("--save", default="thresholds.json", help="Where to write threshold JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    scores = _load_scores_from_csvs(args.csv, args.model)
    print_score_report(scores, title="Observed model scores")
    result = derive_thresholds(scores)
    print_threshold_report(result)
    save_thresholds(args.save, result)
    print(f"\nSaved thresholds to: {args.save}")


if __name__ == "__main__":
    main()
