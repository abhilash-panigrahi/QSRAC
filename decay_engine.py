import math

ALPHA = 0.02
BETA = 0.08


def compute_trust(
    trust0: float,
    sensitivity: float,
    risk_trend: float,
    time_delta: float,
) -> float:
    decay_rate = (ALPHA * sensitivity) + (BETA * risk_trend)
    trust = trust0 * math.exp(-decay_rate * time_delta)
    return max(0.0, min(1.0, trust))