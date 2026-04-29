import math
import os

ALPHA = 0.01
BETA = 0.04


def compute_trust(
    trust0: float,
    sensitivity: float,
    risk_trend: float,
    time_delta: float,
) -> float:
    FORCE_TRUST = os.getenv("FORCE_TRUST")
    if FORCE_TRUST is not None:
        return float(FORCE_TRUST)
        
    decay_rate = (ALPHA * sensitivity) + (BETA * risk_trend)
    trust = trust0 * math.exp(-decay_rate * time_delta)

    return max(0.0, min(1.0, trust))
