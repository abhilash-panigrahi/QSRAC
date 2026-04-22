"""
config.py — Centralised environment configuration for QSRAC.

Loaded once at import time. Any missing or empty required variable raises
RuntimeError immediately so the process refuses to start rather than running
in an insecure half-configured state.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def _require(name: str) -> str:
    """Return env var value or raise RuntimeError at startup."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"[QSRAC] Required environment variable '{name}' is missing or empty. "
            f"Set it in .env or the process environment before starting."
        )
    return value

def _optional(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


# ── 1. Required Secrets ───────────────────────────────────────────────────────
# Process will not start if any are absent

SECRET_KEY: str = _require("SECRET_KEY")
SIGNING_PRIVATE_KEY: str = _require("SIGNING_PRIVATE_KEY")
SIGNING_PUBLIC_KEY: str = _require("SIGNING_PUBLIC_KEY")
EXCHANGE_PRIVATE_KEY: str = _require("EXCHANGE_PRIVATE_KEY")
EXCHANGE_PUBLIC_KEY: str = _require("EXCHANGE_PUBLIC_KEY")


# ── 2. Infrastructure & Networking ───────────────────────────────────────────

REDIS_HOST: str = _optional("REDIS_HOST", "localhost")
REDIS_PORT: int = int(_optional("REDIS_PORT", "6379"))
REDIS_DB:   int = int(_optional("REDIS_DB",   "0"))

APP_HOST: str = _optional("APP_HOST", "0.0.0.0")
APP_PORT: int = int(_optional("APP_PORT", "8000"))


# ── 3. QSRAC Protocol Constants ──────────────────────────────────────────────

SESSION_TTL: int = int(_optional("SESSION_TTL", "3600"))

# Multiplier to convert floats to stable integers for deterministic hashing
PRECISION_MULTIPLIER: int = 1_000_000 

# Centralized audit log path
AUDIT_LOG_PATH: str = _optional("AUDIT_LOG_PATH", "audit.log")

# Protocol Metadata for middleware validation
REQUIRED_CONTEXT_KEYS = [
    "hour_of_day", "request_rate", "failed_attempts", "geo_risk_score",
    "device_trust_score", "sensitivity_level", "is_vpn", "is_tor"
]


# ── 4. Risk ML Thresholds ─────────────────────────────────────────────────────

USE_DYNAMIC_THRESHOLDS: bool = _optional("USE_DYNAMIC_THRESHOLDS", "True").lower() == "true"

# Thresholds represent percentile-based risk cutoffs derived from training data.
# low    → ≥ 70th percentile (safe behavior)
# medium → ≥ 50th percentile
# high   → ≥ 30th percentile
# below  → critical risk

# NOTE: thresholds must satisfy: LOW > MEDIUM > HIGH
RISK_THRESHOLD_LOW: float    = float(_optional("RISK_THRESHOLD_LOW", "0.7"))
RISK_THRESHOLD_MEDIUM: float = float(_optional("RISK_THRESHOLD_MEDIUM", "0.5"))
RISK_THRESHOLD_HIGH: float   = float(_optional("RISK_THRESHOLD_HIGH", "0.3"))

# Clamp thresholds into valid probability range [0.0, 1.0]
RISK_THRESHOLD_LOW = max(min(RISK_THRESHOLD_LOW, 1.0), 0.0)
RISK_THRESHOLD_MEDIUM = max(min(RISK_THRESHOLD_MEDIUM, 1.0), 0.0)
RISK_THRESHOLD_HIGH = max(min(RISK_THRESHOLD_HIGH, 1.0), 0.0)

# Fail-fast security gate
if not (RISK_THRESHOLD_LOW > RISK_THRESHOLD_MEDIUM > RISK_THRESHOLD_HIGH):
    raise RuntimeError(
        "[QSRAC] Invalid risk thresholds: must satisfy LOW > MEDIUM > HIGH"
    )