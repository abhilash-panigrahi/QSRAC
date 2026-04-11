"""
config.py — Centralised environment configuration for QSRAC.

Loaded once at import time.  Any missing or empty required variable raises
RuntimeError immediately so the process refuses to start rather than running
in an insecure half-configured state.

All other modules import their values from here instead of calling os.getenv
directly.  This gives a single, auditable list of every secret the system
touches.
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


# ── Required secrets — process will not start if any are absent ───────────────

SECRET_KEY: str = _require("SECRET_KEY")
DB_PASSWORD: str = _require("DB_PASSWORD")

# ── Optional with safe defaults ───────────────────────────────────────────────

REDIS_HOST: str = _optional("REDIS_HOST", "localhost")
REDIS_PORT: int = int(_optional("REDIS_PORT", "6379"))
REDIS_DB:   int = int(_optional("REDIS_DB",   "0"))

DB_USER: str = _optional("DB_USER", "postgres")
DB_HOST: str = _optional("DB_HOST", "localhost")
DB_PORT: str = _optional("DB_PORT", "5432")
DB_NAME: str = _optional("DB_NAME", "qsrac")

SESSION_TTL: int = int(_optional("SESSION_TTL", "3600"))
APP_HOST:    str = _optional("APP_HOST", "0.0.0.0")
APP_PORT:    int = int(_optional("APP_PORT", "8000"))

# ── Risk ML Thresholds ────────────────────────────────────────────────────────
# NOTE: thresholds must satisfy: LOW > MEDIUM > HIGH (scores are negative)
RISK_THRESHOLD_LOW: float    = float(_optional("RISK_THRESHOLD_LOW", "-0.4449"))
RISK_THRESHOLD_MEDIUM: float = float(_optional("RISK_THRESHOLD_MEDIUM", "-0.5381"))
RISK_THRESHOLD_HIGH: float   = float(_optional("RISK_THRESHOLD_HIGH", "-0.6313"))
if not (RISK_THRESHOLD_LOW > RISK_THRESHOLD_MEDIUM > RISK_THRESHOLD_HIGH):
    raise RuntimeError(
        "[QSRAC] Invalid risk thresholds: must satisfy LOW > MEDIUM > HIGH"
    )
