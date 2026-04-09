import time
import asyncio
import json
import logging
from time import perf_counter
from urllib import request, response

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ml_module import get_risk_score
from envelope import generate_envelope
from redis_lua import validate_and_update, get_session, get_redis_client
from decay_engine import compute_trust
from crypto_provider import verify as verify_token
from audit import log_event_async

log = logging.getLogger(__name__)

SKIP_PATHS = {
    "/health",
    "/login",
    "/docs",
    "/openapi.json",
    "/docs/oauth2-redirect",
    "/mfa/challenge",
    "/mfa/verify"
}

_server_signing_public_key = None


def set_signing_public_key(public_key):
    global _server_signing_public_key
    _server_signing_public_key = public_key


class QSRACMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in SKIP_PATHS:
            return await call_next(request)

        
        # 0. Core Token Signature Verification
        # ONLY Redis interaction permitted for authentication.
        # session_data is used exclusively here and NOT passed to Steps 3-4.
        try:
            session_id = request.headers.get("X-Session-ID")
            if not session_id:
                return JSONResponse(status_code=401, content={"error": "Missing X-Session-ID"})

            core_token_raw = request.headers.get("X-Core-Token")
            if not core_token_raw:
                return JSONResponse(status_code=401, content={"error": "Missing X-Core-Token"})

            session_data = get_session(session_id)
            token_signature = bytes.fromhex(session_data.get("token_signature"))
            if not token_signature:
                return JSONResponse(status_code=401, content={"error": "Missing token signature in session"})

            core_token_bytes = core_token_raw.encode("utf-8")
            if not verify_token(token_signature, core_token_bytes, _server_signing_public_key):
                return JSONResponse(status_code=401, content={"error": "Core token signature invalid"})

        except ValueError as e:
            return JSONResponse(status_code=401, content={"error": str(e)})
        except ConnectionError:
            return JSONResponse(status_code=503, content={"error": "Redis unavailable during token verification"})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Token verification failed: {str(e)}"})

        # 1. Extract context
        try:
            context = {
                "hour_of_day": float(request.headers.get("X-Hour-Of-Day", 12)),
                "request_rate": float(request.headers.get("X-Request-Rate", 1.0)),
                "failed_attempts": float(request.headers.get("X-Failed-Attempts", 0)),
                "geo_risk_score": float(request.headers.get("X-Geo-Risk-Score", 0.0)),
                "device_trust_score": float(request.headers.get("X-Device-Trust-Score", 1.0)),
                "sensitivity_level": float(request.headers.get("X-Sensitivity-Level", 1.0)),
                "is_vpn": float(request.headers.get("X-Is-VPN", 0)),
                "is_tor": float(request.headers.get("X-Is-TOR", 0)),
            }

            sensitivity = context["sensitivity_level"]
            time_delta = float(request.headers.get("X-Time-Delta", 1.0))
            persisted_trust = session_data.get("trust")
            trust0 = float(persisted_trust) if persisted_trust is not None else float(request.headers.get("X-Trust-Init", 1.0))

        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Context extraction failed: {str(e)}"})
        
        # 2. Compute risk
        try:
            risk_level = get_risk_score(context)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Risk computation failed: {str(e)}"})

        # 3 + 4. Envelope generation AND Redis Lua gate — single try block.
        # get_session() is called fresh here, independent of Step 0.
        # On ConnectionError the entire block is skipped — no partial state.
        degraded = False
        envelope_hash = None
        next_seq = None

        try:
            # Fresh Redis read scoped exclusively to Fast Path state
            fast_path_session = get_session(session_id)
            # 🔥 Read client state
            try:
                client_seq = int(request.headers.get("X-QSRAC-Seq", -1))
            except ValueError:
                return JSONResponse(status_code=400, content={
                "error": "Invalid sequence header"
                })
            client_env = request.headers.get("X-QSRAC-Envelope")

            if client_seq == -1 or client_env is None:
                return JSONResponse(status_code=400, content={
                    "error": "Missing QSRAC state headers"
                })
            # Server state
            server_seq = int(fast_path_session["seq"])
            server_prev_hash = fast_path_session["last_hash_1"]
            # Replay protection
            if client_seq != server_seq:
                return JSONResponse(status_code=403, content={
                    "error": "Replay detected (sequence mismatch)"
                })

            if client_env != server_prev_hash:
                return JSONResponse(status_code=403, content={
                    "error": "Replay detected (hash mismatch)"
                })

            core_token_hash = fast_path_session["core_token_hash"]
            session_key_hex = fast_path_session["session_key"]
            prev_hash = fast_path_session["last_hash_1"]
            seq = int(fast_path_session["seq"])
            session_key = bytes.fromhex(session_key_hex)

            risk_trend_map = {"Low": 0.0, "Medium": 0.3, "High": 0.6, "Critical": 1.0}
            risk_trend = risk_trend_map.get(risk_level, 0.5)

            trust_value = compute_trust(
                trust0=trust0,
                sensitivity=sensitivity,
                risk_trend=risk_trend,
                time_delta=time_delta,
            )
            
            next_seq = seq + 1

            envelope_hash, raw_envelope_bytes = generate_envelope(
                session_key=session_key,
                core_token_hash=core_token_hash,
                risk=risk_level,
                context=context,
                prev_hash=prev_hash,
                trust=trust_value,
                seq=next_seq,
            )

            result = validate_and_update(
                session_id=session_id,
                seq=next_seq,
                new_hash=envelope_hash,
                prev_hash=prev_hash,
            )
            if result != "OK":
                return JSONResponse(status_code=403, content={"error": f"Gate rejected: {result}"})


        except ValueError as e:
            return JSONResponse(status_code=403, content={"error": str(e)})
        except ConnectionError:
            if sensitivity >= 3:
                return JSONResponse(status_code=503, content={"error": "Redis unavailable — fail closed"})
            # Degraded mode: all Fast Path Redis state skipped, no seq increment
            degraded = True

        
        # 5. Persist trust (already computed earlier)
        try:
            rc = get_redis_client()
            rc.hset(f"session:{session_id}", "trust", trust_value)
        except Exception as e:
            log.warning("Failed to persist trust [%s]: %s", session_id, e)
        
        
        # 6. Return response
        request.state.context = context
        request.state.risk = risk_level
        request.state.trust = trust_value
        response = await call_next(request)
        response.headers["X-QSRAC-Risk"] = risk_level
        response.headers["X-QSRAC-Trust"] = str(round(trust_value, 4))
        response.headers["X-QSRAC-Mode"] = "DEGRADED" if degraded else "NORMAL"
        if not degraded:
            response.headers["X-QSRAC-Seq"] = str(next_seq)
            response.headers["X-QSRAC-Envelope"] = envelope_hash

        return response
