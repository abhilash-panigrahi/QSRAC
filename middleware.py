import time
import asyncio
import json
import logging
from time import perf_counter

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ml_module import get_risk_score
from envelope import generate_envelope
from redis_lua import validate_and_update, get_session, get_redis_client
from decay_engine import compute_trust
from policy_engine import evaluate_policy_full
from crypto_provider import verify_token
from audit import log_event_async
from role_module import validate_role
from attribute_validator import validate_attributes
from rate_limiter import check_rate_limit

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

        # Latency clock — monotonic, sub-microsecond resolution.
        # Captured before step 0 so total middleware cost is measured.
        _t_start = perf_counter()

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
            token_signature = session_data.get("token_signature")
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

        # 0b. Rate limiting — after identity is confirmed, before any state mutation.
        # Uses key namespace  rl:{session_id}:general — never overlaps gate namespace.
        # Non-blocking: ConnectionError inside check_rate_limit → fail-open.
        if not check_rate_limit(session_id, bucket="general"):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"},
                headers={"Retry-After": "60"},
            )

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

        # 1b. RBAC + ABAC clearance (after context, before risk + gate)
        try:
            core_token_dict = json.loads(core_token_raw)
            role_clearance = validate_role(core_token_dict)
            abac_clearance = validate_attributes(context)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"RBAC/ABAC evaluation failed: {str(e)}"})

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
            error_msg = str(e)
            if "SESSION_NOT_FOUND" in error_msg:
                return JSONResponse(status_code=401, content={"error": "Session not found"})
            elif "REPLAY_DETECTED" in error_msg:
                return JSONResponse(status_code=403, content={"error": "Replay detected"})
            elif "CHAIN_BROKEN" in error_msg:
                return JSONResponse(status_code=403, content={"error": "Hash chain broken"})
            return JSONResponse(status_code=403, content={"error": error_msg})
        except ConnectionError:
            if sensitivity >= 3:
                return JSONResponse(status_code=503, content={"error": "Redis unavailable — fail closed"})
            # Degraded mode: all Fast Path Redis state skipped, no seq increment
            degraded = True

        # Anomaly timestamp — only in normal mode.
        # Failure is non-fatal (best-effort telemetry) but must be logged.
        if risk_level in {"High", "Critical"} and not degraded:
            try:
                rc = get_redis_client()
                rc.hsetnx(f"session:{session_id}", "first_anomaly_timestamp", time.time())
            except Exception as e:
                log.warning("Failed to record anomaly timestamp [%s]: %s", session_id, e)

        # 5. Compute trust
        try:
            
            # Persist updated trust (ONLY after successful gate)
            try:
                rc = get_redis_client()
                rc.hset(f"session:{session_id}", "trust", trust_value)
            except Exception as e:
                log.warning("Failed to persist trust [%s]: %s", session_id, e)            
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Trust computation failed: {str(e)}"})

        # 6. Apply policy (RBAC + ABAC + Risk + Trust)
        try:
            decision = evaluate_policy_full(
                risk_level=risk_level,
                trust_value=trust_value,
                role_clearance=role_clearance,
                abac_clearance=abac_clearance,
            )
            
            # Block timestamp — non-fatal telemetry, must be logged on failure.
            if decision == "Block" and not degraded:
                try:
                    rc = get_redis_client()
                    rc.hset(f"session:{session_id}", "block_timestamp", time.time())
                except Exception as e:
                    log.warning("Failed to record block timestamp [%s]: %s", session_id, e)

            # Latency measured to end of policy evaluation — excludes upstream
            # handler time intentionally (Fast Path cost only).
            _latency_ms = (perf_counter() - _t_start) * 1000.0
            _event_type = "DEGRADED" if degraded else "NORMAL"

            asyncio.create_task(
                log_event_async(
                    session_id,
                    decision,
                    risk_level,
                    trust_value,
                    event_type=_event_type,
                    latency_ms=_latency_ms,
                )
            )

        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Policy evaluation failed: {str(e)}"})

        if decision == "Block":
            return JSONResponse(status_code=403, content={"error": "Access blocked by policy", "decision": "Block"})
        if decision == "Deny":
            return JSONResponse(status_code=403, content={"error": "Access denied by policy", "decision": "Deny"})

        # 7. Return response
        response = await call_next(request)
        response.headers["X-QSRAC-Decision"] = decision
        response.headers["X-QSRAC-Risk"] = risk_level
        response.headers["X-QSRAC-Trust"] = str(round(trust_value, 4))
        response.headers["X-QSRAC-Seq"] = str(next_seq) if not degraded else "degraded"
        response.headers["X-QSRAC-Envelope"] = envelope_hash[:16] if not degraded else "degraded"

        return response
