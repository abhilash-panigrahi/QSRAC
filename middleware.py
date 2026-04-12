import time
import json
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timezone
from policy_engine import evaluate_policy
from ml_module import get_risk_score
from envelope import generate_envelope
from redis_lua import validate_and_update, get_session
from decay_engine import compute_trust
from crypto_provider import verify as verify_token
from audit import log_event_async
from attribute_validator import validate_attributes

# Centralized Protocol Configuration
from config import REQUIRED_CONTEXT_KEYS 

log = logging.getLogger(__name__)

SKIP_PATHS = {
    "/", "/favicon.ico", "/health", "/login", "/docs", 
    "/openapi.json", "/docs/oauth2-redirect", "/mfa/challenge", "/mfa/verify"
}

_server_signing_public_key = None

def set_signing_public_key(public_key):
    global _server_signing_public_key
    _server_signing_public_key = public_key

class QSRACMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path.rstrip("/") or "/"
        if path in SKIP_PATHS:
            return await call_next(request)

        # 0. Header Extraction
        session_id = request.headers.get("X-Session-ID")
        core_token_raw = request.headers.get("X-Core-Token")
        
        if not session_id or not core_token_raw:
            if session_id:
                try:
                    session_data = get_session(session_id)

                    next_seq = int(session_data["seq"]) + 1
                    current_time = time.time()

                    context = {
                        "hour_of_day": float(datetime.now(timezone.utc).hour),
                        "request_rate": float(session_data.get("request_rate", 0.0)),
                        "failed_attempts": float(session_data.get("failed_attempts", 0.0)),
                        "geo_risk_score": float(session_data.get("geo_risk_score", 0.0)),
                        "device_trust_score": float(session_data.get("device_trust_score", 1.0)),
                        "sensitivity_level": float(session_data.get("sensitivity_level", 1.0)),
                        "is_vpn": float(session_data.get("is_vpn", 0.0)),
                        "is_tor": float(session_data.get("is_tor", 0.0)),
                    }

                    trust_value = float(session_data.get("trust", 1.0))

                    envelope_hash, _ = generate_envelope(
                        bytes.fromhex(session_data["session_key"]),
                        session_data["core_token_hash"],
                        "Medium",
                        context,
                        session_data["last_hash_1"],
                        trust_value,
                        next_seq
                    )

                    result = validate_and_update(
                        session_id,
                        next_seq,
                        envelope_hash,
                        session_data["last_hash_1"],
                        current_time,
                        trust_value
                    )

                    if result != "OK":
                        await log_event_async(session_id, "Reject", result, trust_value)    
                        return JSONResponse(
                            status_code=403,
                            content={"error": f"State update failed: {result}"}
                        )

                    await log_event_async(session_id, "Reject", "MissingHeaders", trust_value)
                    return self._finalize_response(
                        JSONResponse(status_code=401, content={"error": "Missing required authentication headers"}),
                        "Medium",
                        trust_value,
                        next_seq,
                        envelope_hash
                    )

                except ValueError as e:
                    await log_event_async(session_id, "Reject", str(e), 0.0)
                    return JSONResponse(status_code=403, content={"error": str(e)})
                except Exception as e:
                    await log_event_async(session_id or "UNKNOWN", "Reject", "StateProgressionError", 0.0)
                    log.error(f"State progression failed: {str(e)}")

            await log_event_async("UNKNOWN", "Reject", "MissingHeaders", 0.0)
            return JSONResponse(status_code=401, content={"error": "Missing required authentication headers"})

        try:
            # 1. Session Fetch & Token Verification
            session_data = get_session(session_id)
            
            # Issue 1: Standardized Token Parsing for main.py
            try:
                core_token_dict = json.loads(core_token_raw)
                request.state.core_token = core_token_dict
                core_token_canonical = json.dumps(
                    core_token_dict,
                    sort_keys=True,
                    separators=(",", ":")
                )
            except json.JSONDecodeError:
                await log_event_async(session_id, "Reject", "InvalidTokenJSON", 0.0)
                return JSONResponse(status_code=400, content={"error": "Invalid Core Token JSON"})

            # Verify cryptographic signature
            token_signature_hex = session_data.get("token_signature")
            if not token_signature_hex:
                await log_event_async(session_id, "Reject", "IncompleteSession", 0.0)
                return JSONResponse(status_code=401, content={"error": "Incomplete Session State"})
            
            token_signature = bytes.fromhex(token_signature_hex)
            if not verify_token(token_signature, core_token_canonical.encode("utf-8"), _server_signing_public_key):
                await log_event_async(session_id, "Reject", "InvalidSignature", 0.0)
                return JSONResponse(status_code=401, content={"error": "Core token signature invalid"})

            # 2. Context Extraction & ABAC Gating
            context = {
                k[2:].replace("-", "_").lower(): float(v) 
                for k, v in request.headers.items() 
                if k.lower().startswith("x-") and k[2:].replace("-", "_").lower() in REQUIRED_CONTEXT_KEYS
            }

            if not set(REQUIRED_CONTEXT_KEYS).issubset(context.keys()):
                await log_event_async(session_id, "Reject", "IncompleteContext", 0.0)
                return JSONResponse(status_code=400, content={"error": "Incomplete QSRAC context"})

            request.state.context = context

            # 3. Risk and Trust Computation
            risk_raw = str(get_risk_score(context)).strip().lower()

            risk_map = {
                "low": "Low",
                "medium": "Medium",
                "high": "High",
                "critical": "Critical"
            }

            risk_level = risk_map.get(risk_raw, "Medium")
            current_time = time.time()
            time_delta = current_time - float(session_data.get("last_req_at", current_time))
            trust_value = compute_trust(
                float(session_data.get("trust", 1.0)), 
                context["sensitivity_level"], 
                {"low": 0.0, "medium": 0.3, "high": 0.6, "critical": 1.0}.get(risk_raw, 0.5), 
                time_delta
            )
            
            request.state.risk, request.state.trust = risk_level, trust_value

            if not validate_attributes(context):
                await log_event_async(session_id, "Reject", "AttributeDenied", trust_value)

                next_seq = int(session_data["seq"]) + 1
                envelope_hash, _ = generate_envelope(
                    bytes.fromhex(session_data["session_key"]),
                    session_data["core_token_hash"],
                    risk_level,
                    context,
                    session_data["last_hash_1"],
                    trust_value,
                    next_seq
                )

                result=validate_and_update(
                    session_id,
                    next_seq,
                    envelope_hash,
                    session_data["last_hash_1"],
                    current_time,
                    trust_value
                )
                if result != "OK":
                    await log_event_async(session_id, "Reject", result, trust_value)
                    return JSONResponse(status_code=403, content={"error": f"State update failed: {result}"})

                return self._finalize_response(
                    JSONResponse(status_code=403, content={"error": "Access denied: Attribute validation failed"}),
                    risk_level,
                    trust_value,
                    next_seq,
                    envelope_hash
                )

            # 4. Fast Path State Evolution (Issue 4 Integration)
            client_seq = int(request.headers.get("X-QSRAC-Seq", -1))
            client_env = request.headers.get("X-QSRAC-Envelope")

            if client_seq == -1 or client_env is None:
                await log_event_async(session_id, "Reject", "MissingStateHeaders", trust_value)

                next_seq = int(session_data["seq"]) + 1
                envelope_hash, _ = generate_envelope(
                    bytes.fromhex(session_data["session_key"]),
                    session_data["core_token_hash"],
                    risk_level,
                    context,
                    session_data["last_hash_1"],
                    trust_value,
                    next_seq
                )

                result= validate_and_update(
                    session_id,
                    next_seq,
                    envelope_hash,
                    session_data["last_hash_1"],
                    current_time,
                    trust_value
                )
                if result != "OK":
                    await log_event_async(session_id, "Reject", result, trust_value)
                    return JSONResponse(status_code=403, content={"error": f"State update failed: {result}"})

                return self._finalize_response(
                    JSONResponse(status_code=400, content={"error": "Missing QSRAC state headers"}),
                    risk_level,
                    trust_value,
                    next_seq,
                    envelope_hash
                )

            # Granular Verification
            if client_seq != int(session_data["seq"]):
                await log_event_async(session_id, "Reject", "ReplayAttempt", trust_value)

                next_seq = int(session_data["seq"]) + 1
                envelope_hash, _ = generate_envelope(
                    bytes.fromhex(session_data["session_key"]),
                    session_data["core_token_hash"],
                    risk_level,
                    context,
                    session_data["last_hash_1"],
                    trust_value,
                    next_seq
                )

                result= validate_and_update(
                    session_id,
                    next_seq,
                    envelope_hash,
                    session_data["last_hash_1"],
                    current_time,
                    trust_value
                )
                if result != "OK":
                    await log_event_async(session_id, "Reject", result, trust_value)
                    return JSONResponse(status_code=403, content={"error": f"State update failed: {result}"})

                return self._finalize_response(
                    JSONResponse(status_code=403, content={"error": "Sequence mismatch: Replay detected"}),
                    risk_level,
                    trust_value,
                    next_seq,
                    envelope_hash
                )

            if client_env != session_data["last_hash_1"]:
                await log_event_async(session_id, "Reject", "StateMismatch", trust_value)

                next_seq = int(session_data["seq"]) + 1
                envelope_hash, _ = generate_envelope(
                    bytes.fromhex(session_data["session_key"]),
                    session_data["core_token_hash"],
                    risk_level,
                    context,
                    session_data["last_hash_1"],
                    trust_value,
                    next_seq
                )

                result= validate_and_update(
                    session_id,
                    next_seq,
                    envelope_hash,
                    session_data["last_hash_1"],
                    current_time,
                    trust_value
                )
                if result != "OK":
                    await log_event_async(session_id, "Reject", result, trust_value)
                    return JSONResponse(status_code=403, content={"error": f"State update failed: {result}"})

                return self._finalize_response(
                    JSONResponse(status_code=403, content={"error": "Hash mismatch: State integrity failed"}),
                    risk_level,
                    trust_value,
                    next_seq,
                    envelope_hash
                )

            next_seq = client_seq + 1
            envelope_hash, _ = generate_envelope(
                bytes.fromhex(session_data["session_key"]), 
                session_data["core_token_hash"], risk_level, context, 
                client_env, trust_value, next_seq
            )

            # Atomic evolution via Lua
            result = validate_and_update(session_id, next_seq, envelope_hash, client_env, current_time, trust_value)
            if result != "OK":
                await log_event_async(session_id, "Reject", result, trust_value)
                return JSONResponse(status_code=403, content={"error": f"Gate rejected update: {result}"})

            # 5. Policy Execution
            decision = evaluate_policy(risk_level, trust_value)

            if decision in {"Deny", "Block"}:
                response = JSONResponse(status_code=403, content={"error": f"Access {decision.lower()}ed"})
            elif decision == "Step-Up":
                response = JSONResponse(status_code=401, content={"error": "Step-Up Authentication Required"})
            elif decision == "Restrict":
                response = JSONResponse(status_code=403, content={"error": "Access Restricted"})
            else:
                response = await call_next(request)

            # Finalize with synchronized headers
            return self._finalize_response(response, risk_level, trust_value, next_seq, envelope_hash)

        except ValueError as e:
            return JSONResponse(status_code=401, content={"error": str(e)})
        except ConnectionError:
            return JSONResponse(status_code=503, content={"error": "Security backend unreachable"})
        except Exception as e:
            log.error(f"Middleware Error: {str(e)}")
            return JSONResponse(status_code=500, content={"error": "Internal authentication failure"})

    def _finalize_response(self, response, risk, trust, seq, env):
        response.headers["X-QSRAC-Risk"] = risk
        response.headers["X-QSRAC-Trust"] = str(round(trust, 6))
        response.headers["X-QSRAC-Mode"] = "NORMAL" if env else "DEGRADED"

        if seq is not None:
            response.headers["X-QSRAC-Seq"] = str(seq)
        if env is not None:
            response.headers["X-QSRAC-Envelope"] = env
        return response