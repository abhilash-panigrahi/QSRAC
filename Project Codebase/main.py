from dotenv import load_dotenv
load_dotenv()
import uuid
import hashlib
import hmac
import json
import logging
from crypto_provider import CRYPTO_MODE, serialize_public_key
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from role_module import validate_role
from attribute_validator import validate_attributes
from policy_engine import evaluate_policy
from pydantic import BaseModel
from datetime import datetime, timezone
from ml_module import init_model

# config must be imported first — raises RuntimeError at startup if any
# required env var is missing, before any other module initialises.
import config
from config import SECRET_KEY, SESSION_TTL, APP_HOST, APP_PORT

from crypto_provider import (
    generate_signing_keypair,
    generate_exchange_keypair,
    kem_encapsulate,
    serialize_exchange_public_key,
    sign_token
)


from envelope import generate_envelope
from redis_lua import (
    create_session,
    get_redis_client,
    get_session,
    validate_and_update,
)
from decay_engine import compute_trust
from audit import log_event_async

log = logging.getLogger(__name__)

MFA_NONCE_TTL = 120
MFA_TRUST_INCREMENT = 0.2

app = FastAPI(title="QSRAC", version="1.0.0")
init_model()
# ── Middleware registration ────────────────────────────────────────────────────

from middleware import QSRACMiddleware, set_signing_public_key
app.add_middleware(QSRACMiddleware)

from crypto_provider import get_signing_public_key
_server_signing_private_key, _ = generate_signing_keypair()
_server_signing_public_key = get_signing_public_key()

_, _server_exchange_public_key = generate_exchange_keypair()

set_signing_public_key(_server_signing_public_key)

# ── Pydantic models ────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    client_public_key: str | None = None


class LoginResponse(BaseModel):
    session_id: str
    core_token: str
    server_public_key: str
    signing_public_key: str
    token_signature: str
    crypto_mode: str  
    init_envelope: str  
    kem_ciphertext: str | None = None
    seq: int
    expires_in: int


# ── MFA models ─────────────────────────────────────────────────────────────────

class MFAVerifyRequest(BaseModel):
    nonce: str
    response: str
    timestamp: float


class MFAChallengeResponse(BaseModel):
    nonce: str
    challenge: str
    timestamp: float



def access_dependency(request: Request):
    context = request.state.context
    core_token = request.state.core_token

    if not validate_role(core_token):
        raise HTTPException(status_code=403, detail="Role denied")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        client = get_redis_client()
        client.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "unavailable"

    return {
        "status": "ok",
        "redis": redis_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/test", dependencies=[Depends(access_dependency)])
def test(request: Request, x_session_id: str = Header(...)):

    decision = evaluate_policy(
        request.state.risk,
        request.state.trust
    )

    if decision in {"Block", "Deny"}:
        raise HTTPException(status_code=403, detail=f"Access {decision.lower()}ed")
    elif decision == "Step-Up":
        raise HTTPException(status_code=401, detail="Step-Up Authentication Required")
    elif decision == "Restrict":
        raise HTTPException(status_code=403, detail="Access Restricted")

    return {
        "msg": "ok",
        "decision": decision,
        "risk": request.state.risk,
        "trust": request.state.trust
    }


@app.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    try:
        session_id = str(uuid.uuid4())

        policy_hash = hashlib.sha256(b"RBAC_ABAC_RISK_TRUST_V1").hexdigest()

        core_token_payload = json.dumps(
            {
                "attributes": {
                    "clearance_level": 2,
                    "department": "engineering",
                },
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "policy_hash": policy_hash,
                "role": "user",
                "session_id": session_id,
                "username": request.username,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        core_token_bytes = core_token_payload.encode("utf-8")
        core_token_hash = hashlib.sha256(core_token_bytes).hexdigest()
        recomputed_hash = hashlib.sha256(core_token_bytes).hexdigest()
        if recomputed_hash != core_token_hash:
            raise Exception("Core token integrity failure")

        try:
            token_signature = sign_token(_server_signing_private_key, core_token_bytes)
        except (ConnectionError, RuntimeError):
            raise HTTPException(status_code=503, detail="Backend unavailable")
        except Exception:
            raise Exception("Token signing failed")       
        
        if not request.client_public_key:
            raise HTTPException(status_code=400, detail="Client public key required")

        client_pub_bytes = bytes.fromhex(request.client_public_key)
        kem_ct, shared_key = kem_encapsulate(client_pub_bytes)
        kem_ciphertext_hex = kem_ct.hex()
        session_key = hashlib.sha256(shared_key).digest()
        session_key_hex = session_key.hex()
        init_hash = hashlib.sha256(b"init").hexdigest()

        # Serialize whatever key type the provider generated (Kyber or ECDH)
        from crypto_provider import serialize_exchange_public_key
        server_pub_bytes = serialize_exchange_public_key(_server_exchange_public_key)
        server_pub_hex = server_pub_bytes.hex()

        # create_session writes core fields + sets TTL atomically via pipeline.
        create_session(
            session_id=session_id,
            core_token_hash=core_token_hash,
            session_key=session_key_hex,
            ttl=SESSION_TTL,
        )

        # Post-create fields written in a single pipeline, with EXPIRE
        # re-asserted at the end so the key TTL is never left at -1
        # (no expiry) due to the additional hset calls outside create_session.
        client = get_redis_client()
        key = f"session:{session_id}"
        pipe = client.pipeline()
        pipe.hset(key, "token_signature", token_signature)
        pipe.expire(key, SESSION_TTL)
        pipe.execute()

        signing_pub_bytes = serialize_public_key(_server_signing_public_key)
        signing_pub_hex = signing_pub_bytes.hex()
        from crypto_provider import CRYPTO_MODE
        crypto_mode = CRYPTO_MODE

        return LoginResponse(
            session_id=session_id,
            core_token=core_token_payload,
            server_public_key=server_pub_hex,
            signing_public_key=signing_pub_hex,
            crypto_mode=crypto_mode,
            init_envelope=init_hash,
            token_signature=token_signature,
            kem_ciphertext=kem_ciphertext_hex,
            seq=0,
            expires_in=SESSION_TTL,
        )

    except HTTPException:
        raise
    except (ConnectionError, RuntimeError):
        raise HTTPException(status_code=503, detail="Backend unavailable")
    except Exception as e:
        log.error("Login failed for user %r: %s", request.username, e)
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")
    
# ── MFA endpoints ──────────────────────────────────────────────────────────────

@app.post("/mfa/challenge", response_model=MFAChallengeResponse)
def mfa_challenge(x_session_id: str = Header(...)):
    """
    Issue a single-use HMAC challenge for MFA state-repair.
    Stores nonce in Redis with TTL=120s.    
    """
    try:
        session_data = get_session(x_session_id)
        session_key = bytes.fromhex(session_data["session_key"])
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except (ConnectionError, RuntimeError):
        raise HTTPException(status_code=503, detail="Backend unavailable")
    except Exception as e:
        log.error("mfa_challenge session fetch failed [%s]: %s", x_session_id, e)
        raise HTTPException(status_code=500, detail=f"Challenge failed: {str(e)}")

    nonce = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).timestamp()

    challenge_input = (nonce + f"{timestamp:.6f}").encode("utf-8")
    challenge = hmac.new(session_key, challenge_input, hashlib.sha256).hexdigest()

    try:
        client = get_redis_client()
        nonce_key = f"mfa:{x_session_id}:{nonce}"
        client.set(nonce_key, "1", ex=MFA_NONCE_TTL)
    except Exception as e:
        log.error("mfa_challenge nonce store failed [%s]: %s", x_session_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to store nonce: {str(e)}")

    return MFAChallengeResponse(nonce=nonce, challenge=challenge, timestamp=timestamp)


@app.post("/mfa/verify")
async def mfa_verify(body: MFAVerifyRequest, x_session_id: str = Header(...)):
    """
    Verify MFA response and perform state-repair transition.
    MUST pass through Redis Lua gate — not a bypass.
    Increments seq, extends hash-chain, adjusts trust upward (bounded).
    """
    # ── Step A: Validate and consume nonce (single-use) ───────────────────────
    try:
        client = get_redis_client()
        nonce_key = f"mfa:{x_session_id}:{body.nonce}"
        deleted = client.delete(nonce_key)
        if deleted == 0:
            raise HTTPException(status_code=401, detail="Invalid or expired nonce")
    except HTTPException:
        raise
    except Exception as e:
        log.error("mfa_verify nonce deletion failed [%s]: %s", x_session_id, e)
        raise HTTPException(status_code=500, detail=f"Nonce validation failed: {str(e)}")

    # ── Step B: Recompute expected HMAC and verify response ───────────────────
    try:
        session_data = get_session(x_session_id)
        session_key = bytes.fromhex(session_data["session_key"])
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except (ConnectionError, RuntimeError):
        raise HTTPException(status_code=503, detail="Backend unavailable")
    except Exception as e:
        log.error("mfa_verify session fetch failed [%s]: %s", x_session_id, e)
        raise HTTPException(status_code=500, detail=f"Session fetch failed: {str(e)}")

    challenge_input = (body.nonce + f"{body.timestamp:.6f}").encode("utf-8")
    expected = hmac.new(session_key, challenge_input, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected, body.response):
        raise HTTPException(status_code=401, detail="MFA response invalid")

    # ── Step C: State-repair transition (MUST go through Redis Lua gate) ──────
    try:
        core_token_hash = session_data["core_token_hash"]
        prev_hash = session_data["last_hash_1"]
        seq = int(session_data["seq"])
        next_seq = seq + 1

        trust0 = float(session_data.get("trust", 1.0))
        current_trust = compute_trust(
            trust0=trust0,
            sensitivity=1.0,
            risk_trend=0.3,
            time_delta=1.0,
        )
        adjusted_trust = min(1.0, current_trust + MFA_TRUST_INCREMENT)

        repair_risk = "Low"
        repair_context = {
            "hour_of_day": float(datetime.now(timezone.utc).hour),
            "request_rate": float(session_data.get("request_rate", 0.0)),
            "failed_attempts": float(session_data.get("failed_attempts", 0.0)),
            "geo_risk_score": float(session_data.get("geo_risk_score", 0.0)),
            "device_trust_score": float(session_data.get("device_trust_score", 1.0)),
            "sensitivity_level": float(session_data.get("sensitivity_level", 1.0)),
            "is_vpn": float(session_data.get("is_vpn", 0.0)),
            "is_tor": float(session_data.get("is_tor", 0.0)),
        }
        if not isinstance(repair_context, dict):
            raise HTTPException(status_code=400, detail="Invalid MFA context")

        if not set(config.REQUIRED_CONTEXT_KEYS).issubset(repair_context.keys()):
            raise HTTPException(status_code=400, detail="Incomplete MFA context")
        repair_context = {k: float(v) for k, v in repair_context.items()}

        envelope_hash, _ = generate_envelope(
            session_key=session_key,
            core_token_hash=core_token_hash,
            risk=repair_risk,
            context=repair_context,
            prev_hash=prev_hash,
            trust=adjusted_trust,
            seq=next_seq,
        )

        result = validate_and_update(
            session_id=x_session_id,
            seq=next_seq,
            new_hash=envelope_hash,
            prev_hash=prev_hash,
            current_time=datetime.now(timezone.utc).timestamp(),
            trust=adjusted_trust
        )
        if result != "OK":
            raise HTTPException(status_code=403, detail=f"Gate rejected repair: {result}")

    except HTTPException:
        raise
    except ValueError as e:
        error_msg = str(e)
        if "SESSION_NOT_FOUND" in error_msg:
            raise HTTPException(status_code=401, detail="Session not found")
        elif "REPLAY_DETECTED" in error_msg:
            raise HTTPException(status_code=403, detail="Replay detected")
        elif "CHAIN_BROKEN" in error_msg:
            raise HTTPException(status_code=403, detail="Hash chain broken")
        raise HTTPException(status_code=403, detail=error_msg)
    except Exception as e:
        log.error("mfa_verify state repair failed [%s]: %s", x_session_id, e)
        raise HTTPException(status_code=500, detail=f"State repair failed: {str(e)}")

    # ── Step D: Async audit log — event_type=REPAIR, latency_ms=None ─────────
    import asyncio
    asyncio.create_task(
        log_event_async(
            x_session_id,
            "REPAIR",
            repair_risk,
            adjusted_trust,
            )
        )
    
    return {
        "status": "repaired",
        "seq": next_seq,
        "envelope": envelope_hash,
        "trust": adjusted_trust,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=False)