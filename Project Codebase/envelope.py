import hashlib
import hmac
import json
from config import PRECISION_MULTIPLIER


def generate_envelope(
    session_key: bytes,
    core_token_hash: str,
    risk: str,
    context: dict,
    prev_hash: str,
    trust: float,
    seq: int,
) -> tuple[str, bytes]:
    """
    Generates a cryptographically bound envelope for state evolution.
    Uses fixed-point arithmetic to ensure serialization determinism.
    """
    
    # Issue 3 Fix: Use PRECISION_MULTIPLIER for fixed-point context
    normalized_context = {
        k: int(round(float(v), 6) * PRECISION_MULTIPLIER) 
        for k, v in context.items()
    }

    context_json = json.dumps(
        normalized_context,
        sort_keys=True,
        separators=(",", ":")
    )
    context_hash = hashlib.sha256(context_json.encode("utf-8")).hexdigest()

    # Issue 3 Fix: Convert trust to a stable fixed-point integer
    stable_trust = int(round(trust, 6) * PRECISION_MULTIPLIER)

    envelope_payload = json.dumps(
        {
            "core_token_hash": core_token_hash,
            "risk": risk,
            "context_hash": context_hash,
            "prev_hash": prev_hash,
            "trust_fixed": stable_trust, 
            "seq": seq,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    
    raw_envelope_bytes = envelope_payload.encode("utf-8")
    session_key = hashlib.sha256(session_key).digest()
    # HMAC-SHA256 ensures the state cannot be tampered with in transit
    envelope_hash = hmac.new(
        session_key,
        raw_envelope_bytes,
        hashlib.sha256,
    ).hexdigest()

    return envelope_hash, raw_envelope_bytes