import hashlib
import hmac
import json


def generate_envelope(
    session_key: bytes,
    core_token_hash: str,
    risk: str,
    context: dict,
    prev_hash: str,
    trust: float,
    seq: int,
) -> tuple[str, bytes]:
    # IEEE Alignment: Force all context values to float to ensure hash consistency
    normalized_context = {k: float(v) for k, v in context.items()}
    
    context_json = json.dumps(normalized_context, sort_keys=True, separators=(",", ":"))
    context_hash = hashlib.sha256(context_json.encode("utf-8")).hexdigest()

    envelope_payload = json.dumps(
        {
            "core_token_hash": core_token_hash,
            "risk": risk,
            "context_hash": context_hash,
            "prev_hash": prev_hash,
            "trust": round(trust, 6),
            "seq": seq,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    raw_envelope_bytes = envelope_payload.encode("utf-8")

    envelope_hash = hmac.new(
        session_key,
        raw_envelope_bytes,
        hashlib.sha256,
    ).hexdigest()

    return envelope_hash, raw_envelope_bytes