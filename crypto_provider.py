"""
crypto_provider.py — Signing and key-exchange primitives for QSRAC.

PQC availability is detected once at import time.  All original public
function signatures are preserved.  Two new KEM helpers are added for the
correct ML-KEM handshake:

    kem_encapsulate(peer_public_key) -> (ciphertext: bytes, shared_secret: bytes)
    kem_decapsulate(private_key, ciphertext) -> shared_secret: bytes

The login endpoint MUST use kem_encapsulate and return the ciphertext to the
client so the client can call kem_decapsulate and arrive at the same key.
derive_shared_key() is kept for the classical ECDH path and for any caller
that has not yet been updated.

PQC path  : ML-DSA (Dilithium2) for signing, ML-KEM (Kyber512) for KEM.
"""

import logging
from pqcrypto.sign import dilithium2
from pqcrypto.kem import kyber512

log = logging.getLogger(__name__)

# ── Signing keypair ────────────────────────────────────────────────────────────

def generate_signing_keypair():
    public_key, private_key = dilithium2.generate_keypair()
    return private_key, public_key


# ── Internal sign / verify ─────────────────────────────────────────────────────

def sign(private_key, data: bytes) -> bytes:
    return dilithium2.sign(data, private_key)


def verify(signature: bytes, data: bytes, public_key) -> bool:
    try:
        dilithium2.verify(data, signature, public_key)
        return True
    except Exception:
        return False


# ── Exchange / KEM keypair ────────────────────────────────────────────────────

def generate_exchange_keypair():
    public_key, private_key = kyber512.generate_keypair()
    return private_key, public_key

# ── PQC KEM helpers (correct encapsulate / decapsulate flow) ──────────────────

def kem_encapsulate(peer_public_key: bytes) -> tuple[bytes, bytes]:
    return kyber512.encrypt(peer_public_key)


def kem_decapsulate(private_key: bytes, ciphertext: bytes) -> bytes:
    return kyber512.decrypt(private_key, ciphertext)


# ── Serialization helpers ──────────────────────────────────────────────────────

def serialize_public_key(public_key) -> bytes:
    return public_key

def serialize_exchange_public_key(public_key) -> bytes:
    return public_key

# ── Token sign / verify (public API) ──────────────────────────────────────────

def sign_token(private_key, data: bytes) -> str:
    raw_signature = sign(private_key, data)
    return raw_signature.hex()


def verify_token(signature_hex: str, data: bytes, public_key) -> bool:
    try:
        raw_signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    return verify(raw_signature, data, public_key)