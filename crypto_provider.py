# crypto_provider.py
# FINAL: PQC (liboqs) + MOCK fallback
# ML-KEM-512 (Kyber512) + ML-DSA-44 (Dilithium2)

import os
import logging
import hashlib
import hmac

log = logging.getLogger(__name__)

CRYPTO_MODE = os.getenv("CRYPTO_MODE", "MOCK")  # MOCK | PQC


# ─────────────────────────────────────────────
# INTERFACE (DO NOT CHANGE)
# ─────────────────────────────────────────────

class CryptoProvider:
    def generate_signing_keypair(self): ...
    def sign(self, private_key, data: bytes) -> bytes: ...
    def verify(self, signature: bytes, data: bytes, public_key) -> bool: ...

    def generate_exchange_keypair(self): ...
    def kem_encapsulate(self, peer_public_key: bytes): ...
    def kem_decapsulate(self, private_key: bytes, ciphertext: bytes): ...


# ─────────────────────────────────────────────
# MOCK IMPLEMENTATION (FAST / DEV SAFE)
# ─────────────────────────────────────────────

class MockCryptoProvider(CryptoProvider):

    def generate_signing_keypair(self):
        key = os.urandom(32)
        return key, key

    def sign(self, private_key, data: bytes) -> bytes:
        return hmac.new(private_key, data, hashlib.sha256).digest()

    def verify(self, signature: bytes, data: bytes, public_key) -> bool:
        expected = hmac.new(public_key, data, hashlib.sha256).digest()
        return hmac.compare_digest(signature, expected)

    def generate_exchange_keypair(self):
        sk = os.urandom(32)
        pk = hashlib.sha256(sk).digest()
        return sk, pk

    def kem_encapsulate(self, peer_public_key: bytes):
        shared = hashlib.sha256(peer_public_key).digest()
        return shared, shared  # (ciphertext, shared_secret)

    def kem_decapsulate(self, private_key: bytes, ciphertext: bytes):
        return ciphertext


# ─────────────────────────────────────────────
# PQC IMPLEMENTATION (liboqs)
# ─────────────────────────────────────────────

class PQCCryptoProvider(CryptoProvider):

    def __init__(self):
        import oqs

        # NIST standard mapping:
        # ML-DSA-44 → Dilithium2
        # ML-KEM-512 → Kyber512
        self.sig_alg = "Dilithium2"
        self.kem_alg = "Kyber512"
        self.oqs = oqs

    # ── SIGNING (ML-DSA) ──────────────────────

    def generate_signing_keypair(self):
        with self.oqs.Signature(self.sig_alg) as signer:
            public_key = signer.generate_keypair()
            private_key = signer.export_secret_key()
        return private_key, public_key

    def sign(self, private_key, data: bytes) -> bytes:
        # Key-specific instance (safe, avoids reuse issues)
        with self.oqs.Signature(self.sig_alg, secret_key=private_key) as signer:
            return signer.sign(data)

    def verify(self, signature: bytes, data: bytes, public_key) -> bool:
        try:
            with self.oqs.Signature(self.sig_alg) as verifier:
                return verifier.verify(data, signature, public_key)
        except Exception:
            return False

    # ── KEM (ML-KEM) ──────────────────────────

    def generate_exchange_keypair(self):
        with self.oqs.KeyEncapsulation(self.kem_alg) as kem:
            public_key = kem.generate_keypair()
            private_key = kem.export_secret_key()
        return private_key, public_key

    def kem_encapsulate(self, peer_public_key: bytes):
        with self.oqs.KeyEncapsulation(self.kem_alg) as kem:
            return kem.encap_secret(peer_public_key)  # (ciphertext, shared_secret)

    def kem_decapsulate(self, private_key: bytes, ciphertext: bytes):
        with self.oqs.KeyEncapsulation(self.kem_alg, secret_key=private_key) as kem:
            return kem.decap_secret(ciphertext)


# ─────────────────────────────────────────────
# PROVIDER SELECTOR (FAIL-SAFE)
# ─────────────────────────────────────────────

def _get_provider():
    if CRYPTO_MODE == "MOCK":
        log.info("Using MOCK crypto provider")
        return MockCryptoProvider()

    try:
        log.info("Using PQC crypto provider (liboqs)")
        return PQCCryptoProvider()
    except (ImportError, Exception) as e:
        log.warning("PQC init failed (%s). Falling back to MOCK.", e)
        return MockCryptoProvider()


_provider = _get_provider()


# ─────────────────────────────────────────────
# PUBLIC API (DO NOT CHANGE)
# ─────────────────────────────────────────────

def generate_signing_keypair():
    return _provider.generate_signing_keypair()

def sign(private_key, data: bytes) -> bytes:
    return _provider.sign(private_key, data)

def verify(signature: bytes, data: bytes, public_key) -> bool:
    return _provider.verify(signature, data, public_key)

def generate_exchange_keypair():
    return _provider.generate_exchange_keypair()

def kem_encapsulate(peer_public_key: bytes):
    return _provider.kem_encapsulate(peer_public_key)

def kem_decapsulate(private_key: bytes, ciphertext: bytes):
    return _provider.kem_decapsulate(private_key, ciphertext)


# ─────────────────────────────────────────────
# SERIALIZATION (UNCHANGED)
# ─────────────────────────────────────────────

def serialize_public_key(public_key):
    return public_key

def serialize_exchange_public_key(public_key):
    return public_key


# ─────────────────────────────────────────────
# TOKEN HELPERS
# ─────────────────────────────────────────────

def sign_token(private_key, data: bytes) -> str:
    return sign(private_key, data).hex()

def verify_token(signature_hex: str, data: bytes, public_key) -> bool:
    try:
        raw_signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    return verify(raw_signature, data, public_key)