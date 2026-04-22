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
# MOCK IMPLEMENTATION (ONLY FOR BENCHMARKS LIKE END-TO-END REQUEST LATENCY, RPS etc....)
# ─────────────────────────────────────────────

class MockCryptoProvider(CryptoProvider):
    def __init__(self):
        self.signing_private_key = None
        self.signing_public_key = None

    def generate_signing_keypair(self):
        key = os.urandom(32)
        self.signing_private_key = key
        self.signing_public_key = key  # symmetric for HMAC
        return key, key

    def sign(self, private_key, data: bytes) -> bytes:
        tmp = data
        for _ in range(2000):
            tmp = hashlib.sha256(tmp).digest()
        return hmac.new(private_key, tmp, hashlib.sha256).digest()

    def verify(self, signature: bytes, data: bytes, public_key) -> bool:
        tmp = data
        for _ in range(2000):
            tmp = hashlib.sha256(tmp).digest()
        expected = hmac.new(public_key, tmp, hashlib.sha256).digest()
        return hmac.compare_digest(signature, expected)

    def generate_exchange_keypair(self):
        sk = os.urandom(32)
        pk = hashlib.sha256(sk).digest()
        return sk, pk

    def kem_encapsulate(self, peer_public_key: bytes):
        ephemeral = os.urandom(32)

        tmp = peer_public_key
        for _ in range(1000):
            tmp = hashlib.sha256(tmp).digest()

        shared = hashlib.sha256(peer_public_key + ephemeral).digest()
        return ephemeral, shared

    def kem_decapsulate(self, private_key: bytes, ciphertext: bytes):
        peer_public_key = hashlib.sha256(private_key).digest()
        shared = hashlib.sha256(peer_public_key + ciphertext).digest()
        return shared

# ─────────────────────────────────────────────
# PQC IMPLEMENTATION (liboqs)
# ─────────────────────────────────────────────

class PQCCryptoProvider(CryptoProvider):
    def __init__(self):
        import oqs
        self.oqs = oqs

        sigs = oqs.get_enabled_sig_mechanisms()
        self.sig_alg = next((s for s in sigs if "ML-DSA-44" in s or "Dilithium2" in s), None)

        kems = oqs.get_enabled_kem_mechanisms()
        self.kem_alg = next((k for k in kems if "ML-KEM-512" in k or "Kyber512" in k), None)

        if not self.sig_alg or not self.kem_alg:
            raise RuntimeError("Required PQC algorithms not found")

        # 🔥 STATEFUL OBJECTS (FIX)
        self.signer = oqs.Signature(self.sig_alg)
        self.kem = oqs.KeyEncapsulation(self.kem_alg)

        # 🔥 GENERATE ONCE — KEEP IN MEMORY
        self.signing_public_key = self.signer.generate_keypair()
        self.exchange_public_key = self.kem.generate_keypair()

        log.info(f"PQC initialized (STATEFUL) Sig: {self.sig_alg}, KEM: {self.kem_alg}")

    # ── SIGNING ──────────────────────
    def generate_signing_keypair(self):
        # return only public (private stays internal)
        return None, self.signing_public_key

    def sign(self, private_key, data: bytes) -> bytes:
        return self.signer.sign(data)

    def verify(self, signature: bytes, data: bytes, public_key) -> bool:
        try:
            verifier = self.oqs.Signature(self.sig_alg)
            return verifier.verify(data, signature, public_key)
        except Exception:
            return False

    # ── KEM ──────────────────────────
    def generate_exchange_keypair(self):
        return None, self.exchange_public_key

    def kem_encapsulate(self, peer_public_key: bytes):
        return self.kem.encap_secret(peer_public_key)

    def kem_decapsulate(self, private_key, ciphertext: bytes):
        return self.kem.decap_secret(ciphertext)

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
        raise RuntimeError(f"PQC initialization failed: {e}")


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

def get_signing_public_key():
    return _provider.signing_public_key


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