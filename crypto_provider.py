# crypto_provider.py
# Interface-driven crypto layer for QSRAC
# Supports: MOCK (host) | PQC (Docker with liboqs/pqcrypto)

import os
import hashlib
import hmac

CRYPTO_MODE = os.getenv("CRYPTO_MODE", "MOCK")  # MOCK | PQC


# ─────────────────────────────────────────────
# INTERFACE (CONTRACT — DO NOT CHANGE)
# ─────────────────────────────────────────────

class CryptoProvider:
    def generate_signing_keypair(self): ...
    def sign(self, private_key, data: bytes) -> bytes: ...
    def verify(self, signature: bytes, data: bytes, public_key) -> bool: ...

    def generate_exchange_keypair(self): ...
    def kem_encapsulate(self, peer_public_key: bytes): ...
    def kem_decapsulate(self, private_key: bytes, ciphertext: bytes): ...


# ─────────────────────────────────────────────
# MOCK IMPLEMENTATION (HOST-SAFE)
# ─────────────────────────────────────────────

class MockCryptoProvider(CryptoProvider):

    # --- Signing (symmetric for correctness) ---
    def generate_signing_keypair(self):
        key = os.urandom(32)
        return key, key   # symmetric key for mock

    def sign(self, private_key, data: bytes) -> bytes:
        return hmac.new(private_key, data, hashlib.sha256).digest()

    def verify(self, signature: bytes, data: bytes, public_key) -> bool:
        expected = hmac.new(public_key, data, hashlib.sha256).digest()
        return hmac.compare_digest(signature, expected)

    # --- KEM (deterministic, consistent shared secret) ---
    def generate_exchange_keypair(self):
        sk = os.urandom(32)
        pk = hashlib.sha256(sk).digest()
        return sk, pk

    def kem_encapsulate(self, peer_public_key: bytes):
        shared = hashlib.sha256(peer_public_key).digest()
        return shared, shared   # ciphertext == shared

    def kem_decapsulate(self, private_key: bytes, ciphertext: bytes):
        return ciphertext


# ─────────────────────────────────────────────
# PQC IMPLEMENTATION (DOCKER ONLY)
# ─────────────────────────────────────────────

class PQCCryptoProvider(CryptoProvider):

    def __init__(self):
        from pqcrypto.sign import dilithium2
        from pqcrypto.kem import kyber512

        self.dilithium2 = dilithium2
        self.kyber512 = kyber512

    # --- Signing ---
    def generate_signing_keypair(self):
        pk, sk = self.dilithium2.generate_keypair()
        return sk, pk

    def sign(self, private_key, data: bytes) -> bytes:
        return self.dilithium2.sign(data, private_key)

    def verify(self, signature: bytes, data: bytes, public_key) -> bool:
        try:
            self.dilithium2.verify(data, signature, public_key)
            return True
        except Exception:
            return False

    # --- KEM ---
    def generate_exchange_keypair(self):
        pk, sk = self.kyber512.generate_keypair()
        return sk, pk

    def kem_encapsulate(self, peer_public_key: bytes):
        return self.kyber512.encrypt(peer_public_key)

    def kem_decapsulate(self, private_key: bytes, ciphertext: bytes):
        return self.kyber512.decrypt(private_key, ciphertext)


# ─────────────────────────────────────────────
# PROVIDER SELECTOR
# ─────────────────────────────────────────────

def _get_provider():
    if CRYPTO_MODE == "PQC":
        return PQCCryptoProvider()
    return MockCryptoProvider()


_provider = _get_provider()


# ─────────────────────────────────────────────
# PUBLIC API (USED BY SYSTEM — DO NOT CHANGE)
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

def serialize_public_key(public_key):
    return public_key

def serialize_exchange_public_key(public_key):
    return public_key

def sign_token(private_key, data: bytes) -> str:
    return sign(private_key, data).hex()

def verify_token(signature_hex: str, data: bytes, public_key) -> bool:
    try:
        return verify(bytes.fromhex(signature_hex), data, public_key)
    except Exception:
        return False