import oqs

# SIGNING (Dilithium2)
with oqs.Signature("Dilithium2") as sig:
    pk = sig.generate_keypair()
    sk = sig.export_secret_key()
    print(f"SIGNING_PRIVATE_KEY={sk.hex()}")
    print(f"SIGNING_PUBLIC_KEY={pk.hex()}")

# KEM (Kyber512)
with oqs.KeyEncapsulation("Kyber512") as kem:
    pk = kem.generate_keypair()
    sk = kem.export_secret_key()
    print(f"EXCHANGE_PRIVATE_KEY={sk.hex()}")
    print(f"EXCHANGE_PUBLIC_KEY={pk.hex()}")