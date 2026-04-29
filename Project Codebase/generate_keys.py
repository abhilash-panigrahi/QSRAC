import oqs
import os

# 1. Identify supported mechanisms
sigs = oqs.get_enabled_sig_mechanisms()
target_sig = next((s for s in sigs if "Dilithium2" in s or "ML-DSA-44" in s), None)

kems = oqs.get_enabled_kem_mechanisms()
target_kem = next((k for k in kems if "Kyber512" in k or "ML-KEM-512" in k), None)

if not target_sig or not target_kem:
    print(f"Error: Could not find required algorithms. Enabled sigs: {sigs}")
else:
    # 2. Generate the keys in memory
    with oqs.Signature(target_sig) as sig:
        pk_sig = sig.generate_keypair()
        sk_sig = sig.export_secret_key()

    with oqs.KeyEncapsulation(target_kem) as kem:
        pk_kem = kem.generate_keypair()
        sk_kem = kem.export_secret_key()

    # 3. Write directly to .env (Appending to avoid deleting existing secrets)
    with open(".env", "a") as f:
        f.write(f"\n# Automated PQC Key Generation - {target_sig} & {target_kem}\n")
        f.write(f"SIGNING_PRIVATE_KEY={sk_sig.hex()}\n")
        f.write(f"SIGNING_PUBLIC_KEY={pk_sig.hex()}\n")
        f.write(f"EXCHANGE_PRIVATE_KEY={sk_kem.hex()}\n")
        f.write(f"EXCHANGE_PUBLIC_KEY={pk_kem.hex()}\n")

    print(f"✅ Success! Lattice-based keys have been written to your .env file.")