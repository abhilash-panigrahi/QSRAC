import requests
import hashlib
import hmac
import json
from crypto_provider import generate_exchange_keypair, kem_decapsulate
import oqs
from envelope import generate_envelope

class QSRACClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.priv, self.pub = generate_exchange_keypair()

        self.session_id = None
        self.session_key = None
        self.seq = 0
        self.envelope = hashlib.sha256(b"init").hexdigest()
        self.core_token = None
        self.core_token_hash = None # Cached for deterministic envelope generation

    def login(self, username):
        """Authenticates and establishes the initial QSRAC state."""
        r = requests.post(f"{self.base_url}/login", json={
            "username": username,
            "client_public_key": self.pub.hex()
        }).json()

        self.session_id = r["session_id"]
        self.core_token = r["core_token"]
        
        # Issue 1: Derive and store core token hash for state chain
        self.core_token_hash = hashlib.sha256(self.core_token.encode("utf-8")).hexdigest()

        # Verify Server Signature
        signing_pub = bytes.fromhex(r["signing_public_key"])
        signature = bytes.fromhex(r.get("token_signature", ""))
        
        sig_alg = next(s for s in oqs.get_enabled_sig_mechanisms() if "ML-DSA-44" in s or "Dilithium2" in s)
        with oqs.Signature(sig_alg) as verifier:
            if not signature or not verifier.verify(self.core_token.encode("utf-8"), signature, signing_pub):
                raise Exception("CRITICAL: Invalid server signature on core token")

        # Establish Session Key via KEM
        ciphertext = bytes.fromhex(r["kem_ciphertext"])
        self.session_key = kem_decapsulate(self.priv, ciphertext)

        # Sync Initial State
        self.seq = r["seq"]
        self.envelope = r["init_envelope"]

    def request(self, path, context):
        """
        Executes an authenticated request and evolves the authorization state.
        Handles state synchronization even on 403 Forbidden responses.
        """
        headers = {
            "X-Session-ID": self.session_id,
            "X-Core-Token": self.core_token,
            "X-QSRAC-Seq": str(self.seq),
            "X-QSRAC-Envelope": self.envelope,
        }

        # Dynamically map context to protocol headers
        context = {k: float(v) for k, v in context.items()}
        for k, v in context.items():
            headers[f"X-{k.replace('_','-')}"] = str(v)

        r = requests.get(f"{self.base_url}{path}", headers=headers)

        # 🔥 STATE SYNCHRONIZATION: Evolution must happen on ALL successful gate checks (200, 403)
        if "X-QSRAC-Seq" in r.headers and "X-QSRAC-Envelope" in r.headers:
            new_seq = int(r.headers["X-QSRAC-Seq"])
            new_env = r.headers["X-QSRAC-Envelope"]
            trust_signal = float(r.headers["X-QSRAC-Trust"])
            risk_signal = str(r.headers["X-QSRAC-Risk"])

            # Validate the new state using fixed-point arithmetic
            expected_hash, _ = generate_envelope(
                session_key=self.session_key,
                core_token_hash=self.core_token_hash,
                risk=risk_signal,
                context=context,
                prev_hash=headers["X-QSRAC-Envelope"],
                trust=trust_signal, # envelope.py handles fixed-point conversion internally
                seq=new_seq
            )

            if new_env != expected_hash:
                raise Exception("CRITICAL: Server state integrity verification failed (Tampering Detected)")

            # Update local state machine
            self.seq = new_seq
            self.envelope = new_env

        return r

# ─────────────────────────────────────────────────────────────────────────────
# Integration Test Script
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    client = QSRACClient("http://127.0.0.1:8000")

    print("[1/6] Establishing Quantum-Safe Session...")
    client.login("test_user")

    # Standard Operational Context
    context = {
        "hour_of_day": 12,
        "request_rate": 1.0,
        "failed_attempts": 0,
        "geo_risk_score": 0.1,
        "device_trust_score": 0.9,
        "sensitivity_level": 2,
        "is_vpn": 0,
        "is_tor": 0
    }

    print("[2/6] Baseline Request...")
    res = client.request("/test", context)
    print(f"Status: {res.status_code} | Trust: {res.headers.get('X-QSRAC-Trust')}")

    print("[3/6] Inducing Trust Decay (High Sensitivity Attack)...")
    context_attack = context.copy()
    context_attack.update({"sensitivity_level": 5, "request_rate": 20.0, "is_tor": 1})
    
    for i in range(3):
        res = client.request("/test", context_attack)
        print(f"Attempt {i+1}: {res.status_code} | Mode: {res.headers.get('X-QSRAC-Mode')}")

    print("[4/6] Solving MFA Challenge...")
    res_mfa = requests.post(f"{client.base_url}/mfa/challenge", 
                            headers={"X-Session-ID": client.session_id}).json()
    
    # Solve challenge using established Session Key
    msg = (res_mfa["nonce"] + f"{res_mfa['timestamp']:.6f}").encode("utf-8")
    response_hmac = hmac.new(client.session_key, msg, hashlib.sha256).hexdigest()

    print("[5/6] Verifying MFA & Recovering State...")
    res_verify = requests.post(
        f"{client.base_url}/mfa/verify",
        headers={"X-Session-ID": client.session_id},
        json={
            "nonce": res_mfa["nonce"],
            "response": response_hmac,
            "timestamp": res_mfa["timestamp"]
        }
    ).json()

    # 🔥 SYNC: Recover state machine from MFA completion
    client.seq = res_verify["seq"]
    client.envelope = res_verify["envelope"]
    print(f"MFA Success. New Seq: {client.seq}")

    print("[6/6] Confirming Access Recovery...")
    res_final = client.request("/test", context)
    print(f"Final Status: {res_final.status_code} (Expected 200)")