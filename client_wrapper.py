import requests
import hashlib
from crypto_provider import generate_exchange_keypair, kem_decapsulate
from crypto_provider import verify

class QSRACClient:
    def __init__(self, base_url):
        self.base_url = base_url

        self.priv, self.pub = generate_exchange_keypair()

        self.session_id = None
        self.session_key = None
        self.seq = 0
        self.envelope = hashlib.sha256(b"init").hexdigest()
        self.core_token = None

    def login(self, username):
        r = requests.post(f"{self.base_url}/login", json={
            "username": username,
            "client_public_key": self.pub.hex()
        }).json()

        self.session_id = r["session_id"]
        self.core_token = r["core_token"]
        signing_pub = bytes.fromhex(r["signing_public_key"])
        token_bytes = self.core_token.encode("utf-8")
        if "token_signature" not in r:
            raise Exception("Missing server signature")

        signature = bytes.fromhex(r["token_signature"])

        if not verify(signature, token_bytes, signing_pub):
            raise Exception("Invalid server signature")

        ciphertext = bytes.fromhex(r["kem_ciphertext"])
        self.session_key = kem_decapsulate(self.priv, ciphertext)

        self.seq = r["seq"]
        self.envelope = r["init_envelope"]

    def request(self, path, context):
        headers = {
            "X-Session-ID": self.session_id,
            "X-Core-Token": self.core_token,
            "X-QSRAC-Seq": str(self.seq),
            "X-QSRAC-Envelope": self.envelope,
        }

        for k, v in context.items():
            headers[f"X-{k.replace('_','-')}"] = str(v)

        r = requests.get(f"{self.base_url}{path}", headers=headers)

        # 🔥 CRITICAL FIX: Sync state on ALL responses (200 + 403)
        if "X-QSRAC-Seq" in r.headers and "X-QSRAC-Envelope" in r.headers:
            new_seq = int(r.headers["X-QSRAC-Seq"])
            new_env = r.headers["X-QSRAC-Envelope"]

            from envelope import generate_envelope

            expected_hash, _ = generate_envelope(
                self.session_key,
                hashlib.sha256(self.core_token.encode()).hexdigest(),
                r.headers["X-QSRAC-Risk"],  # no fallback
                context,
                self.envelope,
                float(r.headers["X-QSRAC-Trust"]),
                new_seq,
            )

            if new_env != expected_hash:
                raise Exception("CRITICAL: Server state verification failed")

            # ✅ ALWAYS update state (even on 403)
            self.seq = new_seq
            self.envelope = new_env

        return r

if __name__ == "__main__":
    client = QSRACClient("http://127.0.0.1:8000")

    print("Logging in...")
    client.login("test_user")

    print("Session ID:", client.session_id)
    print("Seq:", client.seq)
    print("Envelope:", client.envelope)

    # ─────────────────────────────
    # 1. NORMAL REQUEST
    # ─────────────────────────────
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

    print("\nNormal request...")
    res = client.request("/test", context)
    print(res.json())

    # ─────────────────────────────
    # 2. FORCE LOW TRUST
    # ─────────────────────────────
    context_attack = {
        "hour_of_day": 3,
        "request_rate": 15.0,
        "failed_attempts": 8,
        "geo_risk_score": 0.9,
        "device_trust_score": 0.2,
        "sensitivity_level": 5,
        "is_vpn": 1,
        "is_tor": 1
    }

    print("\nTriggering low trust...")
    for _ in range(5):
        res = client.request("/test", context_attack)
        print(res.status_code, res.text)

    # ─────────────────────────────
    # 3. MFA CHALLENGE
    # ─────────────────────────────
    import requests, hmac, hashlib

    print("\nRequesting MFA challenge...")
    res = requests.post(
        "http://127.0.0.1:8000/mfa/challenge",
        headers={"X-Session-ID": client.session_id}
    )
    challenge = res.json()
    print("Challenge:", challenge)

    # ─────────────────────────────
    # 4. SOLVE MFA
    # ─────────────────────────────
    nonce = challenge["nonce"]
    timestamp = challenge["timestamp"]

    msg = (nonce + str(timestamp)).encode()
    response = hmac.new(client.session_key, msg, hashlib.sha256).hexdigest()

    # ─────────────────────────────
    # 5. VERIFY MFA
    # ─────────────────────────────
    print("\nVerifying MFA...")
    res = requests.post(
        "http://127.0.0.1:8000/mfa/verify",
        headers={"X-Session-ID": client.session_id},
        json={
            "nonce": nonce,
            "response": response,
            "timestamp": timestamp
        }
    )

    result = res.json()

    print("MFA result:", result)

    # 🔥 FINAL CRITICAL FIX: sync state after MFA
    client.seq = result["seq"]
    client.envelope = result["envelope"]

    # ─────────────────────────────
    # 6. CONFIRM RECOVERY
    # ─────────────────────────────
    print("\nPost-MFA request...")
    res = client.request("/test", context)
    print(res.json())