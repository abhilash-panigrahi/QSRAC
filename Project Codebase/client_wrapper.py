import requests
import hashlib
import json
import time
from crypto_provider import generate_exchange_keypair, kem_decapsulate
from envelope import generate_envelope


class QSRACClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

        # PQC keypair
        self.priv, self.pub = generate_exchange_keypair()

        # Session state
        self.session_id = None
        self.core_token = None
        self.core_token_hash = None
        self.session_key = None
        self.seq = 0
        self.envelope = hashlib.sha256(b"init").hexdigest()

    # ─────────────────────────────────────────────
    # 🔐 LOGIN
    # ─────────────────────────────────────────────
    def login(self, username: str):
        res = requests.post(f"{self.base_url}/login", json={
            "username": username,
            "client_public_key": self.pub.hex()
        })

        if res.status_code != 200:
            raise Exception(f"Login failed: {res.text}")

        data = res.json()

        self.session_id = data["session_id"]
        self.core_token = data["core_token"]

        # hash token for envelope chain
        self.core_token_hash = hashlib.sha256(
            self.core_token.encode()
        ).hexdigest()

        # derive session key
        ciphertext = bytes.fromhex(data["kem_ciphertext"])
        self.session_key = kem_decapsulate(self.priv, ciphertext)

        # initial state
        self.seq = data["seq"]
        self.envelope = data["init_envelope"]

        print("✅ Login successful")
        print("Session:", self.session_id)

    # ─────────────────────────────────────────────
    # 📡 REQUEST
    # ─────────────────────────────────────────────
    def request(self, path: str, context: dict):
        if not self.session_id:
            raise Exception("Login first")

        headers = {
            "X-Session-ID": self.session_id,
            "X-Core-Token": self.core_token,
            "X-QSRAC-Seq": str(self.seq),
            "X-QSRAC-Envelope": self.envelope,
        }

        # attach context
        for k, v in context.items():
            headers[f"X-{k.replace('_','-')}"] = str(float(v))

        res = requests.get(f"{self.base_url}{path}", headers=headers)

        print(f"➡ Status: {res.status_code}")

        # ── STATE SYNC (IMPORTANT)
        if "X-QSRAC-Seq" in res.headers:
            new_seq = int(res.headers["X-QSRAC-Seq"])
            new_env = res.headers["X-QSRAC-Envelope"]
            trust = float(res.headers["X-QSRAC-Trust"])
            risk = res.headers["X-QSRAC-Risk"]

            # verify server state
            expected_hash, _ = generate_envelope(
                session_key=self.session_key,
                core_token_hash=self.core_token_hash,
                risk=risk,
                context=context,
                prev_hash=self.envelope,
                trust=trust,
                seq=new_seq
            )

            if expected_hash != new_env:
                raise Exception("❌ State tampering detected")

            # update state
            self.seq = new_seq
            self.envelope = new_env

        return res

    # ─────────────────────────────────────────────
    # 🔐 MFA FLOW
    # ─────────────────────────────────────────────
    def solve_mfa(self):
        headers = {"X-Session-ID": self.session_id}

        res = requests.post(f"{self.base_url}/mfa/challenge", headers=headers)
        data = res.json()

        nonce = data["nonce"]
        challenge = data["challenge"]
        timestamp = data["timestamp"]

        # solve challenge
        msg = (nonce + f"{timestamp:.6f}").encode()
        response = hashlib.sha256(self.session_key + msg).hexdigest()

        verify = requests.post(
            f"{self.base_url}/mfa/verify",
            headers=headers,
            json={
                "nonce": nonce,
                "response": response,
                "timestamp": timestamp
            }
        )

        print("🔐 MFA status:", verify.status_code)
        return verify


# ─────────────────────────────────────────────
# 🚀 DEMO RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    client = QSRACClient("http://127.0.0.1:8000")

    client.login("test_user")

    # normal context
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

    print("\n[1] Normal request")
    client.request("/test", context)

    # attack simulation
    attack = context.copy()
    attack.update({
        "request_rate": 25,
        "failed_attempts": 5,
        "is_tor": 1,
        "sensitivity_level": 5
    })

    print("\n[2] Attack simulation")
    for i in range(3):
        client.request("/test", attack)
        time.sleep(1)

    print("\n[3] Solve MFA")
    client.solve_mfa()

    print("\n[4] Post-MFA request")
    client.request("/test", context)