import requests
import hmac
import hashlib
from client_wrapper import QSRACClient

BASE = "http://localhost:8000"

client = QSRACClient(BASE)

# ── 1. Login ─────────────────────────────
client.login("user")

# ── 2. Trigger Step-Up ───────────────────
context = {
    "hour_of_day": 2,
    "request_rate": 10,
    "failed_attempts": 4,
    "geo_risk_score": 0.7,
    "device_trust_score": 0.8,
    "sensitivity_level": 3,
    "is_vpn": 1,
    "is_tor": 0
}

res1 = client.request("/test", context)
print("Before MFA:", res1.json())

# ── 3. MFA Challenge ─────────────────────
chal = requests.post(
    f"{BASE}/mfa/challenge",
    headers={
        "X-Session-ID": client.session_id
    }
).json()

print("Challenge:", chal)

# ── 4. Generate response ────────────────
nonce = chal["nonce"]
timestamp = chal["timestamp"]

msg = (nonce + str(timestamp)).encode()

response = hmac.new(
    client.session_key,
    msg,
    hashlib.sha256
).hexdigest()

# ── 5. Verify MFA ──────────────────────
verify = requests.post(
    f"{BASE}/mfa/verify",
    headers={
        "X-Session-ID": client.session_id
    },
    json={
        "nonce": nonce,
        "response": response,
        "timestamp": timestamp
    }
).json()

print("MFA Verify:", verify)
# 🔥 sync client state after MFA (CRITICAL)
client.seq = verify["seq"]
if "envelope" not in verify:
    raise Exception("Server did not return envelope — restart server")

client.envelope = verify["envelope"]

# ── 6. Retry request ───────────────────
res2 = client.request("/test", context)
print("After MFA:", res2.json())
print("SEQ:", client.seq)
