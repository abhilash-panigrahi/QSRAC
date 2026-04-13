from client_wrapper import QSRACClient

BASE = "http://localhost:8000"

client = QSRACClient(BASE)
client.login("user")

context = {
    "hour_of_day": 10,
    "request_rate": 1,
    "failed_attempts": 0,
    "geo_risk_score": 0.1,
    "device_trust_score": 0.9,
    "sensitivity_level": 1,
    "is_vpn": 0,
    "is_tor": 0
}

# valid request
res1 = client.request("/test", context)
print("First:", res1.status_code)

# tamper envelope
client.envelope = "deadbeef"

res2 = client.request("/test", context)
print("Tamper:", res2.status_code, res2.text)