from client_wrapper import QSRACClient

BASE = "http://localhost:8000"

client = QSRACClient(BASE)

client.login("user")

res = client.request("/test", {
    "hour_of_day": 2,
    "request_rate": 10,
    "failed_attempts": 4,
    "geo_risk_score": 0.7,
    "device_trust_score": 0.8,   # HIGH trust
    "sensitivity_level": 3,
    "is_vpn": 1,
    "is_tor": 0
})

print(res.status_code, res.json())
print("SEQ:", client.seq)