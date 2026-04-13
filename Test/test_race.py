import threading
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

def send_request(i):
    res = client.request("/test", context)
    print(f"[{i}] →", res.status_code)

threads = []

for i in range(5):
    t = threading.Thread(target=send_request, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()