from ml_module import get_risk_score

# 3 handcrafted inputs targeting each bucket
LOW_1 = {
    "hour_of_day": 10, "request_rate": 0.2, "failed_attempts": 0,
    "geo_risk_score": 0.0, "device_trust_score": 0.95,
    "sensitivity_level": 1, "is_vpn": 0, "is_tor": 0
}

LOW_2 = {
    "hour_of_day": 14, "request_rate": 0.4, "failed_attempts": 0,
    "geo_risk_score": 0.05, "device_trust_score": 0.9,
    "sensitivity_level": 1, "is_vpn": 0, "is_tor": 0
}

LOW_3 = {
    "hour_of_day": 9, "request_rate": 0.3, "failed_attempts": 0,
    "geo_risk_score": 0.0, "device_trust_score": 0.85,
    "sensitivity_level": 2, "is_vpn": 0, "is_tor": 0
}

MEDIUM_1 = {
    "hour_of_day": 15, "request_rate": 0.9, "failed_attempts": 0,
    "geo_risk_score": 0.1, "device_trust_score": 0.7,
    "sensitivity_level": 2, "is_vpn": 0, "is_tor": 0
}

MEDIUM_2 = {
    "hour_of_day": 16, "request_rate": 1.1, "failed_attempts": 0,
    "geo_risk_score": 0.15, "device_trust_score": 0.65,
    "sensitivity_level": 2, "is_vpn": 0, "is_tor": 0
}

MEDIUM_3 = {
    "hour_of_day": 14, "request_rate": 1.0, "failed_attempts": 0,
    "geo_risk_score": 0.12, "device_trust_score": 0.68,
    "sensitivity_level": 2, "is_vpn": 0, "is_tor": 0
}

HIGH_1 = {
    "hour_of_day": 2, "request_rate": 2.5, "failed_attempts": 2,
    "geo_risk_score": 0.4, "device_trust_score": 0.4,
    "sensitivity_level": 4, "is_vpn": 1, "is_tor": 0
}

HIGH_2 = {
    "hour_of_day": 4, "request_rate": 3.0, "failed_attempts": 3,
    "geo_risk_score": 0.5, "device_trust_score": 0.35,
    "sensitivity_level": 4, "is_vpn": 1, "is_tor": 0
}

HIGH_3 = {
    "hour_of_day": 1, "request_rate": 2.8, "failed_attempts": 2,
    "geo_risk_score": 0.45, "device_trust_score": 0.3,
    "sensitivity_level": 5, "is_vpn": 1, "is_tor": 0
}

CRITICAL_1 = {
    "hour_of_day": 3, "request_rate": 6.0, "failed_attempts": 5,
    "geo_risk_score": 1.0, "device_trust_score": 0.1,
    "sensitivity_level": 5, "is_vpn": 1, "is_tor": 1
}

CRITICAL_2 = {
    "hour_of_day": 2, "request_rate": 7.0, "failed_attempts": 6,
    "geo_risk_score": 1.0, "device_trust_score": 0.05,
    "sensitivity_level": 5, "is_vpn": 1, "is_tor": 1
}

CRITICAL_3 = {
    "hour_of_day": 1, "request_rate": 8.0, "failed_attempts": 7,
    "geo_risk_score": 1.0, "device_trust_score": 0.1,
    "sensitivity_level": 5, "is_vpn": 1, "is_tor": 1
}

test_cases = [
    ("LOW_1", LOW_1),
    ("LOW_2", LOW_2),
    ("LOW_3", LOW_3),

    ("MEDIUM_1", MEDIUM_1),
    ("MEDIUM_2", MEDIUM_2),
    ("MEDIUM_3", MEDIUM_3),

    ("HIGH_1", HIGH_1),
    ("HIGH_2", HIGH_2),
    ("HIGH_3", HIGH_3),

    ("CRITICAL_1", CRITICAL_1),
    ("CRITICAL_2", CRITICAL_2),
    ("CRITICAL_3", CRITICAL_3),
]

for label, data in test_cases:
    print(f"\nExpected: {label}")
    result = get_risk_score(data)
    print(f"Predicted: {result}")