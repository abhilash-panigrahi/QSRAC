def _map_trust_confidence(trust_value: float) -> str:
    if trust_value >= 0.8:
        return "High"
    elif trust_value >= 0.4:
        return "Low"
    else:
        return "Zero"


DECISION_TABLE = {
    ("Low",      "High"): "Allow",
    ("Low",      "Low"):  "Restrict",
    ("Low",      "Zero"): "Step-Up",
    ("Medium",   "High"): "Restrict",
    ("Medium",   "Low"):  "Step-Up",
    ("Medium",   "Zero"): "Deny",
    ("High",     "High"): "Step-Up",
    ("High",     "Low"):  "Step-up",
    ("High",     "Zero"): "Block",
    ("Critical", "High"): "Deny",
    ("Critical", "Low"):  "Block",
    ("Critical", "Zero"): "Block",
}


def evaluate_policy(risk_level: str, trust_value: float) -> str:
    trust_confidence = _map_trust_confidence(trust_value)
    decision = DECISION_TABLE.get((risk_level, trust_confidence))
    if decision is None:
        return "Block"
    return decision


def evaluate_policy_full(
    risk_level: str,
    trust_value: float,
    role_clearance: bool,
    abac_clearance: bool,
) -> str:
    if not role_clearance:
        return "Deny"
    if not abac_clearance:
        return "Deny"
    return evaluate_policy(risk_level, trust_value)