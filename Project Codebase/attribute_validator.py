import math
def validate_attributes(context: dict) -> bool:
    try:
        sensitivity = float(context.get("sensitivity_level", 0))
        device_trust = float(context.get("device_trust_score", 0))
        geo_risk = float(context.get("geo_risk_score", 0))
        failed_attempts = float(context.get("failed_attempts", 0))
        is_vpn = float(context.get("is_vpn", 0))
        is_tor = float(context.get("is_tor", 0))
                
        if any(math.isnan(x) or math.isinf(x) for x in [
            sensitivity, device_trust, geo_risk, failed_attempts, is_vpn, is_tor
        ]):
            return False
    except (TypeError, ValueError):
        return False

    # core checks
    if not (1 <= sensitivity <= 5):
        return False
    if device_trust < 0.2:
        return False
    if device_trust < 0.4 and geo_risk > 0.7:
        return False

    # additional contextual checks
    if geo_risk > 0.8:
        return False
    if failed_attempts > 5:
        return False

    # network risk rules
    if is_tor == 1 and sensitivity >= 3:
        return False
    if is_vpn == 1 and device_trust < 0.5:
        return False

    return True