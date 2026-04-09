def validate_attributes(context: dict) -> bool:
    try:
        sensitivity = float(context.get("sensitivity_level", 0))
        device_trust = float(context.get("device_trust_score", 0))
    except (TypeError, ValueError):
        return False

    if sensitivity > 5:
        return False
    if device_trust < 0.3:
        return False

    # optional stronger rules
    if context.get("is_tor", 0) == 1 and sensitivity >= 3:
        return False

    return True