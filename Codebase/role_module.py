ALLOWED_ROLES = {"user", "admin"}


def validate_role(core_token: dict) -> bool:
    role = core_token.get("role")
    if not role:
        return False
    return role in ALLOWED_ROLES