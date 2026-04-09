import json
import asyncio
from datetime import datetime, timezone

LOG_FILE = "audit.log"


def log_event(session_id: str, decision: str, risk: str, trust: float):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "decision": decision,
        "risk": risk,
        "trust": trust,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


async def log_event_async(session_id: str, decision: str, risk: str, trust: float):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        log_event,
        session_id,
        decision,
        risk,
        trust,
    )