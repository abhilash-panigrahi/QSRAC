import json
import asyncio
from datetime import datetime, timezone

LOG_FILE = "audit.log"

def log_event(session_id: str, decision: str, risk: str, trust: float, status: str = "OK"):
    """Synchronous logging function for use in executors."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "decision": decision,
        "risk": risk,
        "trust": trust,
        "status": status
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

async def log_event_async(session_id: str, decision: str, risk: str, trust: float, status: str = "OK"):
    """Asynchronous wrapper that offloads I/O to a thread."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, 
        log_event, 
        session_id, 
        decision, 
        risk, 
        trust, 
        status
    )