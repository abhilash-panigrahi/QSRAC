import json
import asyncio
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from config import AUDIT_LOG_PATH

# Centralized log path from configuration
LOG_FILE = AUDIT_LOG_PATH

# Professional log rotation (5MB files, keeps 5 backups)
handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))

audit_logger = logging.getLogger("audit")
audit_logger.addHandler(handler)
audit_logger.setLevel(logging.INFO)

def log_event(session_id: str, decision: str, risk: str, trust: float, status: str = "OK"):
    """
    Synchronous logging function for use in executors.
    Ensures security events are persisted with consistent precision.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "decision": decision,
        "risk": risk,
        "trust": round(float(trust), 6), # Matches precision logic in middleware
        "status": status
    }
    try:
        audit_logger.info(json.dumps(entry))
    except Exception:
        # Fallback to stderr ensures critical security events are never silently lost
        import sys
        print(f"CRITICAL AUDIT FAILURE: {json.dumps(entry)}", file=sys.stderr)

async def log_event_async(session_id: str, decision: str, risk: str, trust: float, status: str = "OK"):
    """
    Asynchronous wrapper that offloads I/O to a thread.
    Prevents the 'Fast Path' from blocking during disk write operations.
    """
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