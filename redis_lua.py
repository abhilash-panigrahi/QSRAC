"""
redis_lua.py — Redis session management and atomic Lua gate for QSRAC.

This module enforces the 'Fast Path' security invariants:
1. Strict sequence increment (Replay Protection).
2. Hash-chain continuity (Integrity Enforcement).
3. Atomic state transitions to eliminate TOCTOU windows.
"""

import logging
import redis
import hashlib
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, SESSION_TTL

log = logging.getLogger(__name__)

_redis_client = None

# ── Atomic gate Lua script ─────────────────────────────────────────────────────
# Issue 4: Includes atomic trust update and defensive corruption checks.
LUA_SCRIPT = """
local key = KEYS[1]
local seq = tonumber(ARGV[1])
local new_hash = ARGV[2]
local prev_hash = ARGV[3]
local timestamp = ARGV[4]
local new_trust = tonumber(ARGV[5]) -- Issue 4: Trust is passed as a numeric string

local exists = redis.call('EXISTS', key)
if exists == 0 then
    return redis.error_reply('SESSION_NOT_FOUND')
end

-- Fetch state fields for validation
local stored_seq_raw = redis.call('HGET', key, 'seq')
local stored_last_hash = redis.call('HGET', key, 'last_hash_1')

-- Defensive check: Ensure critical session fields are not null
if not stored_seq_raw or not stored_last_hash then
    return redis.error_reply('CORRUPTED_SESSION')
end

local stored_seq = tonumber(stored_seq_raw)

-- Sequence increment check for Replay Protection
if seq ~= (stored_seq + 1) then
    return redis.error_reply('REPLAY_DETECTED')
end

-- Hash-chain continuity check for Forward Integrity
if stored_last_hash ~= prev_hash then
    return redis.error_reply('CHAIN_BROKEN')
end

-- Atomic update of state and trust score
redis.call('HSET', key, 'last_hash_2', stored_last_hash)
redis.call('HSET', key, 'last_hash_1', new_hash)
redis.call('HSET', key, 'seq', seq)
redis.call('HSET', key, 'last_req_at', timestamp)
redis.call('HSET', key, 'trust', new_trust)

return 'OK'
"""

_script_sha = None

def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
        )
    return _redis_client

def load_lua_script() -> str:
    global _script_sha
    if _script_sha is None:
        client = get_redis_client()
        _script_sha = client.script_load(LUA_SCRIPT)
    return _script_sha

def validate_and_update(session_id: str, seq: int, new_hash: str, prev_hash: str, current_time: float, trust: float) -> str:
    """
    Atomically verify seq + hash-chain then write new state.
    On success, refreshes the session TTL.
    """
    try:
        client = get_redis_client()
        sha = load_lua_script()
        key = f"session:{session_id}"
        
        result = client.evalsha(sha, 1, key, seq, new_hash, prev_hash, current_time, float(trust))
        
        try:
            client.expire(key, SESSION_TTL)
        except redis.exceptions.ConnectionError:
            # State already committed — do NOT fail request
            pass
        return result
        
    except redis.exceptions.ResponseError as e:
        error_msg = str(e)
        # Granular mapping of Lua error replies to Python ValueErrors
        if "SESSION_NOT_FOUND" in error_msg:
            raise ValueError("SESSION_NOT_FOUND")
        elif "CORRUPTED_SESSION" in error_msg:
            raise ValueError("CORRUPTED_SESSION")
        elif "REPLAY_DETECTED" in error_msg:
            raise ValueError("REPLAY_DETECTED")
        elif "CHAIN_BROKEN" in error_msg:
            raise ValueError("CHAIN_BROKEN")
        else:
            raise ValueError(f"REDIS_ERROR: {error_msg}")
    except redis.exceptions.ConnectionError as e:
        raise ConnectionError(f"REDIS_UNAVAILABLE: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"REDIS_UNAVAILABLE: {str(e)}")

def create_session(session_id: str, core_token_hash: str, session_key: str, ttl: int) -> bool:
    """Initializes session hash and sets TTL atomically."""
    try:
        init_hash = hashlib.sha256(b"init").hexdigest()
        client = get_redis_client()
        key = f"session:{session_id}"
        pipe = client.pipeline()
        pipe.hset(key, mapping={
            "core_token_hash": core_token_hash,
            "last_hash_1": init_hash,
            "last_hash_2": init_hash,
            "seq": 0,
            "session_key": session_key,
            "trust": 1.0,
        })
        pipe.expire(key, ttl)
        pipe.execute()
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to create session: {str(e)}")

def get_session(session_id: str) -> dict:
    """Fetches all fields for a session; raises ValueError if missing."""
    try:
        client = get_redis_client()
        key = f"session:{session_id}"
        data = client.hgetall(key)
        if not data:
            raise ValueError("SESSION_NOT_FOUND")
        return data
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to get session: {str(e)}")

def extend_session_ttl(session_id: str, ttl: int) -> None:
    """Refreshes TTL on an existing session key."""
    try:
        client = get_redis_client()
        key = f"session:{session_id}"
        if not client.expire(key, ttl):
            raise RuntimeError(f"TTL refresh failed — key does not exist: {key}")
    except Exception as e:
        raise RuntimeError(f"Failed to extend session TTL: {str(e)}")

def delete_session(session_id: str) -> bool:
    """Deletes the session key from Redis."""
    try:
        client = get_redis_client()
        key = f"session:{session_id}"
        client.delete(key)
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to delete session: {str(e)}")