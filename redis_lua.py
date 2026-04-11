"""
redis_lua.py — Redis session management and atomic Lua gate for QSRAC.

TTL contract
────────────
Every write path that mutates session state calls EXPIRE with SESSION_TTL
so the Redis key's lifetime stays in sync with the token's expiry regardless
of which code path last touched the key.  The gate Lua script intentionally
does NOT refresh TTL — it is a pure integrity check.  TTL extension is done
by the Python caller (validate_and_update) after a confirmed OK so that a
failed gate never silently extends a session.

Key namespaces
──────────────
  session:{session_id}  — Fast Path session state (this module)
  mfa:{session_id}:*    — MFA nonces                  (main.py)
"""

import logging
import redis

from config import REDIS_HOST, REDIS_PORT, REDIS_DB, SESSION_TTL

log = logging.getLogger(__name__)

_redis_client = None

# ── Atomic gate Lua script ─────────────────────────────────────────────────────
# Enforces strict seq increment and hash-chain continuity.
# No TTL mutation inside Lua — see module docstring.
LUA_SCRIPT = """
local key = KEYS[1]
local seq = tonumber(ARGV[1])
local new_hash = ARGV[2]
local prev_hash = ARGV[3]

local exists = redis.call('EXISTS', key)
if exists == 0 then
    return redis.error_reply('SESSION_NOT_FOUND')
end

local stored_seq = tonumber(redis.call('HGET', key, 'seq'))
local stored_last_hash = redis.call('HGET', key, 'last_hash_1')

if seq ~= (stored_seq + 1) then
    return redis.error_reply('REPLAY_DETECTED')
end

if stored_last_hash ~= prev_hash then
    return redis.error_reply('CHAIN_BROKEN')
end

redis.call('HSET', key, 'last_hash_2', stored_last_hash)
redis.call('HSET', key, 'last_hash_1', new_hash)
redis.call('HSET', key, 'seq', seq)
redis.call('HSET', key, 'last_req_at', ARGV[4])

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


def validate_and_update(session_id: str, seq: int, new_hash: str, prev_hash: str, current_time: float) -> str:
    """
    Atomically verify seq + hash-chain then write new state.
    On success, refreshes the session TTL so active sessions never expire
    mid-use while idle sessions still expire on schedule.
    Raises ValueError for logical errors, ConnectionError for Redis outage.
    """
    try:
        client = get_redis_client()
        sha = load_lua_script()
        key = f"session:{session_id}"
        result = client.evalsha(sha, 1, key, seq, new_hash, prev_hash, current_time)
        # Gate confirmed OK — refresh TTL to match token expiry.
        # A failed gate never reaches this line so TTL is never extended
        # for a rejected request.
        client.expire(key, SESSION_TTL)
        return result
    except redis.exceptions.ResponseError as e:
        error_msg = str(e)
        if "SESSION_NOT_FOUND" in error_msg:
            raise ValueError("SESSION_NOT_FOUND")
        elif "REPLAY_DETECTED" in error_msg:
            raise ValueError("REPLAY_DETECTED")
        elif "CHAIN_BROKEN" in error_msg:
            raise ValueError("CHAIN_BROKEN")
        else:
            raise ValueError(f"REDIS_ERROR: {error_msg}")
    except redis.exceptions.ConnectionError as e:
        raise ConnectionError(f"Redis connection failed: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected Redis error: {str(e)}")


def create_session(session_id: str, core_token_hash: str, session_key: str, ttl: int) -> bool:
    """
    Atomically create session hash and set TTL in a single pipeline.
    The pipeline guarantees EXPIRE is set in the same round-trip as HSET
    so there is no window where the key exists without a TTL.
    """
    try:
        import hashlib
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
    except redis.exceptions.ConnectionError as e:
        raise ConnectionError(f"Redis connection failed: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to create session: {str(e)}")


def extend_session_ttl(session_id: str, ttl: int) -> None:
    """
    Refresh TTL on an existing session key.  Called by main.py after any
    post-create hset operations to ensure the key never loses its expiry.
    Raises RuntimeError on failure — callers must not silently ignore this.
    """
    try:
        client = get_redis_client()
        key = f"session:{session_id}"
        refreshed = client.expire(key, ttl)
        if not refreshed:
            raise RuntimeError(f"TTL refresh failed — key does not exist: {key}")
    except redis.exceptions.ConnectionError as e:
        raise ConnectionError(f"Redis connection failed: {str(e)}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to extend session TTL: {str(e)}")


def get_session(session_id: str) -> dict:
    try:
        client = get_redis_client()
        key = f"session:{session_id}"
        data = client.hgetall(key)
        if not data:
            raise ValueError("SESSION_NOT_FOUND")
        return data
    except ValueError:
        raise
    except redis.exceptions.ConnectionError as e:
        raise ConnectionError(f"Redis connection failed: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to get session: {str(e)}")


def delete_session(session_id: str) -> bool:
    try:
        client = get_redis_client()
        key = f"session:{session_id}"
        client.delete(key)
        return True
    except redis.exceptions.ConnectionError as e:
        raise ConnectionError(f"Redis connection failed: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to delete session: {str(e)}")
