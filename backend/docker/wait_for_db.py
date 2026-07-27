"""Wait until the database TCP endpoint is reachable (container startup helper)."""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from urllib.parse import urlparse

DEBUG_LOG_PATH = os.environ.get("DEBUG_LOG_PATH", "/debug/debug-68dca3.log")
SESSION_ID = "68dca3"


def _debug_log(hypothesis_id: str, message: str, data: dict[str, object]) -> None:
    entry = {
        "sessionId": SESSION_ID,
        "hypothesisId": hypothesis_id,
        "location": "wait_for_db.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(entry)
    print(line)
    try:
        log_dir = os.path.dirname(DEBUG_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def _resolve_host_port() -> tuple[str, int]:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        sys.exit("DATABASE_URL is not set")

    parsed = urlparse(database_url.replace("+asyncpg", ""))
    host = (os.environ.get("DB_WAIT_HOST") or parsed.hostname or "postgres").strip()
    port = parsed.port or 5432
    return host, port


def _try_connect(host: str, port: int) -> tuple[bool, str | None]:
    last_error: str | None = None
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            infos = socket.getaddrinfo(host, port, family, socket.SOCK_STREAM)
        except socket.gaierror as exc:
            last_error = f"getaddrinfo({family}): {exc}"
            continue

        for info in infos:
            try:
                sock = socket.socket(info[0], info[1], info[2])
                sock.settimeout(2)
                sock.connect(info[4])
                sock.close()
                return True, f"connected via family={family} addr={info[4]}"
            except OSError as exc:
                last_error = f"connect({info[4]}): {exc}"
    return False, last_error


def main() -> None:
    host, port = _resolve_host_port()
    _debug_log(
        "H2",
        "wait_config",
        {
            "host": host,
            "port": port,
            "db_wait_host_env": os.environ.get("DB_WAIT_HOST"),
            "database_url_present": bool(os.environ.get("DATABASE_URL")),
        },
    )

    last_error: str | None = None
    for attempt in range(1, 31):
        ok, detail = _try_connect(host, port)
        if ok:
            _debug_log("H1", "connected", {"attempt": attempt, "detail": detail})
            print(f"Database reachable at {host}:{port}")
            return

        last_error = detail
        _debug_log("H3", "attempt_failed", {"attempt": attempt, "last_error": last_error})
        print(f"Waiting for database ({attempt}/30)...")
        time.sleep(2)

    _debug_log("H4", "wait_exhausted", {"host": host, "port": port, "last_error": last_error})
    sys.exit(f"Database not reachable at {host}:{port} after 30 attempts (last error: {last_error})")


if __name__ == "__main__":
    main()
