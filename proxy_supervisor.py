"""Keep the current ModelLabs proxy healthy without replacing live peers."""

from __future__ import annotations

import os
import sys
import time

from model_host_launcher import SUPERVISOR_PID, ensure_proxy


def claim_pidfile() -> None:
    """Make the service-owned supervisor authoritative for future launches."""
    SUPERVISOR_PID.write_text(f"{os.getpid()}\n", encoding="utf-8")
    os.chmod(SUPERVISOR_PID, 0o600)


def main() -> None:
    interval = float(os.environ.get("MODELLABS_PROXY_SUPERVISOR_INTERVAL", "5"))
    if interval < 1:
        raise ValueError("MODELLABS_PROXY_SUPERVISOR_INTERVAL must be at least one second.")
    claim_pidfile()
    try:
        while True:
            try:
                ensure_proxy()
            except Exception as exc:
                print(f"ModelLabs proxy supervisor: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            time.sleep(interval)
    finally:
        try:
            if SUPERVISOR_PID.read_text(encoding="utf-8").strip() == str(os.getpid()):
                SUPERVISOR_PID.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
