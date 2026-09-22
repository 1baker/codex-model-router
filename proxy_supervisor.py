"""Keep the current ModelLabs proxy healthy without replacing live peers."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

from model_host_launcher import PROXY_PORT, ROOT, SUPERVISOR_PID, ensure_proxy


def claim_pidfile() -> None:
    """Make the service-owned supervisor authoritative for future launches."""
    SUPERVISOR_PID.write_text(f"{os.getpid()}\n", encoding="utf-8")
    os.chmod(SUPERVISOR_PID, 0o600)


def port_has_established_connection(port: int) -> bool:
    encoded = f"{port:04X}"
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            rows = table.read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            columns = row.split()
            if len(columns) > 3 and columns[1].rsplit(":", 1)[-1] == encoded and columns[3] == "01":
                return True
    return False


def retire_drained_proxy_generations() -> None:
    """Stop only obsolete proxies with no established client connection."""
    expected_script = str(ROOT / "turn_proxy.py").encode()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal() or int(proc.name) == os.getpid():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().split(b"\0")
            if expected_script not in cmdline:
                continue
            environment = (proc / "environ").read_bytes().split(b"\0")
            setting = next((item for item in environment if item.startswith(b"MODELLABS_PROXY_PORT=")), None)
            if setting is None:
                continue
            port = int(setting.split(b"=", 1)[1])
            if port != PROXY_PORT and not port_has_established_connection(port):
                os.kill(int(proc.name), signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
            continue


def remove_expired_launch_tickets() -> None:
    now = time.time()
    for ticket in (ROOT / "launch-tickets").glob("*.json"):
        try:
            if now - ticket.stat().st_mtime > 24 * 60 * 60:
                ticket.unlink()
        except OSError:
            continue


def main() -> None:
    interval = float(os.environ.get("MODELLABS_PROXY_SUPERVISOR_INTERVAL", "5"))
    if interval < 1:
        raise ValueError("MODELLABS_PROXY_SUPERVISOR_INTERVAL must be at least one second.")
    claim_pidfile()
    try:
        while True:
            try:
                ensure_proxy()
                retire_drained_proxy_generations()
                remove_expired_launch_tickets()
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
