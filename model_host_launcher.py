"""Launch a Codex TUI on the local app-server that provides model control."""

from __future__ import annotations

import fcntl
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import websockets

from host_control import HOST_URL, TOKEN_FILE, _read_token
from paths import ROOT
from thread_owner import require_unowned


LOCK_FILE = ROOT / "host.lock"
LOG_FILE = ROOT / "host.log"
READY_URL = "http://127.0.0.1:45172/readyz"
# A new port permits a no-interruption proxy rollout. Existing managed TUIs
# retain their old connection until they exit or reconnect.
PROXY_PORT = 45177
PROXY_URL = f"ws://127.0.0.1:{PROXY_PORT}"
PROXY_LOG = ROOT / "proxy.log"
SUPERVISOR_PID = ROOT / f"proxy-supervisor-{PROXY_PORT}.pid"
SUPERVISOR_LOG = ROOT / "proxy-supervisor.log"


def ready() -> bool:
    try:
        with urllib.request.urlopen(READY_URL, timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def ensure_host() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ready():
            return
        host_environment = os.environ.copy()
        for key in ("CODEX_THREAD_ID", "CODEX_SESSION_ID", "CODEX_TURN_ID", "CODEX_CI"):
            host_environment.pop(key, None)
        with LOG_FILE.open("ab") as log:
            child = subprocess.Popen(
                ["codex", "app-server", "--listen", HOST_URL,
                 "--ws-auth", "capability-token", "--ws-token-file", str(TOKEN_FILE)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                env=host_environment,
            )
        for _ in range(50):
            if ready():
                return
            if child.poll() is not None:
                raise RuntimeError(f"model host exited with status {child.returncode}; see {LOG_FILE}")
            time.sleep(0.2)
        raise RuntimeError(f"model host did not become ready; see {LOG_FILE}")


def ensure_proxy() -> None:
    ensure_host()
    with LOCK_FILE.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if proxy_ready():
            return
        environment = os.environ.copy()
        for key in ("CODEX_THREAD_ID", "CODEX_SESSION_ID", "CODEX_TURN_ID", "CODEX_CI"):
            environment.pop(key, None)
        environment["MODELLABS_PROXY_PORT"] = str(PROXY_PORT)
        with PROXY_LOG.open("ab") as log:
            child = subprocess.Popen(
                [str(ROOT / "venv/bin/python"), str(ROOT / "turn_proxy.py")],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True, env=environment,
            )
        for _ in range(40):
            if proxy_ready():
                return
            if child.poll() is not None:
                raise RuntimeError(f"ModelLabs proxy exited; see {PROXY_LOG}")
            time.sleep(0.1)
        raise RuntimeError(f"ModelLabs proxy did not become ready; see {PROXY_LOG}")


def ensure_proxy_supervisor() -> None:
    """Start one local watchdog for the new proxy; it never touches old ports."""
    try:
        pid = int(SUPERVISOR_PID.read_text(encoding="utf-8").strip())
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        if b"proxy_supervisor.py" in cmdline:
            return
    except (FileNotFoundError, ValueError, OSError):
        pass
    environment = os.environ.copy()
    for key in ("CODEX_THREAD_ID", "CODEX_SESSION_ID", "CODEX_TURN_ID", "CODEX_CI"):
        environment.pop(key, None)
    environment["MODELLABS_PROXY_PORT"] = str(PROXY_PORT)
    with SUPERVISOR_LOG.open("ab") as log:
        child = subprocess.Popen(
            [str(ROOT / "venv/bin/python"), str(ROOT / "proxy_supervisor.py")],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True, env=environment,
        )
    SUPERVISOR_PID.write_text(f"{child.pid}\n", encoding="utf-8")
    os.chmod(SUPERVISOR_PID, 0o600)


def proxy_ready() -> bool:
    async def probe() -> bool:
        async with websockets.connect(PROXY_URL,
                                      additional_headers={"Authorization": f"Bearer {_read_token()}"},
                                      open_timeout=0.5, close_timeout=0.5) as ws:
            await ws.send(json.dumps({"id": 1, "method": "modellabs/ping", "params": {}}))
            reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
            return reply.get("result", {}).get("service") == "modellabs-proxy"
    try:
        return asyncio.run(probe())
    except (OSError, RuntimeError, TimeoutError, ValueError, websockets.WebSocketException):
        return False


def main() -> None:
    # New prompt-first chats go through ModelLabs before the first model call.
    # Resume and utility commands retain the existing TUI behavior.
    args_in = sys.argv[1:]
    utility_commands = {"resume", "agents", "exec", "review", "login", "logout",
                        "mcp", "plugin", "app-server", "remote-control", "completion",
                        "update", "doctor", "sandbox", "debug", "apply", "queue",
                        "archive", "delete", "fork", "cloud", "features", "help"}
    if not args_in or args_in[0] == "start" or (not args_in[0].startswith("-") and args_in[0] not in utility_commands):
        args = args_in[1:] if args_in and args_in[0] == "start" else args_in
        os.execv(sys.executable, [sys.executable, str(ROOT / "modellabs.py"), "run", *args])
    if args_in and args_in[0] == "resume":
        if len(args_in) < 2 or args_in[1].startswith("-"):
            raise ValueError("Managed resume requires an exact thread UUID, not the session picker or --last.")
        require_unowned(args_in[1])
    token = _read_token()
    ensure_proxy()
    ensure_proxy_supervisor()
    environment = os.environ.copy()
    environment["MODEL_SELECTOR_HOST_TOKEN"] = token
    # A CLI model flag is an explicit user choice. Keep its normal direct-host
    # behavior instead of silently overriding it in the proxy.
    explicit_model = any(arg in {"-m", "--model"} or arg.startswith("--model=") for arg in args_in)
    remote_url = HOST_URL if explicit_model else PROXY_URL
    os.execvpe(
        "codex",
        ["codex", "--remote", remote_url,
         "--remote-auth-token-env", "MODEL_SELECTOR_HOST_TOKEN", *sys.argv[1:]],
        environment,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"codex-model-host: {exc}", file=sys.stderr)
        raise SystemExit(1)
