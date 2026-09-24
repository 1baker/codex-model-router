"""Launch a Codex TUI on the local app-server that provides model control."""

from __future__ import annotations

import fcntl
import asyncio
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import websockets

from authority import path_for as authority_path_for
from host_control import HOST_URL, THREAD_ID_PATTERN, TOKEN_FILE, _read_token
from legacy_thread_handoff import import_closed_thread
from paths import ROOT, proxy_port, proxy_revision, real_codex_binary


LOCK_FILE = ROOT / "host.lock"
LOG_FILE = ROOT / "host.log"
READY_URL = "http://127.0.0.1:45172/readyz"
PROXY_REVISION = proxy_revision()
# Each implementation gets a new loopback generation. Existing managed TUIs
# retain their old connection until they exit or reconnect.
PROXY_PORT = proxy_port(PROXY_REVISION)
PROXY_URL = f"ws://127.0.0.1:{PROXY_PORT}"
PROXY_LOG = ROOT / "proxy.log"
SUPERVISOR_PID = ROOT / f"proxy-supervisor-{PROXY_PORT}.pid"
SUPERVISOR_LOG = ROOT / "proxy-supervisor.log"
CONFIGURED_MCP_SERVERS = {
    "agentBrowser", "cloudflare", "cloudflare-docs", "cloudflare-bindings",
    "cloudflare-builds", "cloudflare-observability", "openaiDeveloperDocs",
    "codegraph", "previews", "codexResearch", "litScout", "modelControl",
}


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
                [str(real_codex_binary()), "app-server", "--listen", HOST_URL,
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
            result = reply.get("result", {})
            return (result.get("service") == "modellabs-proxy"
                    and result.get("revision") == PROXY_REVISION)
    try:
        return asyncio.run(probe())
    except (OSError, RuntimeError, TimeoutError, ValueError, websockets.WebSocketException):
        return False


def _command_index(args: list[str]) -> int | None:
    value_options = {
        "-c", "--config", "--enable", "--disable", "--remote",
        "--remote-auth-token-env", "-i", "--image", "-m", "--model",
        "--local-provider", "-p", "--profile", "-s", "--sandbox", "-C", "--cd",
        "--add-dir", "-a", "--ask-for-approval",
    }
    index = 0
    while index < len(args):
        if args[index] == "--":
            return index + 1 if index + 1 < len(args) else None
        if args[index] in value_options:
            if index + 1 >= len(args):
                raise ValueError(f"Missing value for {args[index]}.")
            index += 2
        elif any(args[index].startswith(f"{option}=") for option in value_options
                 if option.startswith("--")):
            index += 1
        elif args[index].startswith("-"):
            index += 1
        else:
            return index
    return None


def _exec_prompt(args: list[str]) -> str | None:
    """Extract the direct `codex exec` prompt without changing its arguments."""
    command_index = _command_index(args)
    if command_index is None or args[command_index] not in {"exec", "e"}:
        return None
    tail = args[command_index + 1:]
    subcommand, subcommand_index = _exec_subcommand(args)
    if subcommand == "help":
        return None
    if subcommand:
        relative = subcommand_index - command_index - 1
        tail = tail[:relative] + tail[relative + 1:]
    value_options = {
        "-c", "--config", "-i", "--image", "-m", "--model", "--local-provider",
        "-p", "--profile", "-s", "--sandbox", "-C", "--cd", "--add-dir",
        "--thread-source", "--output-schema", "--color", "-o", "--output-last-message",
    }
    positionals: list[str] = []
    index = 0
    while index < len(tail):
        item = tail[index]
        if item == "--":
            positionals.extend(tail[index + 1:])
            break
        if item in value_options:
            index += 2
            continue
        if any(item.startswith(f"{option}=") for option in value_options if option.startswith("--")):
            index += 1
            continue
        if item == "-":
            positionals.append(item)
            index += 1
            continue
        if item.startswith("-"):
            index += 1
            continue
        positionals.append(item)
        index += 1
    if subcommand in {"resume", "fork"}:
        # With --last there is no session-id positional, so the first value is
        # the new prompt. Otherwise the prompt follows the session id.
        delimiter = tail.index("--") if "--" in tail else len(tail)
        prompt_index = 0 if "--last" in tail[:delimiter] else 1
        return positionals[prompt_index] if len(positionals) > prompt_index else None
    return positionals[-1] if positionals else None


def _exec_subcommand(args: list[str]) -> tuple[str | None, int | None]:
    """Find an exec subcommand after options while respecting literal `--`."""
    command_index = _command_index(args)
    if command_index is None or args[command_index] not in {"exec", "e"}:
        return None, None
    value_options = {
        "-c", "--config", "-i", "--image", "-m", "--model", "--local-provider",
        "-p", "--profile", "-s", "--sandbox", "-C", "--cd", "--add-dir",
        "--thread-source", "--output-schema", "--color", "-o", "--output-last-message",
    }
    index = command_index + 1
    while index < len(args):
        item = args[index]
        if item == "--":
            return None, None
        if item in value_options:
            if index + 1 >= len(args):
                raise ValueError(f"Missing value for {item}.")
            index += 2
            continue
        if any(item.startswith(f"{option}=") for option in value_options if option.startswith("--")):
            index += 1
            continue
        if item.startswith("-"):
            index += 1
            continue
        return (item, index) if item in {"resume", "fork", "review", "help"} else (None, None)
    return None, None


def _explicit_setting(args: list[str], setting: str) -> str | None:
    """Return an explicit CLI/config value without consulting ambient config."""
    flag = "--model" if setting == "model" else None
    result = None
    delimiter = args.index("--") if "--" in args else len(args)
    for index, item in enumerate(args[:delimiter]):
        if flag and item in {"-m", flag} and index + 1 < len(args):
            result = args[index + 1]
            continue
        if flag and item.startswith(f"{flag}="):
            result = item.split("=", 1)[1]
            continue
        if flag == "--model" and item.startswith("-m") and len(item) > 2:
            result = item[2:]
            continue
        if item in {"-c", "--config"} and index + 1 < len(args):
            config = args[index + 1]
        elif item.startswith("-c") and not item.startswith("--") and len(item) > 2:
            config = item[2:]
        elif item.startswith("--config="):
            config = item.split("=", 1)[1]
        else:
            continue
        key, separator, value = config.partition("=")
        if separator and key.strip() == setting:
            result = value.strip().strip('"\'')
    return result


def _explicit_mcp_scope(args: list[str]) -> list[str]:
    delimiter = args.index("--") if "--" in args else len(args)
    conflicts: list[str] = []
    for index, item in enumerate(args[:delimiter]):
        if item in {"-c", "--config"} and index + 1 < delimiter:
            config = args[index + 1]
        elif item.startswith("-c") and not item.startswith("--") and len(item) > 2:
            config = item[2:]
        elif item.startswith("--config="):
            config = item.split("=", 1)[1]
        else:
            continue
        key = config.partition("=")[0].strip()
        if key.startswith("mcp_servers."):
            conflicts.append(key)
    return conflicts


def _routed_exec_command(binary: Path, args: list[str], choice: dict) -> list[str]:
    """Build the effective command while preserving explicit caller choices."""
    command_index = _command_index(args)
    if command_index is None:
        raise ValueError("Cannot route an exec invocation without an exec command.")
    conflicts = _explicit_mcp_scope(args)
    if conflicts:
        raise ValueError(
            "ModelLabs refuses caller MCP scope overrides on a routed exec: "
            + ", ".join(conflicts)
        )
    injected: list[str] = []
    if _explicit_setting(args, "model") is None:
        injected.extend(["--model", choice["model"]])
    if _explicit_setting(args, "model_reasoning_effort") is None:
        injected.extend(["--config", f'model_reasoning_effort="{choice["effort"]}"'])
    selected = set(choice.get("servers", []))
    for server in sorted(CONFIGURED_MCP_SERVERS):
        injected.extend(["--config", f'mcp_servers.{server}.enabled={str(server in selected).lower()}'])
    return [str(binary), *args[:command_index + 1], *injected, *args[command_index + 1:]]


def _route_choice(prompt: str, args: list[str], status: str) -> dict:
    model_override = _explicit_setting(args, "model")
    effort_override = _explicit_setting(args, "model_reasoning_effort")
    routed = subprocess.run(
        [sys.executable, str(ROOT / "modellabs.py"), "route",
         *(["--model", model_override] if model_override else []),
         *(["--effort", effort_override] if effort_override else [])],
        input=prompt, text=True, capture_output=True, check=True,
    )
    choice = json.loads(routed.stdout)
    record = {**choice, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "status": status}
    routes = ROOT / "routes.jsonl"
    fd = os.open(routes, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.chmod(routes, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as destination:
        destination.write(json.dumps(record, sort_keys=True) + "\n")
    return choice


def _routed_interactive_args(args: list[str], prompt: str, choice: dict,
                             append_prompt: bool) -> list[str]:
    conflicts = _explicit_mcp_scope(args)
    if conflicts:
        raise ValueError(
            "ModelLabs refuses caller MCP scope overrides on a routed chat: "
            + ", ".join(conflicts)
        )
    insertion = args.index("--") if "--" in args else _command_index(args)
    if insertion is None:
        insertion = len(args)
    injected: list[str] = []
    if _explicit_setting(args, "model") is None:
        injected.extend(["--model", choice["model"]])
    if _explicit_setting(args, "model_reasoning_effort") is None:
        injected.extend(["--config", f'model_reasoning_effort="{choice["effort"]}"'])
    selected = set(choice.get("servers", []))
    for server in sorted(CONFIGURED_MCP_SERVERS):
        injected.extend(["--config", f'mcp_servers.{server}.enabled={str(server in selected).lower()}'])
    result = [*args[:insertion], *injected, *args[insertion:]]
    if append_prompt:
        result.append(prompt)
    return result


def _create_launch_ticket(choice: dict, *, explicit_model: bool, explicit_effort: bool) -> str:
    ticket_id = secrets.token_hex(16)
    directory = ROOT / "launch-tickets"
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    path = directory / f"{ticket_id}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"created_at": time.time(), "model": choice["model"],
                   "effort": choice["effort"], "servers": choice.get("servers", []),
                   "explicit_model": explicit_model,
                   "explicit_effort": explicit_effort}, handle)
        handle.write("\n")
    return ticket_id


def run_routed_exec(args: list[str]) -> int:
    """Fail closed until noninteractive turns have authoritative receipts."""
    prompt = _exec_prompt(args)
    stdin_text = None if sys.stdin.isatty() else sys.stdin.read()
    if stdin_text == "":
        stdin_text = None
    command_index = _command_index(args)
    subcommand, _subcommand_index = _exec_subcommand(args)
    if subcommand in {"resume", "fork"}:
        raise ValueError(
            f"Managed codex exec {subcommand} is refused because safe shared-host ownership "
            "cannot be guaranteed; use interactive codex resume THREAD_ID instead."
        )
    binary = real_codex_binary()
    delimiter = args.index("--") if "--" in args else len(args)
    if prompt is None and any(item in {"-h", "--help", "-V", "--version"}
                              for item in args[:delimiter]):
        command = [str(binary), *args]
        if stdin_text is None:
            os.execve(binary, command, os.environ.copy())
        return subprocess.run(command, input=stdin_text, text=True, env=os.environ.copy()).returncode
    raise ValueError(
        "Managed codex exec is refused until noninteractive admission, completion, cancellation, "
        "and exact-or-unavailable usage receipts can be guaranteed; use the interactive TUI."
    )


def main() -> None:
    # The TUI owns prompt entry. A positional prompt can still be preselected
    # before launch so its initial MCP scope matches the route.
    args_in = sys.argv[1:]
    command_index = _command_index(args_in)
    command_name = args_in[command_index] if command_index is not None else None
    literal_after_delimiter = (command_index is not None and "--" in args_in
                               and command_index == args_in.index("--") + 1)
    if command_name in {"exec", "e"}:
        raise SystemExit(run_routed_exec(args_in))
    passthrough_commands = {"login", "logout", "mcp", "plugin", "app-server",
                            "completion", "update", "doctor", "features", "help"}
    if ((not literal_after_delimiter and command_name in passthrough_commands)
            or (command_name is None and any(
            item in {"-h", "--help", "-V", "--version"} for item in args_in))):
        binary = real_codex_binary()
        os.execve(binary, [str(binary), *args_in], os.environ.copy())
    utility_commands = {"resume", "agents", "exec", "review", "login", "logout",
                        "mcp", "plugin", "app-server", "remote-control", "completion",
                        "update", "doctor", "sandbox", "debug", "apply", "queue",
                        "archive", "delete", "fork", "cloud", "features", "help"}
    if command_name == "start":
        args_in = [*args_in[:command_index], *args_in[command_index + 1:]]
        command_index = _command_index(args_in)
        command_name = args_in[command_index] if command_index is not None else None
    new_chat = literal_after_delimiter or command_name is None or command_name not in utility_commands
    preselected = False
    launch_ticket = None
    if new_chat and command_index is not None:
        explicit_model = _explicit_setting(args_in, "model") is not None
        explicit_effort = _explicit_setting(args_in, "model_reasoning_effort") is not None
        prompt = args_in[command_index]
        if prompt == "--":
            raise ValueError("A new managed chat requires a prompt after --.")
        append_prompt = False
        choice = _route_choice(prompt, args_in, "interactive_preselection")
        args_in = _routed_interactive_args(args_in, prompt, choice, append_prompt)
        preselected = True
        launch_ticket = _create_launch_ticket(choice, explicit_model=explicit_model,
                                               explicit_effort=explicit_effort)
    if command_name == "resume":
        resume_index = command_index + 1
        if len(args_in) <= resume_index or args_in[resume_index].startswith("-"):
            raise ValueError("Managed resume requires an exact thread UUID, not the session picker or --last.")
        resume_thread = args_in[resume_index]
        if not THREAD_ID_PATTERN.fullmatch(resume_thread):
            raise ValueError("Managed resume requires an exact thread UUID, not the session picker or --last.")
        authority_path = authority_path_for(resume_thread)
        if not authority_path.exists() and not authority_path.is_symlink():
            import_closed_thread(resume_thread)
    token = _read_token()
    ensure_proxy()
    ensure_proxy_supervisor()
    environment = os.environ.copy()
    client_mode = None
    binary = real_codex_binary()
    if preselected:
        client_mode = f"launch-{launch_ticket}"
    elif (_explicit_setting(args_in, "model") is not None
          and _explicit_setting(args_in, "model_reasoning_effort") is not None):
        client_mode = "explicit-both"
    elif _explicit_setting(args_in, "model") is not None:
        client_mode = "explicit-model"
    elif _explicit_setting(args_in, "model_reasoning_effort") is not None:
        client_mode = "explicit-effort"
    environment["MODEL_SELECTOR_HOST_TOKEN"] = token + (("." + client_mode) if client_mode else "")
    os.execve(
        binary,
        [str(binary), "--remote", PROXY_URL,
         "--remote-auth-token-env", "MODEL_SELECTOR_HOST_TOKEN", *args_in],
        environment,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"codex-model-host: {exc}", file=sys.stderr)
        raise SystemExit(1)
