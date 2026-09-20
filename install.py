"""Install or upgrade ModelLabs without hard-coded user paths."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path


SOURCE = Path(__file__).resolve().parent
PYTHON_FILES = [
    "adaptive_policy.py", "health_dashboard.py", "host_control.py", "install.py", "model_host_launcher.py", "model_host_mcp.py",
    "modellabs.py", "paths.py", "prompt_hook.py", "proxy_supervisor.py", "telemetry.py",
    "thread_owner.py", "turn_proxy.py", "usage_observer.py",
]


def default_home() -> Path:
    return Path(os.environ.get("MODELLABS_HOME", Path.home() / ".local/share/model-selector")).expanduser()


def set_private(path: Path) -> None:
    path.chmod(0o600)


def create_venv(venv: Path) -> None:
    if (venv / "bin/python").exists():
        return
    if venv.exists():
        backup = venv.with_name(f"{venv.name}.incomplete-{int(time.time())}")
        venv.rename(backup)
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, capture_output=True)
        return
    except subprocess.CalledProcessError:
        if venv.exists():
            backup = venv.with_name(f"{venv.name}.incomplete-{int(time.time())}")
            venv.rename(backup)
    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "venv", "--python", sys.executable, str(venv)], check=True, capture_output=True)
        return
    virtualenv = shutil.which("virtualenv")
    if virtualenv:
        subprocess.run([virtualenv, "--python", sys.executable, str(venv)], check=True, capture_output=True)
        return
    raise RuntimeError("Python venv support is unavailable; install uv, virtualenv, or python3-venv.")


def upsert_toml(text: str, table: str, values: dict[str, str]) -> str:
    lines = text.splitlines()
    header = f"[{table}]"
    try:
        start = lines.index(header)
    except ValueError:
        if lines and lines[-1]:
            lines.append("")
        lines.append(header)
        lines.extend(f"{key} = {value}" for key, value in values.items())
        return "\n".join(lines) + "\n"
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("[")), len(lines))
    for key, value in values.items():
        prefix = f"{key} ="
        match = next((i for i in range(start + 1, end) if lines[i].strip().startswith(prefix)), None)
        if match is None:
            lines.insert(end, f"{key} = {value}")
            end += 1
        else:
            lines[match] = f"{key} = {value}"
    return "\n".join(lines) + "\n"


def configure_codex(codex_home: Path, home: Path) -> None:
    codex_home.mkdir(parents=True, exist_ok=True)
    config = codex_home / "config.toml"
    text = config.read_text(encoding="utf-8") if config.exists() else ""
    command = json.dumps(str(home / "venv/bin/python"))
    args = json.dumps([str(home / "model_host_mcp.py")])
    text = upsert_toml(text, "mcp_servers.modelControl", {"command": command, "args": args,
                                                            "startup_timeout_sec": "10.0", "tool_timeout_sec": "20.0"})
    text = upsert_toml(text, "features", {"step_model_switching": "true"})
    config.write_text(text, encoding="utf-8")

    hooks_path = codex_home / "hooks.json"
    hooks = json.loads(hooks_path.read_text(encoding="utf-8")) if hooks_path.exists() else {"hooks": {}}
    command = f"{home / 'venv/bin/python'} {home / 'prompt_hook.py'}"
    groups = hooks.setdefault("hooks", {}).setdefault("UserPromptSubmit", [{"hooks": []}])
    if not groups:
        groups.append({"hooks": []})
    entries = groups[0].setdefault("hooks", [])
    if not any(item.get("command") == command for item in entries if isinstance(item, dict)):
        entries.append({"command": command, "statusMessage": "Routing prompt with ModelLabs", "timeout": 10,
                        "type": "command"})
    hooks_path.write_text(json.dumps(hooks, indent=2) + "\n", encoding="utf-8")


def write_wrappers(home: Path, bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, module in (("modellabs", "modellabs.py"), ("codex-model-host", "model_host_launcher.py"),
                         ("modellabs-health", "health_dashboard.py")):
        path = bin_dir / name
        path.write_text(f"#!/bin/sh\nexec {home / 'venv/bin/python'} {home / module} \"$@\"\n", encoding="utf-8")
        path.chmod(0o755)


def write_service(home: Path) -> Path:
    unit = (SOURCE / "systemd/modellabs-proxy.service.in").read_text(encoding="utf-8")
    unit = unit.replace("@HOME@", str(Path.home())).replace("@MODELLABS_HOME@", str(home))
    unit = unit.replace("@PYTHON@", str(home / "venv/bin/python"))
    path = Path.home() / ".config/systemd/user/modellabs-proxy.service"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(unit, encoding="utf-8")
    return path


def enable_service() -> bool:
    env = os.environ.copy()
    runtime = Path(f"/run/user/{os.getuid()}")
    if runtime.is_dir():
        env.setdefault("XDG_RUNTIME_DIR", str(runtime))
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime}/bus")
    command = ["systemctl", "--user", "daemon-reload"]
    if subprocess.run(command, env=env, capture_output=True).returncode:
        return False
    return subprocess.run(["systemctl", "--user", "enable", "--now", "modellabs-proxy.service"],
                          env=env, capture_output=True).returncode == 0


def install(args: argparse.Namespace) -> None:
    home = args.home.expanduser().resolve()
    home.mkdir(parents=True, exist_ok=True)
    for name in PYTHON_FILES + ["requirements.txt"]:
        shutil.copy2(SOURCE / name, home / name)
    (home / "systemd").mkdir(exist_ok=True)
    token = home / "host-token"
    if not token.exists():
        token.write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8")
    set_private(token)
    venv = home / "venv"
    create_venv(venv)
    pip = subprocess.run([str(venv / "bin/python"), "-m", "pip", "--version"], capture_output=True)
    if pip.returncode == 0:
        subprocess.run([str(venv / "bin/python"), "-m", "pip", "install", "--quiet", "-r", str(home / "requirements.txt")], check=True)
    else:
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("The created environment has no pip; install uv or recreate it with python3-venv.")
        subprocess.run([uv, "pip", "install", "--quiet", "--python", str(venv / "bin/python"), "-r", str(home / "requirements.txt")], check=True, capture_output=True)
    skill = args.codex_home / "skills/model-selector"
    skill.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / "SKILL.md", skill / "SKILL.md")
    (skill / "agents").mkdir(exist_ok=True)
    shutil.copy2(SOURCE / "openai.yaml", skill / "agents/openai.yaml")
    configure_codex(args.codex_home, home)
    write_wrappers(home, args.bin_dir)
    service = write_service(home)
    service_enabled = False if args.no_service else enable_service()
    state = home / "install-state.json"
    state.write_text(json.dumps({"home": str(home), "service": str(service), "service_enabled": service_enabled}, indent=2) + "\n")
    set_private(state)
    print(json.dumps({"home": str(home), "bin_dir": str(args.bin_dir), "codex_home": str(args.codex_home),
                      "service": str(service), "service_enabled": service_enabled}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Install ModelLabs for the current user")
    parser.add_argument("--home", type=Path, default=default_home())
    parser.add_argument("--codex-home", type=Path, default=Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")))
    parser.add_argument("--bin-dir", type=Path, default=Path.home() / ".local/bin")
    parser.add_argument("--no-service", action="store_true")
    install(parser.parse_args())


if __name__ == "__main__":
    main()
