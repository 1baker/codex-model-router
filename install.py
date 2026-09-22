"""Install or upgrade ModelLabs without hard-coded user paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import shlex
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
    "thread_owner.py", "turn_proxy.py", "usage_observer.py", "smoke_bench.py",
]
SHELL_PATH_START = "# >>> ModelLabs managed Codex route >>>"
SHELL_PATH_END = "# <<< ModelLabs managed Codex route <<<"
OWNED_MANIFEST = "owned-files.json"


def default_home() -> Path:
    return Path(os.environ.get("MODELLABS_HOME", Path.home() / ".local/share/model-selector")).expanduser()


def set_private(path: Path) -> None:
    path.chmod(0o600)


def _digest_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_bytes(path: Path, content: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _service_bytes(home: Path) -> bytes:
    unit = (SOURCE / "systemd/modellabs-proxy.service.in").read_text(encoding="utf-8")
    unit = unit.replace("@HOME@", str(Path.home())).replace("@MODELLABS_HOME@", str(home))
    unit = unit.replace("@PYTHON@", str(home / "venv/bin/python"))
    return unit.encode()


def _install_payloads(home: Path, codex_home: Path, *, include_service: bool = True) -> dict[Path, bytes]:
    payloads = {home / name: (SOURCE / name).read_bytes()
                for name in PYTHON_FILES + ["requirements.txt"]}
    for source in (SOURCE / "benchmarks").rglob("*"):
        if source.is_file():
            payloads[home / "benchmarks" / source.relative_to(SOURCE / "benchmarks")] = source.read_bytes()
    payloads[codex_home / "skills/model-selector/SKILL.md"] = (SOURCE / "SKILL.md").read_bytes()
    payloads[codex_home / "skills/model-selector/agents/openai.yaml"] = (SOURCE / "openai.yaml").read_bytes()
    if include_service:
        payloads[Path.home() / ".config/systemd/user/modellabs-proxy.service"] = _service_bytes(home)
    return payloads


def _legacy_owned(path: Path, source_content: bytes) -> bool:
    """Recognize only narrow artifacts written by pre-manifest ModelLabs."""
    try:
        content = path.read_bytes()
    except OSError:
        return False
    if path.name in PYTHON_FILES:
        return content.splitlines()[:1] == source_content.splitlines()[:1]
    markers = {
        "requirements.txt": b"websockets",
        "SKILL.md": b"name: model-selector",
        "openai.yaml": b"Model Selector",
        "modellabs-proxy.service": b"Description=ModelLabs",
        "smoke.json": b'"schema"',
    }
    marker = markers.get(path.name)
    return marker is not None and marker in content


def preflight_install_payloads(payloads: dict[Path, bytes], home: Path, upstream: Path,
                               codex_home: Path | None = None) -> None:
    manifest_path = home / OWNED_MANIFEST
    manifest_valid = True
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
    except (OSError, ValueError, TypeError):
        manifest = {}
        manifest_valid = not manifest_path.exists()
    try:
        state = json.loads((home / "install-state.json").read_text(encoding="utf-8"))
        legacy_install = Path(state.get("home", "")).expanduser().resolve() == home
    except (OSError, ValueError, TypeError):
        legacy_install = False
    codex_root = (codex_home or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))).expanduser()
    ancillary = [home / OWNED_MANIFEST, home / "host-token", home / "real-codex-path",
                 home / "managed-codex-path", home / "install-state.json",
                 Path.home() / ".profile", Path.home() / ".bashrc",
                 codex_root / "config.toml", codex_root / "hooks.json"]
    for path in ancillary:
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink():
            raise RuntimeError(f"Refusing symlinked ModelLabs mutation destination {path}.")
        try:
            if os.path.samefile(path, upstream):
                raise RuntimeError(f"Refusing destination aliasing upstream Codex: {path}.")
        except OSError as exc:
            raise RuntimeError(f"Cannot validate ModelLabs mutation destination {path}.") from exc
    if manifest_path.exists() and (not manifest_valid or not isinstance(manifest, dict)):
        raise RuntimeError(f"Refusing unrelated ownership manifest {manifest_path}.")
    for path, source_content in payloads.items():
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink():
            raise RuntimeError(f"Refusing to replace symlinked ModelLabs destination {path}.")
        try:
            if os.path.samefile(path, upstream):
                raise RuntimeError(f"Refusing to replace destination aliasing upstream Codex: {path}.")
            actual = _digest_bytes(path.read_bytes())
        except OSError as exc:
            raise RuntimeError(f"Cannot validate ModelLabs destination {path}.") from exc
        if manifest.get(str(path)) == actual:
            continue
        if legacy_install and _legacy_owned(path, source_content):
            continue
        raise RuntimeError(f"Refusing to replace unrelated ModelLabs destination {path}.")


def write_owned_manifest(home: Path, payloads: dict[Path, bytes]) -> None:
    content = json.dumps({"schema": "modellabs.owned_files.v1",
                          "files": {str(path): _digest_bytes(value)
                                    for path, value in sorted(payloads.items(), key=lambda item: str(item[0]))}},
                         indent=2, sort_keys=True).encode() + b"\n"
    _atomic_bytes(home / OWNED_MANIFEST, content, 0o600)


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


def discover_real_codex(home: Path, bin_dir: Path) -> Path:
    candidates: list[Path] = []
    configured = os.environ.get("MODELLABS_REAL_CODEX")
    if configured:
        candidates.append(Path(configured).expanduser())
    state = home / "install-state.json"
    try:
        saved = json.loads(state.read_text(encoding="utf-8")).get("real_codex")
        if saved:
            candidates.append(Path(saved).expanduser())
    except (OSError, ValueError, TypeError):
        pass
    current = shutil.which("codex")
    if current:
        candidates.append(Path(current))
    candidates.extend([
        Path.home() / ".npm-global/bin/codex",
        Path("/usr/local/bin/codex"),
        Path("/snap/bin/codex"),
    ])
    shim = (bin_dir.expanduser().absolute() / "codex")
    recovery = (bin_dir.expanduser().absolute() / "codex-direct")
    for candidate in candidates:
        candidate = candidate.expanduser()
        try:
            resolved = candidate.resolve(strict=True)
            lexical = candidate.absolute()
            # A symlink at the managed path may still be the only legitimate
            # upstream installation. Preserve its resolved target. A regular
            # managed entry and the recovery wrapper are never upstreams.
            if lexical == recovery or (lexical == shim and not shim.is_symlink()):
                continue
            if resolved.is_file() and os.access(resolved, os.X_OK):
                prefix = resolved.read_bytes()[:4096]
                if b"ModelLabs managed wrapper" in prefix or b"ModelLabs recovery wrapper" in prefix:
                    continue
                if b"model_host_launcher.py" in prefix:
                    continue
                if recovery.exists() and os.path.samefile(resolved, recovery):
                    continue
                return resolved
        except OSError:
            continue
    raise RuntimeError("Cannot install the managed codex route because the real Codex executable was not found.")


def write_wrappers(home: Path, bin_dir: Path, real_codex: Path, *, dry_run: bool = False) -> None:
    def atomic_executable(path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o755)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    bin_dir.mkdir(parents=True, exist_ok=True)
    upstream = real_codex.expanduser().resolve(strict=True)
    if not upstream.is_file() or not os.access(upstream, os.X_OK):
        raise RuntimeError("The selected upstream Codex executable is not executable.")
    prefix = upstream.read_bytes()[:4096]
    if (b"ModelLabs managed wrapper" in prefix or b"ModelLabs recovery wrapper" in prefix
            or b"model_host_launcher.py" in prefix):
        raise RuntimeError("A ModelLabs wrapper cannot be used as the upstream Codex executable.")
    direct = bin_dir / "codex-direct"
    if direct.exists() and os.path.samefile(upstream, direct):
        raise RuntimeError("codex-direct cannot be used as its own recovery target.")
    codex = bin_dir / "codex"
    modules = (("modellabs", "modellabs.py"), ("codex-model-host", "model_host_launcher.py"),
               ("modellabs-health", "health_dashboard.py"), ("modellabs-smoke", "smoke_bench.py"))
    contents: dict[Path, str] = {}
    legacy: dict[Path, str] = {}
    for name, module in modules:
        path = bin_dir / name
        legacy[path] = f"#!/bin/sh\nexec {home / 'venv/bin/python'} {home / module} \"$@\"\n"
        contents[path] = (f"#!/bin/sh\n# ModelLabs auxiliary wrapper\n"
                          f"exec {home / 'venv/bin/python'} {home / module} \"$@\"\n")
    contents[codex] = (
        f"#!/bin/sh\n# ModelLabs managed wrapper\nexec {shlex.quote(str(home / 'venv/bin/python'))} "
        f"{shlex.quote(str(home / 'model_host_launcher.py'))} \"$@\"\n"
    )
    contents[direct] = (
        f"#!/bin/sh\n# ModelLabs recovery wrapper\nexec {shlex.quote(str(upstream))} \"$@\"\n"
    )

    # Preflight every destination before replacing any of them. This prevents
    # a partial installation from overwriting unrelated user executables.
    for path, content in contents.items():
        if not path.exists() and not path.is_symlink():
            continue
        if path == codex and path.is_symlink():
            try:
                if path.resolve(strict=True) == upstream:
                    continue
            except OSError:
                pass
        try:
            existing = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"Refusing to replace unrelated executable {path}.") from exc
        if ("# ModelLabs managed wrapper" in existing
                or "# ModelLabs recovery wrapper" in existing
                or "# ModelLabs auxiliary wrapper" in existing
                or existing == legacy.get(path)):
            continue
        raise RuntimeError(f"Refusing to replace unrelated executable {path}.")

    if dry_run:
        return
    for path, content in contents.items():
        atomic_executable(path, content)


def ensure_managed_route_precedence(path: Path, bin_dir: Path) -> None:
    """Idempotently put the ModelLabs shim first for future interactive shells."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = (
        f"{SHELL_PATH_START}\n"
        f"export PATH={shlex.quote(str(bin_dir.expanduser().resolve()))}:\"$PATH\"\n"
        f"{SHELL_PATH_END}"
    )
    if SHELL_PATH_START in text:
        before, remainder = text.split(SHELL_PATH_START, 1)
        if SHELL_PATH_END not in remainder:
            raise RuntimeError(f"Incomplete ModelLabs PATH block in {path}")
        _, after = remainder.split(SHELL_PATH_END, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def write_service(home: Path) -> Path:
    path = Path.home() / ".config/systemd/user/modellabs-proxy.service"
    _atomic_bytes(path, _service_bytes(home))
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
    enabled = subprocess.run(["systemctl", "--user", "enable", "modellabs-proxy.service"],
                             env=env, capture_output=True).returncode == 0
    # Restart only the lightweight supervisor. Release-specific proxy children
    # are detached and keep serving existing TUI connections while the new
    # supervisor starts the new revision on its own loopback port.
    restarted = subprocess.run(["systemctl", "--user", "restart", "modellabs-proxy.service"],
                               env=env, capture_output=True).returncode == 0
    return enabled and restarted


def installed_proxy_port(home: Path) -> int:
    digest = hashlib.sha256()
    for name in ("turn_proxy.py", "telemetry.py", "modellabs.py", "adaptive_policy.py"):
        digest.update(name.encode())
        digest.update((home / name).read_bytes())
    return 46000 + int(digest.hexdigest()[:8], 16) % 16000


def retire_old_supervisors(home: Path, current_port: int) -> None:
    """Stop obsolete watchdogs without touching their live proxy children."""
    expected_script = str(home / "proxy_supervisor.py").encode()
    for pid_file in home.glob("proxy-supervisor-*.pid"):
        if pid_file.name == f"proxy-supervisor-{current_port}.pid":
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if expected_script not in cmdline:
                continue
            os.kill(pid, signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
            continue


def install(args: argparse.Namespace) -> None:
    home = args.home.expanduser().resolve()
    real_codex = discover_real_codex(home, args.bin_dir)
    write_wrappers(home, args.bin_dir, real_codex, dry_run=True)
    payloads = _install_payloads(home, args.codex_home, include_service=not args.no_service)
    preflight_install_payloads(payloads, home, real_codex, args.codex_home)
    home.mkdir(parents=True, exist_ok=True)
    for path, content in payloads.items():
        _atomic_bytes(path, content)
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
    configure_codex(args.codex_home, home)
    (home / "real-codex-path").write_text(str(real_codex) + "\n", encoding="utf-8")
    set_private(home / "real-codex-path")
    (home / "managed-codex-path").write_text(str((args.bin_dir / "codex").absolute()) + "\n", encoding="utf-8")
    set_private(home / "managed-codex-path")
    write_wrappers(home, args.bin_dir, real_codex)
    for shell_file in (Path.home() / ".profile", Path.home() / ".bashrc"):
        ensure_managed_route_precedence(shell_file, args.bin_dir)
    service = (Path.home() / ".config/systemd/user/modellabs-proxy.service"
               if args.no_service else write_service(home))
    write_owned_manifest(home, payloads)
    service_enabled = False if args.no_service else enable_service()
    retire_old_supervisors(home, installed_proxy_port(home))
    state = home / "install-state.json"
    state.write_text(json.dumps({"home": str(home), "service": str(service), "service_enabled": service_enabled,
                                 "real_codex": str(real_codex), "managed_codex": str(args.bin_dir / 'codex')}, indent=2) + "\n")
    set_private(state)
    print(json.dumps({"home": str(home), "bin_dir": str(args.bin_dir), "codex_home": str(args.codex_home),
                      "real_codex": str(real_codex), "managed_codex": str(args.bin_dir / 'codex'),
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
