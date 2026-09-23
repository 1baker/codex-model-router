"""Install or upgrade ModelLabs without hard-coded user paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
    "adaptive_policy.py", "authority.py", "health_dashboard.py", "host_control.py", "install.py", "model_host_launcher.py", "model_host_mcp.py",
    "modellabs.py", "outcome_model.py", "paths.py", "prompt_hook.py", "proxy_supervisor.py", "telemetry.py",
    "protocol_policy.py", "receipt_journal.py", "thread_owner.py", "turn_proxy.py", "usage_observer.py", "smoke_bench.py",
]
SHELL_PATH_START = "# >>> ModelLabs managed Codex route >>>"
SHELL_PATH_END = "# <<< ModelLabs managed Codex route <<<"
OWNED_MANIFEST = "owned-files.json"
PINNED_CODEX_VERSION = "codex-cli 0.156.1"
AUTHORITY_SCHEMA = "modellabs.thread_authority.v1"


def default_home() -> Path:
    return Path(os.environ.get("MODELLABS_HOME", Path.home() / ".local/share/model-selector")).expanduser()


def set_private(path: Path) -> None:
    path.chmod(0o600)


def _digest_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _assert_lexical_destination(path: Path, *, allow_final_symlink: bool = False) -> None:
    lexical = path.expanduser().absolute()
    ancestor = lexical if not allow_final_symlink else lexical.parent
    while ancestor != ancestor.parent:
        if ancestor.is_symlink():
            raise RuntimeError(f"Refusing symlinked or redirected ModelLabs mutation path {ancestor}.")
        ancestor = ancestor.parent


def _atomic_bytes(path: Path, content: bytes, mode: int = 0o644) -> None:
    _assert_lexical_destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_lexical_destination(path)
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
                               codex_home: Path | None = None, bin_dir: Path | None = None) -> None:
    manifest_path = home / OWNED_MANIFEST
    manifest_valid = True
    has_manifest = manifest_path.exists() or manifest_path.is_symlink()
    try:
        manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = manifest_document.get("files") if isinstance(manifest_document, dict) else None
        manifest_valid = (isinstance(manifest_document, dict)
                          and set(manifest_document) == {"schema", "files"}
                          and manifest_document.get("schema") == "modellabs.owned_files.v1"
                          and isinstance(manifest, dict)
                          and all(isinstance(key, str) and isinstance(value, str)
                                  and len(value) == 64
                                  and all(character in "0123456789abcdef" for character in value)
                                  for key, value in manifest.items()))
    except (OSError, ValueError, TypeError):
        manifest = {}
        manifest_valid = not has_manifest
    try:
        state = json.loads((home / "install-state.json").read_text(encoding="utf-8"))
        legacy_install = Path(state.get("home", "")).expanduser().resolve() == home
    except (OSError, ValueError, TypeError):
        legacy_install = False
    codex_root = (codex_home or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))).expanduser()
    selected_bin = (bin_dir or Path.home() / ".local/bin").expanduser().absolute()
    wrappers = [selected_bin / name for name in
                ("codex", "codex-direct", "modellabs", "codex-model-host",
                 "modellabs-health", "modellabs-smoke", "modellabs-learn")]
    authority_dir = home / "thread-authority"
    authority_files = list(authority_dir.glob("*.json")) if authority_dir.is_dir() else []
    mutation_roots = [home, home / "venv", authority_dir, selected_bin, codex_root,
                      Path.home() / ".config/systemd/user"]
    ancillary = [home / OWNED_MANIFEST, home / "host-token", home / "real-codex-path",
                 home / "managed-codex-path", home / "install-state.json",
                 Path.home() / ".profile", Path.home() / ".bashrc",
                 codex_root / "config.toml", codex_root / "hooks.json"] + wrappers
    for path in mutation_roots + ancillary + authority_files + list(payloads):
        lexical = path.expanduser().absolute()
        ancestor = lexical
        while ancestor != ancestor.parent:
            if ancestor.is_symlink():
                if lexical == selected_bin / "codex" and ancestor == lexical:
                    break
                raise RuntimeError(f"Refusing symlinked or redirected ModelLabs mutation path {ancestor}.")
            ancestor = ancestor.parent
    for path in ancillary:
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() and path != selected_bin / "codex":
            raise RuntimeError(f"Refusing symlinked ModelLabs mutation destination {path}.")
        try:
            if path == selected_bin / "codex" and path.is_symlink():
                if path.resolve(strict=True) == upstream:
                    continue
            if os.path.samefile(path, upstream):
                raise RuntimeError(f"Refusing destination aliasing upstream Codex: {path}.")
        except OSError as exc:
            raise RuntimeError(f"Cannot validate ModelLabs mutation destination {path}.") from exc
    if has_manifest and (not manifest_valid or not isinstance(manifest, dict)):
        raise RuntimeError(f"Refusing unrelated ownership manifest {manifest_path}.")
    if has_manifest:
        for recorded_path, recorded_digest in manifest.items():
            owned = Path(recorded_path)
            if not owned.exists() or owned.is_symlink():
                raise RuntimeError(f"Owned manifest entry is missing or redirected: {owned}.")
            current = owned.read_bytes()
            if (_digest_bytes(current) != recorded_digest
                    and current != payloads.get(owned)):
                raise RuntimeError(f"Refusing unrelated modified owned content: {owned}.")
        for path in payloads:
            if path.exists() and str(path) not in manifest:
                raise RuntimeError(f"Owned manifest is missing existing payload {path}.")
    for path in authority_files:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Refusing malformed thread authority {path}.") from exc
        keys = set(value) if isinstance(value, dict) else set()
        legacy = {"thread_id", "model", "effort", "explicit_model", "explicit_effort"}
        current = legacy | {"schema"}
        if frozenset(keys) not in {frozenset(legacy), frozenset(current)}:
            raise RuntimeError(f"Refusing malformed thread authority {path}.")
        if (keys == current and value.get("schema") != AUTHORITY_SCHEMA
                or not isinstance(value.get("thread_id"), str)
                or not isinstance(value.get("explicit_model"), bool)
                or not isinstance(value.get("explicit_effort"), bool)
                or value.get("model") is not None and not isinstance(value.get("model"), str)
                or value.get("effort") is not None and value.get("effort") not in
                   {"none", "low", "medium", "high", "xhigh", "max", "ultra"}
                or value.get("explicit_model") and not value.get("model")
                or value.get("explicit_effort") and value.get("effort") is None):
            raise RuntimeError(f"Refusing malformed thread authority {path}.")
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
        if str(path) in manifest and actual == _digest_bytes(source_content):
            continue
        if not has_manifest and legacy_install and _legacy_owned(path, source_content):
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


def verify_upstream_protocol_version(upstream: Path) -> str:
    result = subprocess.run([str(upstream), "--version"], check=True, capture_output=True,
                            text=True, timeout=15)
    version = getattr(result, "stdout", PINNED_CODEX_VERSION).strip()
    if version != PINNED_CODEX_VERSION:
        raise RuntimeError(
            f"ModelLabs protocol policy is pinned to {PINNED_CODEX_VERSION}; found {version or 'unknown'}."
        )
    return version


def verify_existing_venv_containment(venv: Path) -> None:
    if not venv.exists():
        return
    root = venv.resolve(strict=True)
    python = venv / "bin/python"
    if not python.exists():
        return
    for path in venv.rglob("*"):
        if not path.is_symlink():
            continue
        relative = path.relative_to(venv)
        if (len(relative.parts) == 2 and relative.parts[0] == "bin"
                and (relative.name in {"python", "python3"}
                     or re.fullmatch(r"python3\.\d+", relative.name))):
            target = path.resolve(strict=True)
            if not target.is_file() or not os.access(target, os.X_OK):
                raise RuntimeError(f"Refusing invalid virtualenv interpreter link {path}.")
            continue
        try:
            path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Refusing escaping virtualenv link {path}.") from exc
    result = subprocess.run([str(python), "-c", "import sys; print(sys.prefix)"],
                            check=True, capture_output=True, text=True, timeout=15)
    if Path(result.stdout.strip()).resolve(strict=True) != root:
        raise RuntimeError("Existing virtualenv interpreter does not belong to the ModelLabs venv.")


def migrate_authority_documents(home: Path) -> None:
    directory = home / "thread-authority"
    if not directory.exists():
        return
    for path in directory.glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if "schema" not in value:
            value = {"schema": AUTHORITY_SCHEMA, **value}
            _atomic_bytes(path, (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(), 0o600)


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
    _atomic_bytes(config, text.encode())

    hooks_path = codex_home / "hooks.json"
    if hooks_path.exists():
        hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
        command = f"{home / 'venv/bin/python'} {home / 'prompt_hook.py'}"
        groups = hooks.get("hooks", {}).get("UserPromptSubmit", [])
        changed = False
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                continue
            entries = group["hooks"]
            retained = [item for item in entries if not (isinstance(item, dict) and item.get("command") == command)]
            if len(retained) != len(entries):
                group["hooks"] = retained
                changed = True
        if changed:
            _atomic_bytes(hooks_path, (json.dumps(hooks, indent=2) + "\n").encode())


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
        _assert_lexical_destination(path, allow_final_symlink=path == bin_dir / "codex")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o755)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            _assert_lexical_destination(path, allow_final_symlink=path == bin_dir / "codex")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

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
               ("modellabs-health", "health_dashboard.py"), ("modellabs-smoke", "smoke_bench.py"),
               ("modellabs-learn", "outcome_model.py"))
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
    bin_dir.mkdir(parents=True, exist_ok=True)
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
    _atomic_bytes(path, text.encode())


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
    for name in ("adaptive_policy.py", "authority.py", "host_control.py", "modellabs.py",
                 "paths.py", "protocol_policy.py", "receipt_journal.py", "telemetry.py",
                 "thread_owner.py", "turn_proxy.py"):
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
    home = args.home.expanduser().absolute()
    real_codex = discover_real_codex(home, args.bin_dir)
    payloads = _install_payloads(home, args.codex_home, include_service=not args.no_service)
    preflight_install_payloads(payloads, home, real_codex, args.codex_home, args.bin_dir)
    write_wrappers(home, args.bin_dir, real_codex, dry_run=True)
    upstream_version = verify_upstream_protocol_version(real_codex)
    verify_existing_venv_containment(home / "venv")
    home.mkdir(parents=True, exist_ok=True)
    migrate_authority_documents(home)
    for path, content in payloads.items():
        _atomic_bytes(path, content)
    (home / "systemd").mkdir(exist_ok=True)
    token = home / "host-token"
    if not token.exists():
        _atomic_bytes(token, (secrets.token_urlsafe(32) + "\n").encode(), 0o600)
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
    _atomic_bytes(home / "real-codex-path", (str(real_codex) + "\n").encode(), 0o600)
    set_private(home / "real-codex-path")
    _atomic_bytes(home / "managed-codex-path",
                  (str((args.bin_dir / "codex").absolute()) + "\n").encode(), 0o600)
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
    _atomic_bytes(state, (json.dumps({"home": str(home), "service": str(service),
                                      "service_enabled": service_enabled,
                                      "real_codex": str(real_codex),
                                      "managed_codex": str(args.bin_dir / 'codex'),
                                      "upstream_version": upstream_version}, indent=2) + "\n").encode(), 0o600)
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
