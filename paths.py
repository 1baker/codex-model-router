"""Resolve ModelLabs state without binding the project to one user or host."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


def modellabs_home() -> Path:
    configured = os.environ.get("MODELLABS_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return (data_home / "model-selector").resolve()


ROOT = modellabs_home()
CODEX_HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser().resolve()
REAL_CODEX_PATH_FILE = ROOT / "real-codex-path"
MANAGED_CODEX_PATH_FILE = ROOT / "managed-codex-path"
RUNTIME_SOURCE = Path(__file__).resolve().parent


def proxy_revision() -> str:
    """Identify the exact proxy/accounting implementation loaded at runtime."""
    digest = hashlib.sha256()
    for name in ("adaptive_policy.py", "authority.py", "host_control.py", "modellabs.py",
                 "paths.py", "protocol_policy.py", "receipt_journal.py", "telemetry.py",
                 "thread_owner.py", "turn_proxy.py"):
        digest.update(name.encode())
        digest.update((RUNTIME_SOURCE / name).read_bytes())
    return digest.hexdigest()[:16]


def proxy_port(revision: str) -> int:
    """Use a release-specific loopback generation so old sessions can drain."""
    return 46000 + int(revision[:8], 16) % 16000


def real_codex_binary() -> Path:
    """Return the underlying Codex executable, never the ModelLabs shim."""
    configured = os.environ.get("MODELLABS_REAL_CODEX")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    try:
        saved = REAL_CODEX_PATH_FILE.read_text(encoding="utf-8").strip()
        if saved:
            candidates.append(Path(saved).expanduser())
    except OSError:
        pass
    candidates.extend([
        Path.home() / ".npm-global/bin/codex",
        Path("/usr/local/bin/codex"),
        Path("/snap/bin/codex"),
    ])
    discovered = shutil.which("codex")
    if discovered:
        candidates.append(Path(discovered))
    managed = Path.home() / ".local/bin/codex"
    try:
        saved_managed = MANAGED_CODEX_PATH_FILE.read_text(encoding="utf-8").strip()
        if saved_managed:
            managed = Path(saved_managed).expanduser().absolute()
    except OSError:
        pass
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            if resolved == managed.resolve(strict=False) or not resolved.is_file() or not os.access(resolved, os.X_OK):
                continue
            if managed.exists() and not managed.is_symlink() and os.path.samefile(resolved, managed):
                continue
            prefix = resolved.read_bytes()[:4096]
            if b"ModelLabs managed wrapper" in prefix or b"ModelLabs recovery wrapper" in prefix:
                continue
            if b"model_host_launcher.py" in prefix:
                continue
            return resolved
        except OSError:
            continue
    raise RuntimeError(
        "The real Codex executable is unavailable. Reinstall ModelLabs after installing Codex, "
        "or set MODELLABS_REAL_CODEX to its absolute path."
    )
