"""Resolve ModelLabs state without binding the project to one user or host."""

from __future__ import annotations

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
    shim = Path.home() / ".local/bin/codex"
    for candidate in candidates:
        try:
            if candidate == shim or not candidate.is_file() or not os.access(candidate, os.X_OK):
                continue
            return candidate
        except OSError:
            continue
    raise RuntimeError(
        "The real Codex executable is unavailable. Reinstall ModelLabs after installing Codex, "
        "or set MODELLABS_REAL_CODEX to its absolute path."
    )
