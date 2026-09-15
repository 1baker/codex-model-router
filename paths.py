"""Resolve ModelLabs state without binding the project to one user or host."""

from __future__ import annotations

import os
from pathlib import Path


def modellabs_home() -> Path:
    configured = os.environ.get("MODELLABS_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return (data_home / "model-selector").resolve()


ROOT = modellabs_home()
CODEX_HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser().resolve()
