"""Private, prompt-free routing and completion telemetry."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_PATH = Path("/home/bak3r/.local/share/model-selector/metrics.jsonl")
METRICS_PATH = Path(os.environ.get("MODELLABS_METRICS_PATH", DEFAULT_PATH))


def record(event: str, **fields: Any) -> None:
    """Append metadata only; callers must never supply prompt text or tokens."""
    payload = {"event": event, "recorded_at_ms": int(time.time() * 1000), **fields}
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(METRICS_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.chmod(METRICS_PATH, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as out:
        out.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def usage_from(params: dict[str, Any]) -> Any:
    usage = params.get("usage")
    return usage if isinstance(usage, dict) else None
