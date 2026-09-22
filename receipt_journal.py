"""Crash-durable, idempotent accepted-turn receipt obligations."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any

from paths import ROOT


SCHEMA = "modellabs.receipt_obligation.v1"
JOURNAL_DIR = ROOT / "receipt-obligations"
QUARANTINE_DIR = ROOT / "lifecycle-quarantine"


def _key(thread_id: str, turn_id: str) -> str:
    return hashlib.sha256(f"{thread_id}\0{turn_id}".encode()).hexdigest()


def path_for(thread_id: str, turn_id: str) -> Path:
    return JOURNAL_DIR / f"{_key(thread_id, turn_id)}.json"


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def create(thread_id: str, turn_id: str, route: dict[str, Any]) -> dict[str, Any]:
    path = path_for(thread_id, turn_id)
    if path.exists():
        return load(path)
    value = {
        "schema": SCHEMA, "thread_id": thread_id, "turn_id": turn_id,
        "model": route["model"], "effort": route["effort"], "task_class": route["class"],
        "accepted": False, "terminal": False, "usage": False,
    }
    _atomic(path, value)
    return value


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema", "thread_id", "turn_id", "model", "effort", "task_class",
                "accepted", "terminal", "usage"}
    if (not isinstance(value, dict) or set(value) != required or value.get("schema") != SCHEMA
            or not all(isinstance(value.get(key), str) and value[key] for key in
                       ("thread_id", "turn_id", "model", "effort", "task_class"))
            or not all(isinstance(value.get(key), bool) for key in ("accepted", "terminal", "usage"))):
        raise RuntimeError(f"Invalid receipt obligation {path}.")
    return value


def list_obligations() -> list[tuple[Path, dict[str, Any]]]:
    if not JOURNAL_DIR.exists():
        return []
    return [(path, load(path)) for path in sorted(JOURNAL_DIR.glob("*.json"))]


def unresolved_for_thread(thread_id: str) -> bool:
    if any(value["thread_id"] == thread_id for _path, value in list_obligations()):
        return True
    if not QUARANTINE_DIR.exists():
        return False
    for path in QUARANTINE_DIR.glob("*.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("thread_id") == thread_id:
                return True
        except (OSError, ValueError, TypeError):
            return True
    return False


def quarantine(thread_id: str, request_id: object, method: str) -> Path:
    request_digest = hashlib.sha256(str(request_id).encode()).hexdigest()
    key = hashlib.sha256(f"{thread_id}\0{request_digest}\0{method}".encode()).hexdigest()
    path = QUARANTINE_DIR / f"{key}.json"
    _atomic(path, {"schema": "modellabs.lifecycle_quarantine.v1", "thread_id": thread_id,
                   "request_digest": request_digest, "method": method})
    return path


def clear_quarantine(path: Path) -> None:
    path.unlink(missing_ok=True)
    if path.parent.exists():
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def mark(path: Path, stage: str) -> dict[str, Any]:
    if stage not in {"accepted", "terminal", "usage"}:
        raise ValueError("Invalid receipt stage.")
    value = load(path)
    value[stage] = True
    _atomic(path, value)
    if value["accepted"] and value["terminal"] and value["usage"]:
        path.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return value
