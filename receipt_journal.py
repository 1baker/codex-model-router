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
        parent = os.open(path.parent.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)


def _barrier(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    for directory_path in (path.parent, path.parent.parent):
        directory = os.open(directory_path, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def create(thread_id: str, turn_id: str, route: dict[str, Any]) -> dict[str, Any]:
    path = path_for(thread_id, turn_id)
    if path.exists():
        value = load(path)
        if (value["thread_id"] != thread_id or value["turn_id"] != turn_id
                or value["model"] != route["model"] or value["effort"] != route["effort"]
                or value["task_class"] != route["class"]):
            raise RuntimeError("Receipt obligation identity collision.")
        _barrier(path)
        return value
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


def quarantine(thread_id: str, lifecycle_id: str, method: str) -> Path:
    if not isinstance(lifecycle_id, str) or not lifecycle_id:
        raise ValueError("lifecycle identity must be a nonempty string")
    lifecycle_digest = hashlib.sha256(lifecycle_id.encode()).hexdigest()
    key = hashlib.sha256(f"{thread_id}\0{lifecycle_digest}\0{method}".encode()).hexdigest()
    path = QUARANTINE_DIR / f"{key}.json"
    _atomic(path, {"schema": "modellabs.lifecycle_quarantine.v2", "thread_id": thread_id,
                   "lifecycle_digest": lifecycle_digest, "method": method,
                   "route": None, "turn_id": None})
    return path


def bind_quarantine(path: Path, *, route: dict[str, Any] | None = None,
                    turn_id: str | None = None) -> dict[str, Any]:
    value = load_quarantine(path)
    if route is not None:
        value["route"] = {key: route.get(key) for key in
                          ("model", "effort", "class", "task_bucket", "adaptive_reason",
                           "prompt_sha256", "baseline_turn_id")}
    if turn_id is not None:
        value["turn_id"] = turn_id
    _atomic(path, value)
    return value


def load_quarantine(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict)
            or value.get("schema") != "modellabs.lifecycle_quarantine.v2"
            or not isinstance(value.get("thread_id"), str)
            or not isinstance(value.get("lifecycle_digest"), str)
            or not isinstance(value.get("method"), str)
            or value.get("route") is not None and not isinstance(value.get("route"), dict)
            or value.get("turn_id") is not None and not isinstance(value.get("turn_id"), str)):
        raise RuntimeError(f"Invalid lifecycle quarantine {path}.")
    return value


def list_quarantines() -> list[tuple[Path, dict[str, Any]]]:
    if not QUARANTINE_DIR.exists():
        return []
    return [(path, load_quarantine(path)) for path in sorted(QUARANTINE_DIR.glob("*.json"))]


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


def retire_if_complete(path: Path) -> bool:
    value = load(path)
    if not (value["accepted"] and value["terminal"] and value["usage"]):
        return False
    path.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return True


def confirm_retired(path: Path) -> None:
    if path.exists():
        raise RuntimeError("Receipt obligation is not retired.")
    if path.parent.exists():
        for directory_path in (path.parent, path.parent.parent):
            directory = os.open(directory_path, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
