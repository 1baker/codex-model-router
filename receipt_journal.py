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
RETIREMENT_SCHEMA = "modellabs.receipt_retirement.v1"
QUARANTINE_CLEANUP_SCHEMA = "modellabs.quarantine_cleanup.v1"


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
    state_directories = (JOURNAL_DIR.parent / "receipt-retirements",
                         QUARANTINE_DIR.parent / "quarantine-cleanups")
    for directory in (QUARANTINE_DIR, *state_directories):
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if (value.get("thread_id") == thread_id
                        and (directory == QUARANTINE_DIR
                             or value.get("state") in {"pending", "retired", "complete"})):
                    return True
            except (OSError, ValueError, TypeError, AttributeError):
                return True
    return False


def _load_retirement(marker: Path) -> dict[str, Any]:
    value = json.loads(marker.read_text(encoding="utf-8"))
    if (not isinstance(value, dict)
            or set(value) != {"schema", "state", "obligation", "thread_id", "turn_id"}
            or value.get("schema") != RETIREMENT_SCHEMA
            or value.get("state") not in {"pending", "retired", "confirmed", "complete"}
            or not all(isinstance(value.get(key), str) and value[key]
                       for key in ("obligation", "thread_id", "turn_id"))
            or Path(value["obligation"]).name != value["obligation"]):
        raise RuntimeError(f"Invalid receipt retirement {marker}.")
    if value["state"] == "complete":
        value["state"] = "retired"
    return value


def _retirement_value(path: Path, value: dict[str, Any], state: str) -> dict[str, Any]:
    return {"schema": RETIREMENT_SCHEMA, "state": state, "obligation": path.name,
            "thread_id": value["thread_id"], "turn_id": value["turn_id"]}


def _confirm_retirement(marker: Path, value: dict[str, Any]) -> None:
    retired = {**value, "state": "retired"}
    _atomic(marker, retired)
    try:
        _atomic(marker, {**retired, "state": "confirmed"})
    except Exception:
        # Preserve a discoverable retry claim when a testable I/O failure is
        # reported after replacement but before the directory barrier.
        try:
            _atomic(marker, retired)
        finally:
            raise


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
    if (isinstance(value, dict)
            and value.get("schema") == "modellabs.lifecycle_quarantine.v1"
            and set(value) == {"schema", "thread_id", "request_digest", "method"}
            and all(isinstance(value.get(key), str) and value[key]
                    for key in ("thread_id", "request_digest", "method"))):
        # A v1 record cannot prove a dispatched turn or reconstruct its route.
        # Keep it as an unresolved thread fence throughout a rolling upgrade.
        return {**value, "legacy": True, "route": None, "turn_id": None}
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
    marker = QUARANTINE_DIR.parent / "quarantine-cleanups" / path.name
    if path.exists():
        value = load_quarantine(path)
        cleanup = {"schema": QUARANTINE_CLEANUP_SCHEMA, "state": "pending",
                   "quarantine": path.name, "thread_id": value["thread_id"]}
        _atomic(marker, cleanup)
        path.unlink()
    elif marker.exists():
        cleanup = _load_quarantine_cleanup(marker)
    else:
        path.unlink(missing_ok=True)
        _sync_directories(path.parent, path.parent.parent)
        return
    _sync_directories(path.parent, path.parent.parent)
    _confirm_quarantine_cleanup(marker, cleanup)


def _load_quarantine_cleanup(marker: Path) -> dict[str, Any]:
    value = json.loads(marker.read_text(encoding="utf-8"))
    if (not isinstance(value, dict)
            or set(value) != {"schema", "state", "quarantine", "thread_id"}
            or value.get("schema") != QUARANTINE_CLEANUP_SCHEMA
            or value.get("state") not in {"pending", "retired", "confirmed"}
            or not isinstance(value.get("quarantine"), str)
            or Path(value["quarantine"]).name != value["quarantine"]
            or not isinstance(value.get("thread_id"), str) or not value["thread_id"]):
        raise RuntimeError(f"Invalid quarantine cleanup {marker}.")
    return value


def list_quarantine_cleanups() -> list[tuple[Path, dict[str, Any]]]:
    cleanup_dir = QUARANTINE_DIR.parent / "quarantine-cleanups"
    if not cleanup_dir.exists():
        return []
    result = []
    for marker in sorted(cleanup_dir.glob("*.json")):
        value = _load_quarantine_cleanup(marker)
        if value["state"] != "confirmed":
            result.append((marker, value))
    return result


def confirm_quarantine_cleanup(marker: Path) -> None:
    value = _load_quarantine_cleanup(marker)
    path = QUARANTINE_DIR / value["quarantine"]
    if path.exists():
        current = load_quarantine(path)
        if current["thread_id"] != value["thread_id"]:
            raise RuntimeError("Quarantine cleanup identity mismatch.")
        path.unlink()
    _sync_directories(path.parent, path.parent.parent)
    _confirm_quarantine_cleanup(marker, value)


def _confirm_quarantine_cleanup(marker: Path, value: dict[str, Any]) -> None:
    retired = {**value, "state": "retired"}
    _atomic(marker, retired)
    try:
        _atomic(marker, {**retired, "state": "confirmed"})
    except Exception:
        try:
            _atomic(marker, retired)
        finally:
            raise


def retirement_path_for(path: Path) -> Path:
    return JOURNAL_DIR.parent / "receipt-retirements" / path.name


def _sync_directories(*paths: Path) -> None:
    for directory_path in paths:
        if not directory_path.exists():
            continue
        directory = os.open(directory_path, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _begin_retirement(path: Path, value: dict[str, Any]) -> Path:
    marker = retirement_path_for(path)
    _atomic(marker, _retirement_value(path, value, "pending"))
    return marker


def _finish_retirement(path: Path, value: dict[str, Any]) -> None:
    marker = _begin_retirement(path, value)
    path.unlink(missing_ok=True)
    _sync_directories(path.parent, path.parent.parent)
    _confirm_retirement(marker, _retirement_value(path, value, "retired"))


def list_retirements() -> list[tuple[Path, dict[str, Any]]]:
    retirement_dir = JOURNAL_DIR.parent / "receipt-retirements"
    if not retirement_dir.exists():
        return []
    result = []
    for marker in sorted(retirement_dir.glob("*.json")):
        value = _load_retirement(marker)
        if value["state"] != "confirmed":
            result.append((marker, value))
    return result


def mark(path: Path, stage: str) -> dict[str, Any]:
    if stage not in {"accepted", "terminal", "usage"}:
        raise ValueError("Invalid receipt stage.")
    value = load(path)
    value[stage] = True
    _atomic(path, value)
    if value["accepted"] and value["terminal"] and value["usage"]:
        _finish_retirement(path, value)
    return value


def retire_if_complete(path: Path) -> bool:
    if not path.exists():
        marker = retirement_path_for(path)
        if marker.exists():
            confirm_retired(path)
            return True
        return False
    value = load(path)
    if not (value["accepted"] and value["terminal"] and value["usage"]):
        return False
    _finish_retirement(path, value)
    return True


def confirm_retired(path: Path) -> None:
    marker = retirement_path_for(path)
    if not marker.exists():
        if path.exists():
            raise RuntimeError("Receipt obligation is not retired.")
        _sync_directories(path.parent, path.parent.parent)
        return
    retirement = _load_retirement(marker)
    if path.exists():
        obligation = load(path)
        if (retirement["state"] != "pending"
                or retirement["obligation"] != path.name
                or retirement["thread_id"] != obligation["thread_id"]
                or retirement["turn_id"] != obligation["turn_id"]
                or not all(obligation[stage] for stage in ("accepted", "terminal", "usage"))):
            raise RuntimeError("Receipt obligation is not safely retireable.")
        _finish_retirement(path, obligation)
        return
    _sync_directories(path.parent, path.parent.parent)
    _confirm_retirement(marker, {**retirement, "state": "retired"})
