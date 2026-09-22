"""Strict, lock-serialized explicit model-choice authority."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any

from paths import ROOT


SCHEMA = "modellabs.thread_authority.v1"
EFFORTS = {"none", "low", "medium", "high", "xhigh", "max", "ultra"}


class AuthorityError(RuntimeError):
    pass


def path_for(thread_id: str) -> Path:
    return ROOT / "thread-authority" / f"{hashlib.sha256(thread_id.encode()).hexdigest()}.json"


def lock_path_for(thread_id: str) -> Path:
    return path_for(thread_id).with_suffix(".lock")


def acquire_lock(thread_id: str) -> int:
    directory = path_for(thread_id).parent
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    descriptor = os.open(lock_path_for(thread_id), os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def _validate(payload: Any, thread_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema", "thread_id", "model", "effort", "explicit_model", "explicit_effort"
    }:
        raise AuthorityError("Thread choice authority has an invalid schema.")
    if payload["schema"] != SCHEMA or payload["thread_id"] != thread_id:
        raise AuthorityError("Thread choice authority does not match this managed thread.")
    if not isinstance(payload["explicit_model"], bool) or not isinstance(payload["explicit_effort"], bool):
        raise AuthorityError("Thread choice authority pin flags are invalid.")
    if payload["model"] is not None and not isinstance(payload["model"], str):
        raise AuthorityError("Thread choice authority model is invalid.")
    if payload["effort"] is not None and payload["effort"] not in EFFORTS:
        raise AuthorityError("Thread choice authority effort is invalid.")
    if payload["explicit_model"] and not payload["model"]:
        raise AuthorityError("Pinned model authority is missing its model.")
    if payload["explicit_effort"] and payload["effort"] is None:
        raise AuthorityError("Pinned effort authority is missing its effort.")
    return payload


def read_locked(thread_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(path_for(thread_id).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise AuthorityError("Thread choice authority is unavailable or invalid.") from exc
    return _validate(payload, thread_id)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
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


def initialize_locked(thread_id: str, model: str | None, effort: str | None, *,
                      explicit_model: bool, explicit_effort: bool) -> dict[str, Any]:
    path = path_for(thread_id)
    if path.exists() or path.is_symlink():
        raise AuthorityError("Refusing to replace existing authority during thread creation.")
    payload = _validate({
        "schema": SCHEMA,
        "thread_id": thread_id,
        "model": model if explicit_model else None,
        "effort": effort if explicit_effort else None,
        "explicit_model": explicit_model,
        "explicit_effort": explicit_effort,
    }, thread_id)
    _atomic_write(path, payload)
    return payload


def update_locked(thread_id: str, model: str | None, effort: str | None, *,
                  explicit_model: bool, explicit_effort: bool) -> dict[str, Any]:
    current = read_locked(thread_id)
    payload = _validate({
        "schema": SCHEMA,
        "thread_id": thread_id,
        "model": model if explicit_model else current["model"],
        "effort": effort if explicit_effort else current["effort"],
        "explicit_model": explicit_model or current["explicit_model"],
        "explicit_effort": explicit_effort or current["explicit_effort"],
    }, thread_id)
    _atomic_write(path_for(thread_id), payload)
    return payload
