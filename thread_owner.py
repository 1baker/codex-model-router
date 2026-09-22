"""Refuse a second Codex TUI owner for an existing saved conversation."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

from host_control import HOST_URL, THREAD_ID_PATTERN
from paths import CODEX_HOME, ROOT


SESSIONS = CODEX_HOME / "sessions"


def rollout_for(thread_id: str) -> Path:
    if not THREAD_ID_PATTERN.fullmatch(thread_id):
        raise ValueError("Resume requires an exact thread UUID.")
    matches = list(SESSIONS.glob(f"**/rollout-*-{thread_id}.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"Expected one saved rollout for {thread_id}; found {len(matches)}.")
    return matches[0]


def live_owners(thread_id: str) -> list[int]:
    """Find standalone Codex processes holding this conversation's rollout open."""
    target = rollout_for(thread_id).stat()
    owners: list[int] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / "comm").read_text().strip()
            if command != "codex":
                continue
            args = (proc / "cmdline").read_bytes().split(b"\0")
            if b"app-server" in args and HOST_URL.encode() in args:
                # The managed host keeps its own rollout open after a TUI
                # disconnects. It is the intended owner for this route.
                continue
            for fd in (proc / "fd").iterdir():
                try:
                    opened = fd.stat()
                except (FileNotFoundError, PermissionError, OSError):
                    continue
                if (opened.st_dev, opened.st_ino) == (target.st_dev, target.st_ino):
                    owners.append(int(proc.name))
                    break
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return owners


def require_unowned(thread_id: str) -> None:
    owners = live_owners(thread_id)
    if owners:
        raise RuntimeError(
            f"Thread {thread_id} is already open in standalone Codex process(es) "
            f"{', '.join(map(str, owners))}. Leave that chat first; ModelLabs will not resume a duplicate owner."
        )


def acquire_thread_ownership(thread_id: str, *, existing_thread: bool = True) -> int:
    """Atomically reserve one resumed TUI owner until that process exits."""
    if not THREAD_ID_PATTERN.fullmatch(thread_id):
        raise ValueError("Resume requires an exact thread UUID.")
    if existing_thread:
        rollout_for(thread_id)
    lock_dir = ROOT / "thread-owner-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{thread_id}.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if existing_thread:
            require_unowned(thread_id)
        os.set_inheritable(descriptor, True)
        return descriptor
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError(
            f"Thread {thread_id} is already owned by another managed Codex process."
        ) from error
    except Exception:
        os.close(descriptor)
        raise
