"""Refuse a second Codex TUI owner for an existing saved conversation."""

from __future__ import annotations

import os
from pathlib import Path

from host_control import HOST_URL, THREAD_ID_PATTERN


SESSIONS = Path.home() / ".codex/sessions"


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
