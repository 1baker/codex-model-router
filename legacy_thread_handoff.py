"""Admit one closed, previously standalone Codex thread to ModelLabs.

This is an operator-run handoff, not an automatic fallback in the proxy.
The subsequent managed resume still takes its own ownership lock and checks
that no standalone process has reopened the rollout.
"""

from __future__ import annotations

import argparse
import json
import os

from authority import acquire_lock, initialize_locked, path_for
from paths import ROOT
from thread_owner import acquire_thread_ownership, rollout_for, require_unowned


MANAGED_EVENTS = {"route_accepted", "turn_usage", "turn_completed", "quality_grade"}


def _has_managed_evidence(thread_id: str) -> bool:
    metrics = ROOT / "metrics.jsonl"
    try:
        with metrics.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (isinstance(row, dict) and row.get("thread_id") == thread_id
                        and row.get("event") in MANAGED_EVENTS):
                    return True
    except FileNotFoundError:
        pass
    return False


def import_closed_thread(thread_id: str) -> None:
    rollout = rollout_for(thread_id)
    with rollout.open(encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    meta = first.get("payload") if isinstance(first, dict) else None
    if (not isinstance(first, dict) or first.get("type") != "session_meta"
            or not isinstance(meta, dict)
            or meta.get("id") != thread_id or meta.get("thread_source") != "user"):
        raise RuntimeError("Handoff requires an exact saved user conversation.")
    if _has_managed_evidence(thread_id):
        raise RuntimeError("Refusing to import a thread with prior managed evidence.")

    owner = acquire_thread_ownership(thread_id)
    try:
        authority_lock = acquire_lock(thread_id)
        try:
            if path_for(thread_id).exists() or path_for(thread_id).is_symlink():
                raise RuntimeError("Refusing to replace existing thread authority.")
            require_unowned(thread_id)
            if _has_managed_evidence(thread_id):
                raise RuntimeError("Managed evidence appeared during handoff.")
            initialize_locked(thread_id, None, None,
                              explicit_model=False, explicit_effort=False)
        finally:
            os.close(authority_lock)
    finally:
        os.close(owner)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("thread_id", help="exact UUID of a closed standalone user chat")
    args = parser.parse_args()
    import_closed_thread(args.thread_id)
    print(f"Imported closed standalone thread {args.thread_id}; resume through codex-model-host.")


if __name__ == "__main__":
    main()
