"""Constrained client for switching the model of a live Codex app-server turn."""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Any

import websockets
from paths import ROOT
from authority import AuthorityError, acquire_lock_async, read_locked
from telemetry import record as record_metric


HOST_URL = "ws://127.0.0.1:45172"
TOKEN_FILE = ROOT / "host-token"
THREAD_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
MODELS_CACHE_PATH = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "models_cache.json"
MODEL_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60


class ModelHostError(Exception):
    """An expected control-plane or validation failure."""


def _choice_authority(thread_id: str) -> dict[str, Any]:
    try:
        return read_locked(thread_id)
    except AuthorityError as exc:
        raise ModelHostError(str(exc)) from exc


def _read_token() -> str:
    try:
        mode = TOKEN_FILE.stat().st_mode
        if stat.S_IMODE(mode) & 0o077:
            raise ModelHostError("Model-host token file must be private (mode 0600).")
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ModelHostError("Model-host token file is missing.") from exc
    if not token:
        raise ModelHostError("Model-host token file is empty.")
    return token


def _node_repl_review_requirement(model: str, path: Path | None = None,
                                  now: float | None = None) -> bool | None:
    """Return a fresh listed model's local runtime contract, or unknown."""
    cache_path = path or MODELS_CACHE_PATH
    try:
        metadata = cache_path.stat()
        current = time.time() if now is None else now
        if current - metadata.st_mtime > MODEL_CACHE_MAX_AGE_SECONDS or metadata.st_mtime > current + 300:
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    entries = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return None
    matches = [entry for entry in entries
               if isinstance(entry, dict) and entry.get("slug") == model
               and entry.get("visibility") == "list"]
    if len(matches) != 1:
        return None
    required = matches[0].get("node_repl_auto_review_required")
    return required if isinstance(required, bool) else None


def _ensure_switch_runtime_compatible(source_model: str | None, target_model: str) -> None:
    """Reject only a known active-turn runtime-contract mismatch."""
    if not source_model or source_model == target_model:
        return
    source_review = _node_repl_review_requirement(source_model)
    target_review = _node_repl_review_requirement(target_model)
    if source_review is not None and target_review is not None and source_review != target_review:
        raise ModelHostError(
            f"Cannot switch this active turn from {source_model!r} to {target_model!r}: "
            "the models require different Node REPL auto-review contracts. "
            "Start the next turn on the target model instead."
        )


async def _rpc(ws: Any, method: str, params: dict[str, Any], request_id: int) -> dict[str, Any]:
    await ws.send(json.dumps({"id": request_id, "method": method, "params": params}))
    while True:
        try:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=12))
        except asyncio.TimeoutError as exc:
            raise ModelHostError(f"Model host timed out during {method}.") from exc
        if message.get("id") != request_id:
            continue
        if "error" in message:
            detail = message["error"].get("message", "unknown error")
            raise ModelHostError(f"Model host rejected {method}: {detail}")
        result = message.get("result")
        if not isinstance(result, dict):
            raise ModelHostError(f"Model host returned an invalid {method} response.")
        return result


async def switch_current_turn_model(thread_id: str, model: str, effort: str | None = None) -> dict[str, Any]:
    """Publish a model choice for later steps of one active turn in the given thread."""
    if not THREAD_ID_PATTERN.fullmatch(thread_id):
        raise ModelHostError("thread_id must be the exact current Codex thread UUID.")
    bound_thread = os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_SESSION_ID")
    if bound_thread and bound_thread != thread_id:
        raise ModelHostError("Requested thread differs from this Codex process's thread.")
    if not model or len(model) > 100:
        raise ModelHostError("A valid model ID is required.")
    if effort is not None and effort not in {"none", "low", "medium", "high", "xhigh", "max", "ultra"}:
        raise ModelHostError("Unsupported reasoning effort.")
    # This cross-process transaction lock is shared with the owner TUI proxy.
    # It covers validation, the actual host mutation and confirmed application.
    lock_descriptor = await acquire_lock_async(thread_id)
    try:
        authority = _choice_authority(thread_id)
        if authority.get("explicit_model") and authority.get("model") != model:
            raise ModelHostError("The model is explicitly pinned by this managed thread's user choice.")
        if (effort is not None and authority.get("explicit_effort")
                and authority.get("effort") != effort):
            raise ModelHostError("The reasoning effort is explicitly pinned by this managed thread's user choice.")
        token = _read_token()
        async with websockets.connect(
            HOST_URL,
            additional_headers={"Authorization": f"Bearer {token}"},
            open_timeout=5,
            close_timeout=2,
            max_size=4 * 1024 * 1024,
        ) as ws:
            await _rpc(
                ws,
                "initialize",
                {"clientInfo": {"name": "model-control-mcp", "title": "Model Control", "version": "0.1.0"},
                 "capabilities": {"experimentalApi": True}},
                1,
            )
            await ws.send(json.dumps({"method": "initialized", "params": {}}))
            catalog = await _rpc(ws, "model/list", {}, 2)
            matches = [item for item in catalog.get("data", []) if item.get("id") == model and not item.get("hidden")]
            if len(matches) != 1:
                raise ModelHostError(f"Model {model!r} is not listed as available by this host.")
            if effort is not None:
                supported = {x.get("reasoningEffort") for x in matches[0].get("supportedReasoningEfforts", [])}
                if effort not in supported:
                    raise ModelHostError(f"Model {model!r} does not support effort {effort!r} here.")

            read = await _rpc(ws, "thread/read", {"threadId": thread_id, "includeTurns": False}, 3)
            thread = read.get("thread") or {}
            state = (thread.get("status") or {}).get("type")
            if thread.get("id") != thread_id or state != "active":
                if state == "notLoaded":
                    raise ModelHostError(
                        "This thread is saved on disk but is not loaded by the shared model host. "
                        "Its live CLI process cannot be switched through this endpoint; do not resume a duplicate owner."
                    )
                raise ModelHostError("This thread is not active on the connected model host.")
            turns = await _rpc(
                ws, "thread/turns/list",
                {"threadId": thread_id, "limit": 1, "itemsView": "full", "sortDirection": "desc"}, 4,
            )
            active = [turn for turn in turns.get("data", []) if turn.get("status") == "inProgress"]
            if len(active) != 1:
                raise ModelHostError("Could not identify exactly one active turn in this thread.")
            _ensure_switch_runtime_compatible(thread.get("model"), model)
            discovery = [
                item for item in active[0].get("items", [])
                if item.get("type") == "commandExecution"
                and item.get("status") == "completed"
                and item.get("exitCode") == 0
                and "CODEX_THREAD_ID" in item.get("command", "")
                and item.get("aggregatedOutput", "").strip() == thread_id
            ]
            if not discovery:
                raise ModelHostError("Read CODEX_THREAD_ID from the shell in this active turn before switching.")
            turn_id = active[0].get("id")
            if not isinstance(turn_id, str):
                raise ModelHostError("Active turn ID is missing.")
            params: dict[str, Any] = {"threadId": thread_id, "turnId": turn_id, "model": model}
            if effort is not None:
                params["effort"] = effort
            # A separate control connection may not publish its switch notice to
            # the owning proxy. Persist the attempt before host inference can
            # change, so this turn cannot be counted as a single-model sample.
            try:
                record_metric("turn_model_switch_attempted",
                              receipt_id=f"{thread_id}:{turn_id}:model_switch",
                              thread_id=thread_id, turn_id=turn_id,
                              target_model=model, target_effort=effort,
                              source="model_control")
            except Exception as exc:
                raise ModelHostError("Could not persist the model-switch evidence marker.") from exc
            result = await _rpc(ws, "turn/settings/update", params, 5)
            if result.get("status") != "applied":
                raise ModelHostError(f"Model host did not apply the change: {result.get('status', 'unknown')}.")
            return {
                "status": "applied_to_later_steps",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "model": model,
                "effort": effort,
                "note": "Already captured steps are unchanged; verify the model on a later inference.",
            }
    except (OSError, websockets.WebSocketException) as exc:
        raise ModelHostError("The shared Codex model host is unavailable or rejected the connection.") from exc
    finally:
        os.close(lock_descriptor)
