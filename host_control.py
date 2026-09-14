"""Constrained client for switching the model of a live Codex app-server turn."""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

import websockets


HOST_URL = "ws://127.0.0.1:45172"
TOKEN_FILE = Path("/home/bak3r/.local/share/model-selector/host-token")
THREAD_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class ModelHostError(Exception):
    """An expected control-plane or validation failure."""


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

    token = _read_token()
    try:
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
