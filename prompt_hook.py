"""Codex UserPromptSubmit hook: route a prompt in its existing session.

The hook never starts, resumes, or forks a thread. It attempts a host-side
model change only when the supplied session and active turn exist on the
protected shared app-server. Ordinary local sessions receive advice only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import websockets

from host_control import HOST_URL, THREAD_ID_PATTERN, _read_token, _rpc
from modellabs import ROUTES, record_route, route


CONTINUATIONS = {"ok go", "go ahead", "continue", "yes", "do it"}


def prior_user_prompt(transcript_path: str | None, current: str) -> str | None:
    if not transcript_path:
        return None
    try:
        path = Path(transcript_path).resolve()
        if not path.is_relative_to(Path.home() / ".codex" / "sessions") or not path.is_file():
            return None
        with path.open("rb") as source:
            source.seek(max(0, path.stat().st_size - 1024 * 1024))
            if source.tell():
                source.readline()
            lines = source.readlines()
        for raw in reversed(lines):
            item = json.loads(raw)
            payload = item.get("payload", {})
            if item.get("type") != "response_item" or payload.get("role") != "user":
                continue
            message = "\n".join(c.get("text", "") for c in payload.get("content", []) if c.get("type") == "input_text")
            if message.strip() and message.strip() != current.strip() and message.lower().strip().rstrip(".!?") not in CONTINUATIONS:
                return message
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return None


def initial_route_for(session_id: str, prompt_hash: str) -> dict | None:
    try:
        with ROUTES.open("rb") as source:
            source.seek(max(0, ROUTES.stat().st_size - 512 * 1024))
            if source.tell():
                source.readline()
            lines = source.readlines()
        for line in reversed(lines):
            record = json.loads(line)
            if (record.get("session_id") == session_id and record.get("prompt_sha256") == prompt_hash
                    and record.get("status") == "initial_route_preserved"):
                return None
            if (record.get("thread_id") == session_id and record.get("prompt_sha256") == prompt_hash
                    and record.get("status") == "initial_route"):
                return record
    except (OSError, ValueError, TypeError):
        pass
    return None


async def apply_on_host(session_id: str, turn_id: str, choice: dict) -> bool:
    if not THREAD_ID_PATTERN.fullmatch(session_id) or not THREAD_ID_PATTERN.fullmatch(turn_id):
        return False
    try:
        async with websockets.connect(
            HOST_URL, additional_headers={"Authorization": f"Bearer {_read_token()}"},
            open_timeout=1, close_timeout=1, max_size=4 * 1024 * 1024,
        ) as ws:
            await _rpc(ws, "initialize", {"clientInfo": {"name": "modellabs-hook", "title": "ModelLabs Hook", "version": "0.1"},
                                          "capabilities": {"experimentalApi": True}}, 1)
            await ws.send(json.dumps({"method": "initialized", "params": {}}))
            info = await _rpc(ws, "thread/read", {"threadId": session_id, "includeTurns": False}, 2)
            if (info.get("thread") or {}).get("id") != session_id:
                return False
            turns = await _rpc(ws, "thread/turns/list", {"threadId": session_id, "limit": 1,
                                                         "itemsView": "full", "sortDirection": "desc"}, 3)
            if not any(t.get("id") == turn_id and t.get("status") == "inProgress" for t in turns.get("data", [])):
                return False
            catalog = await _rpc(ws, "model/list", {}, 4)
            model = next((m for m in catalog.get("data", []) if m.get("id") == choice["model"] and not m.get("hidden")), None)
            if not model or choice["effort"] not in {e.get("reasoningEffort") for e in model.get("supportedReasoningEfforts", [])}:
                return False
            result = await _rpc(ws, "turn/settings/update", {"threadId": session_id, "turnId": turn_id,
                                                              "model": choice["model"], "effort": choice["effort"]}, 5)
            return result.get("status") == "applied"
    except Exception:
        return False


def main() -> None:
    payload = json.load(sys.stdin)
    if payload.get("hook_event_name") != "UserPromptSubmit":
        return
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return
    context = prior_user_prompt(payload.get("transcript_path"), prompt) if prompt.lower().strip().rstrip(".!?") in CONTINUATIONS else None
    choice = route(context or prompt)
    choice["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    session_id = payload.get("session_id", "")
    turn_id = payload.get("turn_id", "")
    initial = initial_route_for(session_id, choice["prompt_sha256"])
    if initial:
        choice = {key: initial[key] for key in ("class", "model", "effort", "intelligence_slider",
                                                "servers", "prompt_sha256")}
    applied = False if initial else asyncio.run(apply_on_host(session_id, turn_id, choice))
    record = {**choice, "session_id": session_id, "turn_id": turn_id,
              "status": "initial_route_preserved" if initial else "applied_to_managed_turn" if applied else "advisory_only"}
    record_route(record)
    message = (f"ModelLabs route: {choice['model']}; intelligence slider: "
               f"{choice['intelligence_slider']} ({choice['effort']} reasoning effort); "
               f"suggested MCPs: {', '.join(choice['servers'])}. ")
    if initial:
        message += "The managed launcher already applied the first-turn model, effort, and tool scope; do not override its explicit choice."
    elif applied:
        message += "Host accepted these settings for later steps of this active turn. Check later inference evidence before claiming a model switch."
    else:
        message += "Advisory only: this session is not controllable through the shared model host, or the turn was not updateable. Do not claim a model or tool switch."
    if "agentBrowser" in choice["servers"]:
        message += (
            " For ChatGPT browser-backed intelligence, use AuraCall as the provider bridge with "
            "the semantic selector chatgpt:premium while agent-browser retains browser lifecycle "
            "ownership. Fail closed unless the completed receipt binds an observed Pro/premium "
            "selection, response and assistant-message identities, conversation, runtime profile, "
            "and browser account/profile; a requested selector alone is not execution proof."
        )
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                              "additionalContext": message}}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ModelLabs hook skipped: {type(exc).__name__}", file=sys.stderr)
