"""Authenticated, prompt-aware WebSocket ingress for the shared Codex host."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
from typing import Any

import websockets

from host_control import HOST_URL, _read_token, _rpc
from modellabs import config_for, record_route, route
from telemetry import UsageTracker, record as record_metric, usage_from
from adaptive_policy import adapt, note_followup
from paths import ROOT, proxy_port, proxy_revision


PROXY_REVISION = proxy_revision()
PROXY_PORT = int(os.environ.get("MODELLABS_PROXY_PORT", str(proxy_port(PROXY_REVISION))))
if not 1024 <= PROXY_PORT <= 65535:
    raise ValueError("MODELLABS_PROXY_PORT must be an unprivileged TCP port.")
PROXY_URL = f"ws://127.0.0.1:{PROXY_PORT}"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
CONTINUATIONS = {"ok go", "go ahead", "continue", "yes", "do it"}
CATALOG_TTL_SECONDS = 60.0


def apply_launch_choice(raw: str, choice: dict[str, Any] | None) -> str:
    if choice is None:
        return raw
    try:
        request = json.loads(raw)
        params = request.get("params")
        if request.get("method") != "thread/start" or not isinstance(params, dict):
            return raw
        params["model"] = choice["model"]
        params["config"] = config_for(choice["servers"])
        return json.dumps(request)
    except (TypeError, ValueError, KeyError):
        return raw


async def previous_task(thread_id: str, token: str) -> str | None:
    try:
        async with websockets.connect(HOST_URL, additional_headers={"Authorization": f"Bearer {token}"},
                                      open_timeout=1, max_size=MAX_MESSAGE_BYTES) as ws:
            await _rpc(ws, "initialize", {"clientInfo": {"name": "modellabs-context", "title": "ModelLabs", "version": "0.1"},
                                          "capabilities": {"experimentalApi": True}}, 1)
            await ws.send(json.dumps({"method": "initialized", "params": {}}))
            turns = await _rpc(ws, "thread/turns/list", {"threadId": thread_id, "limit": 10,
                                                         "itemsView": "full", "sortDirection": "desc"}, 2)
            for turn in turns.get("data", []):
                for item in reversed(turn.get("items", [])):
                    if item.get("type") != "userMessage":
                        continue
                    message = "\n".join(c.get("text", "") for c in item.get("content", []) if c.get("type") == "text")
                    if message.strip() and message.lower().strip().rstrip(".!?") not in CONTINUATIONS:
                        return message
    except Exception:
        pass
    return None


def route_request(raw: str, context_prompt: str | None = None,
                  preserve_model: bool = False,
                  preserve_effort: bool = False,
                  explicit_model: bool = False,
                  explicit_effort: bool = False) -> tuple[str, dict | None]:
    try:
        message = json.loads(raw)
        if message.get("method") != "turn/start":
            return raw, None
        params = message.get("params")
        if not isinstance(params, dict):
            return raw, None
        prompt = "\n".join(item.get("text", "") for item in params.get("input", [])
                           if isinstance(item, dict) and item.get("type") == "text")
        if not prompt.strip() and params.get("input"):
            kinds = sorted({str(item.get("type", "unknown")) for item in params["input"] if isinstance(item, dict)})
            prompt = "Analyze " + ", ".join(kinds) + " input."
        if not prompt.strip():
            return raw, None
        note_followup(params.get("threadId"), prompt)
        baseline = route(context_prompt or prompt)
        # A model selected through Codex's settings UI is an explicit user
        # choice even though it is not present in this turn's natural language.
        if explicit_model and params.get("model"):
            baseline["explicit_model"] = True
        if explicit_effort and params.get("effort"):
            baseline["explicit_effort"] = True
        choice = adapt(baseline, thread_id=params.get("threadId"))
        choice["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        if preserve_model and params.get("model"):
            choice["model"] = params["model"]
            if explicit_model:
                choice["explicit_model"] = True
        else:
            params["model"] = choice["model"]
        if preserve_effort and params.get("effort"):
            choice["effort"] = params["effort"]
            choice["intelligence_slider"] = params["effort"]
            if explicit_effort:
                choice["explicit_effort"] = True
        else:
            params["effort"] = choice["effort"]
        context = params.setdefault("additionalContext", {})
        if isinstance(context, dict) and "modellabs" not in context:
            context["modellabs"] = {"kind": "application", "value":
                f"ModelLabs selected the Codex intelligence slider {choice['intelligence_slider']} "
                f"({choice['effort']} reasoning effort). Starting tool shortlist for this task: "
                + ", ".join(choice["servers"])
                + ". Use other tools available in this thread if the task requires them; this shortlist does not grant access."}
        info = {**choice, "thread_id": params.get("threadId"), "status": "submitted_to_host"}
        return json.dumps(message), info
    except (TypeError, ValueError, KeyError):
        return raw, None


def selection_is_listed(catalog: dict[str, Any], choice: dict[str, Any]) -> bool:
    """Reject a stale or unsupported model/effort pair before host admission."""
    for entry in catalog.get("data", []):
        if entry.get("id") != choice.get("model") or entry.get("hidden"):
            continue
        efforts = {item.get("reasoningEffort") for item in entry.get("supportedReasoningEfforts", [])}
        return choice.get("effort") in efforts
    return False


async def live_catalog(token: str) -> dict[str, Any]:
    async with websockets.connect(HOST_URL, additional_headers={"Authorization": f"Bearer {token}"},
                                  open_timeout=2, close_timeout=1, max_size=MAX_MESSAGE_BYTES) as ws:
        await _rpc(ws, "initialize", {"clientInfo": {"name": "modellabs-catalog", "version": "0.1"},
                                      "capabilities": {"experimentalApi": True}}, 1)
        await ws.send(json.dumps({"method": "initialized", "params": {}}))
        return await _rpc(ws, "model/list", {}, 2)


async def record_completion(thread_id: str, turn_id: str, route_info: dict[str, Any], token: str,
                            delivered: asyncio.Event) -> None:
    """Poll the durable turn record when event delivery belongs to another client."""
    deadline = time.monotonic() + 15 * 60
    while time.monotonic() < deadline:
        if delivered.is_set():
            return
        try:
            async with websockets.connect(HOST_URL, additional_headers={"Authorization": f"Bearer {token}"},
                                          open_timeout=2, close_timeout=1, max_size=MAX_MESSAGE_BYTES) as ws:
                await _rpc(ws, "initialize", {"clientInfo": {"name": "modellabs-telemetry", "version": "0.1"},
                                              "capabilities": {"experimentalApi": True}}, 1)
                await ws.send(json.dumps({"method": "initialized", "params": {}}))
                turns = await _rpc(ws, "thread/turns/list", {"threadId": thread_id, "limit": 1,
                                                             "itemsView": "full", "sortDirection": "desc"}, 2)
            turn = next((item for item in turns.get("data", []) if item.get("id") == turn_id), None)
            if turn and turn.get("status") != "inProgress":
                # The live completion notification may arrive while the
                # durable-record RPC is in flight. Claim completion only
                # after that await so the two paths cannot both emit it.
                if delivered.is_set():
                    return
                delivered.set()
                record_metric("turn_completed", thread_id=thread_id, turn_id=turn_id, model=route_info["model"],
                              effort=route_info["effort"], task_class=route_info["class"],
                              elapsed_ms=turn.get("durationMs"), status=turn.get("status"), usage=None)
                return
        except Exception:
            pass
        await asyncio.sleep(1)
    record_metric("turn_completion_timeout", thread_id=thread_id, turn_id=turn_id,
                  model=route_info["model"], effort=route_info["effort"])


async def handler(client: websockets.ServerConnection) -> None:
    token = _read_token()
    authorization = client.request.headers.get("Authorization")
    modes = {
        f"Bearer {token}": "default",
        f"Bearer {token}.preselected": "preselected",
        f"Bearer {token}.explicit-model": "explicit-model",
        f"Bearer {token}.explicit-effort": "explicit-effort",
        f"Bearer {token}.explicit-both": "explicit-both",
    }
    client_mode = modes.get(authorization)
    launch_choice = None
    launch_match = re.fullmatch(rf"Bearer {re.escape(token)}\.launch-([0-9a-f]{{32}})",
                                authorization or "")
    if launch_match:
        ticket = ROOT / "launch-tickets" / f"{launch_match.group(1)}.json"
        try:
            launch_choice = json.loads(ticket.read_text(encoding="utf-8"))
            if time.time() - float(launch_choice["created_at"]) > 5 * 60:
                launch_choice = None
            elif (not isinstance(launch_choice.get("model"), str)
                  or not isinstance(launch_choice.get("effort"), str)
                  or not isinstance(launch_choice.get("servers"), list)):
                launch_choice = None
        except (OSError, ValueError, TypeError, KeyError):
            launch_choice = None
        if launch_choice is not None:
            client_mode = "preselected"
    if client_mode is None:
        await client.close(code=1008, reason="authentication required")
        return
    async with websockets.connect(HOST_URL, additional_headers={"Authorization": f"Bearer {token}"},
                                  max_size=MAX_MESSAGE_BYTES) as upstream:
        pending: dict[object, dict] = {}
        preserve_cli_model = client_mode in {"explicit-model", "explicit-both"}
        preserve_cli_effort = client_mode in {"explicit-effort", "explicit-both"}
        preselected = client_mode == "preselected"
        manual_model_threads: set[str] = set()
        active: dict[tuple[str, str], dict[str, Any]] = {}
        provisional: dict[str, list[dict[str, Any]]] = {}
        catalog: dict[str, Any] | None = None
        catalog_at = 0.0

        async def inbound() -> None:
            nonlocal catalog, catalog_at
            async for raw in client:
                try:
                    ping = json.loads(raw)
                    if ping.get("method") == "modellabs/ping" and "id" in ping:
                        await client.send(json.dumps({"id": ping["id"], "result": {
                            "service": "modellabs-proxy", "revision": PROXY_REVISION}}))
                        continue
                except (TypeError, ValueError, AttributeError):
                    pass
                context = None
                params = {}
                try:
                    raw = apply_launch_choice(raw, launch_choice)
                    request = json.loads(raw)
                    params = request.get("params", {})
                    if (request.get("method") in {"thread/settings/update", "turn/settings/update"}
                            and isinstance(params, dict) and params.get("model") and params.get("threadId")):
                        manual_model_threads.add(params["threadId"])
                    if (request.get("method") == "thread/resume" and isinstance(params, dict)
                            and params.get("model") and params.get("threadId")):
                        manual_model_threads.add(params["threadId"])
                    if request.get("method") == "turn/start" and isinstance(params, dict):
                        prompt = "\n".join(x.get("text", "") for x in params.get("input", [])
                                           if isinstance(x, dict) and x.get("type") == "text")
                        if prompt.lower().strip().rstrip(".!?") in CONTINUATIONS:
                            context = await previous_task(params.get("threadId", ""), token)
                except (TypeError, ValueError, AttributeError):
                    pass
                thread_id = params.get("threadId") if isinstance(params, dict) else None
                manual = thread_id in manual_model_threads
                routed, info = route_request(raw, context,
                                             preselected or preserve_cli_model or manual,
                                             preselected or preserve_cli_effort or manual,
                                             preserve_cli_model or manual,
                                             preserve_cli_effort or manual)
                if info:
                    try:
                        if catalog is None or time.monotonic() - catalog_at > CATALOG_TTL_SECONDS:
                            catalog = await live_catalog(token)
                            catalog_at = time.monotonic()
                        if not selection_is_listed(catalog, info):
                            raise ValueError("selected model or reasoning effort is unavailable on this host")
                    except Exception as exc:
                        request_id = json.loads(routed).get("id")
                        if request_id is not None:
                            await client.send(json.dumps({"id": request_id, "error": {
                                "code": -32001, "message": f"ModelLabs rejected turn before inference: {exc}"}}))
                        record_metric("route_rejected", thread_id=info.get("thread_id"), model=info.get("model"),
                                      effort=info.get("effort"), reason=type(exc).__name__)
                        continue
                    try:
                        request_id = json.loads(routed).get("id")
                        if request_id is not None:
                            pending[request_id] = info
                            if info.get("thread_id"):
                                provisional.setdefault(info["thread_id"], [])
                    except ValueError:
                        pass
                await upstream.send(routed)

        async def account_event(response: dict[str, Any]) -> None:
            params = response.get("params") or {}
            event_turn_id = params.get("turnId") or (params.get("turn") or {}).get("id")
            key = (params.get("threadId"), event_turn_id)
            route_info = active.get(key)
            if not route_info:
                return
            if response.get("method") == "thread/tokenUsage/updated":
                route_info["usage_tracker"].observe(params)
            if response.get("method") == "turn/completed":
                turn = params.get("turn") or {}
                if not route_info["delivered"].is_set():
                    route_info["delivered"].set()
                    record_metric("turn_completed", thread_id=key[0], turn_id=key[1], model=route_info["model"],
                                  effort=route_info["effort"], task_class=route_info["class"],
                                  elapsed_ms=round((time.monotonic() - route_info["started_at"]) * 1000),
                                  status=turn.get("status"), usage=usage_from(params))
                usage, unavailable_reason = route_info["usage_tracker"].outcome()
                if usage is not None:
                    record_metric("turn_usage", thread_id=key[0], turn_id=key[1], model=route_info["model"],
                                  effort=route_info["effort"], usage=usage,
                                  source="proxy_thread_usage_delta")
                else:
                    record_metric("turn_usage_unavailable", thread_id=key[0], turn_id=key[1],
                                  model=route_info["model"], effort=route_info["effort"],
                                  reason=unavailable_reason)
                active.pop(key, None)

        async def outbound() -> None:
            async for raw in upstream:
                try:
                    response = json.loads(raw)
                    info = pending.pop(response.get("id"), None)
                    if info:
                        result = response.get("result") or {}
                        info["status"] = "accepted_by_host" if "error" not in response else "rejected_by_host"
                        info["turn_id"] = (result.get("turn") or {}).get("id")
                        record_route(info)
                        if info["status"] == "accepted_by_host" and info.get("thread_id") and info.get("turn_id"):
                            route_info = {**info, "started_at": time.monotonic(), "delivered": asyncio.Event(),
                                          "usage_tracker": UsageTracker()}
                            active[(info["thread_id"], info["turn_id"])] = route_info
                            record_metric("route_accepted", thread_id=info["thread_id"], turn_id=info["turn_id"],
                                          model=info["model"], effort=info["effort"], task_class=info["class"],
                                          task_bucket=info.get("task_bucket"),
                                          adaptive_reason=info.get("adaptive_reason"))
                            asyncio.create_task(record_completion(info["thread_id"], info["turn_id"], route_info, token,
                                                                  route_info["delivered"]))
                            buffered = provisional.pop(info["thread_id"], [])
                            for event in buffered:
                                await account_event(event)
                    params = response.get("params") or {}
                    event_turn_id = params.get("turnId") or (params.get("turn") or {}).get("id")
                    if (response.get("method") in {"thread/tokenUsage/updated", "turn/completed"}
                            and params.get("threadId") in provisional
                            and (params.get("threadId"), event_turn_id) not in active):
                        provisional[params["threadId"]].append(response)
                    else:
                        await account_event(response)
                except (TypeError, ValueError, KeyError):
                    pass
                await client.send(raw)

        tasks = [asyncio.create_task(inbound()), asyncio.create_task(outbound())]
        done, pending_tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending_tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (thread_id, turn_id), route_info in list(active.items()):
            record_metric("turn_usage_unavailable", thread_id=thread_id, turn_id=turn_id,
                          model=route_info["model"], effort=route_info["effort"],
                          reason="proxy_disconnected_before_accounting_completion")
            active.pop((thread_id, turn_id), None)
        for result in results:
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                print(f"ModelLabs proxy relay ended: {type(result).__name__}", file=sys.stderr, flush=True)


async def main() -> None:
    # This bearer-token control plane is deliberately local-only. Do not place
    # it behind a public Traefik router: Authelia does not replace this client
    # capability token or provide a safe interactive authentication flow for it.
    async with websockets.serve(handler, "127.0.0.1", PROXY_PORT, max_size=MAX_MESSAGE_BYTES):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"ModelLabs proxy: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
