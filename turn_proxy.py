"""Authenticated, prompt-aware WebSocket ingress for the shared Codex host."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from typing import Any

import websockets

from host_control import HOST_URL, _read_token, _rpc
from modellabs import record_route, route
from telemetry import record as record_metric, thread_usage_from, usage_delta, usage_from
from adaptive_policy import adapt, note_followup


PROXY_PORT = int(os.environ.get("MODELLABS_PROXY_PORT", "45173"))
if not 1024 <= PROXY_PORT <= 65535:
    raise ValueError("MODELLABS_PROXY_PORT must be an unprivileged TCP port.")
PROXY_URL = f"ws://127.0.0.1:{PROXY_PORT}"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
CONTINUATIONS = {"ok go", "go ahead", "continue", "yes", "do it"}
CATALOG_TTL_SECONDS = 60.0


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
                  preserve_model: bool = False) -> tuple[str, dict | None]:
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
        if preserve_model and params.get("model"):
            baseline["explicit_model"] = True
            if params.get("effort"):
                baseline["explicit_effort"] = True
        choice = adapt(baseline, thread_id=params.get("threadId"))
        choice["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        if preserve_model and params.get("model"):
            choice["model"] = params["model"]
            choice["explicit_model"] = True
            if params.get("effort"):
                choice["effort"] = params["effort"]
                choice["intelligence_slider"] = params["effort"]
                choice["explicit_effort"] = True
        else:
            params["model"] = choice["model"]
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
    if client.request.headers.get("Authorization") != f"Bearer {token}":
        await client.close(code=1008, reason="authentication required")
        return
    async with websockets.connect(HOST_URL, additional_headers={"Authorization": f"Bearer {token}"},
                                  max_size=MAX_MESSAGE_BYTES) as upstream:
        pending: dict[object, dict] = {}
        manual_model_threads: set[str] = set()
        active: dict[tuple[str, str], dict[str, Any]] = {}
        catalog: dict[str, Any] | None = None
        catalog_at = 0.0

        async def inbound() -> None:
            nonlocal catalog, catalog_at
            async for raw in client:
                try:
                    ping = json.loads(raw)
                    if ping.get("method") == "modellabs/ping" and "id" in ping:
                        await client.send(json.dumps({"id": ping["id"], "result": {"service": "modellabs-proxy"}}))
                        continue
                except (TypeError, ValueError, AttributeError):
                    pass
                context = None
                params = {}
                try:
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
                routed, info = route_request(raw, context, thread_id in manual_model_threads)
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
                    except ValueError:
                        pass
                await upstream.send(routed)

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
                            route_info = {**info, "started_at": time.monotonic(), "delivered": asyncio.Event()}
                            active[(info["thread_id"], info["turn_id"])] = route_info
                            record_metric("route_accepted", thread_id=info["thread_id"], turn_id=info["turn_id"],
                                          model=info["model"], effort=info["effort"], task_class=info["class"],
                                          task_bucket=info.get("task_bucket"),
                                          adaptive_reason=info.get("adaptive_reason"))
                            asyncio.create_task(record_completion(info["thread_id"], info["turn_id"], route_info, token,
                                                                  route_info["delivered"]))
                    params = response.get("params") or {}
                    event_turn_id = params.get("turnId") or (params.get("turn") or {}).get("id")
                    key = (params.get("threadId"), event_turn_id)
                    route_info = active.get(key)
                    if response.get("method") == "thread/tokenUsage/updated" and route_info:
                        token_usage = params.get("tokenUsage") or {}
                        usage = thread_usage_from(params)
                        total = token_usage.get("total")
                        if usage and isinstance(total, dict):
                            if "usage_baseline" not in route_info:
                                route_info["usage_baseline"] = {
                                    field: int(total.get(field, 0)) - int(usage.get(field, 0))
                                    for field in total if isinstance(total.get(field), (int, float))
                                }
                            route_info["usage_total"] = total
                    if response.get("method") == "turn/completed" and route_info:
                        turn = params.get("turn") or {}
                        record_metric("turn_completed", thread_id=key[0], turn_id=key[1], model=route_info["model"],
                                      effort=route_info["effort"], task_class=route_info["class"],
                                      elapsed_ms=round((time.monotonic() - route_info["started_at"]) * 1000),
                                      status=turn.get("status"), usage=usage_from(params))
                        aggregated = usage_delta(route_info.get("usage_baseline", {}), route_info.get("usage_total", {}))
                        if aggregated:
                            record_metric("turn_usage", thread_id=key[0], turn_id=key[1], model=route_info["model"],
                                          effort=route_info["effort"], usage=aggregated,
                                          source="proxy_thread_usage_delta")
                        route_info["delivered"].set()
                        active.pop(key, None)
                except (TypeError, ValueError, KeyError):
                    pass
                await client.send(raw)

        tasks = [asyncio.create_task(inbound()), asyncio.create_task(outbound())]
        done, pending_tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending_tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
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
