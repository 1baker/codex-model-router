"""Authenticated, prompt-aware WebSocket ingress for the shared Codex host."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys

import websockets

from host_control import HOST_URL, _read_token, _rpc
from modellabs import record_route, route


PROXY_URL = "ws://127.0.0.1:45173"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
CONTINUATIONS = {"ok go", "go ahead", "continue", "yes", "do it"}


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
        choice = route(context_prompt or prompt)
        choice["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        if preserve_model and params.get("model"):
            choice["model"] = params["model"]
            if params.get("effort"):
                choice["effort"] = params["effort"]
        else:
            params["model"] = choice["model"]
            params["effort"] = choice["effort"]
        context = params.setdefault("additionalContext", {})
        if isinstance(context, dict) and "modellabs" not in context:
            context["modellabs"] = {"kind": "application", "value":
                "ModelLabs starting tool shortlist for this task: " + ", ".join(choice["servers"])
                + ". Use other tools available in this thread if the task requires them; this shortlist does not grant access."}
        info = {**choice, "thread_id": params.get("threadId"), "status": "submitted_to_host"}
        return json.dumps(message), info
    except (TypeError, ValueError, KeyError):
        return raw, None


async def handler(client: websockets.ServerConnection) -> None:
    token = _read_token()
    if client.request.headers.get("Authorization") != f"Bearer {token}":
        await client.close(code=1008, reason="authentication required")
        return
    async with websockets.connect(HOST_URL, additional_headers={"Authorization": f"Bearer {token}"},
                                  max_size=MAX_MESSAGE_BYTES) as upstream:
        pending: dict[object, dict] = {}
        manual_model_threads: set[str] = set()

        async def inbound() -> None:
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
    async with websockets.serve(handler, "127.0.0.1", 45173, max_size=MAX_MESSAGE_BYTES):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"ModelLabs proxy: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
