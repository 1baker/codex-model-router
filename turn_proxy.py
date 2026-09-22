"""Authenticated, prompt-aware WebSocket ingress for the shared Codex host."""

from __future__ import annotations

import asyncio
import fcntl
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
from telemetry import UsageTracker, record as _record_metric, usage_from
from adaptive_policy import adapt, note_followup
from paths import ROOT, proxy_port, proxy_revision
from thread_owner import acquire_thread_ownership


PROXY_REVISION = proxy_revision()
PROXY_PORT = int(os.environ.get("MODELLABS_PROXY_PORT", str(proxy_port(PROXY_REVISION))))
if not 1024 <= PROXY_PORT <= 65535:
    raise ValueError("MODELLABS_PROXY_PORT must be an unprivileged TCP port.")
PROXY_URL = f"ws://127.0.0.1:{PROXY_PORT}"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
CONTINUATIONS = {"ok go", "go ahead", "continue", "yes", "do it"}
CATALOG_TTL_SECONDS = 60.0
BACKGROUND_TASKS: set[asyncio.Task] = set()
UNRESOLVED_FILE = ROOT / f"proxy-unresolved-{PROXY_PORT}.state"
ACCOUNTING_BLOCKED = False


def record_metric(event: str, **fields: Any) -> None:
    global ACCOUNTING_BLOCKED
    try:
        _record_metric(event, **fields)
    except Exception:
        ACCOUNTING_BLOCKED = True
        raise


def _authority_path(thread_id: str) -> Any:
    return ROOT / "thread-authority" / f"{hashlib.sha256(thread_id.encode()).hexdigest()}.json"


def read_choice_authority(thread_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(_authority_path(thread_id).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if payload.get("thread_id") == thread_id else {}


def write_choice_authority(thread_id: str, model: str | None, effort: str | None, *,
                           explicit_model: bool, explicit_effort: bool) -> None:
    """Persist prompt-free explicit-choice provenance for every mutation path."""
    path = _authority_path(thread_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    lock_descriptor = os.open(path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
    existing: dict[str, Any] = {}
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if candidate.get("thread_id") == thread_id:
            existing = candidate
    except (OSError, ValueError, TypeError):
        pass
    payload = {"thread_id": thread_id,
               "model": model if explicit_model else existing.get("model"),
               "effort": effort if explicit_effort else existing.get("effort"),
               "explicit_model": explicit_model or bool(existing.get("explicit_model")),
               "explicit_effort": explicit_effort or bool(existing.get("explicit_effort"))}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        os.close(lock_descriptor)


def apply_launch_choice(raw: str, choice: dict[str, Any] | None) -> str:
    if choice is None:
        return raw
    try:
        request = json.loads(raw)
        params = request.get("params")
        if request.get("method") != "thread/start" or not isinstance(params, dict):
            return raw
        params["model"] = choice["model"]
        existing = params.get("config", {})
        if not isinstance(existing, dict):
            raise ValueError("thread/start config must be an object")
        merged = dict(existing)
        existing_mcp = existing.get("mcp_servers", {})
        if not isinstance(existing_mcp, dict):
            raise ValueError("thread/start mcp_servers config must be an object")
        route_mcp = config_for(choice["servers"])["mcp_servers"]
        merged_mcp = {name: dict(value) if isinstance(value, dict) else value
                      for name, value in existing_mcp.items()}
        for name, scope in route_mcp.items():
            current = merged_mcp.get(name, {})
            if current is not None and not isinstance(current, dict):
                raise ValueError(f"mcp_servers.{name} must be an object")
            merged_mcp[name] = {**(current or {}), **scope}
        merged["mcp_servers"] = merged_mcp
        params["config"] = merged
        return json.dumps(request)
    except (json.JSONDecodeError, TypeError, KeyError):
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
    except (TypeError, ValueError, KeyError) as exc:
        try:
            if json.loads(raw).get("method") == "turn/start":
                raise ValueError("turn routing failed closed") from exc
        except json.JSONDecodeError:
            pass
        return raw, None


def selection_is_listed(catalog: dict[str, Any], choice: dict[str, Any]) -> bool:
    """Reject a stale or unsupported model/effort pair before host admission."""
    for entry in catalog.get("data", []):
        if entry.get("id") != choice.get("model") or entry.get("hidden"):
            continue
        efforts = {item.get("reasoningEffort") for item in entry.get("supportedReasoningEfforts", [])}
        return choice.get("effort") in efforts
    return False


def is_rpc_response(message: dict[str, Any]) -> bool:
    """Distinguish replies from server requests that use an independent ID space."""
    return ("method" not in message and message.get("id") is not None
            and ("result" in message or "error" in message))


async def live_catalog(token: str) -> dict[str, Any]:
    async with websockets.connect(HOST_URL, additional_headers={"Authorization": f"Bearer {token}"},
                                  open_timeout=2, close_timeout=1, max_size=MAX_MESSAGE_BYTES) as ws:
        await _rpc(ws, "initialize", {"clientInfo": {"name": "modellabs-catalog", "version": "0.1"},
                                      "capabilities": {"experimentalApi": True}}, 1)
        await ws.send(json.dumps({"method": "initialized", "params": {}}))
        return await _rpc(ws, "model/list", {}, 2)


def finalize_route(thread_id: str, turn_id: str, route_info: dict[str, Any], *,
                   status: str, elapsed_ms: int | None, terminal_source: str,
                   usage_complete: bool) -> None:
    """Emit exactly one terminal and one exact-or-unavailable usage receipt."""
    if not route_info["terminal_recorded"].is_set():
        record_metric("turn_completed", thread_id=thread_id, turn_id=turn_id,
                      model=route_info["model"], effort=route_info["effort"],
                      task_class=route_info["class"], elapsed_ms=elapsed_ms,
                      status=status, source=terminal_source, usage=None)
        route_info["terminal_recorded"].set()
    if not route_info["usage_recorded"].is_set():
        if usage_complete:
            usage, unavailable_reason = route_info["usage_tracker"].outcome()
        else:
            usage, unavailable_reason = None, "terminal_usage_boundary_unobserved"
        if usage is not None:
            record_metric("turn_usage", thread_id=thread_id, turn_id=turn_id,
                          model=route_info["model"], effort=route_info["effort"], usage=usage,
                          source="proxy_thread_usage_delta")
        else:
            record_metric("turn_usage_unavailable", thread_id=thread_id, turn_id=turn_id,
                          model=route_info["model"], effort=route_info["effort"],
                          reason=unavailable_reason)
        route_info["usage_recorded"].set()
    route_info["finished"].set()


def track_background(task: asyncio.Task) -> None:
    BACKGROUND_TASKS.add(task)
    UNRESOLVED_FILE.write_text(f"{os.getpid()} {len(BACKGROUND_TASKS)}\n", encoding="utf-8")
    os.chmod(UNRESOLVED_FILE, 0o600)

    def done(completed: asyncio.Task) -> None:
        failed = completed.cancelled()
        if not failed:
            try:
                failed = completed.exception() is not None
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                failed = True
        if failed:
            # Retain the completed task as an unresolved generation claim. A
            # failed persistence/reconciliation attempt must never look settled.
            UNRESOLVED_FILE.write_text(f"{os.getpid()} {len(BACKGROUND_TASKS)}\n", encoding="utf-8")
            return
        BACKGROUND_TASKS.discard(completed)
        if BACKGROUND_TASKS:
            UNRESOLVED_FILE.write_text(f"{os.getpid()} {len(BACKGROUND_TASKS)}\n", encoding="utf-8")
        else:
            UNRESOLVED_FILE.unlink(missing_ok=True)

    task.add_done_callback(done)


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
                turns = await _rpc(ws, "thread/turns/list", {"threadId": thread_id, "limit": 100,
                                                             "itemsView": "full", "sortDirection": "desc"}, 2)
            turn = next((item for item in turns.get("data", []) if item.get("id") == turn_id), None)
            if turn and turn.get("status") != "inProgress":
                if delivered.is_set():
                    return
                finalize_route(thread_id, turn_id, route_info, status=turn.get("status", "unknown"),
                               elapsed_ms=turn.get("durationMs"), terminal_source="durable_poll",
                               usage_complete=False)
                return
        except Exception:
            pass
        await asyncio.sleep(1)
    finalize_route(thread_id, turn_id, route_info, status="completion_unavailable",
                   elapsed_ms=None, terminal_source="durable_poll_timeout", usage_complete=False)


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
                  or not isinstance(launch_choice.get("servers"), list)
                  or not isinstance(launch_choice.get("explicit_model", False), bool)
                  or not isinstance(launch_choice.get("explicit_effort", False), bool)):
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
        initial_preselection_available = client_mode == "preselected"
        launch_scope_available = launch_choice is not None
        ticket_explicit_model = bool((launch_choice or {}).get("explicit_model"))
        ticket_explicit_effort = bool((launch_choice or {}).get("explicit_effort"))
        manual_model_threads: set[str] = set()
        manual_effort_threads: set[str] = set()
        active: dict[tuple[str, str], dict[str, Any]] = {}
        provisional: dict[str, list[dict[str, Any]]] = {}
        ownership_pending: set[object] = set()
        admission_events: dict[object, asyncio.Event] = {}
        authority_pending: dict[object, tuple[str, str | None, str | None]] = {}
        owner_descriptors: dict[str, int] = {}
        catalog: dict[str, Any] | None = None
        catalog_at = 0.0

        def mark_admission_pending(request_id: object) -> None:
            if request_id in admission_events:
                return
            event = asyncio.Event()
            admission_events[request_id] = event

            async def wait_for_receipt() -> None:
                await event.wait()

            track_background(asyncio.create_task(wait_for_receipt()))

        def settle_admission(request_id: object) -> None:
            event = admission_events.pop(request_id, None)
            if event is not None:
                event.set()

        async def inbound() -> None:
            nonlocal catalog, catalog_at, initial_preselection_available, launch_scope_available
            async for raw in client:
                try:
                    ping = json.loads(raw)
                    if ping.get("method") == "modellabs/ping" and "id" in ping:
                        await client.send(json.dumps({"id": ping["id"], "result": {
                            "service": "modellabs-proxy", "revision": PROXY_REVISION}}))
                        continue
                except (TypeError, ValueError, AttributeError):
                    pass
                try:
                    raw = apply_launch_choice(raw, launch_choice if launch_scope_available else None)
                except ValueError as exc:
                    try:
                        request_id = json.loads(raw).get("id")
                    except (TypeError, ValueError, AttributeError):
                        request_id = None
                    if request_id is not None:
                        await client.send(json.dumps({"id": request_id, "error": {
                            "code": -32005, "message": f"ModelLabs rejected thread configuration: {exc}"}}))
                    continue
                context = None
                params = {}
                try:
                    request = json.loads(raw)
                    params = request.get("params", {})
                    request_id = request.get("id")
                    if ACCOUNTING_BLOCKED and request.get("method") in {"thread/start", "thread/resume", "turn/start"}:
                        if request_id is not None:
                            await client.send(json.dumps({"id": request_id, "error": {
                                "code": -32006,
                                "message": "ModelLabs accounting persistence is unhealthy; admission is fail-closed."}}))
                        continue
                    if request.get("method") == "thread/start":
                        launch_scope_available = False
                    if request.get("method") in {"thread/fork", "thread/compact/start"}:
                        if request_id is not None:
                            await client.send(json.dumps({"id": request_id, "error": {
                                "code": -32004,
                                "message": "ModelLabs refuses thread transitions without ownership transfer."}}))
                        continue
                    if request.get("method") == "thread/resume" and isinstance(params, dict):
                        resume_thread = params.get("threadId")
                        if resume_thread and resume_thread not in owner_descriptors:
                            try:
                                owner_descriptors[resume_thread] = acquire_thread_ownership(resume_thread)
                                write_choice_authority(
                                    resume_thread, params.get("model"), params.get("effort"),
                                    explicit_model=preserve_cli_model,
                                    explicit_effort=preserve_cli_effort)
                                if request_id is not None:
                                    mark_admission_pending(request_id)
                            except Exception as exc:
                                if request_id is not None:
                                    await client.send(json.dumps({"id": request_id, "error": {
                                        "code": -32003, "message": str(exc)}}))
                                continue
                    if request.get("method") == "thread/start" and request_id is not None:
                        ownership_pending.add(request_id)
                        mark_admission_pending(request_id)
                    mutating_methods = {"turn/start", "turn/steer", "turn/interrupt",
                                        "thread/settings/update", "turn/settings/update",
                                        "thread/archive", "thread/rollback", "thread/inject_items"}
                    if request.get("method") in mutating_methods:
                        mutation_thread = params.get("threadId") if isinstance(params, dict) else None
                        if not mutation_thread or mutation_thread not in owner_descriptors:
                            if request_id is not None:
                                await client.send(json.dumps({"id": request_id, "error": {
                                    "code": -32003,
                                    "message": "ModelLabs rejected a thread mutation from a non-owner connection."}}))
                            continue
                    if (request.get("method") in {"thread/settings/update", "turn/settings/update"}
                            and isinstance(params, dict) and params.get("threadId")
                            and (params.get("model") is not None or params.get("effort") is not None)):
                        if request_id is not None:
                            authority_pending[request_id] = (params["threadId"], params.get("model"),
                                                             params.get("effort"))
                    if request.get("method") == "turn/start" and isinstance(params, dict):
                        authority = read_choice_authority(params.get("threadId", ""))
                        if authority.get("explicit_model"):
                            params["model"] = authority.get("model")
                            manual_model_threads.add(params["threadId"])
                        if authority.get("explicit_effort"):
                            params["effort"] = authority.get("effort")
                            manual_effort_threads.add(params["threadId"])
                        raw = json.dumps(request)
                        prompt = "\n".join(x.get("text", "") for x in params.get("input", [])
                                           if isinstance(x, dict) and x.get("type") == "text")
                        if prompt.lower().strip().rstrip(".!?") in CONTINUATIONS:
                            context = await previous_task(params.get("threadId", ""), token)
                except (TypeError, ValueError, AttributeError):
                    pass
                thread_id = params.get("threadId") if isinstance(params, dict) else None
                manual_model = thread_id in manual_model_threads
                manual_effort = thread_id in manual_effort_threads
                automatic_initial = initial_preselection_available
                try:
                    routed, info = route_request(raw, context,
                                                 automatic_initial or preserve_cli_model or ticket_explicit_model or manual_model,
                                                 automatic_initial or preserve_cli_effort or ticket_explicit_effort or manual_effort,
                                                 preserve_cli_model or ticket_explicit_model or manual_model,
                                                 preserve_cli_effort or ticket_explicit_effort or manual_effort)
                except Exception as exc:
                    if request_id is not None:
                        await client.send(json.dumps({"id": request_id, "error": {
                            "code": -32001, "message": f"ModelLabs routing failed closed: {type(exc).__name__}"}}))
                    continue
                if request.get("method") == "review/start":
                    if request_id is not None:
                        await client.send(json.dumps({"id": request_id, "error": {
                            "code": -32004, "message": "ModelLabs does not support review inference admission."}}))
                    continue
                if request.get("method") == "turn/start" and info is None:
                    if request_id is not None:
                        await client.send(json.dumps({"id": request_id, "error": {
                            "code": -32001, "message": "ModelLabs could not build a complete route."}}))
                    continue
                if info:
                    if automatic_initial:
                        initial_preselection_available = False
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
                            mark_admission_pending(request_id)
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
                finalize_route(key[0], key[1], route_info, status=turn.get("status", "unknown"),
                               elapsed_ms=round((time.monotonic() - route_info["started_at"]) * 1000),
                               terminal_source="live_event", usage_complete=True)
                active.pop(key, None)

        async def outbound() -> None:
            global ACCOUNTING_BLOCKED
            async for raw in upstream:
                try:
                    response = json.loads(raw)
                    response_id = response.get("id")
                    rpc_response = is_rpc_response(response)
                    authority_update = authority_pending.pop(response_id, None) if rpc_response else None
                    if authority_update and "error" not in response:
                        authority_thread, authority_model, authority_effort = authority_update
                        if authority_model is not None:
                            manual_model_threads.add(authority_thread)
                        if authority_effort is not None:
                            manual_effort_threads.add(authority_thread)
                        write_choice_authority(authority_thread, authority_model, authority_effort,
                                               explicit_model=authority_model is not None,
                                               explicit_effort=authority_effort is not None)
                    if rpc_response and response_id in ownership_pending:
                        ownership_pending.discard(response_id)
                        thread_id = ((response.get("result") or {}).get("thread") or {}).get("id")
                        if thread_id:
                            try:
                                owner_descriptors[thread_id] = acquire_thread_ownership(
                                    thread_id, existing_thread=False)
                                write_choice_authority(
                                    thread_id, (launch_choice or {}).get("model"),
                                    (launch_choice or {}).get("effort"),
                                    explicit_model=ticket_explicit_model,
                                    explicit_effort=ticket_explicit_effort)
                            except Exception as exc:
                                response = {"id": response_id, "error": {"code": -32003, "message": str(exc)}}
                                raw = json.dumps(response)
                        settle_admission(response_id)
                    info = pending.pop(response_id, None) if rpc_response else None
                    if info:
                        result = response.get("result") or {}
                        info["status"] = "accepted_by_host" if "error" not in response else "rejected_by_host"
                        info["turn_id"] = (result.get("turn") or {}).get("id")
                        if info["status"] == "accepted_by_host" and info.get("thread_id") and info.get("turn_id"):
                            terminal_recorded = asyncio.Event()
                            route_info = {**info, "started_at": time.monotonic(),
                                          "delivered": terminal_recorded,
                                          "terminal_recorded": terminal_recorded,
                                          "usage_recorded": asyncio.Event(), "finished": asyncio.Event(),
                                          "usage_tracker": UsageTracker()}
                            active[(info["thread_id"], info["turn_id"])] = route_info
                            record_metric("route_accepted", thread_id=info["thread_id"], turn_id=info["turn_id"],
                                          model=info["model"], effort=info["effort"], task_class=info["class"],
                                          task_bucket=info.get("task_bucket"),
                                          adaptive_reason=info.get("adaptive_reason"))
                            task = asyncio.create_task(record_completion(
                                info["thread_id"], info["turn_id"], route_info, token,
                                route_info["finished"]))
                            track_background(task)
                            buffered = provisional.pop(info["thread_id"], [])
                            for event in buffered:
                                await account_event(event)
                        try:
                            record_route(info)
                        except Exception:
                            ACCOUNTING_BLOCKED = True
                        settle_admission(response_id)
                    elif rpc_response and response_id in admission_events:
                        settle_admission(response_id)
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
                try:
                    await client.send(raw)
                except Exception:
                    # The proxy still owns admission/accounting even after the
                    # presentation socket disappears. Keep consuming upstream.
                    pass

        tasks = [asyncio.create_task(inbound()), asyncio.create_task(outbound())]
        done, pending_tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if tasks[0] in done and (ownership_pending or pending or admission_events):
            while (ownership_pending or pending or admission_events) and not tasks[1].done():
                await asyncio.sleep(0.05)
        for request_id in list(ownership_pending):
            record_metric("route_admission_unresolved", request_id=str(request_id),
                          reason="client_disconnected_before_admission_receipt")
        for request_id, info in list(pending.items()):
            record_metric("route_admission_unresolved", request_id=str(request_id),
                          thread_id=info.get("thread_id"), model=info.get("model"),
                          effort=info.get("effort"), reason="admission_receipt_timeout")
        for task in pending_tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if tasks[0] in done and active:
            await asyncio.gather(*(info["finished"].wait() for info in active.values()))
        for descriptor in owner_descriptors.values():
            os.close(descriptor)
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
