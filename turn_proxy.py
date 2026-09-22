"""Authenticated, prompt-aware WebSocket ingress for the shared Codex host."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import sys
import time
from typing import Any

import websockets

from host_control import HOST_URL, ModelHostError, _read_token, _rpc
from modellabs import config_for, record_route, route
from telemetry import UsageTracker, canonical_receipt, record as _record_metric, usage_from
from adaptive_policy import adapt, note_followup
from paths import ROOT, proxy_port, proxy_revision
from thread_owner import acquire_thread_ownership
from authority import (AuthorityError, acquire_lock as acquire_authority_lock_sync,
                       acquire_lock_async as acquire_authority_lock,
                       initialize_locked as initialize_authority_locked,
                       path_for as authority_path, read_locked as read_authority_locked,
                       update_locked as update_authority_locked)
from protocol_policy import classify as classify_protocol_method
import receipt_journal


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
ACCOUNTING_FAILURES: set[str] = set()


def record_metric(event: str, *, _accounting_claim: str | None = None, **fields: Any) -> None:
    global ACCOUNTING_BLOCKED
    claim = _accounting_claim or f"event:{event}"
    try:
        _record_metric(event, **fields)
    except Exception:
        ACCOUNTING_FAILURES.add(claim)
        ACCOUNTING_BLOCKED = True
        raise
    ACCOUNTING_FAILURES.discard(claim)
    ACCOUNTING_BLOCKED = bool(ACCOUNTING_FAILURES)


def effective_request_settings(params: dict[str, Any]) -> tuple[Any, Any]:
    """Resolve settings exactly as the pinned host: collaboration settings win."""
    collaboration = params.get("collaborationMode")
    settings = None
    if collaboration is not None:
        if not isinstance(collaboration, dict):
            raise ValueError("collaborationMode must be an object")
        settings = collaboration.get("settings")
        if not isinstance(settings, dict):
            raise ValueError("collaborationMode.settings must be an object")
    model = params.get("model")
    effort = params.get("effort")
    if settings is not None:
        if "model" in settings:
            model = settings.get("model")
        if "reasoning_effort" in settings:
            effort = settings.get("reasoning_effort")
    if model is not None and not isinstance(model, str):
        raise ValueError("model must be a string or null")
    if effort is not None and not isinstance(effort, str):
        raise ValueError("effort must be a string or null")
    return model, effort


def _authority_path(thread_id: str) -> Any:
    return authority_path(thread_id)


def read_choice_authority(thread_id: str) -> dict[str, Any]:
    return read_authority_locked(thread_id)


def write_choice_authority(thread_id: str, model: str | None, effort: str | None, *,
                           explicit_model: bool, explicit_effort: bool,
                           initialize: bool = False) -> None:
    """Persist prompt-free explicit-choice provenance for every mutation path."""
    lock_descriptor = acquire_authority_lock_sync(thread_id)
    try:
        operation = initialize_authority_locked if initialize else update_authority_locked
        operation(thread_id, model, effort, explicit_model=explicit_model,
                  explicit_effort=explicit_effort)
    finally:
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
        collaboration = params.get("collaborationMode")
        settings = collaboration.get("settings") if isinstance(collaboration, dict) else None
        incoming_model, incoming_effort = effective_request_settings(params)
        note_followup(params.get("threadId"), prompt)
        baseline = route(context_prompt or prompt)
        # A model selected through Codex's settings UI is an explicit user
        # choice even though it is not present in this turn's natural language.
        if explicit_model and incoming_model:
            baseline["explicit_model"] = True
        if explicit_effort and incoming_effort:
            baseline["explicit_effort"] = True
        choice = adapt(baseline, thread_id=params.get("threadId"))
        choice["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        if preserve_model and incoming_model:
            choice["model"] = incoming_model
            if explicit_model:
                choice["explicit_model"] = True
        if preserve_effort and incoming_effort:
            choice["effort"] = incoming_effort
            choice["intelligence_slider"] = incoming_effort
            if explicit_effort:
                choice["explicit_effort"] = True
        params["model"] = choice["model"]
        params["effort"] = choice["effort"]
        if settings is not None:
            settings["model"] = choice["model"]
            settings["reasoning_effort"] = choice["effort"]
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


def reconcile_receipt_stage(thread_id: str, turn_id: str, route_info: dict[str, Any],
                            stage: str, event: str, fields: dict[str, Any]) -> None:
    """Persist or replay one canonical stage, then close its journal barrier."""
    receipt_id = f"{thread_id}:{turn_id}:{stage}"
    payload = canonical_receipt(receipt_id)
    if payload is None:
        record_metric(event, _accounting_claim=receipt_id,
                      receipt_id=receipt_id, thread_id=thread_id, turn_id=turn_id,
                      **fields)
    else:
        record_metric(payload["event"], _accounting_claim=receipt_id,
                      **{key: value for key, value in payload.items()
                                           if key != "event"})
    journal_path = route_info.get("journal_path")
    if journal_path is None:
        raise RuntimeError("Receipt stage has no durable obligation.")
    if journal_path.exists():
        receipt_journal.mark(journal_path, stage)
    else:
        # A final mark may have unlinked the complete obligation before its
        # directory barrier failed. Only all three canonical receipts prove
        # this is retirement rather than a never-created journal.
        for required in ("accepted", "terminal", "usage"):
            if canonical_receipt(f"{thread_id}:{turn_id}:{required}") is None:
                raise RuntimeError("Receipt obligation disappeared before completion.")
        receipt_journal.confirm_retired(journal_path)


async def live_catalog(token: str) -> dict[str, Any]:
    async with websockets.connect(HOST_URL, additional_headers={"Authorization": f"Bearer {token}"},
                                  open_timeout=2, close_timeout=1, max_size=MAX_MESSAGE_BYTES) as ws:
        await _rpc(ws, "initialize", {"clientInfo": {"name": "modellabs-catalog", "version": "0.1"},
                                      "capabilities": {"experimentalApi": True}}, 1)
        await ws.send(json.dumps({"method": "initialized", "params": {}}))
        return await _rpc(ws, "model/list", {}, 2)


async def latest_host_turn_id(thread_id: str, token: str) -> str | None:
    async with websockets.connect(HOST_URL, additional_headers={"Authorization": f"Bearer {token}"},
                                  open_timeout=2, close_timeout=1, max_size=MAX_MESSAGE_BYTES) as ws:
        await _rpc(ws, "initialize", {"clientInfo": {"name": "modellabs-recovery", "version": "0.1"},
                                      "capabilities": {"experimentalApi": True}}, 1)
        await ws.send(json.dumps({"method": "initialized", "params": {}}))
        try:
            turns = await _rpc(ws, "thread/turns/list", {"threadId": thread_id, "limit": 1,
                                                         "itemsView": "full",
                                                         "sortDirection": "desc"}, 2)
        except ModelHostError as exc:
            message = str(exc)
            if "is not materialized yet" in message and "before first user message" in message:
                return None
            raise
    turn = next(iter(turns.get("data", [])), None)
    return turn.get("id") if isinstance(turn, dict) and isinstance(turn.get("id"), str) else None


def finalize_route(thread_id: str, turn_id: str, route_info: dict[str, Any], *,
                   status: str, elapsed_ms: int | None, terminal_source: str,
                   usage_complete: bool) -> None:
    """Emit exactly one terminal and one exact-or-unavailable usage receipt."""
    if not route_info["terminal_recorded"].is_set():
        reconcile_receipt_stage(thread_id, turn_id, route_info, "terminal", "turn_completed", {
            "model": route_info["model"], "effort": route_info["effort"],
            "task_class": route_info["class"], "elapsed_ms": elapsed_ms, "status": status,
            "source": terminal_source, "usage": None})
        route_info["terminal_recorded"].set()
    if not route_info["usage_recorded"].is_set():
        existing = canonical_receipt(f"{thread_id}:{turn_id}:usage")
        if existing is not None:
            event, fields = existing["event"], {key: value for key, value in existing.items()
                                                  if key not in {"event", "receipt_id",
                                                                 "thread_id", "turn_id"}}
        else:
            if usage_complete:
                usage, unavailable_reason = route_info["usage_tracker"].outcome()
            else:
                usage, unavailable_reason = None, "terminal_usage_boundary_unobserved"
            if usage is not None:
                event, fields = "turn_usage", {"model": route_info["model"],
                    "effort": route_info["effort"], "usage": usage,
                    "source": "proxy_thread_usage_delta"}
            else:
                event, fields = "turn_usage_unavailable", {"model": route_info["model"],
                    "effort": route_info["effort"], "reason": unavailable_reason}
        reconcile_receipt_stage(thread_id, turn_id, route_info, "usage", event, fields)
        route_info["usage_recorded"].set()
    route_info["finished"].set()


def record_accepted_receipt(thread_id: str, turn_id: str, route_info: dict[str, Any]) -> None:
    reconcile_receipt_stage(thread_id, turn_id, route_info, "accepted", "route_accepted", {
        "model": route_info["model"], "effort": route_info["effort"],
        "task_class": route_info["class"], "task_bucket": route_info.get("task_bucket"),
        "adaptive_reason": route_info.get("adaptive_reason")})


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
    diagnostic_at = time.monotonic() + 15 * 60
    while True:
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
        if time.monotonic() >= diagnostic_at:
            record_metric("turn_reconciliation_delayed", thread_id=thread_id, turn_id=turn_id,
                          reason="terminal_state_not_confirmed")
            diagnostic_at = time.monotonic() + 15 * 60
        await asyncio.sleep(1)


async def recover_receipt_obligations(token: str) -> None:
    """Recover crash-left receipts before this generation begins admission."""
    grouped: dict[str, dict[str, list]] = {}
    for path, obligation in receipt_journal.list_obligations():
        grouped.setdefault(obligation["thread_id"], {"obligations": [], "quarantines": [], "retirements": []})[
            "obligations"].append((path, obligation))
    for path, quarantine in receipt_journal.list_quarantines():
        if quarantine["method"] == "turn/start" and quarantine.get("route") is not None:
            grouped.setdefault(quarantine["thread_id"], {"obligations": [], "quarantines": [], "retirements": []})[
                "quarantines"].append((path, quarantine))
        # Legacy v1 quarantines intentionally remain as unresolved ownership
        # fences. They lack route/turn evidence and must never be fabricated.
    for marker, retirement in receipt_journal.list_retirements():
        grouped.setdefault(retirement["thread_id"], {"obligations": [], "quarantines": [], "retirements": []})[
            "retirements"].append((marker, retirement))

    async def recover_thread(thread_id: str, state: dict[str, list]) -> None:
        obligations = state["obligations"]
        quarantines = state["quarantines"]
        retirements = state["retirements"]
        while (any(path.exists() for path, _value in obligations)
               or any(path.exists() for path, _value in quarantines)
               or bool(retirements)):
            descriptor = None
            try:
                descriptor = acquire_thread_ownership(thread_id, existing_thread=False,
                                                      allow_unresolved=True)
            except RuntimeError:
                # A healthy predecessor owns this thread. Serve unrelated
                # threads and retry without stealing until its journal drains.
                await asyncio.sleep(1)
                continue
            completion_tasks: list[asyncio.Task] = []
            try:
                for marker, retirement in list(retirements):
                    receipt_journal.confirm_retired(
                        receipt_journal.JOURNAL_DIR / retirement["obligation"])
                    retirements.remove((marker, retirement))
                for quarantine_path, _quarantine in quarantines:
                    if not quarantine_path.exists():
                        continue
                    quarantine = receipt_journal.load_quarantine(quarantine_path)
                    route = quarantine["route"]
                    turn_id = quarantine.get("turn_id")
                    if turn_id is None:
                        latest = await latest_host_turn_id(thread_id, token)
                        if latest is None or latest == route.get("baseline_turn_id"):
                            raise RuntimeError("Dispatched admission has no authoritative turn yet.")
                        turn_id = latest
                        receipt_journal.bind_quarantine(quarantine_path, turn_id=turn_id)
                    journal_path = receipt_journal.path_for(thread_id, turn_id)
                    receipt_journal.create(thread_id, turn_id, route)
                    info = {**route, "journal_path": journal_path}
                    record_accepted_receipt(thread_id, turn_id, info)
                    if not any(path == journal_path for path, _value in obligations):
                        obligations.append((journal_path, receipt_journal.load(journal_path)))
                    receipt_journal.clear_quarantine(quarantine_path)
                for path, obligation in obligations:
                    if not path.exists():
                        continue
                    obligation = receipt_journal.load(path)
                    if receipt_journal.retire_if_complete(path):
                        continue
                    for stage in ("accepted", "terminal", "usage"):
                        if obligation[stage]:
                            continue
                        receipt_id = f"{thread_id}:{obligation['turn_id']}:{stage}"
                        payload = canonical_receipt(receipt_id)
                        if payload is not None:
                            info = {"journal_path": path}
                            reconcile_receipt_stage(
                                thread_id, obligation["turn_id"], info, stage, payload["event"],
                                {key: value for key, value in payload.items()
                                 if key not in {"event", "receipt_id", "thread_id", "turn_id"}})
                    if not path.exists():
                        continue
                    obligation = receipt_journal.load(path)
                    terminal, usage, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
                    if obligation["terminal"]:
                        terminal.set()
                    if obligation["usage"]:
                        usage.set()
                    info = {"model": obligation["model"], "effort": obligation["effort"],
                            "class": obligation["task_class"], "started_at": time.monotonic(),
                            "delivered": finished, "terminal_recorded": terminal,
                            "usage_recorded": usage, "finished": finished,
                            "usage_tracker": UsageTracker(), "journal_path": path}
                    if not obligation["accepted"]:
                        record_accepted_receipt(thread_id, obligation["turn_id"], info)
                    task = asyncio.create_task(record_completion(thread_id, obligation["turn_id"],
                                                                 info, token, finished))
                    completion_tasks.append(task)
                if completion_tasks:
                    done, pending_tasks = await asyncio.wait(
                        completion_tasks, return_when=asyncio.FIRST_EXCEPTION)
                    failure = next((task.exception() for task in done
                                    if not task.cancelled() and task.exception() is not None), None)
                    if failure is not None:
                        for task in pending_tasks:
                            task.cancel()
                        await asyncio.gather(*completion_tasks, return_exceptions=True)
                        raise failure
                    await asyncio.gather(*pending_tasks)
                return
            except asyncio.CancelledError:
                for task in completion_tasks:
                    task.cancel()
                await asyncio.gather(*completion_tasks, return_exceptions=True)
                raise
            except Exception:
                # Setup is transactional with respect to ownership. Cancel any
                # partially started recovery before releasing and retrying.
                for task in completion_tasks:
                    task.cancel()
                await asyncio.gather(*completion_tasks, return_exceptions=True)
            finally:
                os.close(descriptor)
            await asyncio.sleep(1)

    for thread_id, state in grouped.items():
        track_background(asyncio.create_task(recover_thread(thread_id, state)))


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
        authority_pending: dict[object, dict[str, Any]] = {}
        authority_waiting: dict[str, tuple[object, dict[str, Any]]] = {}
        authority_notifications: dict[str, dict[str, Any]] = {}
        lifecycle_pending: dict[object, Any] = {}
        request_lifecycles: dict[object, str] = {}
        server_request_ids: set[object] = set()
        request_ledger: dict[object, dict[str, Any]] = {}
        owner_descriptors: dict[str, int] = {}
        reserved_descriptors: dict[str, int] = {}
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

        async def settle_admission(lifecycle_id: object) -> None:
            quarantine_path = lifecycle_pending.get(lifecycle_id)
            while quarantine_path is not None:
                try:
                    receipt_journal.clear_quarantine(quarantine_path)
                    lifecycle_pending.pop(lifecycle_id, None)
                    break
                except Exception:
                    await asyncio.sleep(1)
            event = admission_events.pop(lifecycle_id, None)
            if event is not None:
                event.set()

        def begin_lifecycle(request_id: object, method: str) -> str:
            lifecycle_id = f"{method}:{secrets.token_hex(16)}"
            mark_admission_pending(lifecycle_id)
            request_lifecycles[request_id] = lifecycle_id
            return lifecycle_id

        def begin_admission(request_id: object, thread_id: str, method: str) -> str:
            lifecycle_id = begin_lifecycle(request_id, method)
            lifecycle_pending[lifecycle_id] = receipt_journal.quarantine(
                thread_id, lifecycle_id, method)
            return lifecycle_id

        def maybe_release_request_id(request_id: object) -> None:
            entry = request_ledger.get(request_id)
            if (entry is not None and entry.get("forwarded")
                    and request_id not in request_lifecycles
                    and request_id not in authority_pending
                    and request_id not in ownership_pending
                    and request_id not in pending
                    and not any(waiting_id == request_id
                                for waiting_id, _update in authority_waiting.values())):
                request_ledger.pop(request_id, None)

        async def settle_request(request_id: object) -> None:
            lifecycle_id = request_lifecycles.get(request_id)
            if lifecycle_id is not None:
                await settle_admission(lifecycle_id)
                request_lifecycles.pop(request_id, None)
            maybe_release_request_id(request_id)

        async def abandon_request(request_id: object, method: str | None,
                                  params: dict[str, Any]) -> None:
            pending.pop(request_id, None)
            ownership_pending.discard(request_id)
            update = authority_pending.pop(request_id, None)
            if update is not None:
                os.close(update["descriptor"])
            if method == "thread/resume":
                thread_id = params.get("threadId")
                descriptor = reserved_descriptors.pop(thread_id, None)
                if descriptor is not None:
                    os.close(descriptor)
            await settle_request(request_id)

        async def reject_and_abandon(request_id: object, method: str | None,
                                     params: dict[str, Any], code: int, message: str) -> None:
            try:
                await client.send(json.dumps({"id": request_id, "error": {
                    "code": code, "message": message}}))
            finally:
                await abandon_request(request_id, method, params)

        async def reconcile_authority_notification(thread_id: str) -> None:
            waiting = authority_waiting.get(thread_id)
            notification = authority_notifications.get(thread_id)
            if not waiting or not notification:
                return
            waiting_id, update = waiting
            settings = notification.get("threadSettings") or {}
            notified_effort = (settings.get("effort") or settings.get("reasoning_effort")
                               or settings.get("reasoningEffort"))
            if (update["model"] is not None and settings.get("model") != update["model"]
                    or update["effort"] is not None and notified_effort != update["effort"]):
                return
            update_authority_locked(
                thread_id, update["model"], update["effort"],
                explicit_model=update["model"] is not None,
                explicit_effort=update["effort"] is not None)
            if update["model"] is not None:
                manual_model_threads.add(thread_id)
            if update["effort"] is not None:
                manual_effort_threads.add(thread_id)
            os.close(update["descriptor"])
            authority_waiting.pop(thread_id, None)
            authority_notifications.pop(thread_id, None)
            await settle_request(waiting_id)

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
                params: dict[str, Any] = {}
                request_id = None
                method = None
                try:
                    request = json.loads(raw)
                    if not isinstance(request, dict):
                        raise ValueError("request must be an object")
                    params = request.get("params", {})
                    request_id = request.get("id")
                    method = request.get("method")
                    if method is None:
                        if not is_rpc_response(request) or request_id not in server_request_ids:
                            continue
                        server_request_ids.discard(request_id)
                        await upstream.send(raw)
                        continue
                    policy = classify_protocol_method(method)
                    if request_id is not None and request_id in request_ledger:
                        await client.send(json.dumps({"id": request_id, "error": {
                            "code": -32600, "message": "ModelLabs rejects duplicate outstanding request IDs."}}))
                        continue
                    if policy == "unknown":
                        if request_id is not None:
                            await client.send(json.dumps({"id": request_id, "error": {
                                "code": -32601, "message": f"ModelLabs protocol policy rejects {method}."}}))
                        continue
                    if policy == "rejected_inference":
                        if request_id is not None:
                            await client.send(json.dumps({"id": request_id, "error": {
                                "code": -32004, "message": "ModelLabs refuses unmanaged inference or thread transitions."}}))
                        continue
                    if ACCOUNTING_BLOCKED and method in {"thread/start", "thread/resume", "turn/start"}:
                        if request_id is not None:
                            await client.send(json.dumps({"id": request_id, "error": {
                                "code": -32006,
                                "message": "ModelLabs accounting persistence is unhealthy; admission is fail-closed."}}))
                        continue
                    if method == "thread/start":
                        launch_scope_available = False
                    if method == "thread/resume" and isinstance(params, dict):
                        if "history" in params or "path" in params:
                            if request_id is not None:
                                await client.send(json.dumps({"id": request_id, "error": {
                                    "code": -32003,
                                    "message": "ModelLabs refuses path- or history-based resume."}}))
                            continue
                        resume_thread = params.get("threadId")
                        if not isinstance(resume_thread, str) or not resume_thread:
                            if request_id is not None:
                                await client.send(json.dumps({"id": request_id, "error": {
                                    "code": -32003,
                                    "message": "ModelLabs requires threadId and refuses path-based resume."}}))
                            continue
                        acquired = False
                        try:
                            if resume_thread not in owner_descriptors and resume_thread not in reserved_descriptors:
                                reserved_descriptors[resume_thread] = acquire_thread_ownership(resume_thread)
                                acquired = True
                            authority_lock = await acquire_authority_lock(resume_thread)
                            try:
                                read_authority_locked(resume_thread)
                                if request_id is not None and (preserve_cli_model or preserve_cli_effort):
                                    resume_config = params.get("config") or {}
                                    authority_pending[request_id] = {
                                        "method": method, "thread": resume_thread,
                                        "model": params.get("model") if preserve_cli_model else None,
                                        "effort": resume_config.get("model_reasoning_effort") if preserve_cli_effort else None,
                                        "descriptor": authority_lock}
                                    authority_lock = None
                            finally:
                                if authority_lock is not None:
                                    os.close(authority_lock)
                        except Exception as exc:
                            if acquired:
                                os.close(reserved_descriptors.pop(resume_thread))
                            if request_id is not None:
                                await client.send(json.dumps({"id": request_id, "error": {
                                    "code": -32003, "message": str(exc)}}))
                            continue
                        if request_id is not None:
                            begin_admission(request_id, resume_thread, method)
                    if method == "thread/start" and request_id is not None:
                        ownership_pending.add(request_id)
                        # The created thread ID is unknown until the response.
                        begin_lifecycle(request_id, method)
                    if policy == "owner_mutation" or method == "turn/start":
                        mutation_thread = params.get("threadId") if isinstance(params, dict) else None
                        if not mutation_thread or mutation_thread not in owner_descriptors:
                            if request_id is not None:
                                await client.send(json.dumps({"id": request_id, "error": {
                                    "code": -32003,
                                    "message": "ModelLabs rejected a thread mutation from a non-owner connection."}}))
                            continue
                        if (method == "turn/start"
                                and (any(key[0] == mutation_thread for key in active)
                                     or any(item.get("thread_id") == mutation_thread for item in pending.values()))):
                            await client.send(json.dumps({"id": request_id, "error": {
                                "code": -32007,
                                "message": "ModelLabs rejects overlapping turn admission."}}))
                            continue
                        if request_id is None:
                            continue
                        if request_id not in request_lifecycles:
                            begin_admission(request_id, mutation_thread, method)
                    if (method in {"thread/settings/update", "turn/settings/update"}
                            and isinstance(params, dict) and params.get("threadId")):
                        requested_model, requested_effort = effective_request_settings(params)
                        if requested_model is not None or requested_effort is not None:
                            if request_id is not None:
                                descriptor = await acquire_authority_lock(params["threadId"])
                                try:
                                    read_authority_locked(params["threadId"])
                                except Exception:
                                    os.close(descriptor)
                                    raise
                                authority_pending[request_id] = {
                                    "method": method, "thread": params["threadId"],
                                    "model": requested_model, "effort": requested_effort,
                                    "descriptor": descriptor}
                                if method == "thread/settings/update":
                                    lifecycle_id = request_lifecycles[request_id]
                                    authority_pending[request_id]["lifecycle_id"] = lifecycle_id
                    if method == "turn/start" and isinstance(params, dict):
                        descriptor = await acquire_authority_lock(params["threadId"])
                        try:
                            authority = read_authority_locked(params["threadId"])
                        finally:
                            os.close(descriptor)
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
                except (TypeError, ValueError, AttributeError, AuthorityError) as exc:
                    if request_id is not None:
                        await reject_and_abandon(request_id, method, params, -32003, str(exc))
                    continue
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
                        await reject_and_abandon(
                            request_id, method, params, -32001,
                            f"ModelLabs routing failed closed: {type(exc).__name__}")
                    continue
                if method == "turn/start" and info is None:
                    if request_id is not None:
                        await reject_and_abandon(
                            request_id, method, params, -32001,
                            "ModelLabs could not build a complete route.")
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
                            await reject_and_abandon(
                                request_id, method, params, -32001,
                                f"ModelLabs rejected turn before inference: {exc}")
                        record_metric("route_rejected", thread_id=info.get("thread_id"), model=info.get("model"),
                                      effort=info.get("effort"), reason=type(exc).__name__)
                        continue
                    lifecycle_id = request_lifecycles.get(request_id)
                    quarantine_path = lifecycle_pending.get(lifecycle_id)
                    if quarantine_path is not None:
                        try:
                            info["baseline_turn_id"] = await latest_host_turn_id(
                                info["thread_id"], token)
                            receipt_journal.bind_quarantine(quarantine_path, route=info)
                        except Exception as exc:
                            await reject_and_abandon(
                                request_id, method, params, -32001,
                                f"ModelLabs could not persist admission intent: {type(exc).__name__}")
                            continue
                    try:
                        request_id = json.loads(routed).get("id")
                        if request_id is not None:
                            pending[request_id] = info
                            if info.get("thread_id"):
                                provisional.setdefault(info["thread_id"], [])
                    except ValueError:
                        pass
                if request_id is not None:
                    request_ledger[request_id] = {"method": method,
                                                  "thread_id": params.get("threadId") if isinstance(params, dict) else None,
                                                  "forwarded": False}
                await upstream.send(routed)

        async def account_event(response: dict[str, Any]) -> None:
            global ACCOUNTING_BLOCKED
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
                while not route_info["finished"].is_set():
                    try:
                        finalize_route(key[0], key[1], route_info,
                                       status=turn.get("status", "unknown"),
                                       elapsed_ms=round((time.monotonic() - route_info["started_at"]) * 1000),
                                       terminal_source="live_event", usage_complete=True)
                    except Exception:
                        await asyncio.sleep(1)
                        continue
                    break
                active.pop(key, None)

        async def reconcile_and_retire(thread_id: str, turn_id: str,
                                       route_info: dict[str, Any]) -> None:
            await record_completion(thread_id, turn_id, route_info, token, route_info["finished"])
            if route_info["finished"].is_set():
                active.pop((thread_id, turn_id), None)

        async def outbound() -> None:
            global ACCOUNTING_BLOCKED
            async for raw in upstream:
                try:
                    response = json.loads(raw)
                    response_id = response.get("id")
                    rpc_response = is_rpc_response(response)
                    ledger_entry = request_ledger.get(response_id) if rpc_response else None
                    if rpc_response:
                        if ledger_entry is None:
                            continue
                        result = response.get("result") or {}
                        expected_method = ledger_entry["method"]
                        shape_valid = "error" in response
                        if "error" in response:
                            pass
                        elif expected_method == "thread/resume":
                            shape_valid = (((result.get("thread") or {}).get("id"))
                                           == ledger_entry["thread_id"])
                            resume_choice = authority_pending.get(response_id)
                            if resume_choice:
                                shape_valid = (shape_valid
                                    and (resume_choice["model"] is None
                                         or result.get("model") == resume_choice["model"])
                                    and (resume_choice["effort"] is None
                                         or result.get("reasoningEffort") == resume_choice["effort"]))
                        elif expected_method == "thread/start":
                            shape_valid = isinstance((result.get("thread") or {}).get("id"), str)
                        elif expected_method == "turn/start":
                            shape_valid = isinstance((result.get("turn") or {}).get("id"), str)
                        else:
                            shape_valid = shape_valid or isinstance(response.get("result"), dict)
                        if not shape_valid:
                            ACCOUNTING_FAILURES.add(f"protocol-shape:{response_id!r}")
                            ACCOUNTING_BLOCKED = True
                            continue
                    authority_update = authority_pending.get(response_id) if rpc_response else None
                    if authority_update:
                        authority_method = authority_update["method"]
                        if (authority_method == "thread/settings/update"
                                and "error" not in response and response.get("result") == {}):
                            authority_pending.pop(response_id, None)
                            authority_waiting[authority_update["thread"]] = (
                                response_id, authority_update)
                            await reconcile_authority_notification(authority_update["thread"])
                        else:
                            authority_pending.pop(response_id, None)
                            try:
                                applied = (authority_method == "thread/resume" or
                                           (response.get("result") or {}).get("status") == "applied")
                                if "error" not in response and applied:
                                    authority_thread = authority_update["thread"]
                                    authority_model = authority_update["model"]
                                    authority_effort = authority_update["effort"]
                                    update_authority_locked(
                                        authority_thread, authority_model, authority_effort,
                                        explicit_model=authority_model is not None,
                                        explicit_effort=authority_effort is not None)
                                    if authority_model is not None:
                                        manual_model_threads.add(authority_thread)
                                    if authority_effort is not None:
                                        manual_effort_threads.add(authority_thread)
                                elif "error" not in response:
                                    response = {"id": response_id, "error": {
                                        "code": -32005,
                                        "message": "ModelLabs refused to pin an unconfirmed settings mutation."}}
                                    raw = json.dumps(response)
                            finally:
                                os.close(authority_update["descriptor"])
                            if (authority_method == "thread/settings/update"
                                    and "error" in response):
                                await settle_request(response_id)
                    if rpc_response and response_id in ownership_pending:
                        ownership_pending.discard(response_id)
                        thread_id = ((response.get("result") or {}).get("thread") or {}).get("id")
                        ownership_settled = False
                        if thread_id:
                            try:
                                lifecycle_id = request_lifecycles[response_id]
                                owner_descriptors[thread_id] = acquire_thread_ownership(
                                    thread_id, existing_thread=False)
                                write_choice_authority(
                                    thread_id, (launch_choice or {}).get("model"),
                                    (launch_choice or {}).get("effort"),
                                    explicit_model=ticket_explicit_model,
                                    explicit_effort=ticket_explicit_effort, initialize=True)
                                ownership_settled = True
                            except Exception as exc:
                                lifecycle_pending[lifecycle_id] = receipt_journal.quarantine(
                                    thread_id, lifecycle_id, "thread/start-authority")
                                response = {"id": response_id, "error": {"code": -32003, "message": str(exc)}}
                                raw = json.dumps(response)
                        if ownership_settled:
                            await settle_request(response_id)
                    if (rpc_response and ledger_entry
                            and ledger_entry["method"] == "thread/resume"):
                        resume_thread = ledger_entry["thread_id"]
                        descriptor = reserved_descriptors.pop(resume_thread, None)
                        if descriptor is not None:
                            if "error" in response:
                                os.close(descriptor)
                            else:
                                owner_descriptors[resume_thread] = descriptor
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
                            lifecycle_id = request_lifecycles.get(response_id)
                            quarantine_path = lifecycle_pending.get(lifecycle_id)
                            admission_claim = f"admission:{info['thread_id']}:{info['turn_id']}"
                            while True:
                                try:
                                    if quarantine_path is None:
                                        raise RuntimeError("Accepted turn lost its admission intent.")
                                    if quarantine_path.exists():
                                        receipt_journal.bind_quarantine(
                                            quarantine_path, route=route_info,
                                            turn_id=info["turn_id"])
                                    receipt_journal.create(info["thread_id"], info["turn_id"], route_info)
                                    route_info["journal_path"] = receipt_journal.path_for(
                                        info["thread_id"], info["turn_id"])
                                    record_accepted_receipt(info["thread_id"], info["turn_id"], route_info)
                                    ACCOUNTING_FAILURES.discard(admission_claim)
                                    ACCOUNTING_BLOCKED = bool(ACCOUNTING_FAILURES)
                                    break
                                except Exception:
                                    ACCOUNTING_FAILURES.add(admission_claim)
                                    ACCOUNTING_BLOCKED = True
                                    await asyncio.sleep(1)
                            active[(info["thread_id"], info["turn_id"])] = route_info
                            task = asyncio.create_task(reconcile_and_retire(
                                info["thread_id"], info["turn_id"], route_info))
                            track_background(task)
                            await settle_request(response_id)
                            buffered = provisional.pop(info["thread_id"], [])
                            for event in buffered:
                                await account_event(event)
                        try:
                            record_route(info)
                        except Exception:
                            ACCOUNTING_FAILURES.add(
                                f"route-record:{info.get('thread_id')}:{info.get('turn_id')}")
                            ACCOUNTING_BLOCKED = True
                        if info["status"] != "accepted_by_host":
                            await settle_request(response_id)
                    elif (rpc_response and response_id in request_lifecycles
                          and not (authority_update and authority_update["method"] == "thread/settings/update"
                                   and "error" not in response)):
                        await settle_request(response_id)
                    if response.get("method") == "thread/settings/updated":
                        notification = response.get("params") or {}
                        notification_thread = notification.get("threadId")
                        if isinstance(notification_thread, str):
                            authority_notifications[notification_thread] = notification
                            await reconcile_authority_notification(notification_thread)
                    if response.get("method") is not None and response_id is not None:
                        server_request_ids.add(response_id)
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
                if rpc_response and ledger_entry is not None:
                    current = request_ledger.get(response_id)
                    if current is not None:
                        current["forwarded"] = True
                    maybe_release_request_id(response_id)

        tasks = [asyncio.create_task(inbound()), asyncio.create_task(outbound())]
        done, pending_tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if tasks[0] in done and (ownership_pending or pending or admission_events
                                 or authority_pending or authority_waiting):
            while (ownership_pending or pending or admission_events
                   or authority_pending or authority_waiting) and not tasks[1].done():
                await asyncio.sleep(0.05)
        for request_id in list(ownership_pending):
            record_metric("route_admission_unresolved",
                          request_digest=hashlib.sha256(str(request_id).encode()).hexdigest(),
                          reason="client_disconnected_before_admission_receipt")
        for request_id, info in list(pending.items()):
            record_metric("route_admission_unresolved",
                          request_digest=hashlib.sha256(str(request_id).encode()).hexdigest(),
                          thread_id=info.get("thread_id"), model=info.get("model"),
                          effort=info.get("effort"), reason="admission_receipt_timeout")
        for task in pending_tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if active:
            await asyncio.gather(*(info["finished"].wait() for info in active.values()))
        for update in authority_pending.values():
            os.close(update["descriptor"])
        for _request_id, update in authority_waiting.values():
            os.close(update["descriptor"])
        for descriptor in owner_descriptors.values():
            os.close(descriptor)
        for descriptor in reserved_descriptors.values():
            os.close(descriptor)
        for result in results:
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                print(f"ModelLabs proxy relay ended: {type(result).__name__}", file=sys.stderr, flush=True)


async def main() -> None:
    # This bearer-token control plane is deliberately local-only. Do not place
    # it behind a public Traefik router: Authelia does not replace this client
    # capability token or provide a safe interactive authentication flow for it.
    await recover_receipt_obligations(_read_token())
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
