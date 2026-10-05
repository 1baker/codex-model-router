import asyncio
import json
import os
import sys
import unittest
import time
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import turn_proxy
import authority
import receipt_journal
import host_control
import model_host_launcher
import protocol_policy


class CatalogTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def entry(model="gpt-6.1-sol", effort="low", **fields):
        return {"id": model, "supportedReasoningEfforts": [
            {"reasoningEffort": effort}], **fields}

    async def test_paginated_catalog_merges_every_page(self):
        pages = [{"data": [self.entry("gpt-6-sol")], "nextCursor": "two"},
                 {"data": [self.entry()], "nextCursor": None}]
        rpc = AsyncMock(side_effect=pages)
        ws = SimpleNamespace(send=AsyncMock())
        with patch.object(host_control, "_rpc", new=rpc):
            catalog = await host_control.list_models(ws)
        self.assertEqual([item["id"] for item in catalog["data"]],
                         ["gpt-6-sol", "gpt-6.1-sol"])
        self.assertEqual([call.args[2:] for call in rpc.await_args_list],
                         [({}, 2), ({"cursor": "two"}, 3)])

    async def test_missing_cursor_ends_catalog(self):
        with patch.object(host_control, "_rpc", new=AsyncMock(
                return_value={"data": []})) as rpc:
            self.assertEqual(await host_control.list_models(None), {"data": []})
        rpc.assert_awaited_once()

    async def test_malformed_catalog_pages_fail_closed(self):
        for page in ({}, {"data": {}}, {"data": [None]},
                     {"data": [], "nextCursor": 4},
                     {"data": [], "nextCursor": ""}):
            with self.subTest(page=page), patch.object(
                    host_control, "_rpc", new=AsyncMock(return_value=page)):
                with self.assertRaises(host_control.ModelHostError):
                    await host_control.list_models(None)

    async def test_repeated_catalog_cursor_fails_closed(self):
        rpc = AsyncMock(return_value={"data": [], "nextCursor": "same"})
        with patch.object(host_control, "_rpc", new=rpc):
            with self.assertRaisesRegex(host_control.ModelHostError, "repeated"):
                await host_control.list_models(None)
        self.assertEqual(rpc.await_count, 2)

    async def test_catalog_has_a_bounded_page_count(self):
        pages = [{"data": [], "nextCursor": str(i)} for i in range(32)]
        rpc = AsyncMock(side_effect=pages)
        with patch.object(host_control, "_rpc", new=rpc):
            with self.assertRaisesRegex(host_control.ModelHostError, "page limit"):
                await host_control.list_models(None)
        self.assertEqual(rpc.await_count, 32)
        self.assertEqual(rpc.await_args.args[-1], 33)

    async def test_last_allowed_catalog_page_can_finish_normally(self):
        pages = [{"data": [], "nextCursor": str(i)} for i in range(31)]
        pages.append({"data": [self.entry()], "nextCursor": None})
        with patch.object(host_control, "_rpc", new=AsyncMock(side_effect=pages)):
            self.assertEqual(await host_control.list_models(None),
                             {"data": [self.entry()]})

    async def test_live_catalog_uses_the_shared_pagination(self):
        ws = SimpleNamespace(send=AsyncMock())

        class Connection:
            async def __aenter__(self):
                return ws

            async def __aexit__(self, *_args):
                return False

        rpc = AsyncMock(side_effect=[
            {"data": [], "nextCursor": "next"}, {"data": [self.entry()]}])
        with patch.object(turn_proxy.websockets, "connect", return_value=Connection()), \
             patch.object(turn_proxy, "_rpc", new=AsyncMock(return_value={})), \
             patch.object(host_control, "_rpc", new=rpc):
            catalog = await turn_proxy.live_catalog("token")
        self.assertEqual(catalog, {"data": [self.entry()]})
        self.assertEqual(rpc.await_count, 2)

    async def test_aged_missing_id_refreshes_once_and_admits_new_model(self):
        old = {"data": [self.entry("gpt-6-sol")]}
        new = {"data": [self.entry()]}
        choice = {"model": "gpt-6.1-sol", "effort": "low"}
        fetch = AsyncMock(return_value=new)
        with patch.object(turn_proxy.time, "monotonic", return_value=100.0), \
             patch.object(turn_proxy, "live_catalog", new=fetch):
            catalog, timestamp = await turn_proxy.catalog_for_selection("token", old, 95.0, choice)
        fetch.assert_awaited_once_with("token")
        self.assertEqual(timestamp, 100.0)
        self.assertTrue(turn_proxy.selection_is_listed(catalog, choice))

    async def test_fresh_cache_miss_is_rejected_without_a_fetch(self):
        choice = {"model": "gpt-6.1-sol", "effort": "low"}
        fetch = AsyncMock()
        with patch.object(turn_proxy.time, "monotonic", return_value=100.0), \
             patch.object(turn_proxy, "live_catalog", new=fetch):
            catalog, timestamp = await turn_proxy.catalog_for_selection(
                "token", {"data": []}, 96.0, choice)
        fetch.assert_not_awaited()
        self.assertEqual(timestamp, 96.0)
        self.assertFalse(turn_proxy.selection_is_listed(catalog, choice))

    async def test_missing_id_after_refresh_does_not_loop(self):
        choice = {"model": "gpt-6.1-sol", "effort": "low"}
        fetch = AsyncMock(return_value={"data": []})
        with patch.object(turn_proxy.time, "monotonic", return_value=100.0), \
             patch.object(turn_proxy, "live_catalog", new=fetch):
            catalog, timestamp = await turn_proxy.catalog_for_selection(
                "token", {"data": []}, 90.0, choice)
            catalog, timestamp = await turn_proxy.catalog_for_selection(
                "token", catalog, timestamp, choice)
        fetch.assert_awaited_once()
        self.assertFalse(turn_proxy.selection_is_listed(catalog, choice))

    async def test_initial_fetch_does_not_retry_a_missing_id(self):
        fetch = AsyncMock(return_value={"data": []})
        with patch.object(turn_proxy.time, "monotonic", return_value=100.0), \
             patch.object(turn_proxy, "live_catalog", new=fetch):
            await turn_proxy.catalog_for_selection("token", None, 0.0,
                                                  {"model": "gpt-6.1-sol"})
        fetch.assert_awaited_once()

    async def test_expired_catalog_rejects_removed_or_hidden_model(self):
        choice = {"model": "gpt-6.1-sol", "effort": "low"}
        for entries in ([], [self.entry(hidden=True)]):
            with self.subTest(entries=entries), \
                 patch.object(turn_proxy.time, "monotonic", return_value=100.0), \
                 patch.object(turn_proxy, "live_catalog", new=AsyncMock(
                     return_value={"data": entries})) as fetch:
                catalog, _ = await turn_proxy.catalog_for_selection(
                    "token", {"data": [self.entry()]}, 39.0, choice)
            fetch.assert_awaited_once()
            self.assertFalse(turn_proxy.selection_is_listed(catalog, choice))

    async def test_hidden_duplicate_or_unsupported_effort_does_not_force_refresh(self):
        for entries in ([self.entry(hidden=True)], [self.entry(), self.entry()],
                        [self.entry(effort="medium")]):
            with self.subTest(entries=entries), \
                 patch.object(turn_proxy.time, "monotonic", return_value=100.0), \
                 patch.object(turn_proxy, "live_catalog", new=AsyncMock()) as fetch:
                catalog, _ = await turn_proxy.catalog_for_selection(
                    "token", {"data": entries}, 90.0,
                    {"model": "gpt-6.1-sol", "effort": "low"})
            fetch.assert_not_awaited()
            self.assertFalse(turn_proxy.selection_is_listed(
                catalog, {"model": "gpt-6.1-sol", "effort": "low"}))

    async def test_catalog_refresh_errors_propagate_for_admission_rejection(self):
        with patch.object(turn_proxy.time, "monotonic", return_value=100.0), \
             patch.object(turn_proxy, "live_catalog", new=AsyncMock(
                 side_effect=host_control.ModelHostError("offline"))):
            with self.assertRaisesRegex(host_control.ModelHostError, "offline"):
                await turn_proxy.catalog_for_selection("token", None, 0.0,
                                                      {"model": "gpt-6.1-sol"})


class AssistantResultExtractionTests(unittest.TestCase):
    def test_steer_observation_requires_bounded_text_and_exact_turn(self):
        params = {"threadId": "thread-a", "expectedTurnId": "turn-a",
                  "input": [{"type": "text", "text": "Please revise the explanation."}]}
        prepared = turn_proxy.steer_learning_observation(params)
        self.assertIsNotNone(prepared)
        self.assertNotIn("Please revise", str(prepared))
        self.assertIsNone(turn_proxy.steer_learning_observation({**params, "expectedTurnId": None}))
        self.assertIsNone(turn_proxy.steer_learning_observation({
            **params, "input": [{"type": "image", "url": "file:///private"}]}))

    def test_only_completed_agent_messages_are_observed(self):
        message = {"method": "item/completed", "params": {"item": {
            "type": "agentMessage", "text": "The verified result."}}}
        self.assertEqual(turn_proxy.completed_agent_message(message), "The verified result.")
        message["params"]["item"]["type"] = "commandExecution"
        self.assertIsNone(turn_proxy.completed_agent_message(message))
        message["params"]["item"]["type"] = "agentMessage"
        message["method"] = "item/updated"
        self.assertIsNone(turn_proxy.completed_agent_message(message))

    def test_model_reroute_requires_exact_turn_and_server_event(self):
        message = {"method": "model/rerouted", "params": {
            "threadId": "thread-1", "turnId": "turn-1", "fromModel": "gpt-6-sol",
            "toModel": "gpt-6-astra", "reason": "highRiskCyberActivity"}}
        self.assertEqual(turn_proxy.model_reroute_metadata(message, "thread-1", "turn-1"),
                         {"from_model": "gpt-6-sol", "to_model": "gpt-6-astra",
                          "reason": "highRiskCyberActivity"})
        self.assertIsNone(turn_proxy.model_reroute_metadata(message, "thread-2", "turn-1"))
        self.assertIsNone(turn_proxy.model_reroute_metadata(message, "thread-1", "turn-2"))
        self.assertIsNone(turn_proxy.model_reroute_metadata(
            {**message, "method": "turn/completed"}, "thread-1", "turn-1"))
        self.assertIsNone(turn_proxy.model_reroute_metadata(
            {**message, "params": {**message["params"], "toModel": "gpt-6-sol"}},
            "thread-1", "turn-1"))

    def test_host_settings_confirm_only_one_exact_pending_route(self):
        notice = {"method": "thread/settings/updated", "params": {
            "threadId": "thread-1", "threadSettings": {
                "model": "gpt-6-luna", "reasoningEffort": "low"}}}
        pending = {7: {"thread_id": "thread-1", "model": "gpt-6-luna", "effort": "low"}}
        self.assertTrue(turn_proxy.confirm_pending_route_settings(notice, pending))
        self.assertTrue(pending[7]["settings_confirmed"])
        self.assertEqual(pending[7]["settings_confirmation_source"],
                         "host_thread_settings_updated_pre_admission")
        self.assertFalse(turn_proxy.confirm_pending_route_settings(
            notice, {**pending, 8: dict(pending[7])}))
        mismatch = {7: {"thread_id": "thread-1", "model": "gpt-6-sol", "effort": "low"}}
        self.assertFalse(turn_proxy.confirm_pending_route_settings(notice, mismatch))

    def test_execution_observation_requires_completion_usage_confirmation_and_no_reroute(self):
        route = {"model": "gpt-6-luna", "effort": "low", "settings_confirmed": True,
                 "settings_confirmation_source": "host_thread_settings_updated_pre_admission",
                 "reroute_seen": False}
        self.assertEqual(turn_proxy.execution_observation(
            route, status="completed", exact_usage=True), {
                "model": "gpt-6-luna", "effort": "low",
                "source": "host_thread_settings_updated_pre_admission",
                "settings_confirmation": "exact"})
        self.assertIsNone(turn_proxy.execution_observation(
            {**route, "reroute_seen": True}, status="completed", exact_usage=True))
        self.assertIsNone(turn_proxy.execution_observation(
            route, status="failed", exact_usage=True))
        self.assertIsNone(turn_proxy.execution_observation(
            route, status="completed", exact_usage=False))
        for flag in ("switch_attempted", "settings_changed_mid_turn"):
            self.assertIsNone(turn_proxy.execution_observation(
                {**route, flag: True}, status="completed", exact_usage=True))

    def test_active_settings_notice_disqualifies_only_changed_thread(self):
        active = {
            ("thread-1", "turn-1"): {"model": "gpt-6-sol", "effort": "medium"},
            ("thread-2", "turn-2"): {"model": "gpt-6-sol", "effort": "medium"},
        }
        notice = {"method": "thread/settings/updated", "params": {
            "threadId": "thread-1", "threadSettings": {
                "model": "gpt-6-astra", "reasoningEffort": "high"}}}
        self.assertEqual(turn_proxy.note_active_settings_change(notice, active),
                         [("thread-1", "turn-1")])
        self.assertTrue(active[("thread-1", "turn-1")]["settings_changed_mid_turn"])
        self.assertNotIn("settings_changed_mid_turn", active[("thread-2", "turn-2")])
        self.assertEqual(turn_proxy.note_active_settings_change(notice, active), [])


class FakeClient:
    def __init__(self, request, authorization="Bearer token"):
        self.request = SimpleNamespace(headers={"Authorization": authorization}, path="/")
        requests = request if isinstance(request, list) else [request]
        self._requests = [json.dumps(item) for item in requests]
        self._request_ids = [item.get("id") for item in requests]
        self._done_id = requests[-1].get("id")
        self._request_index = 0
        self._response_events = [asyncio.Event() for _ in requests]
        self.done = asyncio.Event()
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._request_index < len(self._requests):
            if self._request_index:
                await self._response_events[self._request_index - 1].wait()
            raw = self._requests[self._request_index]
            self._request_index += 1
            return raw
        await self.done.wait()
        raise StopAsyncIteration

    async def send(self, raw):
        self.sent.append(json.loads(raw))
        response_id = self.sent[-1].get("id")
        for index, request_id in enumerate(self._request_ids):
            if response_id == request_id:
                self._response_events[index].set()
        if self.sent[-1].get("id") == self._done_id:
            self.done.set()

    async def close(self, **_kwargs):
        self.done.set()


class FakeUpstream:
    def __init__(self, messages):
        self.messages = [json.dumps(item) for item in messages]
        self.sent = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    def __aiter__(self):
        return self

    async def __anext__(self):
        while not self.sent:
            await asyncio.sleep(0)
        if self.messages:
            return self.messages.pop(0)
        await asyncio.Future()


class TwoRequestUpstream(FakeUpstream):
    async def __anext__(self):
        while not self.sent or (self._yielded >= 1 and len(self.sent) < 2):
            await asyncio.sleep(0)
        if self.messages:
            self._yielded += 1
            return self.messages.pop(0)
        await asyncio.Future()

    def __init__(self, messages):
        super().__init__(messages)
        self._yielded = 0


class ConnectContext:
    def __init__(self, upstream):
        self.upstream = upstream

    async def __aenter__(self):
        return self.upstream

    async def __aexit__(self, *_args):
        return False


class TurnProxyTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def route_info(delivered):
        return {"model": "gpt-5.6-terra", "effort": "medium", "class": "routine",
                "terminal_recorded": delivered, "usage_recorded": asyncio.Event(),
                "finished": asyncio.Event(), "usage_tracker": turn_proxy.UsageTracker()}

    async def test_ephemeral_thread_history_unavailable_is_a_fail_closed_baseline(self):
        upstream = FakeUpstream([])

        async def rpc(_ws, method, _params, _request_id):
            if method == "initialize":
                return {}
            raise turn_proxy.ModelHostError(
                "Model host rejected thread/turns/list: ephemeral threads do not support thread/turns/list")

        with patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
             patch.object(turn_proxy, "_rpc", side_effect=rpc):
            self.assertIsNone(await turn_proxy.latest_host_turn_id("ephemeral-thread", "token"))

    async def test_previous_task_skips_chained_contextual_followups(self):
        upstream = FakeUpstream([])

        async def rpc(_ws, method, _params, _request_id):
            if method == "initialize":
                return {}
            return {"data": [{"items": [
                {"type": "userMessage", "content": [
                    {"type": "text", "text": "Investigate the intermittent race condition."}]},
                {"type": "userMessage", "content": [
                    {"type": "text", "text": "How does this work?"}]},
            ]}]}

        with patch.object(turn_proxy.websockets, "connect",
                          return_value=ConnectContext(upstream)), \
             patch.object(turn_proxy, "_rpc", side_effect=rpc):
            result = await turn_proxy.previous_task("thread", "token")
        self.assertEqual(result, ("Investigate the intermittent race condition.", "found"))

    async def test_previous_task_reports_lookup_failure(self):
        with patch.object(turn_proxy.websockets, "connect", side_effect=OSError("unavailable")):
            self.assertEqual(await turn_proxy.previous_task("thread", "token"),
                             (None, "error"))

    def test_route_records_context_status_and_digest_without_text(self):
        raw = json.dumps({"id": 1, "method": "turn/start", "params": {
            "threadId": "thread", "input": [{"type": "text", "text": "ok go"}]}})
        _routed, info = turn_proxy.route_request(
            raw, context_prompt="Migrate the production database.",
            context_lookup_status="found")
        self.assertEqual(info["context_lookup_status"], "found")
        self.assertEqual(len(info["context_prompt_sha256"]), 64)
        self.assertNotIn("Migrate", str(info))

    async def test_durable_completion_claim_prevents_late_duplicate(self):
        delivered = asyncio.Event()
        metrics = []

        async def rpc(_ws, method, _params, _request_id):
            if method == "thread/turns/list":
                return {"data": [{"id": "turn", "status": "completed", "durationMs": 12}]}
            return {}

        upstream = FakeUpstream([])
        def reconcile(_thread, _turn, _info, _stage, event, fields):
            metrics.append((event, fields))
        with patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
             patch.object(turn_proxy, "_rpc", side_effect=rpc), \
             patch.object(turn_proxy, "reconcile_receipt_stage", side_effect=reconcile):
            await turn_proxy.record_completion(
                "thread", "turn", self.route_info(delivered),
                "token", delivered)

        self.assertTrue(delivered.is_set())
        self.assertEqual([event for event, _fields in metrics],
                         ["turn_completed", "turn_usage_unavailable"])

    async def test_live_completion_wins_rpc_race_without_duplicate(self):
        delivered = asyncio.Event()
        metrics = []

        async def rpc(_ws, method, _params, _request_id):
            if method == "thread/turns/list":
                delivered.set()
                return {"data": [{"id": "turn", "status": "interrupted", "durationMs": 12}]}
            return {}

        upstream = FakeUpstream([])
        with patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
             patch.object(turn_proxy, "_rpc", side_effect=rpc), \
             patch.object(turn_proxy, "record_metric", side_effect=lambda event, **fields: metrics.append((event, fields))):
            await turn_proxy.record_completion(
                "thread", "turn", self.route_info(delivered),
                "token", delivered)

        self.assertEqual(metrics, [])

    async def test_durable_completion_never_promotes_partial_usage_to_exact(self):
        delivered = asyncio.Event()
        metrics = []
        info = self.route_info(delivered)
        sample = {"inputTokens": 8, "cachedInputTokens": 0, "cacheWriteInputTokens": 0,
                  "outputTokens": 2, "reasoningOutputTokens": 0, "totalTokens": 10}
        info["usage_tracker"].observe({"tokenUsage": {"last": sample, "total": sample}})

        async def rpc(_ws, method, _params, _request_id):
            return {"data": [{"id": "turn", "status": "completed"}]} if method == "thread/turns/list" else {}

        def reconcile(_thread, _turn, _info, _stage, event, fields):
            metrics.append((event, fields))
        with patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(FakeUpstream([]))), \
             patch.object(turn_proxy, "_rpc", side_effect=rpc), \
             patch.object(turn_proxy, "reconcile_receipt_stage", side_effect=reconcile):
            await turn_proxy.record_completion("thread", "turn", info, "token", delivered)
        self.assertEqual([event for event, _ in metrics], ["turn_completed", "turn_usage_unavailable"])
        self.assertEqual(metrics[-1][1]["reason"], "terminal_usage_boundary_unobserved")

    def test_receipt_flags_are_committed_only_after_persistence(self):
        info = self.route_info(asyncio.Event())
        with patch.object(turn_proxy, "reconcile_receipt_stage", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                turn_proxy.finalize_route("thread", "turn", info, status="completed",
                                          elapsed_ms=1, terminal_source="live_event",
                                          usage_complete=True)
        self.assertFalse(info["terminal_recorded"].is_set())
        self.assertFalse(info["usage_recorded"].is_set())
        self.assertFalse(info["finished"].is_set())

    def test_finalize_emits_observed_execution_only_after_exact_usage(self):
        info = self.route_info(asyncio.Event())
        info.update({"settings_confirmed": True,
                     "settings_confirmation_source":
                         "host_thread_settings_updated_pre_admission",
                     "reroute_seen": False, "execution_recorded": asyncio.Event()})
        sample = {"inputTokens": 8, "cachedInputTokens": 0, "cacheWriteInputTokens": 0,
                  "outputTokens": 2, "reasoningOutputTokens": 0, "totalTokens": 10}
        info["usage_tracker"].observe({"tokenUsage": {"last": sample, "total": sample}})
        metrics = []

        def reconcile(_thread, _turn, _info, _stage, event, fields):
            metrics.append((event, fields))

        with patch.object(turn_proxy, "reconcile_receipt_stage", side_effect=reconcile), \
             patch.object(turn_proxy, "record_metric",
                          side_effect=lambda event, **fields: metrics.append((event, fields))):
            turn_proxy.finalize_route("thread", "turn", info, status="completed",
                                      elapsed_ms=1, terminal_source="live_event",
                                      usage_complete=True)
        observed = [fields for event, fields in metrics if event == "route_execution_observed"]
        self.assertEqual(len(observed), 1, metrics)
        self.assertEqual(observed[0]["model"], "gpt-5.6-terra")
        self.assertEqual(observed[0]["settings_confirmation"], "exact")

    def test_switch_marker_blocks_execution_promotion_after_exact_usage(self):
        info = self.route_info(asyncio.Event())
        info.update({"settings_confirmed": True,
                     "settings_confirmation_source":
                         "host_thread_settings_updated_pre_admission",
                     "execution_recorded": asyncio.Event()})
        sample = {"inputTokens": 8, "cachedInputTokens": 0, "cacheWriteInputTokens": 0,
                  "outputTokens": 2, "reasoningOutputTokens": 0, "totalTokens": 10}
        info["usage_tracker"].observe({"tokenUsage": {"last": sample, "total": sample}})
        events = []
        with patch.object(turn_proxy, "canonical_receipt",
                          side_effect=lambda receipt_id: ({"event": "turn_model_switch_attempted"}
                              if receipt_id.endswith(":model_switch") else None)), \
             patch.object(turn_proxy, "reconcile_receipt_stage",
                          side_effect=lambda *_args: events.append(_args[4])), \
             patch.object(turn_proxy, "record_metric",
                          side_effect=lambda event, **_fields: events.append(event)):
            turn_proxy.finalize_route("thread", "turn", info, status="completed",
                                      elapsed_ms=1, terminal_source="live_event",
                                      usage_complete=True)
        self.assertIn("turn_usage", events)
        self.assertNotIn("route_execution_observed", events)
        self.assertTrue(info["switch_attempted"])

    def test_server_request_id_never_resolves_pending_rpc(self):
        self.assertFalse(turn_proxy.is_rpc_response({"id": 7, "method": "item/tool/requestUserInput",
                                                     "params": {}}))
        self.assertTrue(turn_proxy.is_rpc_response({"id": 7, "result": {"turn": {"id": "t"}}}))

    def test_choice_authority_preserves_per_field_user_pins(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory, patch.object(turn_proxy, "ROOT", Path(directory)), \
             patch.object(authority, "ROOT", Path(directory)):
            turn_proxy.write_choice_authority("thread", "gpt-pinned", None,
                                              explicit_model=True, explicit_effort=False, initialize=True)
            turn_proxy.write_choice_authority("thread", None, "high",
                                              explicit_model=False, explicit_effort=True)
            payload = json.loads(turn_proxy._authority_path("thread").read_text(encoding="utf-8"))
        self.assertEqual(payload["model"], "gpt-pinned")
        self.assertEqual(payload["effort"], "high")
        self.assertTrue(payload["explicit_model"])
        self.assertTrue(payload["explicit_effort"])

    async def test_foreign_thread_mutation_is_rejected_before_forwarding(self):
        request = {"id": 9, "method": "turn/start", "params": {
            "threadId": "foreign", "input": [{"type": "text", "text": "hello"}]}}
        client = FakeClient(request)
        upstream = FakeUpstream([])
        with patch.object(turn_proxy, "_read_token", return_value="token"), \
             patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)):
            await asyncio.wait_for(turn_proxy.handler(client), timeout=2)
        self.assertEqual(upstream.sent, [])
        self.assertEqual(client.sent[0]["error"]["code"], -32003)

    def test_launch_choice_scopes_thread_before_creation(self):
        raw = json.dumps({"id": 2, "method": "thread/start", "params": {
            "cwd": "/tmp", "config": {"model_reasoning_effort": "high",
                                           "shell_environment_policy": {"inherit": "none"}}}})
        routed = json.loads(turn_proxy.apply_launch_choice(raw, {
            "model": "gpt-5.6-terra", "servers": ["modelControl"]}))
        self.assertEqual(routed["params"]["model"], "gpt-5.6-terra")
        servers = routed["params"]["config"]["mcp_servers"]
        self.assertTrue(servers["modelControl"]["enabled"])
        self.assertTrue(all(not value["enabled"] for name, value in servers.items()
                            if name != "modelControl"))
        self.assertEqual(routed["params"]["config"]["model_reasoning_effort"], "high")
        self.assertEqual(routed["params"]["config"]["shell_environment_policy"], {"inherit": "none"})

    async def test_launch_ticket_auth_applies_scope_in_handler(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tickets = root / "launch-tickets"
            tickets.mkdir()
            ticket_id = "a" * 32
            (tickets / f"{ticket_id}.json").write_text(json.dumps({
                "created_at": __import__("time").time(), "model": "gpt-5.6-terra",
                "effort": "medium", "servers": ["modelControl"],
                "explicit_model": False, "explicit_effort": False,
            }), encoding="utf-8")
            thread_id = "00000000-0000-4000-8000-000000000002"
            upstream = FakeUpstream([{"id": 2, "result": {"thread": {"id": thread_id}}}])
            client = FakeClient({"id": 2, "method": "thread/start", "params": {"cwd": "/tmp"}},
                                f"Bearer token.launch-{ticket_id}")
            def acquire(*_args, **_kwargs):
                self.assertEqual(list((root / "quarantine").glob("*.json")), [])
                return os.open("/dev/null", os.O_RDONLY)
            with patch.object(turn_proxy, "ROOT", root), patch.object(authority, "ROOT", root), \
                 patch.object(receipt_journal, "QUARANTINE_DIR", root / "quarantine"), \
                 patch.object(turn_proxy, "_read_token", return_value="token"), \
                 patch.object(turn_proxy, "acquire_thread_ownership", side_effect=acquire), \
                 patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)):
                await asyncio.wait_for(turn_proxy.handler(client), timeout=2)
            forwarded = upstream.sent[0]["params"]
            self.assertEqual(forwarded["model"], "gpt-5.6-terra")
            self.assertTrue(forwarded["config"]["mcp_servers"]["modelControl"]["enabled"])

    async def test_first_turn_can_continue_with_unavailable_history_baseline(self):
        from tempfile import TemporaryDirectory
        thread_id = "00000000-0000-4000-8000-000000000123"
        turn_id = "00000000-0000-4000-8000-000000000124"
        client = FakeClient([
            {"id": 1, "method": "thread/start", "params": {
                "cwd": "/tmp", "ephemeral": True}},
            {"id": 2, "method": "turn/start", "params": {
                "threadId": thread_id,
                "input": [{"type": "text", "text": "Return the canary."}]}},
        ])
        upstream = TwoRequestUpstream([
            {"id": 1, "result": {"thread": {"id": thread_id}}},
            {"id": 2, "result": {"turn": {"id": turn_id}}},
        ])

        def route(raw, *_args, **_kwargs):
            request = json.loads(raw)
            request["params"].update({"model": "gpt-6-luna", "effort": "low"})
            return json.dumps(request), {
                "model": "gpt-6-luna", "effort": "low", "class": "simple",
                "thread_id": thread_id, "status": "submitted_to_host",
            }

        async def catalog(_token):
            return {"data": [{"id": "gpt-6-luna", "hidden": False,
                              "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}]}

        async def completed(_thread, _turn, _info, _token, finished):
            finished.set()

        history = AsyncMock(return_value=None)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(turn_proxy, "ROOT", root), patch.object(authority, "ROOT", root), \
                 patch.object(receipt_journal, "ROOT", root), \
                 patch.object(receipt_journal, "JOURNAL_DIR", root / "receipts"), \
                 patch.object(receipt_journal, "QUARANTINE_DIR", root / "quarantine"), \
                 patch.object(turn_proxy, "_read_token", return_value="token"), \
                 patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
                 patch.object(turn_proxy, "acquire_thread_ownership",
                              return_value=os.open("/dev/null", os.O_RDONLY)), \
                 patch.object(turn_proxy, "route_request", side_effect=route), \
                 patch.object(turn_proxy, "live_catalog", side_effect=catalog), \
                 patch.object(turn_proxy, "latest_host_turn_id", new=history), \
                 patch.object(turn_proxy, "record_metric"), \
                 patch.object(turn_proxy, "record_route"), \
                 patch.object(turn_proxy, "capture_codex_turn"), \
                 patch.object(turn_proxy, "record_completion", side_effect=completed):
                await asyncio.wait_for(turn_proxy.handler(client), timeout=2)
        self.assertEqual(len(upstream.sent), 2, client.sent)
        history.assert_awaited_once_with(thread_id, "token")
        self.assertEqual(upstream.sent[1]["params"]["threadId"], thread_id)
        self.assertEqual(client.sent[-1]["result"]["turn"]["id"], turn_id)

    async def test_only_host_accepted_steer_is_captured_as_context(self):
        from tempfile import TemporaryDirectory
        thread_id, turn_id = "thread-steer", "turn-steer"
        resume = {"id": 1, "method": "thread/resume", "params": {"threadId": thread_id}}
        steer = {"id": 2, "method": "turn/steer", "params": {
            "threadId": thread_id, "expectedTurnId": turn_id,
            "input": [{"type": "text", "text": "User correction"}]}}
        upstream = TwoRequestUpstream([
            {"id": 1, "result": {"thread": {"id": thread_id}}},
            {"id": 2, "result": {"turnId": turn_id}},
        ])
        client = FakeClient([resume, steer])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(turn_proxy, "ROOT", root), patch.object(authority, "ROOT", root), \
                 patch.object(receipt_journal, "ROOT", root), \
                 patch.object(receipt_journal, "QUARANTINE_DIR", root / "quarantine"):
                turn_proxy.write_choice_authority(thread_id, None, None,
                                                  explicit_model=False, explicit_effort=False,
                                                  initialize=True)
                with patch.object(turn_proxy, "_read_token", return_value="token"), \
                     patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
                     patch.object(turn_proxy, "acquire_thread_ownership", return_value=os.open("/dev/null", os.O_RDONLY)), \
                     patch.object(turn_proxy, "capture_codex_steer") as capture:
                    await asyncio.wait_for(turn_proxy.handler(client), timeout=2)
        self.assertEqual(upstream.sent[1]["params"], steer["params"])
        capture.assert_called_once()
        self.assertEqual(capture.call_args.args[:2], (thread_id, turn_id))
        self.assertEqual(client.sent[-1]["result"], {"turnId": turn_id})

    async def test_host_rejected_steer_is_not_captured(self):
        from tempfile import TemporaryDirectory
        thread_id, turn_id = "thread-steer", "turn-steer"
        requests = [
            {"id": 1, "method": "thread/resume", "params": {"threadId": thread_id}},
            {"id": 2, "method": "turn/steer", "params": {
                "threadId": thread_id, "expectedTurnId": turn_id,
                "input": [{"type": "text", "text": "User correction"}]}}
        ]
        upstream = TwoRequestUpstream([
            {"id": 1, "result": {"thread": {"id": thread_id}}},
            {"id": 2, "error": {"code": -32000, "message": "No active turn"}},
        ])
        client = FakeClient(requests)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(turn_proxy, "ROOT", root), patch.object(authority, "ROOT", root), \
                 patch.object(receipt_journal, "ROOT", root), \
                 patch.object(receipt_journal, "QUARANTINE_DIR", root / "quarantine"):
                turn_proxy.write_choice_authority(thread_id, None, None,
                                                  explicit_model=False, explicit_effort=False,
                                                  initialize=True)
                with patch.object(turn_proxy, "_read_token", return_value="token"), \
                     patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
                     patch.object(turn_proxy, "acquire_thread_ownership", return_value=os.open("/dev/null", os.O_RDONLY)), \
                     patch.object(turn_proxy, "capture_codex_steer") as capture:
                    await asyncio.wait_for(turn_proxy.handler(client), timeout=2)
        capture.assert_not_called()
        self.assertIn("error", client.sent[-1])

    async def test_usage_and_completion_before_admission_are_processed_once(self):
        from tempfile import TemporaryDirectory
        thread_id, turn_id = "thread-1", "turn-1"
        usage = {"inputTokens": 8, "cachedInputTokens": 0, "cacheWriteInputTokens": 0,
                 "outputTokens": 2, "reasoningOutputTokens": 0, "totalTokens": 10}
        total = {"inputTokens": 108, "cachedInputTokens": 0, "cacheWriteInputTokens": 0,
                 "outputTokens": 2, "reasoningOutputTokens": 0, "totalTokens": 110}
        upstream = TwoRequestUpstream([
            {"method": "thread/settings/updated", "params": {
                "threadId": thread_id, "threadSettings": {
                    "model": "gpt-6-luna", "reasoningEffort": "low"}}},
            {"method": "thread/tokenUsage/updated", "params": {
                "threadId": thread_id, "turnId": turn_id,
                "tokenUsage": {"last": usage, "total": total}}},
            {"method": "model/rerouted", "params": {
                "threadId": thread_id, "turnId": turn_id,
                "fromModel": "gpt-6-luna", "toModel": "gpt-6-sol",
                "reason": "highRiskCyberActivity"}},
            {"method": "turn/completed", "params": {
                "threadId": thread_id, "turn": {"id": turn_id, "status": "completed"}}},
            {"id": 7, "result": {"turn": {"id": turn_id}}},
        ])
        request = {"id": 7, "method": "turn/start", "params": {
            "threadId": thread_id,
            "input": [{"type": "text", "text": "Format values as CSV."}],
        }}
        resume = {"id": 6, "method": "thread/resume", "params": {"threadId": thread_id}}
        client = FakeClient([resume, request])
        metrics = []

        async def catalog(_token):
            return {"data": [{"id": "gpt-6-luna", "hidden": False,
                              "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}]}

        upstream.messages.insert(0, json.dumps({"id": 6, "result": {"thread": {"id": thread_id}}}))
        async def completed(_thread, _turn, _info, _token, finished):
            finished.set()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            journal = root / "receipts"
            with patch.object(turn_proxy, "ROOT", root), patch.object(authority, "ROOT", root), \
                 patch.object(receipt_journal, "ROOT", root), \
                 patch.object(receipt_journal, "JOURNAL_DIR", journal), \
                 patch.object(receipt_journal, "QUARANTINE_DIR", root / "quarantine"):
                turn_proxy.write_choice_authority(thread_id, None, None,
                                                  explicit_model=False, explicit_effort=False,
                                                  initialize=True)
                with patch.object(turn_proxy, "_read_token", return_value="token"), \
                     patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
                     patch.object(turn_proxy, "acquire_thread_ownership", return_value=os.open("/dev/null", os.O_RDONLY)), \
                     patch.object(turn_proxy, "live_catalog", side_effect=catalog), \
                     patch.object(turn_proxy, "latest_host_turn_id", new=AsyncMock(return_value=None)), \
                     patch.object(turn_proxy, "record_metric", side_effect=lambda event, **fields: metrics.append((event, fields))), \
                     patch.object(turn_proxy, "record_route"), \
                     patch.object(turn_proxy, "record_completion", side_effect=completed):
                    await asyncio.wait_for(turn_proxy.handler(client), timeout=2)

        usage_records = [fields for event, fields in metrics if event == "turn_usage"]
        unavailable = [fields for event, fields in metrics if event == "turn_usage_unavailable"]
        self.assertEqual(len(usage_records), 1, metrics)
        self.assertEqual(usage_records[0]["usage"]["totalTokens"], 10)
        self.assertEqual(unavailable, [])
        reroutes = [fields for event, fields in metrics if event == "model_rerouted"]
        self.assertEqual(len(reroutes), 1, metrics)
        self.assertEqual(reroutes[0]["to_model"], "gpt-6-sol")
        self.assertEqual([fields for event, fields in metrics
                          if event == "route_execution_observed"], [])


class BurstClient(FakeClient):
    """Send pipelined requests, including notifications and duplicate IDs."""
    async def __anext__(self):
        if self._request_index < len(self._requests):
            raw = self._requests[self._request_index]
            self._request_index += 1
            return raw
        await self.done.wait()
        raise StopAsyncIteration


class ReplyUpstream(FakeUpstream):
    def __init__(self, response=None):
        super().__init__([])
        self.replies = asyncio.Queue()
        self.response = response or (lambda request: {"id": request["id"], "result": {}})

    async def send(self, raw):
        await super().send(raw)
        await self.replies.put(json.dumps(self.response(self.sent[-1])))

    async def __anext__(self):
        return await self.replies.get()


class LaunchTrustTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def ticket(root, *, choice=None, model=False, effort=False):
        workspace = root / 'project.quoted"back\\slash'
        workspace.mkdir(exist_ok=True)
        with patch.object(model_host_launcher, "ROOT", root):
            binding = model_host_launcher._launch_workspace(["-C", str(workspace)])
            ticket_id = model_host_launcher._create_launch_ticket(
                choice, explicit_model=model, explicit_effort=effort, workspace=binding)
        ticket_path = root / "launch-tickets" / f"{ticket_id}.json"
        return ticket_id, json.loads(ticket_path.read_text()), ticket_path

    @staticmethod
    def request(workspace, request_id=7):
        return {"id": request_id, "method": "config/batchWrite", "params": {
            "edits": [{"keyPath": turn_proxy.trust_edit_key_path(workspace),
                       "value": "trusted", "mergeStrategy": "replace"}],
            "filePath": None, "expectedVersion": None, "reloadUserConfig": True}}

    async def relay(self, root, client, upstream, *, blocked=False):
        with ExitStack() as stack:
            for module in (turn_proxy, authority):
                stack.enter_context(patch.object(module, "ROOT", root))
            stack.enter_context(patch.object(receipt_journal, "QUARANTINE_DIR", root / "quarantine"))
            stack.enter_context(patch.object(receipt_journal, "JOURNAL_DIR", root / "receipts"))
            stack.enter_context(patch.object(turn_proxy, "ACCOUNTING_BLOCKED", blocked))
            stack.enter_context(patch.object(turn_proxy, "ACCOUNTING_FAILURES", set()))
            stack.enter_context(patch.object(turn_proxy, "_read_token", return_value="token"))
            stack.enter_context(patch.object(turn_proxy, "record_metric"))
            stack.enter_context(patch.object(turn_proxy, "acquire_thread_ownership",
                                           side_effect=lambda *_args, **_kw: os.open("/dev/null", os.O_RDONLY)))
            stack.enter_context(patch.object(turn_proxy.websockets, "connect",
                                           return_value=ConnectContext(upstream)))
            await asyncio.wait_for(turn_proxy.handler(client), timeout=2)

    async def test_exact_trust_write_normalizes_only_reload_and_preserves_reply(self):
        for reload in (True, False, "absent"):
            with self.subTest(reload=reload), TemporaryDirectory() as directory:
                root = Path(directory)
                ticket_id, ticket, _ = self.ticket(root)
                request = self.request(ticket["workspace"]["path"])
                request["jsonrpc"] = "2.0"
                if reload == "absent":
                    request["params"].pop("reloadUserConfig")
                else:
                    request["params"]["reloadUserConfig"] = reload
                reply = {"id": 7, "result": {"version": "new-version", "filePath": "/config.toml"}}
                upstream = ReplyUpstream(lambda _request: reply)
                client = FakeClient(request, f"Bearer token.launch-{ticket_id}")
                await self.relay(root, client, upstream)
                expected = json.loads(json.dumps(request))
                expected["params"]["reloadUserConfig"] = False
                self.assertEqual(upstream.sent, [expected])
                self.assertEqual(client.sent, [reply])
                self.assertTrue((root / "launch-tickets" / f'trust-{ticket["trust_grant"]}.spent.json').exists())

    async def test_malformed_or_unrelated_writes_never_forward_or_spend(self):
        mutations = {
            "mixed": lambda r: r["params"]["edits"].append({"keyPath": "model", "value": "other", "mergeStrategy": "replace"}),
            "empty": lambda r: r["params"].update(edits=[]),
            "parent": lambda r: r["params"]["edits"][0].update(keyPath=turn_proxy.trust_edit_key_path("/tmp")),
            "descendant": lambda r: r["params"]["edits"][0].update(keyPath=r["params"]["edits"][0]["keyPath"].replace('.trust_level', '.child.trust_level')),
            "bare-key": lambda r: r["params"]["edits"][0].update(keyPath="projects.tmp.trust_level"),
            "untrusted": lambda r: r["params"]["edits"][0].update(value="untrusted"),
            "nonstring": lambda r: r["params"]["edits"][0].update(value=True),
            "upsert": lambda r: r["params"]["edits"][0].update(mergeStrategy="upsert"),
            "extra-edit-field": lambda r: r["params"]["edits"][0].update(extra=True),
            "extra-param": lambda r: r["params"].update(extra=True),
            "extra-request": lambda r: r.update(extra=True),
            "other-file": lambda r: r["params"].update(filePath="/elsewhere.toml"),
            "version": lambda r: r["params"].update(expectedVersion="sha256:other"),
            "reload-int": lambda r: r["params"].update(reloadUserConfig=1),
            "reload-null": lambda r: r["params"].update(reloadUserConfig=None),
            "notification": lambda r: r.pop("id"),
            "null-id": lambda r: r.update(id=None),
            "generic-write": lambda r: r.update(method="config/value/write"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), TemporaryDirectory() as directory:
                root = Path(directory)
                ticket_id, ticket, _ = self.ticket(root)
                request = self.request(ticket["workspace"]["path"])
                mutate(request)
                client = BurstClient([request, {"id": 99, "method": "config/read", "params": {}}],
                                     f"Bearer token.launch-{ticket_id}")
                upstream = ReplyUpstream()
                await self.relay(root, client, upstream)
                self.assertEqual([r["method"] for r in upstream.sent], ["config/read"])
                self.assertTrue((root / "launch-tickets" / f'trust-{ticket["trust_grant"]}.json').exists())
        self.assertEqual(protocol_policy.classify("config/batchWrite"), "unknown")
        self.assertEqual(protocol_policy.classify("config/value/write"), "unknown")

    async def test_no_launcher_authority_or_legacy_ticket_never_grants_trust(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ticket_id, ticket, path = self.ticket(root)
            legacy = {"created_at": time.time(), "model": "gpt-6.1-sol", "effort": "low",
                      "servers": [], "workspace": ticket["workspace"], "trust_grant": ticket["trust_grant"]}
            path.write_text(json.dumps(legacy))
            for authorization in ("Bearer token", "Bearer token.explicit-both", f"Bearer token.launch-{ticket_id}"):
                client = FakeClient(self.request(ticket["workspace"]["path"]), authorization)
                upstream = ReplyUpstream()
                await self.relay(root, client, upstream)
                self.assertEqual(upstream.sent, [])
                self.assertIn("error", client.sent[0])

    async def test_replay_across_connections_and_host_failure_remain_spent(self):
        for fail in (False, True, "disconnect"):
            with self.subTest(fail=fail), TemporaryDirectory() as directory:
                root = Path(directory)
                ticket_id, ticket, _ = self.ticket(root)
                request = self.request(ticket["workspace"]["path"])
                upstream = ReplyUpstream(lambda r: {"id": r["id"], "error": {"code": -1, "message": "host failure"}}
                                         if fail else {"id": r["id"], "result": {}})
                if fail == "disconnect":
                    upstream.send = AsyncMock(side_effect=ConnectionError("disconnected"))
                client = FakeClient(request, f"Bearer token.launch-{ticket_id}")
                await self.relay(root, client, upstream)
                again = ReplyUpstream()
                replay = FakeClient(request, f"Bearer token.launch-{ticket_id}")
                await self.relay(root, replay, again)
                self.assertEqual(again.sent, [])
                self.assertIn("error", replay.sent[0])

    def test_atomic_grant_claim_has_one_winner(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, ticket, _ = self.ticket(root)
            with patch.object(turn_proxy, "ROOT", root), ThreadPoolExecutor(max_workers=8) as workers:
                results = list(workers.map(turn_proxy.spend_trust_grant, [ticket["trust_grant"]] * 8))
            self.assertEqual(sum(result is not None for result in results), 1)

    async def test_two_handlers_racing_share_one_trust_grant(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ticket_id, ticket, _ = self.ticket(root)
            request = self.request(ticket["workspace"]["path"])
            clients = [FakeClient(request, f"Bearer token.launch-{ticket_id}") for _ in range(2)]
            upstreams = [ReplyUpstream() for _ in range(2)]
            with patch.object(turn_proxy, "ROOT", root), \
                 patch.object(turn_proxy, "_read_token", return_value="token"), \
                 patch.object(turn_proxy.websockets, "connect",
                              side_effect=[ConnectContext(upstream) for upstream in upstreams]):
                await asyncio.wait_for(asyncio.gather(*(turn_proxy.handler(client) for client in clients)), timeout=2)
            self.assertEqual(sum(len(upstream.sent) for upstream in upstreams), 1)
            self.assertEqual(sum("error" in client.sent[0] for client in clients), 1)

    async def test_grant_spends_before_all_lifecycle_rejections(self):
        for method in sorted(protocol_policy.TRUST_SPENDING_METHODS):
            for blocked in (False, True):
                with self.subTest(method=method, blocked=blocked), TemporaryDirectory() as directory:
                    root = Path(directory)
                    choice = {"model": "gpt-6.1-sol", "effort": "low", "servers": []}
                    ticket_id, ticket, _ = self.ticket(root, choice=choice)
                    first = {"id": 1, "method": method, "params": {"config": []}}
                    client = FakeClient([first, self.request(ticket["workspace"]["path"])],
                                        f"Bearer token.launch-{ticket_id}")
                    upstream = ReplyUpstream()
                    await self.relay(root, client, upstream, blocked=blocked)
                    self.assertEqual(upstream.sent, [])
                    self.assertTrue((root / "launch-tickets" / f'trust-{ticket["trust_grant"]}.spent.json').exists())
                    self.assertEqual(len(client.sent), 2)
                    self.assertTrue(all("error" in reply for reply in client.sent))

    async def test_duplicate_id_guard_precedes_lifecycle_spend(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ticket_id, ticket, _ = self.ticket(root)
            requests = [{"id": 1, "method": "config/read", "params": {}},
                        {"id": 1, "method": "thread/fork", "params": {}},
                        self.request(ticket["workspace"]["path"])]
            client = BurstClient(requests, f"Bearer token.launch-{ticket_id}")
            upstream = ReplyUpstream()
            await self.relay(root, client, upstream)
            self.assertEqual([r["method"] for r in upstream.sent], ["config/read", "config/batchWrite"])
            self.assertTrue(any(r.get("error", {}).get("code") == -32600 for r in client.sent))

    async def test_expiry_while_waiting_and_future_or_invalid_grant_rejected(self):
        for created_at in (time.time() - 301, time.time() + 600, float("nan"), True):
            with self.subTest(created_at=created_at), TemporaryDirectory() as directory:
                root = Path(directory)
                ticket_id, ticket, _ = self.ticket(root)
                grant_path = root / "launch-tickets" / f'trust-{ticket["trust_grant"]}.json'
                grant = json.loads(grant_path.read_text())
                grant["created_at"] = created_at
                grant_path.write_text(json.dumps(grant))
                client = FakeClient(self.request(ticket["workspace"]["path"]), f"Bearer token.launch-{ticket_id}")
                upstream = ReplyUpstream()
                await self.relay(root, client, upstream)
                self.assertEqual(upstream.sent, [])
                self.assertIn("error", client.sent[0])

    async def test_workspace_redirection_and_inode_replacement_fail_closed(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink), TemporaryDirectory() as directory:
                root = Path(directory)
                ticket_id, ticket, _ = self.ticket(root)
                workspace = Path(ticket["workspace"]["path"])
                workspace.rename(root / "original")
                if symlink:
                    workspace.symlink_to(root / "original", target_is_directory=True)
                else:
                    workspace.mkdir()
                upstream = ReplyUpstream()
                client = FakeClient(self.request(str(workspace)), f"Bearer token.launch-{ticket_id}")
                await self.relay(root, client, upstream)
                self.assertEqual(upstream.sent, [])
                self.assertIn("error", client.sent[0])

    async def test_grant_expires_while_connected_at_confirmation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(model_host_launcher.time, "time", return_value=1000):
                ticket_id, ticket, _ = self.ticket(root)
            client = FakeClient(self.request(ticket["workspace"]["path"]), f"Bearer token.launch-{ticket_id}")
            upstream = ReplyUpstream()
            with patch.object(turn_proxy.time, "time", side_effect=[1000, 1301]):
                await self.relay(root, client, upstream)
            self.assertEqual(upstream.sent, [])
            self.assertIn("error", client.sent[0])

    async def test_symlink_or_mismatched_grant_does_not_forward(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink), TemporaryDirectory() as directory:
                root = Path(directory)
                ticket_id, ticket, _ = self.ticket(root)
                path = root / "launch-tickets" / f'trust-{ticket["trust_grant"]}.json'
                payload = json.loads(path.read_text())
                payload["ticket"] = "f" * 32
                path.write_text(json.dumps(payload))
                if symlink:
                    path.rename(root / "other.json")
                    path.symlink_to(root / "other.json")
                upstream = ReplyUpstream()
                client = FakeClient(self.request(ticket["workspace"]["path"]), f"Bearer token.launch-{ticket_id}")
                await self.relay(root, client, upstream)
                self.assertEqual(upstream.sent, [])
                self.assertIn("error", client.sent[0])

    def test_ticket_shapes_expiry_and_legacy_never_add_trust(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, ticket, _ = self.ticket(root)
            self.assertIsNone(turn_proxy.parse_launch_ticket(ticket, time.time())["choice"])
            omitted = {k: v for k, v in ticket.items() if k not in {"model", "effort", "servers"}}
            self.assertIsNone(turn_proxy.parse_launch_ticket(omitted, time.time())["choice"])
            full = {**ticket, "model": "gpt-6.1-sol", "effort": "low", "servers": []}
            self.assertEqual(turn_proxy.parse_launch_ticket(full, time.time())["choice"]["model"], "gpt-6.1-sol")
            for mutation in ({"model": "gpt-6.1-sol"}, {"servers": []}, {"extra": True},
                             {"created_at": time.time() - 301}, {"created_at": time.time() + 600},
                             {"created_at": float("nan")}, {"created_at": True}, {"explicit_model": 1}):
                with self.subTest(mutation=mutation):
                    self.assertIsNone(turn_proxy.parse_launch_ticket({**ticket, **mutation}, time.time()))
            legacy = {"created_at": time.time(), "model": "gpt-6.1-sol", "effort": "low", "servers": [],
                      "workspace": ticket["workspace"], "trust_grant": ticket["trust_grant"]}
            self.assertIsNone(turn_proxy.parse_launch_ticket(legacy, time.time())["workspace"])
            self.assertIsNone(turn_proxy.parse_launch_ticket(legacy, time.time())["trust_grant"])
            self.assertIsNone(turn_proxy.parse_launch_ticket({**legacy, "created_at": time.time() + 600}, time.time()))

    async def test_promptless_real_handler_binds_only_host_confirmed_pins(self):
        thread_id = "00000000-0000-4000-8000-000000000044"
        for model, effort in ((True, True), (True, False), (False, True), (False, False)):
            with self.subTest(model=model, effort=effort), TemporaryDirectory() as directory:
                root = Path(directory)
                ticket_id, _, _ = self.ticket(root, model=model, effort=effort)
                request = {"id": 2, "method": "thread/start", "params": {
                    "cwd": str(root), "model": "gpt-6.1-sol", "config": {"model_reasoning_effort": "low"}}}
                reply = {"id": 2, "result": {"thread": {"id": thread_id}, "model": "gpt-6.1-sol", "reasoningEffort": "low"}}
                upstream = ReplyUpstream(lambda _r: reply)
                client = FakeClient(request, f"Bearer token.launch-{ticket_id}")
                await self.relay(root, client, upstream)
                self.assertEqual(upstream.sent, [request])
                self.assertEqual(client.sent, [reply])
                with patch.object(authority, "ROOT", root):
                    payload = json.loads(authority.path_for(thread_id).read_text())
                self.assertEqual(payload["model"], "gpt-6.1-sol" if model else None)
                self.assertEqual(payload["effort"], "low" if effort else None)
                self.assertEqual(payload["explicit_model"], model)
                self.assertEqual(payload["explicit_effort"], effort)

    async def test_promptless_missing_invalid_mismatched_pins_fail_closed(self):
        thread_id = "00000000-0000-4000-8000-000000000045"
        for fields in ({"model": "different", "reasoningEffort": "low"},
                       {"model": "gpt-6.1-sol", "reasoningEffort": "high"},
                       {"model": "gpt-6.1-sol"}, {"model": 1, "reasoningEffort": "low"},
                       {"model": "gpt-6.1-sol", "reasoningEffort": "invalid"}):
            with self.subTest(fields=fields), TemporaryDirectory() as directory:
                root = Path(directory)
                ticket_id, _, _ = self.ticket(root, model=True, effort=True)
                request = {"id": 2, "method": "thread/start", "params": {
                    "model": "gpt-6.1-sol", "config": {"model_reasoning_effort": "low"}}}
                upstream = ReplyUpstream(lambda _r: {"id": 2, "result": {"thread": {"id": thread_id}, **fields}})
                client = FakeClient(request, f"Bearer token.launch-{ticket_id}")
                await self.relay(root, client, upstream)
                self.assertEqual(upstream.sent, [request])
                self.assertIn("error", client.sent[0])
                with patch.object(authority, "ROOT", root):
                    self.assertFalse(authority.path_for(thread_id).exists())

    async def test_mixed_shape_ticket_is_rejected_before_connecting(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ticket_id, ticket, path = self.ticket(root)
            ticket["model"] = "gpt-6.1-sol"
            path.write_text(json.dumps(ticket))
            client = FakeClient({"id": 2, "method": "thread/start", "params": {}}, f"Bearer token.launch-{ticket_id}")
            upstream = ReplyUpstream()
            await self.relay(root, client, upstream)
            self.assertEqual(upstream.sent, [])
            self.assertTrue(client.done.is_set())

    async def test_promptless_explicit_pin_without_cli_value_fails_closed(self):
        thread_id = "00000000-0000-4000-8000-000000000046"
        for params in ({}, {"model": "gpt-6.1-sol"}, {"config": {"model_reasoning_effort": "low"}}):
            with self.subTest(params=params), TemporaryDirectory() as directory:
                root = Path(directory)
                ticket_id, _, _ = self.ticket(root, model=True, effort=True)
                request = {"id": 2, "method": "thread/start", "params": params}
                upstream = ReplyUpstream(lambda _r: {"id": 2, "result": {
                    "thread": {"id": thread_id}, "model": "gpt-6.1-sol", "reasoningEffort": "low"}})
                client = FakeClient(request, f"Bearer token.launch-{ticket_id}")
                await self.relay(root, client, upstream)
                self.assertIn("error", client.sent[0])
                with patch.object(authority, "ROOT", root):
                    self.assertFalse(authority.path_for(thread_id).exists())


if __name__ == "__main__":
    unittest.main()
