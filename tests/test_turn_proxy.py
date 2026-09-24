import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import turn_proxy
import authority
import receipt_journal


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


if __name__ == "__main__":
    unittest.main()
