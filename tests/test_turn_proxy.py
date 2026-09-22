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

    async def test_usage_and_completion_before_admission_are_processed_once(self):
        from tempfile import TemporaryDirectory
        thread_id, turn_id = "thread-1", "turn-1"
        usage = {"inputTokens": 8, "cachedInputTokens": 0, "cacheWriteInputTokens": 0,
                 "outputTokens": 2, "reasoningOutputTokens": 0, "totalTokens": 10}
        total = {"inputTokens": 108, "cachedInputTokens": 0, "cacheWriteInputTokens": 0,
                 "outputTokens": 2, "reasoningOutputTokens": 0, "totalTokens": 110}
        upstream = TwoRequestUpstream([
            {"method": "thread/tokenUsage/updated", "params": {
                "threadId": thread_id, "turnId": turn_id,
                "tokenUsage": {"last": usage, "total": total}}},
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
            return {"data": [{"id": "gpt-5.6-luna", "hidden": False,
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


if __name__ == "__main__":
    unittest.main()
