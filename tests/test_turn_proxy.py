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


class FakeClient:
    def __init__(self, request, authorization="Bearer token"):
        self.request = SimpleNamespace(headers={"Authorization": authorization}, path="/")
        self._request = json.dumps(request)
        self._done_id = request.get("id")
        self._sent_request = False
        self.done = asyncio.Event()
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._sent_request:
            self._sent_request = True
            return self._request
        await self.done.wait()
        raise StopAsyncIteration

    async def send(self, raw):
        self.sent.append(json.loads(raw))
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
        with patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
             patch.object(turn_proxy, "_rpc", side_effect=rpc), \
             patch.object(turn_proxy, "record_metric", side_effect=lambda event, **fields: metrics.append((event, fields))):
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
            with patch.object(turn_proxy, "ROOT", root), \
                 patch.object(turn_proxy, "_read_token", return_value="token"), \
                 patch.object(turn_proxy, "acquire_thread_ownership", side_effect=lambda *_args, **_kwargs: os.open("/dev/null", os.O_RDONLY)), \
                 patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)):
                await asyncio.wait_for(turn_proxy.handler(client), timeout=2)
            forwarded = upstream.sent[0]["params"]
            self.assertEqual(forwarded["model"], "gpt-5.6-terra")
            self.assertTrue(forwarded["config"]["mcp_servers"]["modelControl"]["enabled"])

    async def test_usage_and_completion_before_admission_are_processed_once(self):
        thread_id, turn_id = "thread-1", "turn-1"
        usage = {"inputTokens": 8, "cachedInputTokens": 0, "cacheWriteInputTokens": 0,
                 "outputTokens": 2, "reasoningOutputTokens": 0, "totalTokens": 10}
        total = {"inputTokens": 108, "cachedInputTokens": 0, "cacheWriteInputTokens": 0,
                 "outputTokens": 2, "reasoningOutputTokens": 0, "totalTokens": 110}
        upstream = FakeUpstream([
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
        client = FakeClient(request)
        metrics = []

        async def catalog(_token):
            return {"data": [{"id": "gpt-5.6-luna", "hidden": False,
                              "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}]}

        with patch.object(turn_proxy, "_read_token", return_value="token"), \
             patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
             patch.object(turn_proxy, "live_catalog", side_effect=catalog), \
             patch.object(turn_proxy, "record_metric", side_effect=lambda event, **fields: metrics.append((event, fields))), \
             patch.object(turn_proxy, "record_route"), \
             patch.object(turn_proxy, "record_completion", new=AsyncMock()):
            await asyncio.wait_for(turn_proxy.handler(client), timeout=2)

        usage_records = [fields for event, fields in metrics if event == "turn_usage"]
        unavailable = [fields for event, fields in metrics if event == "turn_usage_unavailable"]
        self.assertEqual(len(usage_records), 1)
        self.assertEqual(usage_records[0]["usage"]["totalTokens"], 10)
        self.assertEqual(unavailable, [])


if __name__ == "__main__":
    unittest.main()
