import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import authority
import install
import receipt_journal
import telemetry
import turn_proxy
from protocol_policy import classify
from test_turn_proxy import ConnectContext, FakeClient, FakeUpstream, TwoRequestUpstream


class ConsequentialGateTests(unittest.IsolatedAsyncioTestCase):
    def test_protocol_policy_rejects_known_unmanaged_and_unknown_methods(self):
        self.assertEqual(classify("thread/name/set"), "owner_mutation")
        self.assertEqual(classify("thread/unarchive"), "owner_mutation")
        self.assertEqual(classify("thread/shellCommand"), "owner_mutation")
        self.assertEqual(classify("review/start"), "rejected_inference")
        self.assertEqual(classify("future/dangerousMutation"), "unknown")

    async def test_unknown_method_and_path_resume_never_reach_host(self):
        for request in (
            {"id": 1, "method": "future/dangerousMutation", "params": {}},
            {"id": 2, "method": "thread/resume", "params": {"path": "/tmp/history"}},
        ):
            client = FakeClient(request)
            upstream = FakeUpstream([])
            with patch.object(turn_proxy, "_read_token", return_value="token"), \
                 patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)):
                await asyncio.wait_for(turn_proxy.handler(client), timeout=2)
            self.assertEqual(upstream.sent, [])
            self.assertIn("error", client.sent[0])

    def test_authority_rejects_missing_corrupt_and_incomplete_documents(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(authority, "ROOT", Path(directory)):
            with self.assertRaises(authority.AuthorityError):
                authority.read_locked("thread")
            path = authority.path_for("thread")
            path.parent.mkdir()
            for value in ({"thread_id": "thread"}, {"schema": authority.SCHEMA,
                          "thread_id": "thread", "model": None, "effort": None,
                          "explicit_model": True, "explicit_effort": False}):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(authority.AuthorityError):
                    authority.read_locked("thread")

    def test_terminal_write_failure_retries_without_duplicate_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics.jsonl"
            journal = root / "journal"
            with patch.object(telemetry, "METRICS_PATH", metrics), \
                 patch.object(receipt_journal, "METRICS_PATH", metrics), \
                 patch.object(receipt_journal, "JOURNAL_DIR", journal):
                info = {"model": "gpt", "effort": "low", "class": "routine",
                        "terminal_recorded": asyncio.Event(), "usage_recorded": asyncio.Event(),
                        "finished": asyncio.Event(), "usage_tracker": telemetry.UsageTracker()}
                receipt_journal.create("thread", "turn", info)
                info["journal_path"] = receipt_journal.path_for("thread", "turn")
                receipt_journal.mark(info["journal_path"], "accepted")
                original_mark = receipt_journal.mark
                attempts = 0

                def fail_once(path, stage):
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        raise OSError("simulated journal failure")
                    return original_mark(path, stage)

                with patch.object(receipt_journal, "mark", side_effect=fail_once):
                    with self.assertRaises(OSError):
                        turn_proxy.finalize_route("thread", "turn", info, status="completed",
                                                  elapsed_ms=1, terminal_source="test",
                                                  usage_complete=False)
                    turn_proxy.finalize_route("thread", "turn", info, status="completed",
                                              elapsed_ms=1, terminal_source="test",
                                              usage_complete=False)
                rows = [json.loads(line) for line in metrics.read_text().splitlines()]
                self.assertEqual(sum(row["event"] == "turn_completed" for row in rows), 1)
                self.assertEqual(sum(row["event"] == "turn_usage_unavailable" for row in rows), 1)
                self.assertFalse(info["journal_path"].exists())

    async def test_recovery_uses_persisted_identity_and_unavailable_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = root / "journal"
            route = {"model": "gpt", "effort": "low", "class": "routine"}
            with patch.object(receipt_journal, "JOURNAL_DIR", journal):
                receipt_journal.create("thread", "turn", route)

                async def finish(thread_id, turn_id, info, _token, _delivered):
                    self.assertEqual((thread_id, turn_id), ("thread", "turn"))
                    self.assertEqual(info["journal_path"], receipt_journal.path_for("thread", "turn"))
                    info["finished"].set()

                descriptor = os.open("/dev/null", os.O_RDONLY)
                with patch.object(turn_proxy, "acquire_thread_ownership", return_value=descriptor), \
                     patch.object(turn_proxy, "record_completion", side_effect=finish), \
                     patch.object(turn_proxy, "track_background", side_effect=lambda _task: None):
                    held = await turn_proxy.recover_receipt_obligations("token")
                    await asyncio.sleep(0)
                self.assertEqual(held, [descriptor])
                os.close(descriptor)

    async def test_route_accepted_metric_failure_keeps_completion_obligation(self):
        thread_id = "00000000-0000-4000-8000-000000000041"
        turn_id = "00000000-0000-4000-8000-000000000042"
        upstream = TwoRequestUpstream([
            {"id": 1, "result": {"thread": {"id": thread_id}}},
            {"id": 2, "result": {"turn": {"id": turn_id}}},
        ])
        client = FakeClient([
            {"id": 1, "method": "thread/resume", "params": {"threadId": thread_id}},
            {"id": 2, "method": "turn/start", "params": {"threadId": thread_id,
                "input": [{"type": "text", "text": "Format this value as CSV."}]}},
        ])
        completion_started = asyncio.Event()

        async def complete(_thread, _turn, _info, _token, finished):
            completion_started.set()
            finished.set()

        async def catalog(_token):
            return {"data": [{"id": "gpt-5.6-luna", "hidden": False,
                              "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}]}

        def metric(event, **_fields):
            if event == "route_accepted":
                raise OSError("simulated accepted-event failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(authority, "ROOT", root), \
                 patch.object(receipt_journal, "JOURNAL_DIR", root / "receipts"), \
                 patch.object(receipt_journal, "QUARANTINE_DIR", root / "quarantine"):
                descriptor = authority.acquire_lock(thread_id)
                try:
                    authority.initialize_locked(thread_id, None, None,
                                                explicit_model=False, explicit_effort=False)
                finally:
                    os.close(descriptor)
                owner = os.open("/dev/null", os.O_RDONLY)
                turn_proxy.ACCOUNTING_BLOCKED = False
                with patch.object(turn_proxy, "_read_token", return_value="token"), \
                     patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
                     patch.object(turn_proxy, "acquire_thread_ownership", return_value=owner), \
                     patch.object(turn_proxy, "live_catalog", side_effect=catalog), \
                     patch.object(turn_proxy, "record_metric", side_effect=metric), \
                     patch.object(turn_proxy, "record_route"), \
                     patch.object(turn_proxy, "record_completion", side_effect=complete):
                    await asyncio.wait_for(turn_proxy.handler(client), timeout=2)
                self.assertTrue(completion_started.is_set())
                self.assertTrue(turn_proxy.ACCOUNTING_BLOCKED)
                self.assertTrue(receipt_journal.path_for(thread_id, turn_id).exists())
                turn_proxy.ACCOUNTING_BLOCKED = False

    async def test_server_request_id_collision_is_correlated_at_relay_boundary(self):
        thread_id = "00000000-0000-4000-8000-000000000051"
        turn_id = "00000000-0000-4000-8000-000000000052"

        class CollisionClient:
            request = SimpleNamespace(headers={"Authorization": "Bearer token"}, path="/")

            def __init__(self):
                self.stage = 0
                self.resume_done = asyncio.Event()
                self.server_request_seen = asyncio.Event()
                self.final_seen = asyncio.Event()
                self.sent = []

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.stage == 0:
                    self.stage += 1
                    return json.dumps({"id": 1, "method": "thread/resume",
                                       "params": {"threadId": thread_id}})
                if self.stage == 1:
                    await self.resume_done.wait()
                    self.stage += 1
                    return json.dumps({"id": 7, "method": "turn/start", "params": {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": "Format values as CSV."}]}})
                if self.stage == 2:
                    await self.server_request_seen.wait()
                    self.stage += 1
                    return json.dumps({"id": 7, "result": {"answers": {}}})
                await self.final_seen.wait()
                raise StopAsyncIteration

            async def send(self, raw):
                value = json.loads(raw)
                self.sent.append(value)
                if value.get("id") == 1:
                    self.resume_done.set()
                elif value.get("method") == "item/tool/requestUserInput":
                    self.server_request_seen.set()
                elif value.get("id") == 7 and "result" in value:
                    self.final_seen.set()

            async def close(self, **_kwargs):
                self.final_seen.set()

        class CollisionUpstream(FakeUpstream):
            def __init__(self):
                super().__init__([
                    {"id": 1, "result": {"thread": {"id": thread_id}}},
                    {"id": 7, "method": "item/tool/requestUserInput", "params": {}},
                    {"id": 7, "result": {"turn": {"id": turn_id}}},
                ])
                self.yielded = 0

            async def __anext__(self):
                required_sends = (1, 2, 3)[self.yielded] if self.yielded < 3 else 99
                while len(self.sent) < required_sends:
                    await asyncio.sleep(0)
                if self.messages:
                    self.yielded += 1
                    return self.messages.pop(0)
                await asyncio.Future()

        async def catalog(_token):
            return {"data": [{"id": "gpt-5.6-luna", "hidden": False,
                              "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}]}

        async def complete(_thread, _turn, _info, _token, finished):
            finished.set()

        client, upstream = CollisionClient(), CollisionUpstream()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(authority, "ROOT", root), \
                 patch.object(receipt_journal, "JOURNAL_DIR", root / "receipts"), \
                 patch.object(receipt_journal, "QUARANTINE_DIR", root / "quarantine"):
                descriptor = authority.acquire_lock(thread_id)
                try:
                    authority.initialize_locked(thread_id, None, None,
                                                explicit_model=False, explicit_effort=False)
                finally:
                    os.close(descriptor)
                with patch.object(turn_proxy, "_read_token", return_value="token"), \
                     patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
                     patch.object(turn_proxy, "acquire_thread_ownership",
                                  return_value=os.open("/dev/null", os.O_RDONLY)), \
                     patch.object(turn_proxy, "live_catalog", side_effect=catalog), \
                     patch.object(turn_proxy, "record_metric"), \
                     patch.object(turn_proxy, "record_route"), \
                     patch.object(turn_proxy, "record_completion", side_effect=complete):
                    try:
                        await asyncio.wait_for(turn_proxy.handler(client), timeout=2)
                    except TimeoutError:
                        self.fail(f"relay stalled: client={client.sent!r} upstream={upstream.sent!r} stage={client.stage}")
        self.assertTrue(any(item.get("method") == "item/tool/requestUserInput" for item in client.sent))
        self.assertTrue(any((item.get("result") or {}).get("turn", {}).get("id") == turn_id
                            for item in client.sent))
        self.assertEqual(upstream.sent[-1], {"id": 7, "result": {"answers": {}}})

    def test_installer_rejects_malformed_manifest_and_redirected_roots_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "home"
            home.mkdir()
            upstream = root / "codex"
            upstream.write_text("#!/bin/sh\n", encoding="utf-8")
            upstream.chmod(0o755)
            manifest = home / install.OWNED_MANIFEST
            payload = {home / "turn_proxy.py": b"new"}
            for value in ({"schema": "modellabs.owned_files.v1"}, [],
                          {"schema": "modellabs.owned_files.v1", "files": {"x": 7}}):
                manifest.write_text(json.dumps(value), encoding="utf-8")
                before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
                with self.assertRaises(RuntimeError):
                    install.preflight_install_payloads(payload, home, upstream, root / "codex-home", root / "bin")
                after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
                self.assertEqual(before, after)

            manifest.unlink()
            unrelated = root / "unrelated"
            unrelated.mkdir()
            (root / "bin").symlink_to(unrelated, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlinked|redirected"):
                install.preflight_install_payloads(payload, home, upstream, root / "codex-home", root / "bin")
            (root / "bin").unlink()
            (home / "venv").symlink_to(unrelated, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlinked|redirected"):
                install.preflight_install_payloads(payload, home, upstream, root / "codex-home", root / "bin")

    def test_dry_run_wrapper_preflight_does_not_create_bin_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home, bin_dir = root / "home", root / "missing-bin"
            upstream = root / "codex"
            upstream.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            upstream.chmod(0o755)
            install.write_wrappers(home, bin_dir, upstream, dry_run=True)
            self.assertFalse(bin_dir.exists())


if __name__ == "__main__":
    unittest.main()
