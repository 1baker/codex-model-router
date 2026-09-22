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
    def test_turn_route_normalizes_nested_effective_settings(self):
        request = {"id": 1, "method": "turn/start", "params": {
            "threadId": "thread", "model": "stale-top", "effort": "low",
            "collaborationMode": {"settings": {
                "model": "gpt-5.6-terra", "reasoning_effort": "high"}},
            "input": [{"type": "text", "text": "Refactor this module."}]}}
        choice = {"model": "gpt-5.6-sol", "effort": "medium",
                  "intelligence_slider": "medium", "servers": [], "class": "coding",
                  "explicit_model": False, "explicit_effort": False}
        with patch.object(turn_proxy, "route", return_value=choice.copy()), \
             patch.object(turn_proxy, "adapt", side_effect=lambda value, **_kwargs: value):
            routed, _info = turn_proxy.route_request(json.dumps(request))
        params = json.loads(routed)["params"]
        self.assertEqual((params["model"], params["effort"]), ("gpt-5.6-sol", "medium"))
        self.assertEqual(params["collaborationMode"]["settings"], {
            "model": "gpt-5.6-sol", "reasoning_effort": "medium"})

        with patch.object(turn_proxy, "route", return_value=choice.copy()), \
             patch.object(turn_proxy, "adapt", side_effect=lambda value, **_kwargs: value):
            routed, _info = turn_proxy.route_request(
                json.dumps(request), preserve_model=True, preserve_effort=True,
                explicit_model=True, explicit_effort=True)
        params = json.loads(routed)["params"]
        self.assertEqual((params["model"], params["effort"]), ("stale-top", "low"))
        self.assertEqual(params["collaborationMode"]["settings"], {
            "model": "stale-top", "reasoning_effort": "low"})

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
            {"id": 3, "method": "thread/resume", "params": {
                "threadId": "00000000-0000-4000-8000-000000000099", "path": "/tmp/history"}},
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
                turn_proxy.ACCOUNTING_BLOCKED = False

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
                tasks = []
                with patch.object(turn_proxy, "acquire_thread_ownership", return_value=descriptor), \
                     patch.object(turn_proxy, "record_completion", side_effect=finish), \
                     patch.object(turn_proxy, "track_background", side_effect=tasks.append):
                    held = await turn_proxy.recover_receipt_obligations("token")
                    await asyncio.gather(*tasks)
                self.assertIsNone(held)

    async def test_recovery_reuses_canonical_receipts_and_retires_complete_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, journal = root / "metrics.jsonl", root / "journal"
            route = {"model": "gpt", "effort": "low", "class": "routine"}
            with patch.object(telemetry, "METRICS_PATH", metrics), \
                 patch.object(receipt_journal, "JOURNAL_DIR", journal):
                receipt_journal.create("thread", "turn", route)
                for stage, event in (("accepted", "route_accepted"),
                                     ("terminal", "turn_completed"),
                                     ("usage", "turn_usage_unavailable")):
                    telemetry.record(event, receipt_id=f"thread:turn:{stage}",
                                     thread_id="thread", turn_id="turn")
                descriptor = os.open("/dev/null", os.O_RDONLY)
                tasks = []
                with patch.object(turn_proxy, "acquire_thread_ownership", return_value=descriptor), \
                     patch.object(turn_proxy, "track_background", side_effect=tasks.append):
                    await turn_proxy.recover_receipt_obligations("token")
                    await asyncio.gather(*tasks)
                self.assertFalse(receipt_journal.path_for("thread", "turn").exists())
                rows = [json.loads(line) for line in metrics.read_text().splitlines()]
                self.assertEqual(len(rows), 3)

    async def test_recovery_setup_failure_releases_owner_before_retry(self):
        path = Path("/tmp/recovery-obligation.json")
        obligation = {"schema": receipt_journal.SCHEMA, "thread_id": "thread",
                      "turn_id": "turn", "model": "gpt", "effort": "low",
                      "task_class": "routine", "accepted": True,
                      "terminal": True, "usage": True}
        descriptors = [os.open("/dev/null", os.O_RDONLY) for _ in range(2)]
        calls = 0

        def load(_path):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("simulated setup failure")
            return obligation.copy()

        tasks = []
        with patch.object(receipt_journal, "list_obligations", return_value=[(path, obligation)]), \
             patch.object(Path, "exists", side_effect=[True, True, True, False]), \
             patch.object(receipt_journal, "load", side_effect=load), \
             patch.object(turn_proxy, "acquire_thread_ownership", side_effect=descriptors), \
             patch.object(turn_proxy.asyncio, "sleep", new=AsyncMock()), \
             patch.object(turn_proxy, "track_background", side_effect=tasks.append):
            await turn_proxy.recover_receipt_obligations("token")
            await asyncio.gather(*tasks)
        for descriptor in descriptors:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

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

    def test_telemetry_repairs_torn_tail_and_retries_log_fsync(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            metrics.write_bytes(b'{"event":"torn"')
            with patch.object(telemetry, "METRICS_PATH", metrics):
                telemetry.record("turn_completed", receipt_id="r1", thread_id="t", turn_id="u")
                rows = [json.loads(line) for line in metrics.read_text().splitlines()]
                self.assertEqual([row["receipt_id"] for row in rows], ["r1"])

                real_fsync = os.fsync
                calls = 0

                def fail_log_once(fd):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        raise OSError("simulated log fsync failure")
                    return real_fsync(fd)

                with patch.object(telemetry.os, "fsync", side_effect=fail_log_once):
                    with self.assertRaises(OSError):
                        telemetry.record("turn_completed", receipt_id="r2", thread_id="t", turn_id="v")
                telemetry.record("turn_completed", receipt_id="r2", thread_id="t", turn_id="v")
                rows = [json.loads(line) for line in metrics.read_text().splitlines()]
                self.assertEqual(sum(row.get("receipt_id") == "r2" for row in rows), 1)

    def test_telemetry_retries_short_writes_until_payload_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "write-all"
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
            real_write = os.write

            def short_write(fd, content):
                length = max(1, len(content) // 2)
                return real_write(fd, content[:length])

            try:
                with patch.object(telemetry.os, "write", side_effect=short_write):
                    telemetry._write_all(descriptor, b"complete-payload")
            finally:
                os.close(descriptor)
            self.assertEqual(path.read_bytes(), b"complete-payload")

    def test_existing_venv_rejects_escaping_site_packages_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv = root / "venv"
            site = venv / "lib/python3.12"
            site.mkdir(parents=True)
            outside = root / "outside"
            outside.mkdir()
            (site / "site-packages").symlink_to(outside, target_is_directory=True)
            (venv / "bin").mkdir()
            (venv / "bin/python").symlink_to(Path(sys.executable))
            with self.assertRaisesRegex(RuntimeError, "escaping virtualenv"):
                install.verify_existing_venv_containment(venv)

            (site / "site-packages").unlink()
            (venv / "bin/python-escape").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "escaping virtualenv"):
                install.verify_existing_venv_containment(venv)

    async def test_async_authority_lock_cancellation_does_not_strand_descriptor(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(authority, "ROOT", Path(directory)):
            held = authority.acquire_lock("thread")
            waiter = asyncio.create_task(authority.acquire_lock_async("thread"))
            await asyncio.sleep(0.06)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            os.close(held)
            descriptor = await asyncio.wait_for(authority.acquire_lock_async("thread"), timeout=1)
            os.close(descriptor)

    async def test_queued_thread_setting_commits_only_after_applied_notification(self):
        thread_id = "00000000-0000-4000-8000-000000000071"
        client = FakeClient([
            {"id": 1, "method": "thread/resume", "params": {"threadId": thread_id}},
            {"id": 2, "method": "thread/settings/update", "params": {
                "threadId": thread_id, "model": "gpt-5.6-terra", "effort": "low"}},
        ])
        upstream = TwoRequestUpstream([
            {"id": 1, "result": {"thread": {"id": thread_id}}},
            {"id": 2, "result": {}},
            {"method": "thread/settings/updated", "params": {"threadId": thread_id,
                "threadSettings": {"model": "gpt-5.6-terra", "effort": "low"}}},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(authority, "ROOT", root), \
                 patch.object(receipt_journal, "QUARANTINE_DIR", root / "quarantine"):
                fd = authority.acquire_lock(thread_id)
                try:
                    authority.initialize_locked(thread_id, "gpt-5.6-sol", "high",
                                                explicit_model=True, explicit_effort=True)
                finally:
                    os.close(fd)
                with patch.object(turn_proxy, "_read_token", return_value="token"), \
                     patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
                     patch.object(turn_proxy, "acquire_thread_ownership",
                                  return_value=os.open("/dev/null", os.O_RDONLY)):
                    await asyncio.wait_for(turn_proxy.handler(client), timeout=2)
                value = authority.read_locked(thread_id)
                self.assertEqual((value["model"], value["effort"]), ("gpt-5.6-terra", "low"))

    async def test_explicit_resume_choice_replaces_older_pin_before_next_turn(self):
        thread_id = "00000000-0000-4000-8000-000000000081"
        client = FakeClient({"id": 1, "method": "thread/resume", "params": {
            "threadId": thread_id, "model": "gpt-5.6-terra",
            "config": {"model_reasoning_effort": "low"}}},
            authorization="Bearer token.explicit-both")
        upstream = FakeUpstream([{"id": 1, "result": {"thread": {"id": thread_id},
                                                         "model": "gpt-5.6-terra",
                                                         "reasoningEffort": "low"}}])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(authority, "ROOT", root), \
                 patch.object(receipt_journal, "QUARANTINE_DIR", root / "quarantine"):
                fd = authority.acquire_lock(thread_id)
                try:
                    authority.initialize_locked(thread_id, "gpt-5.6-sol", "high",
                                                explicit_model=True, explicit_effort=True)
                finally:
                    os.close(fd)
                with patch.object(turn_proxy, "_read_token", return_value="token"), \
                     patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
                     patch.object(turn_proxy, "acquire_thread_ownership",
                                  return_value=os.open("/dev/null", os.O_RDONLY)):
                    await asyncio.wait_for(turn_proxy.handler(client), timeout=2)
                value = authority.read_locked(thread_id)
                self.assertEqual((value["model"], value["effort"]), ("gpt-5.6-terra", "low"))

    async def test_resume_error_does_not_admit_pipelined_mutation_or_poison_proxy(self):
        thread_id = "00000000-0000-4000-8000-000000000082"

        class EagerClient:
            request = SimpleNamespace(headers={"Authorization": "Bearer token"}, path="/")

            def __init__(self):
                self.items = iter([
                    {"id": 1, "method": "thread/resume", "params": {"threadId": thread_id}},
                    {"id": 2, "method": "thread/name/set", "params": {
                        "threadId": thread_id, "name": "too early"}},
                ])
                self.done = asyncio.Event()
                self.sent = []

            def __aiter__(self): return self

            async def __anext__(self):
                try:
                    return json.dumps(next(self.items))
                except StopIteration:
                    await self.done.wait()
                    raise StopAsyncIteration

            async def send(self, raw):
                value = json.loads(raw)
                self.sent.append(value)
                if value.get("id") == 1:
                    self.done.set()

            async def close(self, **_kwargs): self.done.set()

        client = EagerClient()
        upstream = FakeUpstream([{"id": 1, "error": {"code": -32000, "message": "missing"}}])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(authority, "ROOT", root), \
                 patch.object(receipt_journal, "QUARANTINE_DIR", root / "quarantine"):
                descriptor = authority.acquire_lock(thread_id)
                try:
                    authority.initialize_locked(thread_id, None, None,
                                                explicit_model=False, explicit_effort=False)
                finally:
                    os.close(descriptor)
                turn_proxy.ACCOUNTING_BLOCKED = False
                with patch.object(turn_proxy, "_read_token", return_value="token"), \
                     patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
                     patch.object(turn_proxy, "acquire_thread_ownership",
                                  return_value=os.open("/dev/null", os.O_RDONLY)):
                    await asyncio.wait_for(turn_proxy.handler(client), timeout=2)
        self.assertEqual([item["id"] for item in upstream.sent], [1])
        self.assertEqual(next(item for item in client.sent if item.get("id") == 2)["error"]["code"],
                         -32003)
        self.assertFalse(turn_proxy.ACCOUNTING_BLOCKED)

    async def test_reused_rpc_id_cannot_settle_waiting_settings_lifecycle(self):
        thread_id = "00000000-0000-4000-8000-000000000083"

        class ReuseClient:
            request = SimpleNamespace(headers={"Authorization": "Bearer token"}, path="/")

            def __init__(self):
                self.stage = 0
                self.resume = asyncio.Event()
                self.settings_response = asyncio.Event()
                self.notification = asyncio.Event()
                self.id2_responses = 0
                self.sent = []

            def __aiter__(self): return self

            async def __anext__(self):
                if self.stage == 0:
                    self.stage += 1
                    return json.dumps({"id": 1, "method": "thread/resume",
                                       "params": {"threadId": thread_id}})
                if self.stage == 1:
                    await self.resume.wait()
                    self.stage += 1
                    return json.dumps({"id": 2, "method": "thread/settings/update", "params": {
                        "threadId": thread_id, "collaborationMode": {"settings": {
                            "model": "gpt-5.6-terra", "reasoning_effort": "low"}}}})
                if self.stage == 2:
                    await self.settings_response.wait()
                    self.stage += 1
                    return json.dumps({"id": 2, "method": "model/list", "params": {}})
                await self.notification.wait()
                raise StopAsyncIteration

            async def send(self, raw):
                value = json.loads(raw)
                self.sent.append(value)
                if value.get("id") == 1:
                    self.resume.set()
                elif value.get("id") == 2:
                    self.id2_responses += 1
                    if self.id2_responses == 1:
                        self.settings_response.set()
                elif value.get("method") == "thread/settings/updated":
                    self.notification.set()

            async def close(self, **_kwargs): self.notification.set()

        class ReuseUpstream(FakeUpstream):
            def __init__(self):
                super().__init__([
                    {"id": 1, "result": {"thread": {"id": thread_id}}},
                    {"id": 2, "result": {}},
                    {"id": 2, "result": {"data": []}},
                    {"method": "thread/settings/updated", "params": {
                        "threadId": thread_id, "threadSettings": {
                            "model": "gpt-5.6-terra", "reasoning_effort": "low"}}},
                ])
                self.yielded = 0
                self.notification_sent = False

            async def __anext__(self):
                if self.yielded >= 4:
                    await asyncio.Future()
                required = (1, 2, 3, 3)[self.yielded]
                while len(self.sent) < required:
                    await asyncio.sleep(0)
                value = self.messages.pop(0)
                self.yielded += 1
                if self.yielded == 4:
                    self.notification_sent = True
                return value

        client, upstream, cleared_after_notification = ReuseClient(), ReuseUpstream(), []
        original_clear = receipt_journal.clear_quarantine

        def clear(path):
            cleared_after_notification.append(upstream.notification_sent)
            original_clear(path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(authority, "ROOT", root), \
                 patch.object(receipt_journal, "QUARANTINE_DIR", root / "quarantine"):
                descriptor = authority.acquire_lock(thread_id)
                try:
                    authority.initialize_locked(thread_id, "gpt-5.6-sol", "high",
                                                explicit_model=True, explicit_effort=True)
                finally:
                    os.close(descriptor)
                with patch.object(turn_proxy, "_read_token", return_value="token"), \
                     patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)), \
                     patch.object(turn_proxy, "acquire_thread_ownership",
                                  return_value=os.open("/dev/null", os.O_RDONLY)), \
                     patch.object(receipt_journal, "clear_quarantine", side_effect=clear):
                    await asyncio.wait_for(turn_proxy.handler(client), timeout=2)
                value = authority.read_locked(thread_id)
        self.assertEqual(cleared_after_notification, [False, True])
        self.assertEqual((value["model"], value["effort"]), ("gpt-5.6-terra", "low"))

    async def test_duplicate_outstanding_client_id_is_rejected_before_forwarding(self):
        class DuplicateClient:
            request = SimpleNamespace(headers={"Authorization": "Bearer token"}, path="/")

            def __init__(self):
                self.items = iter([
                    {"id": "private-id", "method": "thread/list", "params": {}},
                    {"id": "private-id", "method": "model/list", "params": {}},
                ])
                self.done = asyncio.Event()
                self.sent = []

            def __aiter__(self): return self

            async def __anext__(self):
                try:
                    return json.dumps(next(self.items))
                except StopIteration:
                    await self.done.wait()
                    raise StopAsyncIteration

            async def send(self, raw):
                self.sent.append(json.loads(raw))
                self.done.set()

            async def close(self, **_kwargs): self.done.set()

        client, upstream = DuplicateClient(), FakeUpstream([])
        with patch.object(turn_proxy, "_read_token", return_value="token"), \
             patch.object(turn_proxy.websockets, "connect", return_value=ConnectContext(upstream)):
            await asyncio.wait_for(turn_proxy.handler(client), timeout=2)
        self.assertEqual(len(upstream.sent), 1)
        self.assertEqual(client.sent[0]["error"]["code"], -32600)


if __name__ == "__main__":
    unittest.main()
