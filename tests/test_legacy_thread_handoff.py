import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import authority
import legacy_thread_handoff
import thread_owner


THREAD = "00000000-0000-4000-8000-000000000123"


class LegacyThreadHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sessions = self.root / "sessions"
        self.sessions.mkdir()
        self.rollout = self.sessions / f"rollout-test-{THREAD}.jsonl"
        self.rollout.write_text(json.dumps({"type": "session_meta", "payload": {
            "id": THREAD, "thread_source": "user"}}) + "\n", encoding="utf-8")
        for target, name, value in ((authority, "ROOT", self.root),
                                    (thread_owner, "ROOT", self.root),
                                    (thread_owner, "SESSIONS", self.sessions),
                                    (legacy_thread_handoff, "ROOT", self.root)):
            applied = patch.object(target, name, value)
            applied.start()
            self.addCleanup(applied.stop)

    def test_imports_only_once_without_pinning_a_model(self):
        legacy_thread_handoff.import_closed_thread(THREAD)
        self.assertEqual(authority.read_locked(THREAD), {
            "schema": authority.SCHEMA, "thread_id": THREAD,
            "model": None, "effort": None,
            "explicit_model": False, "explicit_effort": False})
        with self.assertRaisesRegex(RuntimeError, "existing thread authority"):
            legacy_thread_handoff.import_closed_thread(THREAD)

    def test_refuses_managed_evidence_without_writing_authority(self):
        (self.root / "metrics.jsonl").write_text(json.dumps({
            "event": "route_accepted", "thread_id": THREAD}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "prior managed evidence"):
            legacy_thread_handoff.import_closed_thread(THREAD)
        self.assertFalse(authority.path_for(THREAD).exists())

    def test_refuses_live_owner_without_writing_authority(self):
        with patch.object(thread_owner, "require_unowned",
                          side_effect=RuntimeError("standalone owner is live")):
            with self.assertRaisesRegex(RuntimeError, "standalone owner is live"):
                legacy_thread_handoff.import_closed_thread(THREAD)
        self.assertFalse(authority.path_for(THREAD).exists())

    def test_refuses_non_user_thread(self):
        self.rollout.write_text(json.dumps({"type": "session_meta", "payload": {
            "id": THREAD, "thread_source": "subagent"}}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "saved user conversation"):
            legacy_thread_handoff.import_closed_thread(THREAD)


if __name__ == "__main__":
    unittest.main()
