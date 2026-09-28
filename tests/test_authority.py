import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import authority
import modellabs


THREAD = "00000000-0000-4000-8000-000000000321"


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        applied = patch.object(authority, "ROOT", self.root)
        applied.start()
        self.addCleanup(applied.stop)
        self.descriptor = authority.acquire_lock(THREAD)
        self.addCleanup(os.close, self.descriptor)

    def test_clear_all_pins_returns_thread_to_adaptive_authority(self):
        authority.initialize_locked(
            THREAD, "gpt-5.6-sol", "xhigh",
            explicit_model=True, explicit_effort=True,
        )
        result = authority.clear_pins_locked(THREAD)
        self.assertEqual(result, {
            "schema": authority.SCHEMA,
            "thread_id": THREAD,
            "model": None,
            "effort": None,
            "explicit_model": False,
            "explicit_effort": False,
        })
        self.assertEqual(authority.read_locked(THREAD), result)

    def test_clear_one_pin_preserves_the_other_explicit_choice(self):
        authority.initialize_locked(
            THREAD, "gpt-5.6-sol", "xhigh",
            explicit_model=True, explicit_effort=True,
        )
        result = authority.clear_pins_locked(THREAD, clear_effort=False)
        self.assertIsNone(result["model"])
        self.assertFalse(result["explicit_model"])
        self.assertEqual(result["effort"], "xhigh")
        self.assertTrue(result["explicit_effort"])

    def test_refuses_a_noop_clear(self):
        authority.initialize_locked(
            THREAD, None, None,
            explicit_model=False, explicit_effort=False,
        )
        with self.assertRaisesRegex(authority.AuthorityError, "At least one"):
            authority.clear_pins_locked(
                THREAD, clear_model=False, clear_effort=False,
            )


class ModelLabsAuthorityCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        applied = patch.object(authority, "ROOT", self.root)
        applied.start()
        self.addCleanup(applied.stop)
        descriptor = authority.acquire_lock(THREAD)
        try:
            authority.initialize_locked(
                THREAD, "gpt-5.6-sol", "xhigh",
                explicit_model=True, explicit_effort=True,
            )
        finally:
            os.close(descriptor)

    def test_unpin_command_clears_both_choices_under_lock(self):
        arguments = ["modellabs", "unpin", "--thread-id", THREAD]
        with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()) as output:
            modellabs.main()
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "unpinned")
        self.assertEqual(result["field"], "all")
        self.assertFalse(result["explicit_model"])
        self.assertFalse(result["explicit_effort"])
        self.assertIsNone(result["model"])
        self.assertIsNone(result["effort"])


if __name__ == "__main__":
    unittest.main()
