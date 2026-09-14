import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modellabs import route


class RoutingTests(unittest.TestCase):
    def test_selects_the_lowest_sufficient_effort(self):
        cases = [
            ("Format these three values as CSV.", "gpt-5.6-luna", "low"),
            ("Implement a small validated parser.", "gpt-5.6-terra", "medium"),
            ("Investigate an intermittent race condition and fix it.", "gpt-5.6-sol", "high"),
            ("Design a production cross-system security architecture.", "gpt-6-astra", "xhigh"),
            ("Perform an adversarial audit of this production migration.", "gpt-6-astra", "max"),
            ("Create an ultra exhaustive independent review of this architecture.", "gpt-6-astra", "ultra"),
        ]
        for prompt, model, effort in cases:
            with self.subTest(prompt=prompt):
                choice = route(prompt)
                self.assertEqual((choice["model"], choice["effort"]), (model, effort))

    def test_explicit_effort_wins(self):
        choice = route("Format these values as CSV with reasoning effort ultra.")
        self.assertEqual(choice["effort"], "max")


if __name__ == "__main__":
    unittest.main()
