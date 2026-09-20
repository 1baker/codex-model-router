import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modellabs import route
from install import upsert_toml
from turn_proxy import selection_is_listed
from adaptive_policy import adapt, record_outcome
from health_dashboard import summarize


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
                self.assertEqual(choice["intelligence_slider"], effort)

    def test_explicit_effort_wins(self):
        choice = route("Format these values as CSV with reasoning effort ultra.")
        self.assertEqual(choice["effort"], "max")
        self.assertTrue(choice["explicit_effort"])

    def test_explicit_model_is_never_adapted(self):
        choice = route("Use Sol to implement a small validated parser.")
        self.assertTrue(choice["explicit_model"])
        self.assertEqual(adapt(choice, records=[])["adaptive_reason"], "explicit_user_override")

    def test_adaptation_is_shadow_by_default(self):
        choice = route("Implement a small validated parser.")
        records = [{"event": "outcome_signal", "outcome": "retry", "source": "explicit",
                    "task_bucket": choice["task_bucket"], "recorded_at_ms": 2_000_000_000_000}] * 4
        result = adapt(choice, records=records, now_ms=2_000_000_000_000)
        self.assertEqual(result["model"], choice["model"])
        self.assertEqual(result["adaptive_reason"], "shadow_retry_escalation")

    def test_explicit_outcome_only(self):
        records = []
        record_outcome(records, "thread", "turn", "verified", {"class": "routine", "model": "gpt-5.6-terra", "effort": "medium", "task_bucket": "routine:base"})
        self.assertEqual(records[0]["source"], "explicit")

    def test_dashboard_counts_accepted_turns_not_every_event(self):
        summary = summarize([
            {"event": "route_accepted", "model": "gpt-5.6-terra", "effort": "medium", "thread_id": "t"},
            {"event": "response_usage", "model": "gpt-5.6-terra", "usage": {"totalTokens": 23}},
            {"event": "turn_completed", "model": "gpt-5.6-terra"},
        ])
        self.assertEqual(summary["accepted_turn_counts"], {"gpt-5.6-terra": 1})
        self.assertEqual(summary["token_totals"], {"gpt-5.6-terra": 23})

    def test_catalog_gate_requires_listed_model_and_effort(self):
        catalog = {"data": [{"id": "gpt-6-astra", "hidden": False,
                             "supportedReasoningEfforts": [{"reasoningEffort": "high"},
                                                           {"reasoningEffort": "ultra"}]}]}
        self.assertTrue(selection_is_listed(catalog, {"model": "gpt-6-astra", "effort": "ultra"}))
        self.assertFalse(selection_is_listed(catalog, {"model": "gpt-6-astra", "effort": "max"}))
        self.assertFalse(selection_is_listed(catalog, {"model": "gpt-5.6-luna", "effort": "low"}))

    def test_toml_upsert_preserves_existing_sections(self):
        source = '[features]\nmemories = true\n\n[mcp_servers.other]\ncommand = "other"\n'
        result = upsert_toml(source, "features", {"step_model_switching": "true"})
        result = upsert_toml(result, "mcp_servers.modelControl", {"command": '"python"'})
        self.assertIn("memories = true", result)
        self.assertIn("step_model_switching = true", result)
        self.assertIn('[mcp_servers.other]', result)
        self.assertIn('[mcp_servers.modelControl]', result)


if __name__ == "__main__":
    unittest.main()
