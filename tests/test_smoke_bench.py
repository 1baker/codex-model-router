import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_policy import adapt
from modellabs import route
from smoke_bench import grade, materialize, parse_codex_jsonl, ranked


class SmokeBenchTests(unittest.TestCase):
    def test_materialize_and_grade_real_product(self):
        scenario = {"files": {"product.json": "[1,2]", "verify.py": "import json; assert json.load(open('product.json')) == [1,2]"},
                    "verify": [sys.executable, "verify.py"], "expected_final": '{"product":"ready"}'}
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            materialize(workspace, scenario)
            result = grade(workspace, scenario, '{"product":"ready"}',
                           {"input_tokens": 10, "output_tokens": 2, "reasoning_output_tokens": 1})
        self.assertTrue(result["product_pass"])
        self.assertEqual(result["satisfaction_score"], 100)
        self.assertEqual(result["total_tokens"], 12)

    def test_parse_and_rank_puts_correctness_before_efficiency(self):
        output = '\n'.join((json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}}),
                            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 3}})))
        usage, final = parse_codex_jsonl(output)
        self.assertEqual((usage["input_tokens"], final), (20, "done"))
        rows = ranked([{"product_pass": False, "total_tokens": 1, "satisfaction_score": 0},
                       {"product_pass": True, "total_tokens": 100, "satisfaction_score": 100}])
        self.assertTrue(rows[0]["product_pass"])

    def test_preliminary_benchmark_is_shadow_only(self):
        choice = route("Implement a small validated parser.")
        evidence = [{"event": "benchmark_result", "task_class": "routine", "model": "gpt-5.6-sol",
                     "effort": "medium", "product_pass": True, "satisfaction_score": 100,
                     "total_tokens": 50, "recorded_at_ms": 2_000_000_000_000},
                    {"event": "benchmark_result", "task_class": "routine", "model": "gpt-5.6-terra",
                     "effort": "medium", "product_pass": True, "satisfaction_score": 100,
                     "total_tokens": 100, "recorded_at_ms": 2_000_000_000_000}]
        with patch.dict("os.environ", {"MODELLABS_ADAPTIVE_MODE": "enforce"}):
            result = adapt(choice, records=evidence, now_ms=2_000_000_000_000)
        self.assertEqual(result["model"], "gpt-5.6-terra")
        self.assertEqual(result["adaptive_recommendation"]["model"], "gpt-5.6-sol")
        self.assertEqual(result["adaptive_reason"], "shadow_benchmark_preliminary")

    def test_benchmark_requires_baseline_on_same_scenario(self):
        choice = route("Implement a small validated parser.")
        evidence = [{"event": "benchmark_result", "scenario_id": "other", "task_class": "routine",
                     "model": "gpt-5.6-sol", "effort": "medium", "product_pass": True,
                     "satisfaction_score": 100, "total_tokens": 10,
                     "recorded_at_ms": 2_000_000_000_000}]
        result = adapt(choice, records=evidence, now_ms=2_000_000_000_000)
        self.assertNotIn("adaptive_recommendation", result)

    def test_verified_benchmark_can_apply_after_two_scenarios(self):
        choice = route("Investigate an intermittent race condition and fix it.")
        rows = []
        for model, effort, tokens in (("gpt-5.6-sol", "high", 120),
                                      ("gpt-6-astra", "medium", 80)):
            for scenario in ("one", "two", "two"):
                rows.append({"event": "benchmark_result", "scenario_id": scenario,
                             "task_class": "difficult", "model": model, "effort": effort,
                             "product_pass": True, "satisfaction_score": 100,
                             "total_tokens": tokens, "recorded_at_ms": 2_000_000_000_000})
        with patch.dict("os.environ", {"MODELLABS_ADAPTIVE_MODE": "enforce"}):
            result = adapt(choice, records=rows, now_ms=2_000_000_000_000)
        self.assertEqual((result["model"], result["effort"]), ("gpt-6-astra", "medium"))
        self.assertEqual(result["adaptive_reason"], "benchmark_verified_efficiency")


if __name__ == "__main__":
    unittest.main()
