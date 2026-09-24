import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from health_dashboard import active_tmux_session, attach_runtime_modes, read_records, summarize


class HealthDashboardTests(unittest.TestCase):
    def test_auto_session_uses_current_tmux_session(self):
        with patch("health_dashboard.subprocess.check_output", return_value="byobu\n"):
            self.assertEqual(active_tmux_session(), "byobu")

    def test_auto_session_falls_back_to_largest_live_session(self):
        def output(command, **_kwargs):
            if command[1] == "display-message":
                raise OSError("outside tmux")
            return "old\t2\t0\nbyobu\t7\t0\n"
        with patch("health_dashboard.subprocess.check_output", side_effect=output):
            self.assertEqual(active_tmux_session(), "byobu")

    def test_tab_runtime_mode_uses_live_descendants_not_saved_mode_alone(self):
        tabs = [
            {"index": "0", "mode": "model-host", "pane_pid": "100"},
            {"index": "1", "mode": "resume", "pane_pid": "200"},
            {"index": "2", "mode": "model-host", "pane_pid": "300"},
        ]
        attach_runtime_modes(tabs, "\n".join((
            "101 100 /bin/bash wrapper",
            "102 101 node /opt/codex/bin/codex.js --remote ws://127.0.0.1:50000 --remote-auth-token-env MODEL_SELECTOR_HOST_TOKEN",
            "201 200 node /opt/bin/codex resume some-thread",
            "301 300 node /opt/bin/codex resume another-thread",
        )))
        self.assertEqual([tab["runtime_mode"] for tab in tabs],
                         ["remote", "standalone", "standalone"])
        self.assertEqual([tab["mode_matches_runtime"] for tab in tabs], [True, True, False])
        self.assertNotIn("command", tabs[0])

    def test_summarize_uses_metadata_without_prompt_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            path.write_text("\n".join((
                json.dumps({"event": "route_accepted", "thread_id": "thread-a", "turn_id": "turn-a", "model": "gpt-5.6-sol", "effort": "high", "recorded_at_ms": 1}),
                json.dumps({"event": "turn_usage", "thread_id": "thread-a", "turn_id": "turn-a", "model": "gpt-5.6-sol", "effort": "high", "recorded_at_ms": 2, "usage": {"totalTokens": 42}}),
                json.dumps({"event": "turn_completed", "thread_id": "thread-a", "turn_id": "turn-a", "model": "gpt-5.6-sol", "effort": "high", "elapsed_ms": 125}),
                "not-json",
            )) + "\n", encoding="utf-8")
            summary = summarize(read_records(path))
        self.assertEqual(summary["accepted_turn_counts"], {"gpt-5.6-sol": 1})
        self.assertEqual(summary["token_totals"], {"gpt-5.6-sol": 42})
        self.assertEqual(summary["latest_by_thread"]["thread-a"]["model"], "gpt-5.6-sol")
        self.assertEqual(summary["scorecards"][0]["latency_p50_ms"], 125)

    def test_summarize_reports_benchmark_product_passes(self):
        summary = summarize([
            {"event": "benchmark_result", "task_class": "routine", "model": "gpt-5.6-terra",
             "effort": "medium", "product_pass": True, "exact_final_response": True,
             "total_tokens": 100, "elapsed_ms": 250},
            {"event": "benchmark_result", "task_class": "routine", "model": "gpt-5.6-terra",
             "effort": "medium", "product_pass": False, "exact_final_response": True,
             "total_tokens": 50, "elapsed_ms": 150},
        ])
        card = summary["benchmark_scorecards"][0]
        self.assertEqual((card["runs"], card["product_passes"], card["pass_rate"]), (2, 1, 0.5))
        self.assertEqual((card["average_tokens"], card["average_elapsed_ms"]), (75, 200))

    def test_summarize_grades_verified_quality_and_token_efficiency(self):
        records = [
            {"event": "route_accepted", "thread_id": "t", "turn_id": "cheap", "model": "gpt-5.6-terra", "effort": "medium", "task_class": "routine", "task_bucket": "routine:base"},
            {"event": "turn_usage", "thread_id": "t", "turn_id": "cheap", "model": "gpt-5.6-terra", "usage": {"totalTokens": 100}},
            {"event": "quality_grade", "source": "explicit", "thread_id": "t", "source_turn_id": "cheap", "quality_score": 95, "verification": "passed"},
            {"event": "route_accepted", "thread_id": "t", "turn_id": "costly", "model": "gpt-5.6-sol", "effort": "high", "task_class": "routine", "task_bucket": "routine:base"},
            {"event": "turn_usage", "thread_id": "t", "turn_id": "costly", "model": "gpt-5.6-sol", "usage": {"totalTokens": 200}},
            {"event": "quality_grade", "source": "explicit", "thread_id": "t", "source_turn_id": "costly", "quality_score": 95, "verification": "passed"},
        ]
        grades = {item["turn_id"]: item for item in summarize(records)["graded_turns"]}
        self.assertEqual((grades["cheap"]["quality_grade"], grades["cheap"]["token_efficiency_grade"], grades["cheap"]["overall_grade"]), ("A", "A", "A"))
        self.assertEqual(grades["costly"]["token_efficiency_score"], 50.0)
        self.assertEqual(grades["costly"]["overall_score"], 86.0)


if __name__ == "__main__":
    unittest.main()
