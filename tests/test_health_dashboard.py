import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from health_dashboard import read_records, summarize


class HealthDashboardTests(unittest.TestCase):
    def test_summarize_uses_metadata_without_prompt_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            path.write_text("\n".join((
                json.dumps({"event": "route_accepted", "thread_id": "thread-a", "model": "gpt-5.6-sol", "effort": "high", "recorded_at_ms": 1}),
                json.dumps({"event": "response_usage", "thread_id": "thread-a", "model": "gpt-5.6-sol", "effort": "high", "recorded_at_ms": 2, "usage": {"totalTokens": 42}}),
                "not-json",
            )) + "\n", encoding="utf-8")
            summary = summarize(read_records(path))
        self.assertEqual(summary["route_counts"], {"gpt-5.6-sol": 2})
        self.assertEqual(summary["token_totals"], {"gpt-5.6-sol": 42})
        self.assertEqual(summary["latest_by_thread"]["thread-a"]["event"], "response_usage")


if __name__ == "__main__":
    unittest.main()
