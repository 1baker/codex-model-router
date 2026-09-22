import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telemetry import aggregate_usage, thread_usage_from, usage_delta


class TelemetryTests(unittest.TestCase):
    def test_thread_usage_reads_live_last_sample(self):
        params = {"tokenUsage": {"last": {"inputTokens": 10, "totalTokens": 12},
                                  "total": {"inputTokens": 30, "totalTokens": 35}}}
        self.assertEqual(thread_usage_from(params), {"inputTokens": 10, "totalTokens": 12})

    def test_cumulative_delta_ignores_repeated_snapshots(self):
        baseline = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
        final = {"inputTokens": 30, "outputTokens": 5, "totalTokens": 35}
        self.assertEqual(usage_delta(baseline, final)["totalTokens"], 35)
        self.assertIsNone(usage_delta(final, baseline))

    def test_aggregate_usage_sums_each_upstream_completion(self):
        self.assertEqual(aggregate_usage([
            {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
            {"inputTokens": 20, "outputTokens": 3, "totalTokens": 23},
        ]), {"inputTokens": 30, "outputTokens": 5, "totalTokens": 35})


if __name__ == "__main__":
    unittest.main()
