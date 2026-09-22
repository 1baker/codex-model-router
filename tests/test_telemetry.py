import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telemetry import UsageTracker, aggregate_usage, thread_usage_from, usage_delta


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

    @staticmethod
    def snapshot(total: int, last: int, *, input_total: int | None = None) -> dict:
        return {"tokenUsage": {
            "total": {"inputTokens": total if input_total is None else input_total,
                      "totalTokens": total},
            "last": {"inputTokens": last, "totalTokens": last},
        }}

    def test_tracker_accepts_repeated_and_equal_valued_snapshots_once(self):
        tracker = UsageTracker()
        tracker.observe(self.snapshot(110, 10))
        tracker.observe(self.snapshot(110, 10))
        tracker.observe(self.snapshot(160, 50))
        usage, reason = tracker.outcome()
        self.assertIsNone(reason)
        self.assertEqual(usage["totalTokens"], 60)

    def test_tracker_missing_evidence_is_explicitly_unavailable(self):
        self.assertEqual(UsageTracker().outcome(), (None, "no_usage_evidence"))

    def test_tracker_malformed_or_partial_evidence_fails_closed(self):
        malformed = UsageTracker()
        malformed.observe({"tokenUsage": {"total": {"totalTokens": "12"},
                                           "last": {"totalTokens": 2}}})
        self.assertEqual(malformed.outcome(), (None, "malformed_usage_snapshot"))

    def test_tracker_decrease_above_baseline_permanently_invalidates(self):
        tracker = UsageTracker()
        tracker.observe(self.snapshot(110, 10))
        tracker.observe(self.snapshot(160, 50))
        tracker.observe(self.snapshot(130, 20))
        tracker.observe(self.snapshot(200, 70))
        self.assertEqual(tracker.outcome(), (None, "cumulative_counter_decreased"))

    def test_tracker_reset_followed_by_recovery_remains_unavailable(self):
        tracker = UsageTracker()
        tracker.observe(self.snapshot(110, 10))
        tracker.observe(self.snapshot(3, 3))
        tracker.observe(self.snapshot(20, 17))
        self.assertEqual(tracker.outcome(), (None, "cumulative_counter_decreased"))


if __name__ == "__main__":
    unittest.main()
