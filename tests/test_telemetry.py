import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telemetry import aggregate_usage


class TelemetryTests(unittest.TestCase):
    def test_aggregate_usage_sums_each_upstream_completion(self):
        self.assertEqual(aggregate_usage([
            {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
            {"inputTokens": 20, "outputTokens": 3, "totalTokens": 23},
        ]), {"inputTokens": 30, "outputTokens": 5, "totalTokens": 35})


if __name__ == "__main__":
    unittest.main()
