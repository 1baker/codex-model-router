import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_policy import adapt
from modellabs import route

class AdaptiveTests(unittest.TestCase):
 def test_retries_recommend_escalation_only_in_shadow_mode(self):
  choice = route('Implement a small validated parser.')
  evidence = [{'event':'outcome_signal','source':'explicit','outcome':'retry',
               'task_bucket':choice['task_bucket'],'recorded_at_ms':2_000_000_000_000}] * 4
  result = adapt(choice, records=evidence, now_ms=2_000_000_000_000)
  self.assertEqual(result['model'], 'gpt-5.6-terra')
  self.assertEqual(result['adaptive_recommendation']['model'], 'gpt-5.6-sol')

if __name__ == '__main__': unittest.main()
