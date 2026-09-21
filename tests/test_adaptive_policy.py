import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_policy import adapt, record_explicit_grade
from modellabs import route

class AdaptiveTests(unittest.TestCase):
 def test_retries_recommend_escalation_only_in_shadow_mode(self):
  choice = route('Implement a small validated parser.')
  evidence = [{'event':'outcome_signal','source':'explicit','outcome':'retry',
               'thread_id':'thread','task_bucket':choice['task_bucket'],
               'recorded_at_ms':2_000_000_000_000}] * 4
  with patch.dict('os.environ', {'MODELLABS_ADAPTIVE_MODE': 'shadow'}):
   result = adapt(choice, thread_id='thread', records=evidence, now_ms=2_000_000_000_000)
  self.assertEqual(result['model'], 'gpt-5.6-terra')
  self.assertEqual(result['adaptive_recommendation']['model'], 'gpt-5.6-sol')

 def test_new_chat_does_not_reuse_another_threads_operator_outcome(self):
  choice = route('Implement a small validated parser.')
  evidence = [{'event':'outcome_signal','source':'explicit','outcome':'retry',
               'thread_id':'different-thread','task_bucket':choice['task_bucket'],
               'recorded_at_ms':2_000_000_000_000}] * 4
  result = adapt(choice, records=evidence, now_ms=2_000_000_000_000)
  self.assertNotIn('adaptive_recommendation', result)

 def test_grade_requires_verified_route_and_records_retry_below_release_quality(self):
  records = [{'event': 'route_accepted', 'thread_id': 'thread', 'turn_id': 'turn',
              'task_class': 'routine', 'model': 'gpt-5.6-terra', 'effort': 'medium',
              'task_bucket': 'routine:base'}]
  with patch('adaptive_policy.read_records', return_value=records), patch('adaptive_policy.record') as record:
   grade = record_explicit_grade('thread', 'turn', 85, 'passed')
  self.assertEqual(grade['outcome'], 'retry')
  self.assertEqual(record.call_count, 2)

if __name__ == '__main__': unittest.main()
