import os, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class AdaptiveTests(unittest.TestCase):
 def test_retries_escalate_but_consequential_does_not_downgrade(self):
  with tempfile.TemporaryDirectory() as d:
   os.environ['MODELLABS_METRICS_PATH'] = str(Path(d)/'metrics.jsonl')
   import importlib, telemetry, adaptive_policy
   importlib.reload(telemetry); importlib.reload(adaptive_policy)
   for _ in range(2): telemetry.record('outcome_signal', task_class='routine', outcome='retry')
   self.assertEqual(adaptive_policy.adapt({'class':'routine','model':'gpt-5.6-terra','effort':'medium'})['model'], 'gpt-5.6-sol')
   self.assertEqual(adaptive_policy.adapt({'class':'consequential','model':'gpt-6-astra','effort':'xhigh'}), {'class':'consequential','model':'gpt-6-astra','effort':'xhigh'})

if __name__ == '__main__': unittest.main()
