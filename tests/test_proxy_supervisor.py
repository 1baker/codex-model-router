"""Conservative proxy-generation retirement defaults."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import proxy_supervisor


class ProxySupervisorTests(unittest.TestCase):
    def test_old_proxies_are_not_scanned_without_explicit_opt_in(self):
        for value in (None, "", "true", "0"):
            environment = {} if value is None else {
                "MODELLABS_RETIRE_DRAINED_PROXY_GENERATIONS": value}
            with self.subTest(value=value), \
                 patch.dict(os.environ, environment, clear=True), \
                 patch.object(proxy_supervisor.Path, "iterdir",
                              side_effect=AssertionError("old proxy scan attempted")):
                proxy_supervisor.retire_drained_proxy_generations()


if __name__ == "__main__":
    unittest.main()
