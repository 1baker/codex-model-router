"""Reviewed Codex protocol versions and fail-closed installer admission."""

import hashlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import install
from protocol_policy import SUPPORTED_CODEX_VERSIONS, classify


class ProtocolVersionTests(unittest.TestCase):
    def test_reviewed_manifest_matches_supported_versions_and_policy(self):
        manifest = json.loads((ROOT / "protocol_manifests/reviewed.json").read_text())
        self.assertEqual(manifest["schema"], "modellabs.reviewed-codex-protocol.v1")
        self.assertEqual(set(manifest["versions"]), SUPPORTED_CODEX_VERSIONS)
        methods = manifest["client_request_methods"]
        self.assertEqual(len(methods), 167)
        self.assertEqual(methods, sorted(set(methods)))
        self.assertEqual(classify("thread/unsubscribe"), "owner_mutation")
        self.assertEqual(classify("mcpServerStatus/list"), "read")
        self.assertEqual(classify("config/mcpServer/reload"), "unknown")
        self.assertEqual(classify("future/unreviewed"), "unknown")
        # Every deliberately denied schema method is reviewed as one set.
        unknown = [method for method in methods if classify(method) == "unknown"]
        self.assertEqual(len(unknown), 103)
        self.assertEqual(hashlib.sha256(("\n".join(unknown) + "\n").encode()).hexdigest(),
                         "e156ed815a5708ab33c2bba12510f7669d09acd5d9d921a5ea8da2bac79d3751")
        versions = manifest["versions"]
        self.assertEqual(versions["codex-cli 0.159.1"], versions["codex-cli 0.160.0"])
        self.assertNotEqual(versions["codex-cli 0.158.0"], versions["codex-cli 0.160.0"])

    def test_installer_accepts_only_reviewed_exact_versions(self):
        for version in sorted(SUPPORTED_CODEX_VERSIONS):
            with self.subTest(version=version), patch.object(
                install.subprocess, "run", return_value=SimpleNamespace(stdout=version + "\n")
            ):
                self.assertEqual(install.verify_upstream_protocol_version(Path("/unused")), version)
        for stdout in ("codex-cli 0.160.1\n", "codex-cli 0.161.0\n", "", " \n", None):
            with self.subTest(stdout=stdout), patch.object(
                install.subprocess, "run", return_value=SimpleNamespace(stdout=stdout)
            ):
                with self.assertRaises(RuntimeError):
                    install.verify_upstream_protocol_version(Path("/unused"))
        with patch.object(install.subprocess, "run", return_value=SimpleNamespace()):
            with self.assertRaises(RuntimeError):
                install.verify_upstream_protocol_version(Path("/unused"))


if __name__ == "__main__":
    unittest.main()
