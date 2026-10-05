import os
import json
import hashlib
import sys
import io
import ast
import shutil
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modellabs import route
import modellabs
import install as installer
from install import (SHELL_PATH_START, discover_real_codex,
                     ensure_managed_route_precedence, upsert_toml, write_wrappers)
from paths import real_codex_binary
import paths
from turn_proxy import route_request, selection_is_listed
import turn_proxy as turn_proxy_module
from adaptive_policy import adapt, feedback, record_outcome
import adaptive_policy
from health_dashboard import summarize
import model_host_launcher
import host_control
from model_host_launcher import (_command_index, _exec_prompt, _explicit_setting,
                                 _exec_subcommand, _create_launch_ticket, _routed_exec_command,
                                 _routed_interactive_args, run_routed_exec)
import thread_owner


class RoutingTests(unittest.TestCase):
    def test_explicit_model_ids_do_not_need_a_router_allowlist(self):
        requests = {
            "use gpt-6.1-sol": "gpt-6.1-sol",
            "Use GPT-6.1 Sol": "gpt-6.1-sol",
            "switch to gpt 6.1 sol": "gpt-6.1-sol",
            "route to the GPT-5.5": "gpt-5.5",
            "use gpt-5.5.": "gpt-5.5",
            "run gpt 5.5": "gpt-5.5",
            "use gpt-7-sol-2026-10-04": "gpt-7-sol-2026-10-04",
            "use gpt 6.1 for review": "gpt-6.1",
            "use Sol": "gpt-5.6-sol",
            "use Luna": "gpt-5.6-luna",
            "use Terra": "gpt-5.6-terra",
            "use Astra": "gpt-6-astra",
        }
        for prompt, model in requests.items():
            with self.subTest(prompt=prompt):
                choice = route(prompt)
                self.assertEqual(choice["model"], model)
                self.assertTrue(choice["explicit_model"])

    def test_model_mentions_without_a_selection_verb_are_not_pins(self):
        self.assertFalse(route("Explain gpt-6.1-sol.")["explicit_model"])
        self.assertFalse(route("Write a gpt-like answer.")["explicit_model"])

    def test_unknown_explicit_model_is_not_silently_replaced(self):
        choice = route("use gpt-99-sol with reasoning effort low")
        self.assertEqual(choice["model"], "gpt-99-sol")
        self.assertTrue(choice["explicit_model"])
        self.assertFalse(selection_is_listed({"data": []}, choice))

    def test_catalog_selection_rejects_hidden_or_duplicate_visible_ids(self):
        entry = {"id": "gpt-6.1-sol", "supportedReasoningEfforts": [
            {"reasoningEffort": "low"}]}
        choice = {"model": entry["id"], "effort": "low"}
        for data in ([], [{**entry, "hidden": True}], [entry, dict(entry)],
                     [None], [{**entry, "supportedReasoningEfforts": None}]):
            with self.subTest(data=data):
                self.assertFalse(selection_is_listed({"data": data}, choice))
        self.assertTrue(selection_is_listed({"data": [entry]}, choice))

    def test_every_generation_dependency_changes_the_proxy_revision(self):
        payloads = {name: f"VALUE = {name!r}\n".encode()
                    for name in installer.PROXY_GENERATION_FILES}
        baseline = installer._proxy_revision(payloads)
        for name in installer.PROXY_GENERATION_FILES:
            with self.subTest(name=name):
                changed = {**payloads, name: payloads[name] + b"# changed\n"}
                self.assertNotEqual(installer._proxy_revision(changed), baseline)

    def test_runtime_and_installer_hash_the_same_ordered_generation_files(self):
        self.assertEqual(paths.PROXY_REVISION_FILES, installer.PROXY_GENERATION_FILES)
        self.assertEqual(paths.PROXY_REVISION_FILES, installer.PROXY_REVISION_FILES)
        self.assertEqual(paths.PROXY_REVISION_FILES, turn_proxy_module.PROXY_GENERATION_FILES)

    def test_source_generation_matches_installer_and_launcher_revision_and_port(self):
        source = paths.RUNTIME_SOURCE
        payloads = {name: (source / name).read_bytes()
                    for name in installer.PROXY_GENERATION_FILES}
        revision = paths.proxy_revision()
        self.assertEqual(revision, installer._proxy_revision(payloads))
        self.assertEqual(revision, model_host_launcher.PROXY_REVISION)
        self.assertEqual(paths.proxy_port(revision), installer.installed_proxy_port(source))
        self.assertEqual(model_host_launcher.PROXY_PORT, installer.installed_proxy_port(source))

    def test_every_bundled_dependency_changes_runtime_and_installer_identity(self):
        payloads = {name: f"VALUE = {name!r}\n".encode()
                    for name in installer.PROXY_GENERATION_FILES}
        with patch.object(paths.Path, "read_bytes", lambda path: payloads[path.name]):
            baseline = paths.proxy_revision()
            self.assertEqual(baseline, installer._proxy_revision(payloads))
            for name in installer.PROXY_GENERATION_FILES:
                with self.subTest(name=name):
                    original = payloads[name]
                    payloads[name] += b"# changed\n"
                    self.assertNotEqual(paths.proxy_revision(), baseline)
                    self.assertEqual(paths.proxy_revision(), installer._proxy_revision(payloads))
                    payloads[name] = original

    def test_runtime_hashes_its_bundle_directory_not_the_mutable_install_home(self):
        import importlib.util
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            payloads = {name: (paths.RUNTIME_SOURCE / name).read_bytes()
                        for name in installer.PROXY_GENERATION_FILES}
            payloads["smoke_bench.py"] += b"\n# bundle-only dependency change\n"
            for name, content in payloads.items():
                (bundle / name).write_bytes(content)
            spec = importlib.util.spec_from_file_location("isolated_bundle_paths", bundle / "paths.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.RUNTIME_SOURCE, bundle)
            self.assertEqual(module.proxy_revision(), installer._proxy_revision(payloads))
            self.assertEqual(module.proxy_port(module.proxy_revision()), installer.installed_proxy_port(bundle))
            self.assertNotEqual(module.proxy_revision(), paths.proxy_revision())

    def test_proxy_generation_contains_transitive_local_imports(self):
        root = Path(__file__).resolve().parents[1]
        local_modules = {path.stem for path in root.glob("*.py")}
        bundled = {Path(name).stem for name in installer.PROXY_GENERATION_FILES}
        pending = ["turn_proxy"]
        visited = set()
        while pending:
            module = pending.pop()
            if module in visited:
                continue
            visited.add(module)
            tree = ast.parse((root / f"{module}.py").read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            pending.extend(sorted((imported & local_modules) - visited))
        self.assertEqual(visited, bundled)
        self.assertEqual(set(turn_proxy_module.PROXY_GENERATION_FILES),
                         set(installer.PROXY_GENERATION_FILES))

    def test_proxy_source_only_pins_old_and_new_generation_bundles(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, home = root / "source", root / "installed"
            source.mkdir()
            home.mkdir()
            manifest = {"schema": "modellabs.owned_files.v1", "files": {}}
            old_proxy = b"VALUE = 'old proxy'\n"
            new_proxy = b"VALUE = 'new proxy'\n"
            for name in installer.PROXY_GENERATION_FILES:
                content = old_proxy if name == "turn_proxy.py" else f"VALUE = {name!r}\n".encode()
                target = home / name
                target.write_bytes(content)
                (source / name).write_bytes(
                    new_proxy if name == "turn_proxy.py" else content)
                manifest["files"][str(target)] = hashlib.sha256(content).hexdigest()
            manifest_path = home / installer.OWNED_MANIFEST
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(installer, "SOURCE", source), \
                 patch.object(installer, "_validate_proxy_bundle_imports"):
                result = installer.install_proxy_source_only(
                    home, hashlib.sha256(old_proxy).hexdigest())
            self.assertEqual((home / "turn_proxy.py").read_bytes(), new_proxy)
            self.assertEqual(Path(result["backup"]).read_bytes(), old_proxy)
            self.assertEqual(result["services_restarted"], "false")
            self.assertEqual(result["changed_files"], ["turn_proxy.py"])
            self.assertNotEqual(result["old_revision"], result["new_revision"])
            for generation, expected in ((result["old_revision"], old_proxy),
                                         (result["new_revision"], new_proxy)):
                entry = home / "proxy-generations" / str(generation) / "turn_proxy.py"
                self.assertEqual(entry.read_bytes(), expected)
                self.assertEqual(entry.stat().st_mode & 0o777, 0o600)
            selected = turn_proxy_module.pinned_proxy_generation(
                home, {"MODELLABS_PROXY_PORT": str(result["old_port"])})
            self.assertEqual(selected[0].read_bytes(), old_proxy)
            self.assertEqual(selected[1][turn_proxy_module.PROXY_GENERATION_ENTRY_ENV], "1")
            self.assertEqual(selected[1]["MODELLABS_HOME"], str(home))
            self.assertEqual(set(installer.PROXY_GENERATION_FILES),
                             set(turn_proxy_module.PROXY_GENERATION_FILES))
            with patch.object(turn_proxy_module.os, "execve", side_effect=SystemExit) as execute:
                with self.assertRaises(SystemExit):
                    turn_proxy_module.dispatch_pinned_proxy_generation(
                        home, {"MODELLABS_PROXY_PORT": str(result["new_port"])})
            self.assertEqual(Path(execute.call_args.args[1][1]).read_bytes(), new_proxy)
            self.assertEqual(execute.call_args.args[2]["MODELLABS_HOME"], str(home))
            selected[0].write_bytes(b"tampered\n")
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                turn_proxy_module.pinned_proxy_generation(
                    home, {"MODELLABS_PROXY_PORT": str(result["old_port"])})

    def test_proxy_source_only_publishes_dependency_only_generation_change(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, home = root / "source", root / "installed"
            source.mkdir()
            home.mkdir()
            manifest = {"schema": "modellabs.owned_files.v1", "files": {}}
            old_policy = b"VALUE = 'old policy'\n"
            new_policy = b"VALUE = 'new policy'\n"
            proxy = b"VALUE = 'unchanged proxy'\n"
            for name in installer.PROXY_GENERATION_FILES:
                content = (old_policy if name == "adaptive_policy.py" else
                           proxy if name == "turn_proxy.py" else
                           f"VALUE = {name!r}\n".encode())
                target = home / name
                target.write_bytes(content)
                (source / name).write_bytes(
                    new_policy if name == "adaptive_policy.py" else content)
                manifest["files"][str(target)] = hashlib.sha256(content).hexdigest()
            manifest_path = home / installer.OWNED_MANIFEST
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(installer, "SOURCE", source), \
                 patch.object(installer, "_validate_proxy_bundle_imports"):
                result = installer.install_proxy_source_only(
                    home, hashlib.sha256(proxy).hexdigest())
            self.assertEqual(result["changed_files"], ["adaptive_policy.py"])
            self.assertEqual((home / "adaptive_policy.py").read_bytes(), new_policy)
            self.assertEqual((home / "turn_proxy.py").read_bytes(), proxy)
            self.assertEqual(Path(result["backups"]["adaptive_policy.py"]).read_bytes(),
                             old_policy)
            self.assertNotEqual(result["old_revision"], result["new_revision"])
            new_bundle = home / "proxy-generations" / str(result["new_revision"])
            self.assertEqual((new_bundle / "adaptive_policy.py").read_bytes(), new_policy)
            self.assertEqual((new_bundle / "turn_proxy.py").read_bytes(), proxy)

    def test_proxy_import_preflight_rejects_missing_symbol(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            home = Path(directory)
            interpreter = home / "venv/bin/python"
            interpreter.parent.mkdir(parents=True)
            shutil.copy2(sys.executable, interpreter)
            with self.assertRaisesRegex(RuntimeError, "cannot import name 'missing'"):
                installer._validate_proxy_bundle_imports(home, {
                    "turn_proxy.py": b"from modellabs import missing\n",
                    "modellabs.py": b"VALUE = 1\n",
                })

    def test_proxy_import_preflight_rejects_before_publication(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, home = root / "source", root / "installed"
            source.mkdir()
            home.mkdir()
            manifest = {"schema": "modellabs.owned_files.v1", "files": {}}
            for name in installer.PROXY_GENERATION_FILES:
                content = f"VALUE = {name!r}\n".encode()
                (home / name).write_bytes(content)
                (source / name).write_bytes(
                    b"from modellabs import missing\n" if name == "turn_proxy.py" else content)
                manifest["files"][str(home / name)] = hashlib.sha256(content).hexdigest()
            manifest_path = home / installer.OWNED_MANIFEST
            original_manifest = json.dumps(manifest).encode()
            manifest_path.write_bytes(original_manifest)
            original_proxy = (home / "turn_proxy.py").read_bytes()
            with patch.object(installer, "SOURCE", source), \
                 patch.object(installer, "_validate_proxy_bundle_imports",
                              side_effect=RuntimeError("import failed")):
                with self.assertRaisesRegex(RuntimeError, "import failed"):
                    installer.install_proxy_source_only(
                        home, hashlib.sha256(original_proxy).hexdigest())
            self.assertEqual((home / "turn_proxy.py").read_bytes(), original_proxy)
            self.assertEqual(manifest_path.read_bytes(), original_manifest)
            self.assertFalse((home / "proxy-generations").exists())

    def test_supervisor_source_only_requires_exact_owned_baseline(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, home = root / "source", root / "installed"
            source.mkdir()
            home.mkdir()
            old, new = b"VALUE = 'old'\n", b"VALUE = 'new'\n"
            digest = lambda value: hashlib.sha256(value).hexdigest()
            (source / "proxy_supervisor.py").write_bytes(new)
            target = home / "proxy_supervisor.py"
            target.write_bytes(old)
            proxy = home / "turn_proxy.py"
            proxy.write_bytes(b"live proxy untouched\n")
            manifest_path = home / installer.OWNED_MANIFEST
            manifest = {"schema": "modellabs.owned_files.v1", "files": {
                str(target): digest(old), str(proxy): digest(proxy.read_bytes())}}
            manifest_path.write_text(json.dumps(manifest))
            with patch.object(installer, "SOURCE", source):
                with self.assertRaisesRegex(RuntimeError, "inspected baseline"):
                    installer.install_supervisor_source_only(home, "0" * 64)
                manifest["files"][str(target)] = "0" * 64
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, "owned-manifest parity"):
                    installer.install_supervisor_source_only(home, digest(old))
                manifest["files"][str(target)] = digest(old)
                manifest_path.write_text(json.dumps(manifest))
                result = installer.install_supervisor_source_only(home, digest(old))
            self.assertEqual(target.read_bytes(), new)
            self.assertEqual(proxy.read_bytes(), b"live proxy untouched\n")
            self.assertEqual(Path(result["backup"]).read_bytes(), old)
            self.assertEqual(Path(result["backup"]).stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(manifest_path.read_text())["files"][str(target)], digest(new))
            self.assertEqual(result["running_process_changed"], "false")
            self.assertEqual(result["services_restarted"], "false")

    def test_learning_only_install_preserves_proxy_and_requires_exact_baseline(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, home = root / "source", root / "installed"
            source.mkdir()
            home.mkdir()
            old, new = b"VALUE = 'old'\n", b"VALUE = 'new'\n"
            (source / "outcome_model.py").write_bytes(new)
            target = home / "outcome_model.py"
            target.write_bytes(old)
            digest = lambda value: hashlib.sha256(value).hexdigest()
            generation_contents = {}
            for name in installer.PROXY_GENERATION_FILES:
                if name == "outcome_model.py":
                    generation_contents[name] = old
                    continue
                content = (b"proxy stays unchanged\n" if name == "turn_proxy.py"
                           else f"VALUE = {name!r}\n".encode())
                (home / name).write_bytes(content)
                generation_contents[name] = content
            other = home / "turn_proxy.py"
            manifest_path = home / installer.OWNED_MANIFEST
            manifest = {"schema": "modellabs.owned_files.v1", "files": {
                str(home / name): digest(content)
                for name, content in generation_contents.items()}}
            manifest_path.write_text(json.dumps(manifest))
            with patch.object(installer, "SOURCE", source):
                with self.assertRaisesRegex(RuntimeError, "inspected baseline"):
                    installer.install_learning_only(home, "0" * 64)
                self.assertEqual(target.read_bytes(), old)
                result = installer.install_learning_only(home, digest(old))
            self.assertEqual(target.read_bytes(), new)
            self.assertEqual(other.read_bytes(), b"proxy stays unchanged\n")
            self.assertEqual(Path(result["backup"]).read_bytes(), old)
            self.assertEqual(Path(result["backup"]).stat().st_mode & 0o777, 0o600)
            updated = json.loads(manifest_path.read_text())
            self.assertEqual(updated["files"][str(target)], digest(new))
            self.assertEqual(updated["files"][str(other)], digest(other.read_bytes()))
            self.assertEqual(result["services_restarted"], "false")
            self.assertEqual(result["running_process_changed"], "false")
            self.assertNotEqual(result["old_revision"], result["new_revision"])
            self.assertEqual(
                (home / "proxy-generations" / result["new_revision"] /
                 "outcome_model.py").read_bytes(), new)

    def test_learning_only_pair_install_is_guarded_and_leaves_proxy_running(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, home = root / "source", root / "installed"
            source.mkdir()
            home.mkdir()
            digest = lambda value: hashlib.sha256(value).hexdigest()
            old_learner, old_smoke = b"VALUE = 'learner old'\n", b"VALUE = 'smoke old'\n"
            new_learner, new_smoke = b"VALUE = 'learner new'\n", b"VALUE = 'smoke new'\n"
            for name, old, new in (("outcome_model.py", old_learner, new_learner),
                                   ("smoke_bench.py", old_smoke, new_smoke)):
                (home / name).write_bytes(old)
                (source / name).write_bytes(new)
            generation_contents = {"outcome_model.py": old_learner,
                                   "smoke_bench.py": old_smoke}
            for name in installer.PROXY_GENERATION_FILES:
                if name in {"outcome_model.py", "smoke_bench.py"}:
                    continue
                content = (b"untouched\n" if name == "turn_proxy.py"
                           else f"VALUE = {name!r}\n".encode())
                (home / name).write_bytes(content)
                generation_contents[name] = content
            proxy = home / "turn_proxy.py"
            manifest_path = home / installer.OWNED_MANIFEST
            manifest_path.write_text(json.dumps({"schema": "modellabs.owned_files.v1",
                                                "files": {
                                                    **{str(home / name): digest(content)
                                                       for name, content in generation_contents.items()},
                                                    str(home / "smoke_bench.py"): digest(old_smoke)}}))
            with patch.object(installer, "SOURCE", source):
                with self.assertRaisesRegex(RuntimeError, "inspected baseline"):
                    installer.install_learning_only(home, digest(old_learner), "0" * 64)
                self.assertEqual((home / "outcome_model.py").read_bytes(), old_learner)
                self.assertEqual((home / "smoke_bench.py").read_bytes(), old_smoke)
                result = installer.install_learning_only(home, digest(old_learner), digest(old_smoke))
            self.assertEqual((home / "outcome_model.py").read_bytes(), new_learner)
            self.assertEqual((home / "smoke_bench.py").read_bytes(), new_smoke)
            self.assertEqual(proxy.read_bytes(), b"untouched\n")
            self.assertEqual(Path(result["smoke_backup"]).read_bytes(), old_smoke)
            self.assertEqual(result["services_restarted"], "false")
            self.assertNotEqual(result["old_revision"], result["new_revision"])

    def test_learning_only_publishes_benchmark_manifest_in_same_guarded_transaction(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, home = root / "source", root / "installed"
            (source / "benchmarks").mkdir(parents=True)
            (home / "benchmarks").mkdir(parents=True)
            digest = lambda value: hashlib.sha256(value).hexdigest()
            learner = b"VALUE = 'same learner'\n"
            old_manifest = b'{"schema":"modellabs.smoke.v1","scenarios":[]}\n'
            new_manifest = b'{"schema":"modellabs.smoke.v1","scenarios":[{}]}\n'
            (source / "outcome_model.py").write_bytes(learner)
            (source / "benchmarks/smoke.json").write_bytes(new_manifest)
            (home / "outcome_model.py").write_bytes(learner)
            (home / "benchmarks/smoke.json").write_bytes(old_manifest)
            manifest_path = home / installer.OWNED_MANIFEST
            manifest_path.write_text(json.dumps({"schema": "modellabs.owned_files.v1",
                "files": {str(home / "outcome_model.py"): digest(learner),
                          str(home / "benchmarks/smoke.json"): digest(old_manifest)}}))
            with patch.object(installer, "SOURCE", source):
                result = installer.install_learning_only(
                    home, digest(learner), expected_benchmark_sha256=digest(old_manifest))
            self.assertEqual((home / "benchmarks/smoke.json").read_bytes(), new_manifest)
            self.assertEqual(Path(result["benchmark_backup"]).read_bytes(), old_manifest)
            self.assertEqual(result["benchmark_sha256"], digest(new_manifest))
            self.assertNotIn("new_revision", result)

    def test_followup_feedback_is_short_direct_and_never_a_grade(self):
        self.assertEqual(feedback("okay that is now working"), "verified")
        self.assertEqual(feedback("this is still not working"), "retry")
        self.assertEqual(feedback("tests passed"), "verified")
        self.assertEqual(feedback("tests failed"), "retry")
        self.assertIsNone(feedback("Please continue; the prior version still needs citation checks."))
        self.assertIsNone(feedback("Review a file that says 'fixed' in its title."))
        self.assertIsNone(feedback("This is a quoted failure report.\nPlease analyze it."))

    def test_grade_updates_local_learner_without_reversing_grade_on_failure(self):
        arguments = ["modellabs", "grade", "--thread-id", "thread", "--turn-id", "turn",
                     "--quality-score", "95", "--verification", "passed"]
        with patch.object(sys, "argv", arguments), \
                patch("adaptive_policy.record_explicit_grade", return_value={"quality_score": 95}) as grade, \
                patch("outcome_model.sync_codex_grades", return_value={"graded": 1}) as sync, \
                patch("outcome_model.train_codex_model", return_value={"status": "insufficient_comparable_outcomes"}), \
                redirect_stdout(io.StringIO()) as output:
            modellabs.main()
        self.assertEqual(grade.call_count, 1)
        self.assertEqual(sync.call_count, 1)
        self.assertEqual(json.loads(output.getvalue())["learning_sync"]["status"], "ok")
        with patch.object(sys, "argv", arguments), \
                patch("adaptive_policy.record_explicit_grade", return_value={"quality_score": 95}) as grade, \
                patch("outcome_model.sync_codex_grades", side_effect=OSError("private store unavailable")), \
                redirect_stdout(io.StringIO()) as output:
            modellabs.main()
        self.assertEqual(grade.call_count, 1)
        self.assertEqual(json.loads(output.getvalue())["learning_sync"]["status"], "failed")

    def test_configure_codex_removes_only_modellabs_prompt_hook(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            codex_home, home = root / "codex", root / "model-selector"
            codex_home.mkdir()
            command = f"{home / 'venv/bin/python'} {home / 'prompt_hook.py'}"
            hooks = {"hooks": {"UserPromptSubmit": [{"hooks": [
                {"command": "/bin/other-hook", "type": "command"},
                {"command": command, "type": "command"},
            ]}]}}
            (codex_home / "hooks.json").write_text(json.dumps(hooks), encoding="utf-8")
            installer.configure_codex(codex_home, home)
            result = json.loads((codex_home / "hooks.json").read_text(encoding="utf-8"))
            self.assertEqual(result["hooks"]["UserPromptSubmit"][0]["hooks"],
                             [{"command": "/bin/other-hook", "type": "command"}])
            installer.configure_codex(codex_home, home)
            self.assertEqual(result, json.loads((codex_home / "hooks.json").read_text(encoding="utf-8")))

    def test_bare_codex_starts_tui_without_reading_prompt(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory, \
             patch.object(model_host_launcher, "ROOT", Path(directory)), \
             patch.object(sys, "argv", ["codex"]), \
             patch.object(model_host_launcher, "_read_token", return_value="token"), \
             patch.object(model_host_launcher, "ensure_proxy"), \
             patch.object(model_host_launcher, "ensure_proxy_supervisor"), \
             patch.object(model_host_launcher, "real_codex_binary", return_value=Path("/real/codex")), \
             patch.object(model_host_launcher, "_route_choice", side_effect=AssertionError("prompt read")), \
             patch.object(model_host_launcher.os, "execve") as launch:
            model_host_launcher.main()
            tickets = list((Path(directory) / "launch-tickets").glob("*.json"))
            payloads = [json.loads(path.read_text()) for path in tickets]
            ticket = next(item for item in payloads if item["schema"] == "modellabs.launch_ticket.v2")
            self.assertIsNone(ticket["model"])
            self.assertIsNone(ticket["effort"])
            self.assertIsNone(ticket["servers"])
        binary, args, environment = launch.call_args.args
        self.assertEqual(binary, Path("/real/codex"))
        self.assertEqual(args, ["/real/codex", "--remote", model_host_launcher.PROXY_URL,
                                "--remote-auth-token-env", "MODEL_SELECTOR_HOST_TOKEN"])
        self.assertRegex(environment["MODEL_SELECTOR_HOST_TOKEN"], r"^token\.launch-[0-9a-f]{32}$")

    def test_managed_resume_imports_a_closed_standalone_thread_before_launch(self):
        from tempfile import TemporaryDirectory
        thread_id = "00000000-0000-4000-8000-000000000009"
        with TemporaryDirectory() as directory, \
             patch.object(sys, "argv", ["codex", "resume", thread_id]), \
             patch.object(model_host_launcher, "authority_path_for",
                          return_value=Path(directory) / "missing-authority.json"), \
             patch.object(model_host_launcher, "import_closed_thread") as handoff, \
             patch.object(model_host_launcher, "_read_token", return_value="token"), \
             patch.object(model_host_launcher, "ensure_proxy"), \
             patch.object(model_host_launcher, "ensure_proxy_supervisor"), \
             patch.object(model_host_launcher, "real_codex_binary",
                          return_value=Path("/real/codex")), \
             patch.object(model_host_launcher.os, "execve") as launch:
            model_host_launcher.main()
        handoff.assert_called_once_with(thread_id)
        self.assertEqual(launch.call_args.args[1][-2:], ["resume", thread_id])

    def test_installer_ships_legacy_thread_handoff(self):
        self.assertIn("legacy_thread_handoff.py", installer.PYTHON_FILES)

    def test_selects_the_lowest_sufficient_effort(self):
        cases = [
            ("Format these three values as CSV.", "gpt-6-luna", "low"),
            ("Implement a small validated parser.", "gpt-6-sol", "medium"),
            ("Investigate an intermittent race condition and fix it.", "gpt-6-sol", "high"),
            ("Design a production cross-system security architecture.", "gpt-6-astra", "xhigh"),
            ("Perform an adversarial audit of this production migration.", "gpt-6-astra", "max"),
            ("Create an ultra exhaustive independent review of this architecture.", "gpt-6-astra", "ultra"),
        ]
        for prompt, model, effort in cases:
            with self.subTest(prompt=prompt):
                choice = route(prompt)
                self.assertEqual((choice["model"], choice["effort"]), (model, effort))
                self.assertEqual(choice["intelligence_slider"], effort)

    def test_short_conversational_requests_use_luna_low(self):
        for prompt in [
            "Okay, so now how does this work?",
            "How could we improve this then?",
            "Routing should happen automatically.",
            "Can you clarify that?",
        ]:
            with self.subTest(prompt=prompt):
                choice = route(prompt)
                self.assertEqual((choice["class"], choice["model"], choice["effort"]),
                                 ("simple", "gpt-6-luna", "low"))

    def test_go_ahead_variants_keep_consequential_context(self):
        context = "Migrate the production database."
        for prompt in ("proceed", "ship it", "yes please", "ok, go ahead", "do that"):
            with self.subTest(prompt=prompt):
                choice = route(prompt, context_prompt=context)
                self.assertEqual((choice["class"], choice["model"], choice["effort"]),
                                 ("consequential", "gpt-6-astra", "xhigh"))
                self.assertTrue(choice["context_inherited"])

    def test_direct_continuation_keeps_prior_effort_even_at_same_class(self):
        choice = route("ok go", context_prompt="Do an exhaustive independent review of the README.")
        self.assertEqual((choice["class"], choice["effort"]), ("routine", "ultra"))
        self.assertTrue(choice["context_inherited"])
        explicit = route("ok go with reasoning effort low",
                         context_prompt="Do an exhaustive independent review of the README.")
        self.assertEqual(explicit["effort"], "low")

    def test_contextual_followups_preserve_prior_task_risk(self):
        difficult = route(
            "How could we improve this then?",
            context_prompt="Investigate the intermittent race condition and find the root cause.",
        )
        self.assertEqual((difficult["class"], difficult["model"], difficult["effort"]),
                         ("difficult", "gpt-6-sol", "high"))
        self.assertTrue(difficult["context_inherited"])

        consequential = route(
            "Ok go",
            context_prompt="Design a production cross-system security architecture.",
        )
        self.assertEqual((consequential["class"], consequential["model"], consequential["effort"]),
                         ("consequential", "gpt-6-astra", "xhigh"))
        self.assertTrue(consequential["context_inherited"])

        simple = route(
            "Ok go",
            context_prompt="Format these three values as CSV.",
        )
        self.assertEqual((simple["class"], simple["model"], simple["effort"]),
                         ("simple", "gpt-6-luna", "low"))
        self.assertTrue(simple["context_inherited"])

    def test_contextual_followup_keeps_prior_tool_shortlist(self):
        choice = route(
            "How does this work?",
            context_prompt="Review the website and produce a PDF preview.",
        )
        self.assertIn("agentBrowser", choice["servers"])
        self.assertIn("previews", choice["servers"])

    def test_explicit_effort_wins(self):
        choice = route("Format these values as CSV with reasoning effort ultra.")
        self.assertEqual(choice["effort"], "ultra")
        self.assertTrue(choice["explicit_effort"])

    def test_new_models_and_intelligence_choices(self):
        for prompt, model, effort in [
            ("Use GPT-6 Sol with reasoning effort ultra to review this.", "gpt-6-sol", "ultra"),
            ("Use GPT-6 Luna with intelligence none to format this.", "gpt-6-luna", "none"),
            ("Use Sol with reasoning effort max to investigate this.", "gpt-5.6-sol", "max"),
            ("Use GPT-5.6 Sol with reasoning effort high to investigate this.", "gpt-5.6-sol", "high"),
        ]:
            with self.subTest(prompt=prompt):
                choice = route(prompt)
                self.assertEqual((choice["model"], choice["effort"]), (model, effort))
                self.assertTrue(choice["explicit_model"])
                self.assertTrue(choice["explicit_effort"])

    def test_explicit_model_is_never_adapted(self):
        choice = route("Use Sol to implement a small validated parser.")
        self.assertTrue(choice["explicit_model"])
        self.assertEqual(adapt(choice, records=[])["adaptive_reason"], "explicit_user_override")

    def test_adaptation_is_shadow_by_default(self):
        choice = route("Implement a small validated parser.")
        records = [{"event": "outcome_signal", "outcome": "retry", "source": "explicit",
                    "thread_id": "thread", "task_bucket": choice["task_bucket"],
                    "recorded_at_ms": 2_000_000_000_000}] * 4
        with patch.dict("os.environ", {"MODELLABS_ADAPTIVE_MODE": "shadow"}):
            result = adapt(choice, thread_id="thread", records=records, now_ms=2_000_000_000_000)
        self.assertEqual(result["model"], choice["model"])
        self.assertEqual(result["adaptive_reason"], "shadow_retry_escalation")

    def test_managed_policy_is_digest_bound_and_pilot_is_ten_percent(self):
        from tempfile import TemporaryDirectory
        choice = route("Build a complete tested command line parser from the supplied fixture.")
        evidence = {"comparison_id": "a" * 64, "task_class": choice["class"],
                    "baseline_arm": [choice["model"], choice["effort"]],
                    "recommended_arm": ["gpt-6-luna", "low"],
                    "prospective_products": 8, "prospective_wins": 8,
                    "prospective_losses": 0, "checkpoint_sha256": "b" * 64}
        value = {**evidence, "evidence_sha256": hashlib.sha256(json.dumps(
            evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                 "validated_at_ms": 1}
        artifact = {"schema": "modellabs.managed_routing_policy.v2",
                    "checkpoints": {}, "validations": {"a" * 64: value}}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            recommendation = adaptive_policy.managed_policy_recommendation(choice, path)
            self.assertEqual(recommendation[0], {"model": "gpt-6-luna", "effort": "low"})
            path.write_text(json.dumps({**artifact, "validations": {"a" * 64: {
                **value, "prospective_losses": 1}}}), encoding="utf-8")
            self.assertIsNone(adaptive_policy.managed_policy_recommendation(choice, path))
        managed = ({"model": "gpt-6-luna", "effort": "low"},
                   {"source": "managed_randomized_policy", "provisional": False})
        selected = {**choice, "prompt_sha256": "0" * 64}
        with patch("adaptive_policy.managed_policy_recommendation", return_value=managed), \
                patch.dict("os.environ", {"MODELLABS_ADAPTIVE_MODE": "pilot"}):
            result = adapt(selected, records=[])
        self.assertEqual((result["model"], result["effort"], result["adaptive_reason"]),
                         ("gpt-6-luna", "low", "managed_policy_pilot"))
        held_out = {**choice, "prompt_sha256": "f" * 64}
        with patch("adaptive_policy.managed_policy_recommendation", return_value=managed), \
                patch.dict("os.environ", {"MODELLABS_ADAPTIVE_MODE": "pilot"}):
            result = adapt(held_out, records=[])
        self.assertEqual((result["model"], result["effort"]),
                         (choice["model"], choice["effort"]))
        self.assertEqual(result["adaptive_reason"], "shadow_managed_policy_pilot_holdout")

    def test_explicit_outcome_only(self):
        records = []
        record_outcome(records, "thread", "turn", "verified", {"class": "routine", "model": "gpt-5.6-terra", "effort": "medium", "task_bucket": "routine:base"})
        self.assertEqual(records[0]["source"], "explicit")

    def test_dashboard_counts_accepted_turns_not_every_event(self):
        summary = summarize([
            {"event": "route_accepted", "model": "gpt-5.6-terra", "effort": "medium", "thread_id": "t"},
            {"event": "turn_usage", "model": "gpt-5.6-terra", "usage": {"totalTokens": 23}},
            {"event": "turn_completed", "model": "gpt-5.6-terra"},
        ])
        self.assertEqual(summary["accepted_turn_counts"], {"gpt-5.6-terra": 1})
        self.assertEqual(summary["token_totals"], {"gpt-5.6-terra": 23})

    def test_catalog_gate_requires_listed_model_and_effort(self):
        catalog = {"data": [{"id": "gpt-6-astra", "hidden": False,
                             "supportedReasoningEfforts": [{"reasoningEffort": "high"},
                                                           {"reasoningEffort": "ultra"}]}]}
        self.assertTrue(selection_is_listed(catalog, {"model": "gpt-6-astra", "effort": "ultra"}))
        self.assertFalse(selection_is_listed(catalog, {"model": "gpt-6-astra", "effort": "max"}))
        self.assertFalse(selection_is_listed(catalog, {"model": "gpt-5.6-luna", "effort": "low"}))

    def test_new_model_efforts_follow_the_live_catalog(self):
        catalog = {"data": [
            {"id": "gpt-6-sol", "hidden": False,
             "supportedReasoningEfforts": [{"reasoningEffort": "ultra"}]},
            {"id": "gpt-6-luna", "hidden": False,
             "supportedReasoningEfforts": [{"reasoningEffort": "max"}]},
        ]}
        self.assertTrue(selection_is_listed(catalog, route("Use GPT-6 Sol with reasoning effort ultra.")))
        self.assertTrue(selection_is_listed(catalog, route("Use GPT-6 Luna with reasoning effort max.")))
        self.assertFalse(selection_is_listed(catalog, route("Use GPT-6 Luna with reasoning effort ultra.")))
        self.assertFalse(selection_is_listed(catalog, route("Use GPT-6 Luna with intelligence none.")))

    def test_toml_upsert_preserves_existing_sections(self):
        source = '[features]\nmemories = true\n\n[mcp_servers.other]\ncommand = "other"\n'
        result = upsert_toml(source, "features", {"step_model_switching": "true"})
        result = upsert_toml(result, "mcp_servers.modelControl", {"command": '"python"'})
        self.assertIn("memories = true", result)
        self.assertIn("step_model_switching = true", result)
        self.assertIn('[mcp_servers.other]', result)
        self.assertIn('[mcp_servers.modelControl]', result)

    def test_shell_route_precedence_is_idempotent(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".bashrc"
            path.write_text('export PATH="$HOME/.npm-global/bin:$PATH"\n', encoding="utf-8")
            bin_dir = Path(directory) / "bin"
            ensure_managed_route_precedence(path, bin_dir)
            ensure_managed_route_precedence(path, bin_dir)
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count(SHELL_PATH_START), 1)
            self.assertTrue(text.rstrip().endswith("# <<< ModelLabs managed Codex route <<<"))
            self.assertIn(f'export PATH={bin_dir}:"$PATH"', text)

    def test_real_codex_override_avoids_managed_shim(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            binary = Path(directory) / "codex-real"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(0o755)
            with patch.dict("os.environ", {"MODELLABS_REAL_CODEX": str(binary)}):
                self.assertEqual(real_codex_binary(), binary)

    def test_exec_prompt_is_found_without_confusing_option_values(self):
        args = ["exec", "--ephemeral", "--cd", "/tmp", "--skip-git-repo-check", "Reply exactly: OK"]
        self.assertEqual(_exec_prompt(args), "Reply exactly: OK")
        self.assertEqual(_exec_prompt(["exec", "resume", "thread-id", "continue"]), "continue")
        self.assertEqual(_command_index(["--profile", "p", "exec", "hello"]), 2)
        self.assertEqual(_exec_prompt(["--profile", "p", "exec", "hello"]), "hello")
        self.assertEqual(_exec_prompt(["exec", "--", "-literal prompt"]), "-literal prompt")
        self.assertEqual(_exec_prompt(["exec", "resume", "thread-id", "continue here"]), "continue here")
        self.assertEqual(_exec_prompt(["exec", "resume", "--last", "continue latest"]), "continue latest")
        self.assertEqual(_exec_prompt(["exec", "resume", "thread-id", "-"]), "-")

    def test_effective_exec_command_preserves_explicit_model_and_effort(self):
        args = ["--profile", "p", "exec", "--model=gpt-explicit",
                "--config", 'model_reasoning_effort="high"', "hello"]
        choice = {"model": "gpt-routed", "effort": "low", "servers": ["modelControl"]}
        command = _routed_exec_command(Path("/real/codex"), args, choice)
        self.assertEqual(_explicit_setting(args, "model"), "gpt-explicit")
        self.assertEqual(_explicit_setting(args, "model_reasoning_effort"), "high")
        self.assertNotIn("gpt-routed", command)
        self.assertNotIn('model_reasoning_effort="low"', command)
        for server in model_host_launcher.CONFIGURED_MCP_SERVERS:
            expected = f"mcp_servers.{server}.enabled={str(server == 'modelControl').lower()}"
            self.assertIn(expected, command)

    def test_interactive_preselection_scopes_tools_before_literal_prompt(self):
        args = ["-C", "/tmp", "--", "-literal prompt"]
        choice = {"model": "gpt-routed", "effort": "medium", "servers": ["modelControl"]}
        routed = _routed_interactive_args(args, "-literal prompt", choice, False)
        delimiter = routed.index("--")
        self.assertEqual(routed[delimiter + 1], "-literal prompt")
        self.assertLess(routed.index("--model"), delimiter)
        for server in model_host_launcher.CONFIGURED_MCP_SERVERS:
            expected = f"mcp_servers.{server}.enabled={str(server == 'modelControl').lower()}"
            self.assertIn(expected, routed[:delimiter])

    def test_launch_ticket_distinguishes_automatic_from_explicit_choice(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory, patch.object(model_host_launcher, "ROOT", Path(directory)):
            choice = {"model": "gpt-5.6-terra", "effort": "medium", "servers": ["modelControl"]}
            ticket = _create_launch_ticket(choice, explicit_model=False, explicit_effort=True)
            payload = json.loads((Path(directory) / "launch-tickets" / f"{ticket}.json").read_text())
            self.assertFalse(payload["explicit_model"])
            self.assertTrue(payload["explicit_effort"])

    def test_launch_workspace_exact_identity_and_alias_rejections(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / 'project.quoted"back\\slash'
            workspace.mkdir()
            info = workspace.stat()
            for args in (["-C", str(workspace)], ["--cd", str(workspace)],
                         [f"--cd={workspace}"]):
                with self.subTest(args=args):
                    self.assertEqual(model_host_launcher._launch_workspace(args),
                                     {"path": str(workspace), "dev": info.st_dev, "ino": info.st_ino})
            alias = root / "alias"
            alias.symlink_to(workspace, target_is_directory=True)
            for selected in (str(alias), str(workspace) + "/", str(root) + "/./project",
                             str(root) + "/bad\npath", str(root) + "/bad\udcffpath"):
                with self.subTest(selected=repr(selected)):
                    self.assertIsNone(model_host_launcher._launch_workspace(["-C", selected]))
            with patch.object(model_host_launcher.os, "getcwd", return_value=str(workspace)):
                self.assertEqual(model_host_launcher._launch_workspace([])["path"], str(workspace))
                self.assertEqual(model_host_launcher._launch_workspace(["--", "--cd=/other"])["path"], str(workspace))
                self.assertEqual(model_host_launcher._launch_workspace(["-c", "--cd=/other"])["path"], str(workspace))
                self.assertIsNone(model_host_launcher._launch_workspace(["-C" + str(workspace)]))

    def test_launch_tickets_and_grants_are_private_and_cleanup_compatible(self):
        from tempfile import TemporaryDirectory
        import proxy_supervisor
        with TemporaryDirectory() as directory, \
             patch.object(model_host_launcher, "ROOT", Path(directory)):
            root = Path(directory)
            workspace = model_host_launcher._launch_workspace(["-C", str(root)])
            ticket_id = _create_launch_ticket(None, explicit_model=True, explicit_effort=True,
                                            workspace=workspace)
            ticket_path = root / "launch-tickets" / f"{ticket_id}.json"
            ticket = json.loads(ticket_path.read_text())
            grant_path = root / "launch-tickets" / f'trust-{ticket["trust_grant"]}.json'
            self.assertEqual(ticket["workspace"], workspace)
            self.assertEqual(json.loads(grant_path.read_text())["ticket"], ticket_id)
            self.assertEqual(ticket_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(grant_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(ticket_path.parent.stat().st_mode & 0o777, 0o700)
            with patch.object(turn_proxy_module, "ROOT", root):
                spent = turn_proxy_module.spend_trust_grant(ticket["trust_grant"])
            old = __import__("time").time() - 25 * 60 * 60
            for path in (ticket_path, spent):
                os.utime(path, (old, old))
            with patch.object(proxy_supervisor, "ROOT", root):
                proxy_supervisor.remove_expired_launch_tickets()
            self.assertEqual(list(ticket_path.parent.glob("*.json")), [])

    def test_new_launch_forms_mint_one_ticket_without_inventing_promptless_route(self):
        from tempfile import TemporaryDirectory
        choice = {"model": "gpt-6.1-sol", "effort": "low", "servers": ["modelControl"]}
        for form, prompt in (([], False), (["start"], False), (["hello"], True),
                             (["start", "hello"], True), (["--", "-literal"], True)):
            with self.subTest(form=form), TemporaryDirectory() as directory, \
                 patch.object(model_host_launcher, "ROOT", Path(directory)), \
                 patch.object(sys, "argv", ["codex", "-C", directory, "-m", "gpt-6.1-sol",
                                           "-c", 'model_reasoning_effort="low"', *form]), \
                 patch.object(model_host_launcher, "_read_token", return_value="token"), \
                 patch.object(model_host_launcher, "ensure_proxy"), \
                 patch.object(model_host_launcher, "ensure_proxy_supervisor"), \
                 patch.object(model_host_launcher, "real_codex_binary", return_value=Path("/real/codex")), \
                 patch.object(model_host_launcher, "_route_choice", return_value=choice) as route_call, \
                 patch.object(model_host_launcher.os, "execve") as launch:
                model_host_launcher.main()
                payloads = [json.loads(path.read_text()) for path in
                            (Path(directory) / "launch-tickets").glob("*.json")]
                self.assertEqual(len(payloads), 2)
                ticket = next(item for item in payloads if item["schema"] == "modellabs.launch_ticket.v2")
                self.assertEqual(ticket["workspace"]["path"], directory)
                self.assertTrue(ticket["explicit_model"])
                self.assertTrue(ticket["explicit_effort"])
                self.assertEqual(ticket["model"], choice["model"] if prompt else None)
                self.assertEqual(route_call.call_count, int(prompt))
                self.assertRegex(launch.call_args.args[2]["MODEL_SELECTOR_HOST_TOKEN"],
                                 r"^token\.launch-[0-9a-f]{32}$")

    def test_proxy_preserves_preselection_without_marking_user_override(self):
        request = {"id": 1, "method": "turn/start", "params": {
            "threadId": "thread", "input": [{"type": "text", "text": "Format values as CSV."}],
            "model": "gpt-6-astra", "effort": "high",
        }}
        routed, info = route_request(json.dumps(request),
                                     preserve_model=True, preserve_effort=True)
        params = json.loads(routed)["params"]
        self.assertEqual((params["model"], params["effort"]), ("gpt-6-astra", "high"))
        self.assertFalse(info.get("explicit_model", False))
        self.assertFalse(info.get("explicit_effort", False))

    def test_turn_routing_exception_fails_closed(self):
        request = json.dumps({"id": 1, "method": "turn/start", "params": {
            "threadId": "thread", "input": [{"type": "text", "text": "hello"}]}})
        with patch("turn_proxy.route", side_effect=ValueError("bad adaptive evidence")):
            with self.assertRaisesRegex(ValueError, "failed closed"):
                route_request(request)

    def test_exec_settings_respect_delimiter_precedence_and_scope_guard(self):
        literal = ["exec", "--", "--model=gpt-literal"]
        self.assertIsNone(_explicit_setting(literal, "model"))
        repeated = ["-c", "model_reasoning_effort=low", "exec", "-c",
                    "model_reasoning_effort=high", "hello"]
        self.assertEqual(_explicit_setting(repeated, "model_reasoning_effort"), "high")
        self.assertEqual(_explicit_setting(["-mgpt-attached", "hello"], "model"), "gpt-attached")
        self.assertEqual(_explicit_setting(["-cmodel_reasoning_effort=high", "hello"],
                                           "model_reasoning_effort"), "high")
        choice = {"model": "gpt-routed", "effort": "medium", "servers": ["modelControl"]}
        command = _routed_exec_command(Path("/real/codex"), literal, choice)
        self.assertIn("gpt-routed", command)
        with self.assertRaisesRegex(ValueError, "MCP scope overrides"):
            _routed_exec_command(Path("/real/codex"),
                                 ["exec", "-c", "mcp_servers.cloudflare.enabled=true", "hello"],
                                 choice)
        with self.assertRaisesRegex(ValueError, "MCP scope overrides"):
            _routed_exec_command(Path("/real/codex"),
                                 ["exec", "-cmcp_servers.cloudflare.enabled=true", "hello"],
                                 choice)

    def test_review_and_resume_delimiter_prompts_are_classified_correctly(self):
        self.assertEqual(_exec_prompt(["exec", "review", "focus on races"]), "focus on races")
        self.assertEqual(_exec_prompt(["exec", "resume", "thread-id", "--", "--last"]), "--last")
        self.assertEqual(_exec_subcommand(["exec", "--json", "resume", "thread-id", "continue"])[0],
                         "resume")
        self.assertEqual(_exec_prompt(["exec", "--", "help"]), "help")

    def test_noninteractive_inference_fails_closed_without_receipts(self):
        with patch.object(sys, "stdin", SimpleNamespace(isatty=lambda: True)):
            with self.assertRaisesRegex(ValueError, "exact-or-unavailable usage receipts"):
                run_routed_exec(["exec", "hello"])
        with patch.object(sys, "stdin", SimpleNamespace(isatty=lambda: True)):
            with self.assertRaisesRegex(ValueError, "exec resume"):
                run_routed_exec(["exec", "--json", "resume", "thread-id", "continue"])
        with patch.object(sys, "stdin", SimpleNamespace(isatty=lambda: True)), \
             patch.object(model_host_launcher, "real_codex_binary", return_value=Path("/real/codex")):
            with self.assertRaisesRegex(ValueError, "exact-or-unavailable usage receipts"):
                run_routed_exec(["exec", "--", "--help"])

    def test_explicit_authority_blocks_agent_model_override(self):
        import asyncio
        thread_id = "00000000-0000-4000-8000-000000000004"
        with patch.dict(os.environ, {"CODEX_THREAD_ID": thread_id}), \
             patch.object(host_control, "_choice_authority", return_value={
                "explicit_model": True, "model": "gpt-pinned",
                "explicit_effort": True, "effort": "high"}):
            with self.assertRaisesRegex(host_control.ModelHostError, "explicitly pinned"):
                asyncio.run(host_control.switch_current_turn_model(
                    thread_id, "gpt-other", "high"))

    def test_switch_compatibility_preflight_rejects_known_runtime_mismatch(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            cache = Path(directory) / "models_cache.json"
            cache.write_text(json.dumps({"models": [
                {"slug": "gpt-6-sol", "visibility": "list",
                 "node_repl_auto_review_required": True},
                {"slug": "gpt-6-luna", "visibility": "list",
                 "node_repl_auto_review_required": False},
            ]}), encoding="utf-8")
            now = cache.stat().st_mtime
            with patch.object(host_control, "MODELS_CACHE_PATH", cache), \
                 patch.object(host_control.time, "time", return_value=now):
                with self.assertRaisesRegex(host_control.ModelHostError,
                                            "different Node REPL auto-review contracts"):
                    host_control._ensure_switch_runtime_compatible(
                        "gpt-6-sol", "gpt-6-luna")

    def test_active_turn_switch_marks_evidence_before_host_update(self):
        import asyncio
        thread_id = "00000000-0000-4000-8000-000000000004"
        turn_id = "turn-4"

        class Connection:
            async def __aenter__(self):
                return SimpleNamespace(send=AsyncMock())

            async def __aexit__(self, *_args):
                return False

        marker_calls = []
        request_ids = []

        async def rpc(_ws, method, _params, _request_id):
            request_ids.append(_request_id)
            if method == "model/list":
                if not _params:
                    return {"data": [], "nextCursor": "page-two"}
                self.assertEqual(_params, {"cursor": "page-two"})
                return {"data": [{"id": "gpt-6-sol", "hidden": False,
                                   "supportedReasoningEfforts": [{"reasoningEffort": "high"}]}]}
            if method == "thread/read":
                return {"thread": {"id": thread_id, "status": {"type": "active"},
                                   "model": "gpt-6-sol"}}
            if method == "thread/turns/list":
                return {"data": [{"id": turn_id, "status": "inProgress", "items": [
                    {"type": "commandExecution", "status": "completed", "exitCode": 0,
                     "command": "printf %s \"$CODEX_THREAD_ID\"",
                     "aggregatedOutput": thread_id}]}]}
            if method == "turn/settings/update":
                self.assertEqual(len(marker_calls), 1)
                self.assertEqual(marker_calls[0][1]["receipt_id"],
                                 f"{thread_id}:{turn_id}:model_switch")
                return {"status": "applied"}
            return {}

        descriptor = os.open("/dev/null", os.O_RDONLY)
        with patch.dict(os.environ, {"CODEX_THREAD_ID": thread_id}), \
             patch.object(host_control, "acquire_lock_async", new=AsyncMock(return_value=descriptor)), \
             patch.object(host_control, "_choice_authority", return_value={}), \
             patch.object(host_control, "_read_token", return_value="token"), \
             patch.object(host_control, "_ensure_switch_runtime_compatible"), \
             patch.object(host_control.websockets, "connect", return_value=Connection()), \
             patch.object(host_control, "_rpc", side_effect=rpc), \
             patch.object(host_control, "record_metric",
                          side_effect=lambda event, **fields: marker_calls.append((event, fields))):
            result = asyncio.run(host_control.switch_current_turn_model(
                thread_id, "gpt-6-sol", "high"))
        self.assertEqual(result["status"], "applied_to_later_steps")
        self.assertEqual(marker_calls[0][0], "turn_model_switch_attempted")
        self.assertEqual(request_ids, [1, 2, 3, 34, 35, 36])

    def test_switch_compatibility_preflight_defers_when_unknown_or_compatible(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            cache = Path(directory) / "models_cache.json"
            cache.write_text(json.dumps({"models": [
                {"slug": "gpt-6-sol", "visibility": "list",
                 "node_repl_auto_review_required": True},
                {"slug": "gpt-6.1-sol", "visibility": "list",
                 "node_repl_auto_review_required": True},
            ]}), encoding="utf-8")
            now = cache.stat().st_mtime
            with patch.object(host_control, "MODELS_CACHE_PATH", cache), \
                 patch.object(host_control.time, "time", return_value=now):
                host_control._ensure_switch_runtime_compatible(
                    "gpt-6-sol", "gpt-6.1-sol")
                host_control._ensure_switch_runtime_compatible(
                    "gpt-6-sol", "gpt-unknown")
            with patch.object(host_control, "MODELS_CACHE_PATH", cache), \
                 patch.object(host_control.time, "time",
                              return_value=now + host_control.MODEL_CACHE_MAX_AGE_SECONDS + 1):
                host_control._ensure_switch_runtime_compatible(
                    "gpt-6-sol", "gpt-6-luna")

    def test_global_option_operands_do_not_become_commands(self):
        args = ["--enable", "search", "--remote", "ws://example", "-a", "never",
                "--model=gpt-explicit", "exec", "hello"]
        self.assertEqual(_command_index(args), 7)
        self.assertEqual(_exec_prompt(args), "hello")

    def test_managed_resume_lock_rejects_a_racing_second_owner(self):
        from tempfile import TemporaryDirectory
        thread_id = "00000000-0000-4000-8000-000000000001"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rollout = root / f"rollout-test-{thread_id}.jsonl"
            rollout.write_text("{}\n", encoding="utf-8")
            with patch.object(thread_owner, "ROOT", root), \
                 patch.object(thread_owner, "rollout_for", return_value=rollout), \
                 patch.object(thread_owner, "require_unowned", return_value=None):
                descriptor = thread_owner.acquire_thread_ownership(thread_id)
                try:
                    with self.assertRaisesRegex(RuntimeError, "already owned"):
                        thread_owner.acquire_thread_ownership(thread_id)
                finally:
                    os.close(descriptor)

    def test_fresh_managed_owner_blocks_resume_owner(self):
        from tempfile import TemporaryDirectory
        thread_id = "00000000-0000-4000-8000-000000000003"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(thread_owner, "ROOT", root):
                descriptor = thread_owner.acquire_thread_ownership(thread_id, existing_thread=False)
                try:
                    with self.assertRaisesRegex(RuntimeError, "already owned"):
                        thread_owner.acquire_thread_ownership(thread_id, existing_thread=False)
                finally:
                    os.close(descriptor)

    def test_wrapper_install_replaces_symlink_without_touching_target(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real-codex"
            real.write_text("#!/bin/sh\necho real\n", encoding="utf-8")
            real.chmod(0o755)
            original = real.read_bytes()
            bin_dir = root / "bin"
            bin_dir.mkdir()
            (bin_dir / "codex").symlink_to(real)
            home = root / "home"
            (home / "venv/bin").mkdir(parents=True)
            write_wrappers(home, bin_dir, real)
            self.assertFalse((bin_dir / "codex").is_symlink())
            self.assertEqual(real.read_bytes(), original)
            self.assertIn("ModelLabs managed wrapper", (bin_dir / "codex").read_text(encoding="utf-8"))
            self.assertIn("ModelLabs recovery wrapper", (bin_dir / "codex-direct").read_text(encoding="utf-8"))
            self.assertIn(str(real), (bin_dir / "codex-direct").read_text(encoding="utf-8"))

    def test_wrapper_install_refuses_unrelated_destination_before_any_change(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real-codex"
            real.write_text("#!/bin/sh\necho real\n", encoding="utf-8")
            real.chmod(0o755)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            unrelated = bin_dir / "codex-direct"
            unrelated.write_text("#!/bin/sh\necho user-owned\n", encoding="utf-8")
            unrelated.chmod(0o755)
            home = root / "home"
            (home / "venv/bin").mkdir(parents=True)
            before = unrelated.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "unrelated executable"):
                write_wrappers(home, bin_dir, real)
            self.assertEqual(unrelated.read_bytes(), before)
            self.assertFalse((bin_dir / "codex").exists())

    def test_discovery_preserves_a_legitimate_upstream_symlink(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            home, bin_dir = root / "home", root / "bin"
            bin_dir.mkdir()
            real = root / "real-codex"
            real.write_text("#!/bin/sh\necho real\n", encoding="utf-8")
            real.chmod(0o755)
            (bin_dir / "codex").symlink_to(real)
            with patch.dict("os.environ", {"PATH": str(bin_dir)}, clear=False):
                self.assertEqual(discover_real_codex(home, bin_dir), real)

    def test_recovery_wrapper_is_rejected_as_upstream_override(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            home, bin_dir = root / "home", root / "bin"
            bin_dir.mkdir()
            direct = bin_dir / "codex-direct"
            direct.write_text("#!/bin/sh\n# ModelLabs recovery wrapper\nexec /real/codex \"$@\"\n",
                              encoding="utf-8")
            direct.chmod(0o755)
            with patch.dict("os.environ", {"MODELLABS_REAL_CODEX": str(direct), "PATH": ""}, clear=False):
                self.assertNotEqual(discover_real_codex(home, bin_dir), direct)

    def test_complete_install_replaces_upstream_symlink_without_mutation(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            home, bin_dir, codex_home = root / "home", root / "bin", root / "codex-home"
            bin_dir.mkdir()
            real = root / "real-codex"
            real.write_text("#!/bin/sh\necho real\n", encoding="utf-8")
            real.chmod(0o755)
            original = real.read_bytes()
            (bin_dir / "codex").symlink_to(real)
            args = SimpleNamespace(home=home, bin_dir=bin_dir, codex_home=codex_home,
                                   no_service=True)

            def fake_venv(path):
                (path / "bin").mkdir(parents=True)
                python = path / "bin/python"
                python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                python.chmod(0o755)

            completed = SimpleNamespace(returncode=0, stdout="codex-cli 0.160.0\n")
            with patch.dict("os.environ", {"PATH": str(bin_dir)}, clear=False), \
                 patch.object(installer, "create_venv", side_effect=fake_venv), \
                 patch.object(installer, "write_service", return_value=root / "service"), \
                 patch.object(installer.subprocess, "run", return_value=completed):
                installer.install(args)
            self.assertEqual(real.read_bytes(), original)
            self.assertFalse((bin_dir / "codex").is_symlink())
            self.assertIn(str(real), (bin_dir / "codex-direct").read_text(encoding="utf-8"))

    def test_install_payload_preflight_refuses_unrelated_and_upstream_aliases(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            home, codex_home = root / "home", root / "codex"
            home.mkdir()
            upstream = root / "real-codex"
            upstream.write_text("#!/bin/sh\n", encoding="utf-8")
            upstream.chmod(0o755)
            payloads = installer._install_payloads(home, codex_home, include_service=False)
            unrelated = home / "adaptive_policy.py"
            unrelated.write_text("user file\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unrelated"):
                installer.preflight_install_payloads(payloads, home, upstream, codex_home)
            unrelated.unlink()
            unrelated.symlink_to(upstream)
            with self.assertRaisesRegex(RuntimeError, "symlinked"):
                installer.preflight_install_payloads(payloads, home, upstream, codex_home)

    def test_manifest_adopts_only_an_exact_unmanifested_service(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            home, codex_home, bin_dir = root / "home", root / "codex", root / "bin"
            home.mkdir()
            upstream = root / "real-codex"
            upstream.write_text("#!/bin/sh\n", encoding="utf-8")
            upstream.chmod(0o755)
            service = root / "config/systemd/user/modellabs-proxy.service"
            service.parent.mkdir(parents=True)
            expected = b"[Unit]\nDescription=ModelLabs exact test service\n"
            service.write_bytes(expected)
            payloads = {service: expected}
            manifest = {"schema": "modellabs.owned_files.v1", "files": {}}
            (home / installer.OWNED_MANIFEST).write_text(
                json.dumps(manifest), encoding="utf-8")
            installer.preflight_install_payloads(
                payloads, home, upstream, codex_home, bin_dir)
            service.write_bytes(expected + b"# changed\n")
            with self.assertRaisesRegex(RuntimeError, "missing existing payload"):
                installer.preflight_install_payloads(
                    payloads, home, upstream, codex_home, bin_dir)

    def test_manifest_digest_and_parent_symlink_fail_closed(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            home, codex_home = root / "home", root / "codex"
            home.mkdir()
            upstream = root / "real-codex"
            upstream.write_text("#!/bin/sh\n", encoding="utf-8")
            upstream.chmod(0o755)
            payloads = installer._install_payloads(home, codex_home, include_service=False)
            target = home / "turn_proxy.py"
            target.write_bytes(payloads[target])
            manifest = {"schema": "modellabs.owned_files.v1", "files": {
                str(target): installer._digest_bytes(payloads[target])}}
            (home / installer.OWNED_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
            target.write_bytes(payloads[target] + b"\n# changed\n")
            with self.assertRaisesRegex(RuntimeError, "unrelated"):
                installer.preflight_install_payloads(payloads, home, upstream, codex_home)
            manifest["files"][str(target)] = installer._digest_bytes(b"previous owned version")
            (home / installer.OWNED_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
            target.write_bytes(payloads[target])
            installer.preflight_install_payloads(payloads, home, upstream, codex_home)
            target.unlink()
            (home / installer.OWNED_MANIFEST).unlink()
            redirected = root / "redirected"
            redirected.mkdir()
            (home / "benchmarks").symlink_to(redirected, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "redirected"):
                installer.preflight_install_payloads(payloads, home, upstream, codex_home)


if __name__ == "__main__":
    unittest.main()
