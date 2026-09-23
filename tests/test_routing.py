import os
import json
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modellabs import route
import install as installer
from install import (SHELL_PATH_START, discover_real_codex,
                     ensure_managed_route_precedence, upsert_toml, write_wrappers)
from paths import real_codex_binary
from turn_proxy import route_request, selection_is_listed
from adaptive_policy import adapt, record_outcome
from health_dashboard import summarize
import model_host_launcher
import host_control
from model_host_launcher import (_command_index, _exec_prompt, _explicit_setting,
                                 _exec_subcommand, _create_launch_ticket, _routed_exec_command,
                                 _routed_interactive_args, run_routed_exec)
import thread_owner


class RoutingTests(unittest.TestCase):
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
        with patch.object(sys, "argv", ["codex"]), \
             patch.object(model_host_launcher, "_read_token", return_value="token"), \
             patch.object(model_host_launcher, "ensure_proxy"), \
             patch.object(model_host_launcher, "ensure_proxy_supervisor"), \
             patch.object(model_host_launcher, "real_codex_binary", return_value=Path("/real/codex")), \
             patch.object(model_host_launcher, "_route_choice", side_effect=AssertionError("prompt read")), \
             patch.object(model_host_launcher.os, "execve") as launch:
            model_host_launcher.main()
        binary, args, environment = launch.call_args.args
        self.assertEqual(binary, Path("/real/codex"))
        self.assertEqual(args, ["/real/codex", "--remote", model_host_launcher.PROXY_URL,
                                "--remote-auth-token-env", "MODEL_SELECTOR_HOST_TOKEN"])
        self.assertEqual(environment["MODEL_SELECTOR_HOST_TOKEN"], "token")

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

            completed = SimpleNamespace(returncode=0)
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
