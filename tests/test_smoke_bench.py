import json
import io
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_policy import adapt
from modellabs import SERVERS, route
from smoke_bench import (browser_prompt_variant, explicit_inline_revision, grade, load_manifest, materialize, run_one,
                         parse_codex_jsonl, product_definition, ranked, stable_digest)
import smoke_bench


class SmokeBenchTests(unittest.TestCase):
    def test_managed_benchmark_thread_disables_unneeded_mcp_servers(self):
        params = smoke_bench.managed_thread_start_params(Path("/tmp/managed-benchmark"))
        self.assertEqual(params["cwd"], "/tmp/managed-benchmark")
        self.assertEqual({key: params[key] for key in (
            "ephemeral", "sandbox", "approvalPolicy")}, {
                "ephemeral": True, "sandbox": "workspace-write",
                "approvalPolicy": "never"})
        config = params["config"]
        self.assertEqual(set(config), {"mcp_servers"})
        self.assertEqual(set(config["mcp_servers"]), SERVERS)
        self.assertTrue(all(value == {"enabled": False}
                            for value in config["mcp_servers"].values()))

    def test_daily_managed_cohorts_and_registered_experiments_are_frozen(self):
        manifest = load_manifest(Path(__file__).resolve().parents[1] / "benchmarks/smoke.json")
        cohorts = {"routine_luna_vs_sol_development": [],
                   "routine_luna_vs_sol_prospective": []}
        for scenario in manifest["scenarios"]:
            if scenario.get("managed_cohort") in cohorts:
                cohorts[scenario["managed_cohort"]].append(scenario)
        self.assertEqual({name: len(rows) for name, rows in cohorts.items()},
                         {"routine_luna_vs_sol_development": 8,
                          "routine_luna_vs_sol_prospective": 8})
        products = [{stable_digest(product_definition(scenario))
                     for scenario in rows} for rows in cohorts.values()]
        self.assertTrue(all(len(group) == 8 for group in products))
        self.assertTrue(products[0].isdisjoint(products[1]))
        self.assertTrue(all(route(scenario["prompt"])["class"] == "routine"
                            for rows in cohorts.values() for scenario in rows))
        experiments = manifest["managed_experiments"]
        self.assertEqual([row["id"] for row in experiments], [
            "routine-gpt6-luna-low-vs-sol-medium",
            "routine-gpt56-luna-low-vs-sol-medium",
            "routine-gpt6-sol-low-vs-medium",
            "routine-gpt6-astra-medium-vs-sol-medium",
            "difficult-gpt6-astra-medium-vs-sol-high",
        ])
        self.assertTrue(all(len(row["development_scenario_ids"]) >= 8
                            and len(row["prospective_scenario_ids"]) >= 8
                            for row in experiments))
        difficult = experiments[-1]
        difficult_products = [next(scenario for scenario in manifest["scenarios"]
                                    if scenario["id"] == identifier)
                              for identifier in (difficult["development_scenario_ids"]
                                                 + difficult["prospective_scenario_ids"])]
        self.assertTrue(all(route(scenario["prompt"])["class"] == "difficult"
                            for scenario in difficult_products))
        self.assertEqual({(route(scenario["prompt"])["model"],
                           route(scenario["prompt"])["effort"])
                          for scenario in difficult_products},
                         {("gpt-6-sol", "high")})

    def test_manifest_rejects_managed_experiment_without_routed_baseline(self):
        source = Path(__file__).resolve().parents[1] / "benchmarks/smoke.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        data["managed_experiments"][0]["arms"] = [
            ["gpt-6-luna", "low"], ["gpt-6-astra", "medium"]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "routed baseline"):
                load_manifest(path)

    def test_daily_managed_selection_holds_prospective_products_until_checkpoint(self):
        scenarios = [
            {"id": "dev-a", "prompt": "Build a complete tested command line utility.",
             "task_class": "routine", "managed_cohort": "routine_luna_vs_sol_development"},
            {"id": "dev-b", "prompt": "Build a complete tested command line parser.",
             "task_class": "routine", "managed_cohort": "routine_luna_vs_sol_development"},
            {"id": "future", "prompt": "Build a complete tested data converter.",
             "task_class": "routine", "managed_cohort": "routine_luna_vs_sol_prospective"},
        ]
        arms = {("gpt-6-luna", "low"), ("gpt-6-sol", "medium")}
        comparison = {"left_arm": ["gpt-6-luna", "low"],
                      "right_arm": ["gpt-6-sol", "medium"],
                      "task_class": "routine", "development_checkpoint_created": False}
        with patch("outcome_model.train_managed_routing_policy",
                   return_value={"comparisons": [comparison]}), \
                patch("outcome_model.managed_scenario_progress",
                      return_value={"dev-a": 2, "dev-b": 0}), \
                patch("outcome_model.managed_scenario_attempts",
                      return_value={"dev-a": 2, "dev-b": 0}):
            selected, selected_arms, phase = smoke_bench.daily_managed_selection(scenarios)
        self.assertEqual((selected["id"], set(selected_arms), phase),
                         ("dev-b", arms, "development"))
        comparison["development_checkpoint_created"] = True
        with patch("outcome_model.train_managed_routing_policy",
                   return_value={"comparisons": [comparison]}), \
                patch("outcome_model.managed_scenario_progress",
                      return_value={"future": 0}), \
                patch("outcome_model.managed_scenario_attempts",
                      return_value={"future": 0}):
            selected, _selected_arms, phase = smoke_bench.daily_managed_selection(scenarios)
        self.assertEqual((selected["id"], phase), ("future", "prospective"))

    def test_daily_managed_selection_skips_class_mismatch_and_quarantines_failures(self):
        scenarios = [
            {"id": "mismatch", "prompt": "Build a complete tested command line utility.",
             "task_class": "simple", "managed_cohort": "routine_luna_vs_sol_development"},
            {"id": "quarantined", "prompt": "Build a complete tested data converter.",
             "task_class": "routine", "managed_cohort": "routine_luna_vs_sol_development"},
            {"id": "healthy", "prompt": "Build a complete tested record parser.",
             "task_class": "routine", "managed_cohort": "routine_luna_vs_sol_development"},
        ]
        with patch("outcome_model.train_managed_routing_policy",
                   return_value={"comparisons": []}), \
                patch("outcome_model.managed_scenario_progress",
                      return_value={"quarantined": 0, "healthy": 0}), \
                patch("outcome_model.managed_scenario_attempts",
                      return_value={"quarantined": 2, "healthy": 0}):
            selected, _arms, phase = smoke_bench.daily_managed_selection(scenarios)
        self.assertEqual((selected["id"], phase), ("healthy", "development"))

    def test_daily_managed_selection_reports_exhausted_failure_budget(self):
        scenarios = [{"id": "quarantined",
                      "prompt": "Build a complete tested data converter.",
                      "task_class": "routine",
                      "managed_cohort": "routine_luna_vs_sol_development"}]
        with patch("outcome_model.train_managed_routing_policy",
                   return_value={"comparisons": []}), \
                patch("outcome_model.managed_scenario_progress",
                      return_value={"quarantined": 0}), \
                patch("outcome_model.managed_scenario_attempts",
                      return_value={"quarantined": 2}):
            selected, _arms, phase = smoke_bench.daily_managed_selection(scenarios)
        self.assertIsNone(selected)
        self.assertEqual(phase, "development_failure_budget_exhausted")

    def test_registered_daily_managed_selection_advances_after_inconclusive_pair(self):
        scenarios = [
            {"id": "dev-a", "prompt": "Build a complete tested command line utility.",
             "task_class": "routine"},
            {"id": "dev-b", "prompt": "Build a complete tested command line parser.",
             "task_class": "routine"},
        ]
        first_arms = [["gpt-6-luna", "low"], ["gpt-6-sol", "medium"]]
        second_arms = [["gpt-5.6-luna", "low"], ["gpt-6-sol", "medium"]]
        experiments = [
            {"id": "finished-tie", "task_class": "routine", "arms": first_arms,
             "development_scenario_ids": ["dev-a"],
             "prospective_scenario_ids": ["dev-b"], "enabled": True},
            {"id": "next-pair", "task_class": "routine", "arms": second_arms,
             "development_scenario_ids": ["dev-a"],
             "prospective_scenario_ids": ["dev-b"], "enabled": True},
        ]
        comparisons = [{"left_arm": first_arms[0], "right_arm": first_arms[1],
                        "task_class": "routine", "development_checkpoint_created": False}]

        def progress(rows, arms):
            complete = 3 if set(arms) == {tuple(value) for value in first_arms} else 0
            return {row["id"]: complete for row in rows}

        with patch("outcome_model.train_managed_routing_policy",
                   return_value={"comparisons": comparisons}), \
                patch("outcome_model.managed_scenario_progress", side_effect=progress), \
                patch("outcome_model.managed_scenario_attempts", side_effect=progress):
            selected, arms, phase = smoke_bench.daily_managed_selection(
                scenarios, experiments)
        self.assertEqual(selected["id"], "dev-a")
        self.assertEqual(set(arms), {tuple(value) for value in second_arms})
        self.assertEqual(phase, "development:next-pair")

    def test_private_prompt_registry_is_bounded_and_owner_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_root = root / "prompt-experiments"
            prompt_root.mkdir()
            prompt = prompt_root / "revision.txt"
            prompt.write_text("Build a complete tested parser.", encoding="utf-8")
            prompt.chmod(0o600)
            registry = root / "prompt-experiments.json"
            registry.write_text(json.dumps({
                "schema": "modellabs.prompt_experiment_registry.v1",
                "experiments": [{"id": "reviewed-1", "scenario_id": "routine_cli",
                                 "guard_id": "guard-1", "prompt_file": str(prompt),
                                 "enabled": True}],
            }), encoding="utf-8")
            registry.chmod(0o600)
            with patch.object(smoke_bench, "ROOT", root):
                loaded = smoke_bench.load_prompt_registry(registry)
                self.assertEqual(loaded["experiments"][0]["prompt_file"], str(prompt.resolve()))
                registry.chmod(0o644)
                with self.assertRaisesRegex(ValueError, "owner-only"):
                    smoke_bench.load_prompt_registry(registry)
                registry.write_text("null\n", encoding="utf-8")
                registry.chmod(0o600)
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    smoke_bench.load_prompt_registry(registry)

    def test_daily_prompt_selection_uses_only_pending_reviewed_development_product(self):
        scenarios = [
            {"id": "dev-a", "prompt": "Build a complete tested command line parser.",
             "task_class": "routine", "managed_cohort": "routine_luna_vs_sol_development"},
            {"id": "dev-b", "prompt": "Build a complete tested record converter.",
             "task_class": "routine", "managed_cohort": "routine_luna_vs_sol_development"},
            {"id": "future", "prompt": "Build a complete tested data formatter.",
             "task_class": "routine", "managed_cohort": "routine_luna_vs_sol_prospective"},
        ]
        registry = {"experiments": [
            {"id": "a", "scenario_id": "dev-a", "guard_id": "guard-a",
             "prompt_file": "/private/a", "enabled": True},
            {"id": "b", "scenario_id": "dev-b", "guard_id": "guard-b",
             "prompt_file": "/private/b", "enabled": True},
            {"id": "future", "scenario_id": "future", "guard_id": "guard-future",
             "prompt_file": "/private/future", "enabled": True},
        ]}

        def variant(base, guard_id, _prompt_file):
            return {**base, "id": f"{base['id']}-candidate", "prompt": base["prompt"] + " Exact output.",
                    "prompt_author": "chatgpt", "variant_of": base["id"],
                    "browser_origin_prompt": base["prompt"],
                    "browser_review": {"guard_id": guard_id, "response_id": "resp_1"}}

        def counts(base, _candidate, _arms):
            return ({"attempts": 3, "complete": 3, "failed_or_incomplete": 0}
                    if base["id"] == "dev-a" else
                    {"attempts": 1, "complete": 1, "failed_or_incomplete": 0})

        with patch.object(smoke_bench, "browser_prompt_variant", side_effect=variant), \
                patch("outcome_model.managed_prompt_experiment_counts", side_effect=counts), \
                patch("outcome_model.train_managed_prompt_policy",
                      return_value={"comparisons": []}):
            selected, arms, phase = smoke_bench.daily_prompt_selection(scenarios, registry)
        self.assertEqual([item["id"] for item in selected], ["dev-b", "dev-b-candidate"])
        self.assertEqual(set(arms), {("gpt-6-luna", "low"), ("gpt-6-sol", "medium")})
        self.assertEqual(phase, "prompt_development:b")

    def test_explicit_inline_revision_is_not_inferred_from_ordinary_feedback(self):
        state = {"review_format": "inline", "status": "completed", "nonce": "n",
                 "verdict": {"nonce": "n", "pass": False, "suggested_next_prompt": None,
                             "summary": "A complete revised prompt would be:\n\n“Use exact stdout.”"},
                 "evaluation": {"valid": True, "nonce_matched": True}}
        self.assertEqual(explicit_inline_revision(state), "Use exact stdout.")
        self.assertIsNone(explicit_inline_revision({**state, "verdict": {
            **state["verdict"], "summary": "Consider clarifying stdout."}}))
        self.assertIsNone(explicit_inline_revision({**state, "verdict": {
            **state["verdict"], "summary": state["verdict"]["summary"] * 2}}))
        self.assertIsNone(explicit_inline_revision({**state, "evaluation": {
            "valid": True, "nonce_matched": False}}))
        self.assertIsNone(explicit_inline_revision({**state, "review_format": "file"}))

    def test_browser_variant_requires_exact_completed_review_quote(self):
        import hashlib
        base = {"id": "base", "prompt": "Original task", "comparison_id": "same"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            origin = root / "origin.txt"
            revision = root / "revision.txt"
            origin.write_text("Original task\n", encoding="utf-8")
            revision.write_text("Better task\n", encoding="utf-8")
            digest = hashlib.sha256(origin.read_bytes()).hexdigest()
            source = {"path": str(origin), "sha256": digest}
            state = {"schema": "codex.pro_guard_run.v1", "guard_id": "review-1",
                     "status": "completed", "review_format": "inline", "response_id": "resp_123",
                     "conversation_url": "https://chatgpt.com/c/123", "nonce": "nonce",
                     "verdict": {"nonce": "nonce", "pass": False,
                                 "summary": "A complete revised prompt would be:\n\n“Better task”"},
                     "evaluation": {"valid": True, "nonce_matched": True},
                     "source_files": {"origin_prompt": source, "generation_prompt": source,
                                      "artifact": source},
                     "learning_trace": {"root_guard_id": "review-1", "parent_guard_id": None,
                                        "origin_prompt_sha256": digest,
                                        "generation_prompt_sha256": digest}}
            guard = root / "review-1.json"
            guard.write_text(json.dumps(state), encoding="utf-8")
            variant = browser_prompt_variant(base, "review-1", revision, root)
            self.assertEqual(variant["prompt_author"], "chatgpt")
            self.assertEqual(variant["browser_review"]["response_id"], "resp_123")
            output = root / "materialized.txt"
            receipt = smoke_bench.materialize_browser_revision("review-1", output, root)
            self.assertEqual(output.read_text(), "Better task\n")
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(receipt["guard_id"], "review-1")
            with self.assertRaises(FileExistsError):
                smoke_bench.materialize_browser_revision("review-1", output, root)
            revision.write_text("Different task", encoding="utf-8")
            with self.assertRaises(ValueError):
                browser_prompt_variant(base, "review-1", revision, root)
            revision.write_text("Better task", encoding="utf-8")
            origin.write_text("Changed task", encoding="utf-8")
            with self.assertRaises(ValueError):
                browser_prompt_variant(base, "review-1", revision, root)

    def test_benchmark_receipt_binds_unique_run_identity(self):
        scenario = {"id": "test", "task_class": "simple", "prompt": "test prompt",
                    "files": {"verify.py": "pass\n"}, "protected_files": ["verify.py"],
                    "verify": ["python3", "verify.py"], "expected_final": "ready"}
        with patch.object(smoke_bench.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout='', stderr='')), \
                patch.object(smoke_bench, "grade", return_value={"product_pass": True,
                                                                  "total_tokens": 10}), \
                patch.object(smoke_bench, "record") as record, \
                patch("outcome_model.capture_benchmark_prompt_run", return_value=True) as capture:
            result = run_one(scenario, "gpt-6-luna", "low", "suite", "codex-direct", True, 3)
        self.assertEqual(result["sequence_index"], 3)
        self.assertEqual(record.call_args.args, ("benchmark_result",))
        self.assertEqual(record.call_args.kwargs["receipt_id"], f"benchmark:{result['run_id']}")
        self.assertEqual(result["learning_capture"], "captured")
        capture.assert_called_once()

    def test_repeated_prompt_comparison_alternates_run_order(self):
        scenarios = [{"id": "original", "matrix": [["gpt-6-luna", "low"]]},
                     {"id": "revised", "matrix": [["gpt-6-luna", "low"]]}]
        calls = []
        def fake_run(scenario, model, effort, suite_id, codex, record_metrics,
                     sequence_index, repetition_index):
            calls.append((scenario["id"], sequence_index))
            return {"scenario_id": scenario["id"], "product_pass": True,
                    "satisfaction_score": 100, "total_tokens": 10}
        with patch.object(sys, "argv", ["smoke_bench", "--repeat", "2"]), \
                patch.object(smoke_bench, "load_manifest", return_value={"scenarios": scenarios}), \
                patch.object(smoke_bench, "run_one", side_effect=fake_run), \
                redirect_stdout(io.StringIO()):
            smoke_bench.main()
        self.assertEqual(calls, [("original", 0), ("revised", 1),
                                 ("revised", 2), ("original", 3)])

    def test_managed_matrix_is_precommitted_before_randomized_arms_run(self):
        scenario = {"id": "managed", "matrix": [["gpt-6-sol", "medium"],
                                                    ["gpt-6-luna", "low"]]}
        timeline = []

        def create(suite_id, block_id, selected, arms, repetition, **_prompt_identity):
            timeline.append(("precommit", list(arms), repetition))
            return {"block_id": block_id,
                    "arms": [["gpt-6-luna", "low"], ["gpt-6-sol", "medium"]],
                    "commitment": "a" * 64, "assignment_probability": 0.5}

        def run(selected, model, effort, suite_id, block, arm_index,
                sequence_index, repetition_index, record_metrics):
            timeline.append(("run", model, effort, arm_index))
            return {"scenario_id": selected["id"], "product_pass": True,
                    "satisfaction_score": 100, "total_tokens": 10}

        with patch.object(sys, "argv", ["smoke_bench", "--managed", "--record-metrics"]), \
                patch.object(smoke_bench, "load_manifest", return_value={"scenarios": [scenario]}), \
                patch("model_host_launcher.ensure_proxy"), \
                patch("model_host_launcher.ensure_proxy_supervisor"), \
                patch("outcome_model.create_managed_benchmark_block", side_effect=create), \
                patch.object(smoke_bench, "run_managed_one", side_effect=run), \
                redirect_stdout(io.StringIO()):
            smoke_bench.main()
        self.assertEqual(timeline, [
            ("precommit", [("gpt-6-sol", "medium"), ("gpt-6-luna", "low")], 0),
            ("run", "gpt-6-luna", "low", 0),
            ("run", "gpt-6-sol", "medium", 1)])

    def test_managed_browser_pair_precommits_prompt_and_all_model_blocks_first(self):
        product = {"files": {"verify.py": "pass\n"}, "protected_files": ["verify.py"],
                   "verify": ["python3", "verify.py"], "expected_final": "ready",
                   "task_class": "routine", "comparison_id": "pair",
                   "matrix": [["gpt-6-luna", "low"], ["gpt-6-sol", "medium"]]}
        base = {**product, "id": "base", "prompt": "baseline", "prompt_author": "benchmark"}
        candidate = {**product, "id": "candidate", "prompt": "candidate",
                     "prompt_author": "chatgpt", "variant_of": "base"}
        timeline = []

        def create_prompt(suite_id, prompt_set_id, scenarios, repetition):
            timeline.append(("prompt", repetition))
            return {"prompt_set_id": prompt_set_id, "commitment": "p" * 64,
                    "prompts": [{"scenario_sha256": stable_digest(candidate)},
                                {"scenario_sha256": stable_digest(base)}]}

        def create_block(suite_id, block_id, scenario, arms, repetition, **identity):
            timeline.append(("block", scenario["id"], identity["prompt_arm_index"]))
            return {"block_id": block_id, "arms": [list(arm) for arm in arms],
                    "commitment": "b" * 64, "assignment_probability": 0.5,
                    "prompt_set_id": identity["prompt_set_id"],
                    "prompt_arm_index": identity["prompt_arm_index"]}

        def run(scenario, model, effort, suite_id, block, arm_index,
                sequence_index, repetition_index, record_metrics):
            timeline.append(("run", scenario["id"], arm_index))
            return {"scenario_id": scenario["id"], "product_pass": True,
                    "satisfaction_score": 100, "total_tokens": 10}

        with patch.object(sys, "argv", ["smoke_bench", "--managed", "--record-metrics"]), \
                patch.object(smoke_bench, "load_manifest",
                             return_value={"scenarios": [base, candidate]}), \
                patch("model_host_launcher.ensure_proxy"), \
                patch("model_host_launcher.ensure_proxy_supervisor"), \
                patch("outcome_model.create_managed_prompt_set", side_effect=create_prompt), \
                patch("outcome_model.create_managed_benchmark_block", side_effect=create_block), \
                patch.object(smoke_bench, "run_managed_one", side_effect=run), \
                redirect_stdout(io.StringIO()):
            smoke_bench.main()
        self.assertEqual(timeline[:3], [("prompt", 0), ("block", "candidate", 0),
                                        ("block", "base", 1)])
        self.assertTrue(all(item[0] == "run" for item in timeline[3:]))

    def test_managed_matrix_refuses_unrecorded_or_single_arm_runs(self):
        scenario = {"id": "managed", "matrix": [["gpt-6-luna", "low"]]}
        with patch.object(smoke_bench, "load_manifest", return_value={"scenarios": [scenario]}):
            with patch.object(sys, "argv", ["smoke_bench", "--managed"]), \
                    self.assertRaises(SystemExit):
                smoke_bench.main()
            with patch.object(sys, "argv", ["smoke_bench", "--managed", "--record-metrics"]), \
                    patch("model_host_launcher.ensure_proxy"), \
                    patch("model_host_launcher.ensure_proxy_supervisor"), \
                    self.assertRaises(SystemExit):
                smoke_bench.main()

    @staticmethod
    def benchmark_row(scenario: str, model: str, effort: str, tokens: int) -> dict:
        return {"event": "benchmark_result", "schema": "modellabs.benchmark-result.v2",
                "scenario_id": scenario, "scenario_sha256": "a" * 64,
                "model": model, "effort": effort, "product_pass": True,
                "model_provenance": "observed_per_request", "observed_model": model,
                "fixture_integrity": True, "codex_exit_code": 0,
                "verifier_exit_code": 0, "satisfaction_score": 100,
                "total_tokens": tokens, "recorded_at_ms": 2_000_000_000_000}

    def test_materialize_and_grade_real_product(self):
        scenario = {"files": {"product.json": "[1,2]", "verify.py": "import json; assert json.load(open('product.json')) == [1,2]"},
                    "protected_files": ["verify.py"],
                    "verify": ["python3", "verify.py"], "expected_final": '{"product":"ready"}'}
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            materialize(workspace, scenario)
            result = grade(workspace, scenario, '{"product":"ready"}',
                           {"input_tokens": 10, "output_tokens": 2, "reasoning_output_tokens": 1})
        self.assertTrue(result["product_pass"])
        self.assertEqual(result["satisfaction_score"], 100)
        self.assertEqual(result["total_tokens"], 12)

    def test_modified_verifier_and_agent_failure_cannot_pass(self):
        scenario = {"files": {"product.json": "[1,2]", "verify.py": "raise AssertionError('original')"},
                    "protected_files": ["verify.py"],
                    "verify": ["python3", "verify.py"], "expected_final": "ready"}
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            materialize(workspace, scenario)
            (workspace / "verify.py").write_text("pass\n", encoding="utf-8")
            result = grade(workspace, scenario, "ready", {"input_tokens": 10})
            self.assertFalse(result["fixture_integrity"])
            self.assertFalse(result["product_pass"])
            scenario["files"]["verify.py"] = "pass\n"
            materialize(workspace, scenario)
            failed = grade(workspace, scenario, "ready", {"input_tokens": 10}, 1)
            self.assertTrue(failed["fixture_integrity"])
            self.assertFalse(failed["product_pass"])

    def test_verifier_isolates_agent_written_code_from_parent_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside-marker"
            scenario = {"files": {"verify.py": "import generated\n"},
                        "protected_files": ["verify.py"],
                        "verify": ["python3", "verify.py"], "expected_final": "ready"}
            materialize(workspace, scenario)
            (workspace / "generated.py").write_text(
                "from pathlib import Path\ntry:\n Path(" + repr(str(outside))
                + ").write_text('unsafe')\nexcept OSError:\n pass\n", encoding="utf-8")
            result = grade(workspace, scenario, "ready", {"input_tokens": 10})
            self.assertTrue(result["product_pass"])
            self.assertFalse(outside.exists())

    def test_manifest_requires_protected_fixtures(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "smoke.json"
            manifest.write_text(json.dumps({"schema": "modellabs.smoke.v1",
                                            "scenarios": [{"files": {"verify.py": "pass"}}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protected fixture"):
                load_manifest(manifest)

    def test_prompt_variant_shares_exact_product_but_not_prompt(self):
        scenarios = {row["id"]: row for row in load_manifest(
            Path(__file__).resolve().parents[1] / "benchmarks/smoke.json")["scenarios"]}
        original = scenarios["simple_transform"]
        revised = scenarios["simple_transform_clarified"]
        self.assertEqual(stable_digest(product_definition(original)),
                         stable_digest(product_definition(revised)))
        self.assertNotEqual(stable_digest(original), stable_digest(revised))
        self.assertEqual(revised["variant_of"], original["id"])

    def test_parse_and_rank_puts_correctness_before_efficiency(self):
        output = '\n'.join((json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}}),
                            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 3}})))
        usage, final = parse_codex_jsonl(output)
        self.assertEqual((usage["input_tokens"], final), (20, "done"))
        rows = ranked([{"product_pass": False, "total_tokens": 1, "satisfaction_score": 0},
                       {"product_pass": True, "total_tokens": 100, "satisfaction_score": 100}])
        self.assertTrue(rows[0]["product_pass"])

    def test_preliminary_benchmark_is_shadow_only(self):
        choice = route("Implement a small validated parser.")
        evidence = [{**self.benchmark_row("routine-a", "gpt-5.6-sol", "medium", 50),
                     "task_class": "routine"},
                    {**self.benchmark_row("routine-a", "gpt-6-sol", "medium", 100),
                     "task_class": "routine"}]
        with patch.dict("os.environ", {"MODELLABS_ADAPTIVE_MODE": "enforce"}):
            result = adapt(choice, records=evidence, now_ms=2_000_000_000_000)
        self.assertEqual(result["model"], "gpt-6-sol")
        self.assertEqual(result["adaptive_recommendation"]["model"], "gpt-5.6-sol")
        self.assertEqual(result["adaptive_reason"], "shadow_benchmark_preliminary")

    def test_benchmark_requires_baseline_on_same_scenario(self):
        choice = route("Implement a small validated parser.")
        evidence = [{**self.benchmark_row("other", "gpt-5.6-sol", "medium", 10),
                     "task_class": "routine"}]
        result = adapt(choice, records=evidence, now_ms=2_000_000_000_000)
        self.assertNotIn("adaptive_recommendation", result)

    def test_benchmark_rejects_legacy_or_changed_scenario_definition(self):
        choice = route("Implement a small validated parser.")
        baseline = {**self.benchmark_row("same-id", "gpt-6-sol", "medium", 100),
                    "task_class": "routine"}
        cheaper = {**self.benchmark_row("same-id", "gpt-5.6-sol", "medium", 40),
                   "task_class": "routine", "scenario_sha256": "b" * 64}
        self.assertNotIn("adaptive_recommendation",
                         adapt(choice, records=[baseline, cheaper], now_ms=2_000_000_000_000))
        cheaper["scenario_sha256"] = baseline["scenario_sha256"]
        cheaper["schema"] = "modellabs.benchmark-result.v1"
        self.assertNotIn("adaptive_recommendation",
                         adapt(choice, records=[baseline, cheaper], now_ms=2_000_000_000_000))

    def test_requested_only_model_evidence_cannot_enforce_benchmark_route(self):
        choice = route("Investigate an intermittent race condition and fix it.")
        rows = []
        for model, effort, tokens in (("gpt-6-sol", "high", 120),
                                      ("gpt-6-astra", "medium", 80)):
            for scenario in ("one", "two", "two"):
                rows.append({**self.benchmark_row(scenario, model, effort, tokens),
                             "task_class": "difficult", "model_provenance": "cli_requested_only",
                             "observed_model": None})
        with patch.dict("os.environ", {"MODELLABS_ADAPTIVE_MODE": "enforce"}):
            result = adapt(choice, records=rows, now_ms=2_000_000_000_000)
        self.assertEqual((result["model"], result["effort"]), ("gpt-6-sol", "high"))
        self.assertEqual(result["adaptive_reason"], "shadow_benchmark_preliminary")

    def test_caller_claimed_execution_cannot_apply_after_two_scenarios(self):
        choice = route("Investigate an intermittent race condition and fix it.")
        rows = []
        for model, effort, tokens in (("gpt-6-sol", "high", 120),
                                      ("gpt-6-astra", "medium", 80)):
            for scenario in ("one", "two", "two"):
                rows.append({**self.benchmark_row(scenario, model, effort, tokens),
                             "task_class": "difficult"})
        with patch.dict("os.environ", {"MODELLABS_ADAPTIVE_MODE": "enforce"}):
            result = adapt(choice, records=rows, now_ms=2_000_000_000_000)
        self.assertEqual((result["model"], result["effort"]), ("gpt-6-sol", "high"))
        self.assertEqual(result["adaptive_recommendation"],
                         {"model": "gpt-6-astra", "effort": "medium"})
        self.assertFalse(result["adaptive_evidence"]["model_execution_verified"])
        self.assertEqual(result["adaptive_reason"], "shadow_benchmark_preliminary")


if __name__ == "__main__":
    unittest.main()
