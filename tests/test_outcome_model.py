import json
import hashlib
import tempfile
import unittest
from datetime import datetime, timedelta
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import outcome_model


class OutcomeModelTests(unittest.TestCase):
    def private_store(self, root: Path) -> ExitStack:
        stack = ExitStack()
        private = root / "learning"
        stack.enter_context(patch.object(outcome_model, "LEARNING_ROOT", private))
        stack.enter_context(patch.object(outcome_model, "KEY_PATH", private / "feature-key"))
        stack.enter_context(patch.object(outcome_model, "DB_PATH", private / "episodes.sqlite3"))
        stack.enter_context(patch.object(outcome_model, "MODEL_PATH", private / "review-model.json"))
        stack.enter_context(patch.object(outcome_model, "CODEX_MODEL_PATH", private / "codex-model.json"))
        stack.enter_context(patch.object(outcome_model, "ITERATION_MODEL_PATH", private / "iteration-model.json"))
        stack.enter_context(patch.object(outcome_model, "IMPROVEMENT_MODEL_PATH", private / "improvement-model.json"))
        stack.enter_context(patch.object(outcome_model, "REVIEW_EVAL_PATH", private / "review-evaluation-checkpoint.json"))
        stack.enter_context(patch.object(outcome_model, "ITERATION_EVAL_PATH", private / "iteration-evaluation-checkpoint.json"))
        stack.enter_context(patch.object(outcome_model, "IMPROVEMENT_EVAL_PATH", private / "improvement-evaluation-checkpoint.json"))
        stack.enter_context(patch.object(outcome_model, "CODEX_EVAL_PATH", private / "codex-evaluation-checkpoint.json"))
        return stack

    def test_codex_validation_checkpoint_freezes_supported_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                key = outcome_model._key()

                def add_turn(number, observed_at_ms, model):
                    features = json.dumps(outcome_model._features(f"prompt {number}", "", 1, key))
                    with outcome_model._connect() as connection:
                        connection.execute("""INSERT INTO codex_turns
                            (thread_key,turn_key,group_id,accepted_at_ms,prompt_digest,features,
                             selected_model,effort,task_class,quality_score,verification,total_tokens,
                             result_digest)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (f"thread-{number}", f"turn-{number}", f"group-{number}", observed_at_ms,
                             f"prompt-{number}", features, model, "low", "simple",
                             95 if number % 2 else 40, "passed" if number % 2 else "failed",
                             100 + number, f"result-{number}"))

                start_ms = int(datetime.fromisoformat("2026-09-01T00:00:00+00:00").timestamp() * 1000)
                for number in range(40):
                    add_turn(number, start_ms + number * 1000,
                             "gpt-6-luna" if number % 2 else "gpt-6-sol")
                with patch.object(outcome_model, "sync_codex_grades", return_value={}), \
                        patch.object(outcome_model, "sync_session_grades", return_value={}):
                    first = outcome_model.train_codex_model()
                    self.assertEqual(first["prospective_holdout_turns"], 0)
                    frozen = outcome_model.CODEX_EVAL_PATH.read_bytes()
                    checkpoint = json.loads(frozen)
                    self.assertEqual(len(checkpoint["eligible_arms"]), 2)
                    future_ms = int((datetime.fromisoformat(checkpoint["created_at"])
                                     + timedelta(minutes=1)).timestamp() * 1000)
                    for number in range(40, 60):
                        add_turn(number, future_ms + number * 1000,
                                 "gpt-6-luna" if number % 2 else "gpt-6-sol")
                    later = outcome_model.train_codex_model()
                    self.assertEqual(later["prospective_holdout_turns"], 20)
                    self.assertTrue(later["prospective_comparable_arms"])
                    self.assertEqual(outcome_model.CODEX_EVAL_PATH.read_bytes(), frozen)

    def test_review_validation_checkpoint_excludes_future_and_backfilled_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                key = outcome_model._key()

                def add_episode(number, group, submitted_at):
                    features = json.dumps(outcome_model._features("private goal", f"candidate {number}", 1, key))
                    with outcome_model._connect() as connection:
                        connection.execute("""INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                           (f"episode-{number}", group, "auracall_pro_guard", submitted_at,
                                            1, "review_goal", f"prompt-{number}", f"revision-{number}",
                                            features, None, None, None, 95 if number % 2 else 45,
                                            number % 2, "nonce_bound_pro_guard", f"response-{number}",
                                            f"guard-{number}", outcome_model.FEATURE_VERSION))

                for number in range(24):
                    add_episode(number, f"development-{number}", f"2026-09-01T00:{number:02d}:00+00:00")
                first = outcome_model.train_review_model()
                self.assertEqual(first["prospective_holdout_episodes"], 0)
                initial_model = json.loads(outcome_model.MODEL_PATH.read_text())
                frozen = outcome_model.REVIEW_EVAL_PATH.read_bytes()
                checkpoint = json.loads(frozen)
                start = datetime.fromisoformat(checkpoint["created_at"])
                for number in range(24, 44):
                    add_episode(number, f"future-{number}",
                                (start + timedelta(seconds=number)).isoformat())
                second = outcome_model.train_review_model()
                self.assertEqual(second["prospective_holdout_episodes"], 20)
                adapted_model = json.loads(outcome_model.MODEL_PATH.read_text())
                self.assertNotEqual((initial_model["weights"], initial_model["bias"]),
                                    (adapted_model["weights"], adapted_model["bias"]))
                self.assertEqual(second["prospective_holdout_task_groups"], 20)
                self.assertEqual(second["prospective_checkpoint_development_episodes"], 24)
                self.assertEqual(outcome_model.REVIEW_EVAL_PATH.read_bytes(), frozen)
                with outcome_model._connect() as connection:
                    for episode_id, submitted_at, parent in (
                            ("episode-0", "2026-09-01T00:00:00+00:00", None),
                            ("episode-24", (start + timedelta(seconds=24)).isoformat(), "episode-0")):
                        connection.execute("""INSERT INTO iteration_traces
                            (episode_id,root_key,parent_episode_id,origin_prompt_digest,
                             generation_prompt_digest,prompt_author,prompt_features,
                             submitted_at,passed) VALUES (?,?,?,?,?,?,?,?,?)""",
                            (episode_id, "shared-verified-root", parent, "origin", episode_id,
                             "user", "{}", submitted_at, 0))
                lineage_corrected = outcome_model.train_review_model()
                self.assertEqual(lineage_corrected["task_groups"], 43)
                self.assertEqual(lineage_corrected["prospective_holdout_episodes"], 19)
                self.assertEqual(lineage_corrected["prospective_holdout_task_groups"], 19)
                self.assertEqual(outcome_model.REVIEW_EVAL_PATH.read_bytes(), frozen)
                add_episode(44, "late-backfill", "2026-09-02T00:00:00+00:00")
                third = outcome_model.train_review_model()
                self.assertEqual(third["prospective_holdout_episodes"], 19)
                self.assertEqual(outcome_model.REVIEW_EVAL_PATH.read_bytes(), frozen)

    def test_guard_import_binds_response_nonce_and_stores_no_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard_root, runs_root = root / "guards", root / "runs"
            guard_root.mkdir()
            response_id = "resp_fixture123"
            run_dir = runs_root / response_id
            run_dir.mkdir(parents=True)
            guard = {"status": "completed", "response_id": response_id, "guard_id": "fixture-r1",
                     "nonce": "private-nonce", "round": 1, "submitted_at": "2026-09-23T00:00:00Z",
                     "submission_fingerprint": "fingerprint",
                     "evaluation": {"valid": True, "nonce_matched": True, "passed": True, "score": 96}}
            record = {"runId": response_id, "bundle": {"run": {"id": response_id, "status": "succeeded",
                      "initialInputs": {"model": "gpt-5.2", "instructions":
                          "Review the supplied artifact against this goal:\nsecret original prompt\n\nFRESHNESS_NONCE: private-nonce",
                          "requestInput": "private revised candidate",
                          "metadata": {"workflow": "codex-pro-guard", "guard_id": "fixture-r1",
                                       "guard_nonce": "private-nonce", "round": 1,
                                       "submission_fingerprint": "fingerprint"}}}}}
            (guard_root / "fixture.json").write_text(json.dumps(guard), encoding="utf-8")
            (run_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")
            origin = root / "origin.txt"
            generation = root / "generation.txt"
            origin.write_text("private user starting prompt", encoding="utf-8")
            generation.write_text("private browser adjusted prompt", encoding="utf-8")
            guard["source_files"] = {"origin_prompt": {"path": str(origin),
                                  "sha256": hashlib.sha256(origin.read_bytes()).hexdigest()},
                                    "generation_prompt": {"path": str(generation),
                                  "sha256": hashlib.sha256(generation.read_bytes()).hexdigest()}}
            guard["learning_trace"] = {"root_guard_id": "fixture-r1", "parent_guard_id": None,
                                       "parent_response_id": None, "prompt_author": "chatgpt",
                                       "origin_prompt_sha256": guard["source_files"]["origin_prompt"]["sha256"],
                                       "generation_prompt_sha256": guard["source_files"]["generation_prompt"]["sha256"]}
            guard["learning_trace_digest"] = hashlib.sha256(json.dumps(guard["learning_trace"],
                                                              sort_keys=True,
                                                              separators=(",", ":")).encode()).hexdigest()
            record["bundle"]["run"]["initialInputs"]["metadata"]["learning_trace_digest"] = guard["learning_trace_digest"]
            (run_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")
            guard["verdict"] = {"summary": "private reviewer result", "blocking_findings": [],
                                "tests_or_checks_required": []}
            (guard_root / "fixture.json").write_text(json.dumps(guard), encoding="utf-8")
            with self.private_store(root):
                first = outcome_model.import_auracall(guard_root, runs_root)
                self.assertEqual(first["imported"], 1)
                self.assertEqual(first["trace_imported"], 1)
                self.assertEqual(outcome_model.import_auracall(guard_root, runs_root, "fixture")["unchanged"], 1)
                with self.assertRaises(ValueError):
                    outcome_model.import_auracall(guard_root, runs_root, "../fixture")
                self.assertEqual(outcome_model.import_auracall(guard_root, runs_root)["unchanged"], 1)
                raw = outcome_model.DB_PATH.read_bytes()
                self.assertNotIn(b"secret original prompt", raw)
                self.assertNotIn(b"private revised candidate", raw)
                self.assertNotIn(b"private browser adjusted prompt", raw)
                self.assertNotIn(b"private reviewer result", raw)
                self.assertEqual(outcome_model.status()["sources"]["auracall_pro_guard"]["passing"], 1)
                self.assertEqual(outcome_model.status()["iterations"]["linked_rounds"], 1)
                self.assertEqual(outcome_model.train_iteration_model()["status"], "insufficient_linked_rounds")
                guard.pop("learning_trace")
                guard.pop("learning_trace_digest")
                (guard_root / "fixture.json").write_text(json.dumps(guard), encoding="utf-8")
                self.assertEqual(outcome_model.import_auracall(guard_root, runs_root)["trace_missing"], 1)
                guard["nonce"] = "different-nonce"
                (guard_root / "fixture.json").write_text(json.dumps(guard), encoding="utf-8")
                self.assertEqual(outcome_model.import_auracall(guard_root, runs_root)["ineligible"], 1)

    def test_codex_capture_joins_only_explicit_grade_and_exact_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.private_store(root):
                prepared = outcome_model.prepare_codex_prompt("private user prompt")
                outcome_model.capture_codex_turn("thread-1", "turn-1", prepared, "gpt-6-sol", "medium", "routine")
                metrics = root / "metrics.jsonl"
                metrics.write_text("\n".join(json.dumps(row) for row in [
                    {"event": "outcome_observation", "thread_id": "thread-1", "turn_id": "turn-1", "outcome": "verified"},
                    {"event": "quality_grade", "thread_id": "thread-1", "source_turn_id": "turn-1",
                     "source": "explicit", "quality_score": 95, "verification": "passed"},
                    {"event": "turn_completed", "thread_id": "thread-1", "turn_id": "turn-1",
                     "status": "completed"},
                    {"event": "turn_usage", "thread_id": "thread-1", "turn_id": "turn-1",
                     "model": "gpt-6-sol", "effort": "medium",
                     "source": "proxy_thread_usage_delta", "usage": {"totalTokens": 123}},
                ]) + "\n", encoding="utf-8")
                self.assertEqual(outcome_model.sync_codex_grades(metrics), {"graded": 1, "usage_matched": 1})
                self.assertEqual(outcome_model.sync_codex_grades(metrics), {"graded": 0, "usage_matched": 0})
                self.assertEqual(outcome_model.train_codex_model()["graded_with_usage"], 0)
                self.assertTrue(outcome_model.capture_codex_result("thread-1", "turn-1",
                                                                  "private Codex final result"))
                self.assertEqual(outcome_model.status()["codex"],
                                 {"accepted_turns": 1, "thread_groups": 1,
                                  "graded_turns": 1, "exact_usage_turns": 1,
                                  "result_captured_turns": 1,
                                  "complete_prompt_result_usage_grade_turns": 1})
                self.assertNotIn(b"private user prompt", outcome_model.DB_PATH.read_bytes())
                self.assertNotIn(b"private Codex final result", outcome_model.DB_PATH.read_bytes())
                report = outcome_model.train_codex_model()
                self.assertEqual(report["status"], "insufficient_comparable_outcomes")
                self.assertEqual(report["graded_with_usage"], 1)

    def test_managed_usage_requires_completed_turn_and_proxy_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.private_store(root):
                prepared = outcome_model.prepare_codex_prompt("private prompt")
                outcome_model.capture_codex_turn("thread", "turn", prepared,
                                                 "gpt-6-sol", "medium", "routine")
                metrics = root / "metrics.jsonl"
                accepted = {"event": "turn_completed", "thread_id": "thread",
                            "turn_id": "turn", "status": "failed"}
                usage = {"event": "turn_usage", "thread_id": "thread", "turn_id": "turn",
                         "model": "gpt-6-sol", "effort": "medium",
                         "source": "legacy_raw", "usage": {"totalTokens": 100}}

                def write_metrics():
                    metrics.write_text("\n".join(json.dumps(row) for row in (accepted, usage)) + "\n")

                write_metrics()
                self.assertEqual(outcome_model.sync_codex_grades(metrics)["usage_matched"], 0)
                accepted["status"] = "completed"
                write_metrics()
                self.assertEqual(outcome_model.sync_codex_grades(metrics)["usage_matched"], 0)
                usage["source"] = "proxy_thread_usage_delta"
                write_metrics()
                self.assertEqual(outcome_model.sync_codex_grades(metrics)["usage_matched"], 1)

    def test_linked_round_import_uses_exact_parent_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guards, runs = root / "guards", root / "runs"
            guards.mkdir()
            origin, first_prompt, second_prompt = root / "origin", root / "first", root / "second"
            for path, value in ((origin, "user prompt"), (first_prompt, "first generation prompt"),
                                (second_prompt, "revised browser prompt")):
                path.write_text(value, encoding="utf-8")

            def source(path):
                return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

            def make_round(guard_id, response_id, round_number, generation, parent=None):
                sources = {"origin_prompt": source(origin), "generation_prompt": source(generation)}
                trace = {"root_guard_id": "first", "parent_guard_id": parent,
                         "parent_response_id": f"resp_{parent}" if parent else None,
                         "prompt_author": "chatgpt" if parent else "user",
                         "origin_prompt_sha256": sources["origin_prompt"]["sha256"],
                         "generation_prompt_sha256": sources["generation_prompt"]["sha256"]}
                digest = hashlib.sha256(json.dumps(trace, sort_keys=True,
                                                   separators=(",", ":")).encode()).hexdigest()
                nonce = f"nonce-{guard_id}"
                state = {"status": "completed", "response_id": response_id, "guard_id": guard_id,
                         "nonce": nonce, "round": round_number,
                         "submitted_at": f"2026-09-23T00:0{round_number}:00Z",
                         "submission_fingerprint": f"fingerprint-{guard_id}",
                         "learning_trace": trace, "learning_trace_digest": digest,
                         "source_files": sources,
                         "verdict": {"summary": f"feedback from {guard_id}",
                                     "suggested_next_prompt": "revised browser prompt" if not parent else None,
                                     "blocking_findings": [], "tests_or_checks_required": []},
                         "evaluation": {"valid": True, "nonce_matched": True,
                                        "passed": bool(parent), "score": 95 if parent else 60}}
                metadata = {"workflow": "codex-pro-guard", "guard_id": guard_id,
                            "guard_nonce": nonce, "round": round_number,
                            "submission_fingerprint": f"fingerprint-{guard_id}",
                            "learning_trace_digest": digest}
                record = {"runId": response_id, "bundle": {"run": {"id": response_id,
                          "status": "succeeded", "initialInputs": {"model": "gpt-5.2",
                          "instructions": f"Review the supplied artifact against this goal:\nshared goal\n\nFRESHNESS_NONCE: {nonce}",
                          "requestInput": f"artifact {guard_id}", "metadata": metadata}}}}
                (guards / f"{guard_id}.json").write_text(json.dumps(state), encoding="utf-8")
                run_dir = runs / response_id
                run_dir.mkdir(parents=True)
                (run_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")

            make_round("first", "resp_first", 1, first_prompt)
            make_round("second", "resp_second", 2, second_prompt, "first")
            with self.private_store(root):
                prepared = outcome_model.prepare_codex_prompt("first generation prompt")
                outcome_model.capture_codex_turn("thread-first", "turn-first", prepared,
                                                 "gpt-6-sol", "medium", "routine")
                outcome_model.capture_codex_result("thread-first", "turn-first", "artifact first")
                imported = outcome_model.import_auracall(guards, runs)
                self.assertEqual(imported["trace_imported"], 2)
                self.assertEqual(imported["codex_link_imported"], 1)
                self.assertEqual(outcome_model.status()["iterations"]["revisions_with_parent_feedback"], 1)
                self.assertEqual(outcome_model.status()["iterations"]["adopted_browser_suggestions"], 1)
                with outcome_model._connect() as connection:
                    parent_context = connection.execute("""SELECT parent_result_features,parent_quality_score
                                           FROM iteration_traces WHERE parent_episode_id IS NOT NULL""").fetchone()
                self.assertEqual(parent_context[1], 60)
                self.assertEqual(json.loads(parent_context[0]),
                                 outcome_model._features("artifact first", "", 1, outcome_model._key()))
                self.assertEqual(outcome_model.status()["browser_codex_links"]
                                 ["exact_prompt_result_review_links"], 1)
                self.assertEqual(outcome_model.import_auracall(guards, runs, "first")
                                 ["codex_link_unchanged"], 1)
                outcome_model.capture_codex_turn("thread-other", "turn-other", prepared,
                                                 "gpt-6-sol", "medium", "routine")
                outcome_model.capture_codex_result("thread-other", "turn-other", "different answer")
                self.assertEqual(outcome_model.import_auracall(guards, runs, "first")
                                 ["codex_link_unchanged"], 1)
                outcome_model.capture_codex_turn("thread-duplicate", "turn-duplicate", prepared,
                                                 "gpt-6-sol", "medium", "routine")
                outcome_model.capture_codex_result("thread-duplicate", "turn-duplicate", "artifact first")
                self.assertEqual(outcome_model.import_auracall(guards, runs, "first")
                                 ["codex_link_ambiguous"], 1)
                self.assertEqual(outcome_model.status()["browser_codex_links"]
                                 ["exact_prompt_result_review_links"], 0)
                with outcome_model._connect() as connection:
                    child = connection.execute("SELECT parent_episode_id,parent_feedback_features FROM iteration_traces WHERE parent_episode_id IS NOT NULL").fetchone()
                self.assertIsNotNone(child[1])
                self.assertNotIn(b"feedback from first", outcome_model.DB_PATH.read_bytes())
                self.assertNotIn(b"artifact first", outcome_model.DB_PATH.read_bytes())
                with outcome_model._connect() as connection:
                    connection.execute("""UPDATE iteration_traces SET parent_result_features=NULL,
                                        parent_quality_score=NULL WHERE parent_episode_id IS NOT NULL""")
                self.assertEqual(outcome_model.import_auracall(guards, runs, "second")["trace_unchanged"], 1)
                with outcome_model._connect() as connection:
                    recovered = connection.execute("""SELECT parent_result_features,parent_quality_score
                                           FROM iteration_traces WHERE parent_episode_id IS NOT NULL""").fetchone()
                self.assertEqual(recovered, parent_context)

    def test_iteration_model_trains_and_predicts_without_claiming_causality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.private_store(root):
                key = outcome_model._key()
                with outcome_model._connect() as connection:
                    for number in range(40):
                        prompt = outcome_model._features("task context", f"candidate {number}", 1, key)
                        connection.execute("""INSERT INTO iteration_traces
                                           (episode_id,root_key,parent_episode_id,origin_prompt_digest,
                                            generation_prompt_digest,prompt_author,prompt_features,
                                            parent_feedback_features,result_feedback_features,submitted_at,passed,
                                            adopted_parent_suggestion)
                                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                                           (f"episode-{number}", f"root-{number}", None,
                                            f"origin-{number}", f"generation-{number}", "user",
                                            json.dumps(prompt), None, None,
                                            f"2026-09-23T00:{number:02d}:00Z", number % 2, 0))
                    connection.execute("""INSERT INTO iteration_traces
                                       (episode_id,root_key,parent_episode_id,origin_prompt_digest,
                                        generation_prompt_digest,prompt_author,prompt_features,
                                        parent_feedback_features,result_feedback_features,
                                        submitted_at,passed,adopted_parent_suggestion,
                                        parent_result_features,parent_quality_score)
                                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                       ("episode-0-revision", "root-0", "episode-0", "origin-0",
                                        "generation-0-revision", "chatgpt",
                                        json.dumps(outcome_model._features("task context", "browser revision", 2, key)),
                                        None, None, "2026-09-23T01:00:00Z", 1, 1,
                                        json.dumps(outcome_model._features("prior answer", "", 1, key)), 50))
                insufficient = outcome_model.train_iteration_model()
                self.assertEqual(insufficient["status"], "insufficient_context_complete_revisions")
                self.assertEqual(insufficient["context_complete_revision_task_groups"], 1)
                self.assertFalse(outcome_model.ITERATION_MODEL_PATH.exists())
                with outcome_model._connect() as connection:
                    for number in (*range(1, 18), 32, 33):
                        connection.execute("""INSERT INTO iteration_traces
                                           (episode_id,root_key,parent_episode_id,origin_prompt_digest,
                                            generation_prompt_digest,prompt_author,prompt_features,
                                            parent_feedback_features,result_feedback_features,
                                            submitted_at,passed,adopted_parent_suggestion,
                                            parent_result_features,parent_quality_score)
                                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                           (f"episode-{number}-revision", f"root-{number}",
                                            f"episode-{number}", f"origin-{number}",
                                            f"generation-{number}-revision", "chatgpt",
                                            json.dumps(outcome_model._features("task context", "browser revision", 2, key)),
                                            None, None, f"2026-09-23T01:{number:02d}:00Z",
                                            number % 2, number % 2,
                                            json.dumps(outcome_model._features("prior answer", "", 1, key)), 50))
                prepared = outcome_model.prepare_codex_prompt("exact parent generation prompt")
                outcome_model.capture_codex_turn("linked-thread", "linked-turn", prepared,
                                                 "gpt-6-sol", "medium", "routine")
                outcome_model.capture_codex_result("linked-thread", "linked-turn", "actual Codex final answer")
                with outcome_model._connect() as connection:
                    self.assertEqual(outcome_model._link_review_to_codex(
                        connection, "episode-0", prepared["prompt_digest"],
                        outcome_model._digest(key, "actual Codex final answer")), "imported")
                report = outcome_model.train_iteration_model()
                self.assertEqual(report["trained_rounds"], 60)
                self.assertEqual(report["context_complete_revision_task_groups"], 20)
                self.assertEqual(report["linked_parent_codex_results"], 1)
                self.assertGreater(len(json.loads(outcome_model.ITERATION_MODEL_PATH.read_text())["weights"]),
                                   4 * (outcome_model.FEATURE_COUNT + 3) + 5)
                self.assertEqual(report["holdout_rounds"], 10)
                self.assertEqual(report["training_context_complete_revision_task_groups"], 18)
                self.assertEqual(report["holdout_context_complete_revision_task_groups"], 2)
                self.assertEqual(report["prospective_holdout_rounds"], 0)
                self.assertEqual(len(json.loads(outcome_model.ITERATION_EVAL_PATH.read_text())
                                     ["development_groups"]), 40)
                frozen = outcome_model.ITERATION_EVAL_PATH.read_bytes()
                start = datetime.fromisoformat(json.loads(frozen)["created_at"])
                with outcome_model._connect() as connection:
                    for number in range(40, 60):
                        prompt = outcome_model._features("task context", f"candidate {number}", 1, key)
                        connection.execute("""INSERT INTO iteration_traces
                                           (episode_id,root_key,parent_episode_id,origin_prompt_digest,
                                            generation_prompt_digest,prompt_author,prompt_features,
                                            parent_feedback_features,result_feedback_features,submitted_at,passed,
                                            adopted_parent_suggestion)
                                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                                           (f"episode-{number}", f"root-{number}", None,
                                            f"origin-{number}", f"generation-{number}", "user",
                                            json.dumps(prompt), None, None,
                                            (start + timedelta(seconds=number)).isoformat(), number % 2, 0))
                later = outcome_model.train_iteration_model()
                self.assertEqual(later["prospective_holdout_rounds"], 20)
                self.assertEqual(later["prospective_context_complete_revision_task_groups"], 0)
                self.assertFalse(later["validated_for_shadow"])
                self.assertEqual(outcome_model.ITERATION_EVAL_PATH.read_bytes(), frozen)
                with outcome_model._connect() as connection:
                    for number in range(40, 48):
                        connection.execute("""INSERT INTO iteration_traces
                                           (episode_id,root_key,parent_episode_id,origin_prompt_digest,
                                            generation_prompt_digest,prompt_author,prompt_features,
                                            parent_feedback_features,result_feedback_features,
                                            submitted_at,passed,adopted_parent_suggestion,
                                            parent_result_features,parent_quality_score)
                                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                           (f"episode-{number}-revision", f"root-{number}",
                                            f"episode-{number}", f"origin-{number}",
                                            f"generation-{number}-revision", "chatgpt",
                                            json.dumps(outcome_model._features("task context", "browser revision", 2, key)),
                                            None, None, (start + timedelta(seconds=number + 100)).isoformat(),
                                            number % 2, 1,
                                            json.dumps(outcome_model._features("prior answer", "", 1, key)), 50))
                revised_later = outcome_model.train_iteration_model()
                self.assertEqual(revised_later["prospective_context_complete_revision_task_groups"], 8)
                self.assertEqual(outcome_model.ITERATION_EVAL_PATH.read_bytes(), frozen)
                prediction = outcome_model.predict_iteration("task context", "candidate 41", "user", 1)
                self.assertFalse(prediction["causal_prompt_comparison"])
                self.assertIn("predicted_review_pass_probability", prediction)
                with_feedback = outcome_model.predict_iteration("task context", "candidate 42",
                                                                "chatgpt", 2, "reviewer fixes")
                self.assertIn("predicted_review_pass_probability", with_feedback)
                with_result = outcome_model.predict_iteration("task context", "candidate 42",
                                                               "chatgpt", 2, "reviewer fixes",
                                                               "earlier answer", 65,
                                                               "candidate 42")
                self.assertTrue(with_result["parent_result_supplied"])
                self.assertTrue(with_result["parent_quality_score_supplied"])
                self.assertTrue(with_result["adopted_parent_suggestion"])
                with_codex_result = outcome_model.predict_iteration("task context", "candidate 42",
                                                                     "chatgpt", 2, "reviewer fixes",
                                                                     "reviewed evidence packet", 65,
                                                                     "candidate 42", "actual Codex final answer")
                self.assertTrue(with_codex_result["parent_codex_result_supplied"])
                self.assertEqual(with_codex_result["linked_parent_codex_results_in_training"], 1)
                other_suggestion = outcome_model.predict_iteration("task context", "candidate 42",
                                                                   "chatgpt", 2,
                                                                   parent_suggestion="different candidate")
                self.assertFalse(other_suggestion["adopted_parent_suggestion"])
                prompt_features = outcome_model._features("task context", "candidate 42", 2, key)
                adopted_vector = outcome_model._iteration_vector(prompt_features, None,
                                                                  "chatgpt", adopted_parent_suggestion=True)
                self.assertIn(str(3 * (outcome_model.FEATURE_COUNT + 3) + 5), adopted_vector)
                with self.assertRaises(ValueError):
                    outcome_model.predict_iteration("task context", "candidate 42",
                                                    "user", 1, parent_result="earlier answer")
                with self.assertRaises(ValueError):
                    outcome_model.predict_iteration("task context", "candidate 42",
                                                    "user", 1, parent_suggestion="candidate 42")

    def test_revision_improvement_model_uses_parent_score_and_future_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                key = outcome_model._key()

                def add_pair(number, child_at):
                    root = f"root-{number}"
                    first_at = (child_at - timedelta(seconds=1)).isoformat()
                    child_at_text = child_at.isoformat()
                    score = (95 if number % 4 == 1 else 80) if number % 2 else 30
                    prompt = json.dumps(outcome_model._features("original task", f"revision {number}", 2, key))
                    parent_result = json.dumps(outcome_model._features("previous answer", "", 1, key))
                    with outcome_model._connect() as connection:
                        for suffix, at, round_number, grade in (
                                ("first", first_at, 1, 50),
                                ("revision", child_at_text, 2, score)):
                            connection.execute("""INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                               (f"episode-{number}-{suffix}", root, "auracall_pro_guard", at,
                                                round_number, "review_goal", f"prompt-{number}",
                                                f"result-{number}-{suffix}", prompt, None, None, None,
                                                grade, int(grade >= 90), "nonce_bound_pro_guard",
                                                f"response-{number}-{suffix}", f"guard-{number}-{suffix}",
                                                outcome_model.FEATURE_VERSION))
                        connection.execute("""INSERT INTO iteration_traces
                                           (episode_id,root_key,parent_episode_id,origin_prompt_digest,
                                            generation_prompt_digest,prompt_author,prompt_features,
                                            submitted_at,passed)
                                           VALUES (?,?,?,?,?,?,?,?,?)""",
                                           (f"episode-{number}-first", root, None, f"origin-{number}",
                                            f"generation-{number}-first", "user", prompt, first_at, 0))
                        connection.execute("""INSERT INTO iteration_traces
                                           (episode_id,root_key,parent_episode_id,origin_prompt_digest,
                                            generation_prompt_digest,prompt_author,prompt_features,
                                            submitted_at,passed,adopted_parent_suggestion,
                                            parent_result_features,parent_quality_score)
                                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                                           (f"episode-{number}-revision", root,
                                            f"episode-{number}-first", f"origin-{number}",
                                            f"generation-{number}-revision", "chatgpt", prompt,
                                            child_at_text, int(score >= 90), 1, parent_result, 50))

                base = datetime.fromisoformat("2026-09-01T00:00:00+00:00")
                for number in range(24):
                    add_pair(number, base + timedelta(minutes=number))
                prepared = outcome_model.prepare_codex_prompt("exact earlier prompt")
                outcome_model.capture_codex_turn("improvement-thread", "improvement-turn", prepared,
                                                 "gpt-6-sol", "medium", "routine")
                outcome_model.capture_codex_result("improvement-thread", "improvement-turn",
                                                   "actual earlier Codex answer")
                with outcome_model._connect() as connection:
                    self.assertEqual(outcome_model._link_review_to_codex(
                        connection, "episode-0-first", prepared["prompt_digest"],
                        outcome_model._digest(key, "actual earlier Codex answer")), "imported")
                self.assertEqual(outcome_model.train_iteration_model()["trained_rounds"], 48)
                first = outcome_model.train_improvement_model()
                self.assertEqual(first["scored_revisions"], 24)
                self.assertEqual(first["linked_parent_codex_results"], 1)
                self.assertEqual(first["prospective_holdout_revisions"], 0)
                frozen = outcome_model.IMPROVEMENT_EVAL_PATH.read_bytes()
                self.assertEqual(json.loads(frozen)["baseline_rate"], 0.5)
                start = datetime.fromisoformat(json.loads(frozen)["created_at"])
                late_at = (start + timedelta(seconds=1)).isoformat()
                with outcome_model._connect() as connection:
                    connection.execute("""INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                       ("episode-0-late", "root-0", "auracall_pro_guard", late_at,
                                        3, "review_goal", "prompt-0", "result-0-late", "{}",
                                        None, None, None, 70, 0, "nonce_bound_pro_guard",
                                        "response-0-late", "guard-0-late", outcome_model.FEATURE_VERSION))
                    connection.execute("""INSERT INTO iteration_traces
                                       (episode_id,root_key,parent_episode_id,origin_prompt_digest,
                                        generation_prompt_digest,prompt_author,prompt_features,
                                        submitted_at,passed,parent_result_features,parent_quality_score)
                                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                                       ("episode-0-late", "root-0", "episode-0-revision", "origin-0",
                                        "generation-0-late", "codex",
                                        json.dumps(outcome_model._features("original task", "late revision", 3, key)),
                                        late_at, 0,
                                        json.dumps(outcome_model._features("previous answer", "", 1, key)), 30))
                self.assertEqual(outcome_model.train_improvement_model()["prospective_holdout_revisions"], 0)
                for number in range(24, 44):
                    add_pair(number, start + timedelta(minutes=number))
                later = outcome_model.train_improvement_model()
                self.assertEqual(later["prospective_holdout_revisions"], 20)
                self.assertEqual(later["prospective_holdout_task_groups"], 20)
                self.assertEqual(outcome_model.IMPROVEMENT_EVAL_PATH.read_bytes(), frozen)
                predicted = outcome_model.predict_iteration(
                    "original task", "revision 45", "chatgpt", 2,
                    parent_result="previous answer", parent_quality_score=50)
                self.assertIsInstance(predicted["predicted_score_improvement_probability"], float)
                self.assertFalse(predicted["causal_prompt_comparison"])
                without_parent = outcome_model.predict_iteration("original task", "revision 45", "chatgpt", 2)
                self.assertIsNone(without_parent["predicted_score_improvement_probability"])

    def test_standalone_session_import_keeps_prompt_result_ungraded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = root / "sessions"
            sessions.mkdir()
            thread_id = "01234567-89ab-cdef-0123-456789abcdef"
            turn_id = "11111111-2222-3333-4444-555555555555"
            rows = [
                {"type": "session_meta", "payload": {"id": thread_id}},
                {"type": "event_msg", "payload": {"type": "task_started", "turn_id": turn_id}},
                {"type": "response_item", "payload": {"type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": "<codex_internal_context>not the user's prompt"}]}},
                {"type": "response_item", "payload": {"type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": "private actual user prompt"}]}},
                {"type": "turn_context", "payload": {"turn_id": turn_id, "model": "gpt-6-sol",
                    "effort": "medium"}},
                {"type": "token_usage_record", "payload": {"turn_id": turn_id,
                    "turn_token_usage": {"total_tokens": 321}}},
                {"type": "event_msg", "timestamp": "2026-09-23T00:01:00Z", "payload": {
                    "type": "task_complete", "turn_id": turn_id,
                    "last_agent_message": "private final answer", "completed_at": 1780274115}},
            ]
            (sessions / f"rollout-example-{thread_id}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            with self.private_store(root):
                first = outcome_model.import_codex_sessions(sessions, thread_id)
                self.assertEqual(first["imported"], 1)
                self.assertEqual(outcome_model.import_codex_sessions(sessions, thread_id)["unchanged"], 1)
                summary = outcome_model.status()["standalone_codex"]
                self.assertEqual(summary["completed_prompt_result_pairs"], 1)
                self.assertEqual(summary["reported_usage_pairs"], 1)
                self.assertEqual(summary["complete_explicit_grade_exact_usage"], 0)
                raw = outcome_model.DB_PATH.read_bytes()
                self.assertNotIn(b"private actual user prompt", raw)
                self.assertNotIn(b"private final answer", raw)
                rows.insert(4, {"type": "response_item", "payload": {"type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": "later user text"}]}})
                (sessions / f"rollout-example-{thread_id}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
                self.assertEqual(outcome_model.import_codex_sessions(sessions, thread_id)["turn_ineligible"], 1)
                rows.pop(4)
                rows.insert(3, {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "earlier inference"}]}})
                (sessions / f"rollout-example-{thread_id}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
                self.assertEqual(outcome_model.import_codex_sessions(sessions, thread_id)["turn_ineligible"], 1)

    def test_retrospective_grade_join_requires_exact_identity_and_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = root / "sessions"
            sessions.mkdir()
            thread_id = "01234567-89ab-cdef-0123-456789abcdef"
            turn_id = "11111111-2222-3333-4444-555555555555"
            rows = [
                {"type": "session_meta", "payload": {"id": thread_id}},
                {"type": "event_msg", "payload": {"type": "task_started", "turn_id": turn_id}},
                {"type": "response_item", "payload": {"type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": "private graded prompt"}]}},
                {"type": "turn_context", "payload": {"turn_id": turn_id, "model": "gpt-6-luna",
                    "effort": "low"}},
                {"type": "token_usage_record", "payload": {"turn_id": turn_id,
                    "turn_token_usage": {"total_tokens": 321}}},
                {"type": "event_msg", "payload": {"type": "task_complete", "turn_id": turn_id,
                    "last_agent_message": "private graded answer", "completed_at": 1780274115}},
            ]
            (sessions / f"rollout-example-{thread_id}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            common = {"thread_id": thread_id, "model": "gpt-6-luna", "effort": "low"}
            metrics = [
                {**common, "event": "route_accepted", "turn_id": turn_id, "task_class": "simple"},
                {**common, "event": "turn_completed", "turn_id": turn_id, "status": "completed"},
                {**common, "event": "turn_usage", "turn_id": turn_id,
                 "source": "proxy_thread_usage_delta", "usage": {"totalTokens": 320}},
                {**common, "event": "quality_grade", "source_turn_id": turn_id,
                 "source": "explicit", "quality_score": 100, "verification": "passed"},
            ]
            metrics_path = root / "metrics.jsonl"
            with self.private_store(root):
                self.assertEqual(outcome_model.import_codex_sessions(sessions, thread_id)["imported"], 1)
                metrics_path.write_text("\n".join(json.dumps(row) for row in metrics) + "\n")
                self.assertEqual(outcome_model.sync_session_grades(metrics_path)["joined"], 0)
                metrics[2]["usage"]["totalTokens"] = 321
                metrics_path.write_text("\n".join(json.dumps(row) for row in metrics) + "\n")
                self.assertEqual(outcome_model.sync_session_grades(metrics_path)["joined"], 1)
                self.assertEqual(outcome_model.sync_session_grades(metrics_path)["unchanged"], 1)
                self.assertEqual(outcome_model.status()["standalone_codex"]
                                 ["complete_explicit_grade_exact_usage"], 1)
                report = outcome_model.train_codex_model()
                self.assertEqual(report["graded_with_usage"], 1)
                self.assertEqual(report["status"], "insufficient_comparable_outcomes")
                self.assertNotIn(b"private graded prompt", outcome_model.DB_PATH.read_bytes())
                self.assertNotIn(b"private graded answer", outcome_model.DB_PATH.read_bytes())


if __name__ == "__main__":
    unittest.main()
