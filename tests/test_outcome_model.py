import json
import hashlib
import tempfile
import unittest
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
        return stack

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
                guard["nonce"] = "different-nonce"
                (guard_root / "fixture.json").write_text(json.dumps(guard), encoding="utf-8")
                self.assertEqual(outcome_model.import_auracall(guard_root, runs_root)["ineligible"], 1)

    def test_codex_capture_joins_only_explicit_grade_and_exact_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.private_store(root):
                prepared = outcome_model.prepare_codex_prompt("private user prompt")
                outcome_model.capture_codex_turn("thread-1", "turn-1", prepared, "gpt-6-sol", "medium", "routine")
                self.assertTrue(outcome_model.capture_codex_result("thread-1", "turn-1",
                                                                  "private Codex final result"))
                metrics = root / "metrics.jsonl"
                metrics.write_text("\n".join(json.dumps(row) for row in [
                    {"event": "outcome_observation", "thread_id": "thread-1", "turn_id": "turn-1", "outcome": "verified"},
                    {"event": "quality_grade", "thread_id": "thread-1", "source_turn_id": "turn-1",
                     "source": "explicit", "quality_score": 95, "verification": "passed"},
                    {"event": "turn_usage", "thread_id": "thread-1", "turn_id": "turn-1",
                     "usage": {"totalTokens": 123}},
                ]) + "\n", encoding="utf-8")
                self.assertEqual(outcome_model.sync_codex_grades(metrics), {"graded": 1, "usage_matched": 1})
                self.assertEqual(outcome_model.sync_codex_grades(metrics), {"graded": 0, "usage_matched": 0})
                self.assertEqual(outcome_model.status()["codex"],
                                 {"accepted_turns": 1, "thread_groups": 1,
                                  "graded_turns": 1, "exact_usage_turns": 1,
                                  "result_captured_turns": 1})
                self.assertNotIn(b"private user prompt", outcome_model.DB_PATH.read_bytes())
                self.assertNotIn(b"private Codex final result", outcome_model.DB_PATH.read_bytes())
                self.assertEqual(outcome_model.train_codex_model()["status"],
                                 "insufficient_comparable_outcomes")

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
                imported = outcome_model.import_auracall(guards, runs)
                self.assertEqual(imported["trace_imported"], 2)
                self.assertEqual(outcome_model.status()["iterations"]["revisions_with_parent_feedback"], 1)
                self.assertEqual(outcome_model.status()["iterations"]["adopted_browser_suggestions"], 1)
                with outcome_model._connect() as connection:
                    child = connection.execute("SELECT parent_episode_id,parent_feedback_features FROM iteration_traces WHERE parent_episode_id IS NOT NULL").fetchone()
                self.assertIsNotNone(child[1])
                self.assertNotIn(b"feedback from first", outcome_model.DB_PATH.read_bytes())

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
                                            parent_feedback_features,result_feedback_features,submitted_at,passed)
                                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                                           (f"episode-{number}", f"root-{number}", None,
                                            f"origin-{number}", f"generation-{number}", "user",
                                            json.dumps(prompt), None, None,
                                            f"2026-09-23T00:{number:02d}:00Z", number % 2))
                report = outcome_model.train_iteration_model()
                self.assertEqual(report["trained_rounds"], 40)
                self.assertEqual(report["holdout_rounds"], 8)
                prediction = outcome_model.predict_iteration("task context", "candidate 41", "user", 1)
                self.assertFalse(prediction["causal_prompt_comparison"])
                self.assertIn("predicted_review_pass_probability", prediction)
                with_feedback = outcome_model.predict_iteration("task context", "candidate 42",
                                                                "chatgpt", 2, "reviewer fixes")
                self.assertIn("predicted_review_pass_probability", with_feedback)

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
