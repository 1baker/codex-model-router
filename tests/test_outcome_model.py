import json
import hashlib
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import outcome_model
from smoke_bench import browser_prompt_variant, product_definition, stable_digest


class OutcomeModelTests(unittest.TestCase):
    def test_file_review_browser_revision_requires_sent_attachment_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, record = self.file_review_round(root, 1)
            origin = Path(state["source_files"]["origin_prompt"]["path"])
            generation = Path(state["source_files"]["generation_prompt"]["path"])
            generation.write_bytes(origin.read_bytes())
            digest = hashlib.sha256(origin.read_bytes()).hexdigest()
            state["source_files"]["generation_prompt"]["sha256"] = digest
            state["learning_trace"]["generation_prompt_sha256"] = digest
            state["schema"] = "codex.pro_guard_run.v1"
            state["conversation_url"] = "https://chatgpt.com/c/file-review-fixture"
            state["verdict"].update({"pass": False, "score": 87,
                                     "suggested_next_prompt": "A narrower task"})
            state["evaluation"].update({"passed": False, "score": 87})
            review_file = Path(state["review_artifact"]["path"])
            review_file.write_text("```json\n" + json.dumps(state["verdict"]) + "\n```\n")
            state["review_artifact"]["sha256"] = hashlib.sha256(review_file.read_bytes()).hexdigest()
            attachments = record["bundle"]["run"]["initialInputs"]["attachments"]
            paths = [Path(item["uri"].removeprefix("file://")).as_posix() for item in attachments]
            browser_run = {"service": "chatgpt", "runtimeProfileId": "agent-browser-chatgpt",
                           "tabUrl": state["conversation_url"],
                           "promptTransport": {"attachments": [{"path": path} for path in paths]},
                           "attachmentUiReceipt": {
                               "schema": "auracall.browser_attachment_ui_receipt.v1",
                               "attachmentPaths": paths, "uploadCompletion": "confirmed",
                               "sentUserTurnAttachments": "confirmed", "submittedUserId": "user-fixture"}}
            record["bundle"]["steps"] = [{"status": "succeeded", "output": {
                "structuredData": {"browserRun": browser_run}}}]
            guard_path = root / "guards/file-1.json"
            record_path = root / "runs/resp_file_1/record.json"
            guard_path.write_text(json.dumps(state))
            record_path.write_text(json.dumps(record))
            revision = root / "revision.txt"
            revision.write_text("A narrower task")
            base = {"id": "base", "prompt": "private original request", "comparison_id": "same"}
            variant = browser_prompt_variant(base, "file-1", revision, root / "guards", root / "runs")
            self.assertEqual(variant["prompt_author"], "chatgpt")
            revision.write_text("An inferred rewrite")
            with self.assertRaisesRegex(ValueError, "does not match"):
                browser_prompt_variant(base, "file-1", revision, root / "guards", root / "runs")
            revision.write_text("A narrower task")
            del browser_run["attachmentUiReceipt"]
            record_path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "UI proof"):
                browser_prompt_variant(base, "file-1", revision, root / "guards", root / "runs")
            browser_run["attachmentUiReceipt"] = {"schema": "auracall.browser_attachment_ui_receipt.v1",
                "attachmentPaths": paths, "uploadCompletion": "confirmed",
                "sentUserTurnAttachments": "confirmed", "submittedUserId": "user-fixture"}
            record["bundle"]["sharedState"]["artifacts"] = []
            record_path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                browser_prompt_variant(base, "file-1", revision, root / "guards", root / "runs")

    def file_review_round(self, root, number, parent=None, reference_bytes=None):
        guard_id, response_id = f"file-{number}", f"resp_file_{number}"
        guards, runs = root / "guards", root / "runs"
        guards.mkdir(exist_ok=True)
        directory = guards / f"{guard_id}-files"
        directory.mkdir()
        def source(name, text):
            path = directory / name
            path.write_bytes(text.encode())
            return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "fileName": name}
        sources = {"origin_prompt": source("origin.md", "private original request"),
                   "generation_prompt": source("generation.md", "private generation " + str(number))}
        goal, artifact = "private review goal\n", f"private exact candidate {number}\r\n"
        prompt, chat = "Please review the attached candidate.", "The candidate is useful. See the attached review."
        files = {"goal": source("review-goal.md", goal), "artifact": source("candidate.md", artifact),
                 "guide": source("review-guide.md", "private guide"), "prompt": source("review-prompt.md", prompt)}
        sources.update({name: files[name] for name in ("goal", "artifact")})
        trace = {"root_guard_id": "file-1", "parent_guard_id": parent["guard_id"] if parent else None,
                 "parent_response_id": parent["response_id"] if parent else None, "prompt_author": "codex",
                 "origin_prompt_sha256": sources["origin_prompt"]["sha256"],
                 "generation_prompt_sha256": sources["generation_prompt"]["sha256"]}
        trace_digest = hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        verdict = {"schema": "codex.pro_guard_review_file.v1", "nonce": f"nonce-{number}",
                   "submission_fingerprint": f"fingerprint-{number}", "goal_sha256": files["goal"]["sha256"],
                   "artifact_sha256": files["artifact"]["sha256"], "pass": True, "score": 95,
                   "summary": "private grading feedback", "blocking_findings": [], "tests_or_checks_required": [],
                   "suggested_next_prompt": None, "confidence": "high"}
        state = {"guard_id": guard_id, "response_id": response_id, "round": number, "status": "completed",
                 "submitted_at": f"2026-09-24T00:0{number}:00Z", "nonce": verdict["nonce"],
                 "review_format": "file", "submission_fingerprint": verdict["submission_fingerprint"],
                 "source_files": sources, "learning_trace": trace, "learning_trace_digest": trace_digest,
                 "file_review": {"schema": "codex.pro_guard_file_handoff.v1", "prompt_author": "codex", "files": files},
                 "review_artifact": source("review-record.md", "```json\n" + json.dumps(verdict) + "\n```\n"),
                 "review_chat": source("review-chat.md", chat), "review_text": chat, "verdict": verdict,
                 "evaluation": {"valid": True, "nonce_matched": True, "passed": True, "score": 95}}
        metadata = {"workflow": "codex-pro-guard", "guard_id": guard_id, "guard_nonce": state["nonce"],
                    "round": number, "submission_fingerprint": state["submission_fingerprint"],
                    "learning_trace_digest": trace_digest,
                    "guardFileReview": {"schema": state["file_review"]["schema"], "prompt_author": "codex",
                                        "files": {name: {"fileName": row["fileName"], "sha256": row["sha256"]} for name, row in files.items()}}}
        outputs = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": chat}]}]
        record = {"runId": response_id, "bundle": {
            "run": {"id": response_id, "status": "succeeded", "initialInputs": {
                "model": "chatgpt:premium", "instructions": "", "requestInput": prompt, "metadata": metadata,
                "attachments": [{"id": f"guard-{name}", "mimeType": "text/markdown", "fileName": files[name]["fileName"],
                                 "uri": Path(files[name]["path"]).as_uri()} for name in ("goal", "artifact", "guide")]}},
            "sharedState": {"structuredOutputs": [{"key": "response.output", "value": outputs}],
                            "artifacts": [{"kind": "file", "path": state["review_artifact"]["path"]}]}}}
        if reference_bytes is not None:
            original = root / f"original-{number}.pdf"
            snapshot = directory / "final.pdf"
            original.write_bytes(reference_bytes); snapshot.write_bytes(reference_bytes)
            identity = {"fileName": "final.pdf", "mimeType": "application/pdf", "size": len(reference_bytes),
                        "sha256": hashlib.sha256(reference_bytes).hexdigest()}
            state["reference_sources"] = [{**identity, "path": str(original)}]
            state["file_review"]["references"] = [{**identity, "path": str(snapshot)}]
            metadata["guardFileReview"]["references"] = [identity]
            record["bundle"]["run"]["initialInputs"]["attachments"].append({"id": "guard-reference-0",
                "mimeType": "application/pdf", "fileName": "final.pdf", "uri": snapshot.as_uri()})
        (runs / response_id).mkdir(parents=True)
        (runs / response_id / "record.json").write_text(json.dumps(record))
        (guards / f"{guard_id}.json").write_text(json.dumps(state))
        return state, record

    def test_binary_reference_review_import_is_bound_private_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_binary = b"%PDF-1.7\n\xff\x00private-document-contents"
            state, record = self.file_review_round(root, 1, reference_bytes=private_binary)
            with self.private_store(root):
                result = outcome_model.import_auracall(root / "guards", root / "runs")
                self.assertEqual(result["imported"], 1)
                self.assertEqual(result["trace_imported"], 1)
                self.assertEqual(outcome_model.import_auracall(root / "guards", root / "runs")["unchanged"], 1)
                self.assertNotIn(private_binary, outcome_model.DB_PATH.read_bytes())
                row = outcome_model._guard_row(root / "guards/file-1.json", root / "runs", outcome_model._key())
                self.assertNotEqual(row[7], outcome_model._digest(outcome_model._key(), "private exact candidate 1\r\n"))

    def test_binary_reference_import_rejects_missing_or_changed_documents(self):
        for failure in ("source", "snapshot", "manifest", "attachment", "mime", "extra", "symlink", "size"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state, record = self.file_review_round(root, 1, reference_bytes=b"%PDF-1.7\n\xff\x00fixture")
                inputs = record["bundle"]["run"]["initialInputs"]
                snapshot = Path(state["file_review"]["references"][0]["path"])
                if failure == "source": Path(state["reference_sources"][0]["path"]).write_bytes(b"changed")
                elif failure == "snapshot": snapshot.write_bytes(b"changed")
                elif failure == "manifest": inputs["metadata"]["guardFileReview"]["references"] = []
                elif failure == "attachment": inputs["attachments"].pop()
                elif failure == "mime": inputs["attachments"][-1]["mimeType"] = "text/plain"
                elif failure == "extra": inputs["attachments"].append(dict(inputs["attachments"][-1]))
                elif failure == "size": state["file_review"]["references"][0]["size"] += 1
                else:
                    snapshot.unlink(); snapshot.symlink_to(state["reference_sources"][0]["path"])
                (root / "guards/file-1.json").write_text(json.dumps(state))
                (root / "runs" / state["response_id"] / "record.json").write_text(json.dumps(record))
                with self.private_store(root):
                    result = outcome_model.import_auracall(root / "guards", root / "runs")
                    self.assertEqual(result["imported"], 0)
                    self.assertEqual(result["trace_imported"], 0)
                    self.assertEqual(result["rejected"], 1)

    def test_same_overview_with_different_document_is_a_distinct_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            digests = []
            key = b"fixture-key-not-a-real-secret"
            for name, raw in (("first", b"%PDF-1.7\nfirst"), ("second", b"%PDF-1.7\nother")):
                root = Path(directory) / name
                root.mkdir()
                self.file_review_round(root, 1, reference_bytes=raw)
                row = outcome_model._guard_row(root / "guards/file-1.json", root / "runs", key)
                digests.append(row[7])
            self.assertNotEqual(*digests)

    def test_file_review_learning_preserves_candidate_identity_and_parent_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, record = self.file_review_round(root, 1)
            self.file_review_round(root, 2, parent)
            with self.private_store(root):
                imported = outcome_model.import_auracall(root / "guards", root / "runs")
                self.assertEqual(imported["imported"], 2)
                self.assertEqual(imported["trace_imported"], 2)
                self.assertEqual(imported["browser_result_imported"], 2)
                self.assertEqual(outcome_model.import_auracall(root / "guards", root / "runs")["trace_unchanged"], 2)
                with outcome_model._connect() as connection:
                    features, score = connection.execute("SELECT parent_result_features,parent_quality_score FROM iteration_traces WHERE parent_episode_id IS NOT NULL").fetchone()
                    digest = connection.execute("SELECT revision_digest FROM episodes WHERE source_response_id=?", (parent["response_id"],)).fetchone()
                exact = "private exact candidate 1\r\n"
                self.assertEqual(json.loads(features), outcome_model._features(exact, "", 1, outcome_model._key()))
                self.assertEqual(score, 95)
                self.assertEqual(digest[0], outcome_model._digest(outcome_model._key(), exact))
                raw = outcome_model.DB_PATH.read_bytes()
                for private_text in (exact, "private original request", "private grading feedback", parent["review_text"]):
                    self.assertNotIn(private_text.encode(), raw)

    def test_file_review_import_rejects_missing_or_changed_provenance(self):
        for failure in ("attachment", "prompt", "candidate", "verdict", "chat", "download", "metadata", "score"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state, record = self.file_review_round(root, 1)
                inputs = record["bundle"]["run"]["initialInputs"]
                if failure == "attachment": inputs["attachments"] = inputs["attachments"][:1]
                elif failure == "prompt": inputs["requestInput"] = "changed"
                elif failure == "candidate": Path(state["source_files"]["artifact"]["path"]).write_text("changed")
                elif failure == "verdict": state["verdict"]["summary"] = "changed"
                elif failure == "chat": state["review_text"] = "changed"
                elif failure == "download": record["bundle"]["sharedState"]["artifacts"] = []
                elif failure == "metadata": inputs["metadata"]["guardFileReview"] = {}
                else: state["evaluation"]["score"] = 99
                (root / "guards" / "file-1.json").write_text(json.dumps(state))
                (root / "runs" / state["response_id"] / "record.json").write_text(json.dumps(record))
                with self.private_store(root):
                    result = outcome_model.import_auracall(root / "guards", root / "runs")
                    self.assertEqual(result["imported"], 0)
                    self.assertEqual(result["trace_imported"], 0)
                    self.assertEqual(result["rejected"], 1)

    def test_recovered_browser_result_is_ungraded_keyed_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response_id = "resp_recovery_fixture"
            run_dir = root / "runs" / response_id
            run_dir.mkdir(parents=True)
            nonce = "private-recovery-nonce-1234"
            prompt = f"Review the private artifact. FRESHNESS_NONCE: {nonce}"
            answer = json.dumps({"nonce": nonce, "score": 91, "summary": "private result"})
            metadata = {"schema": "guard", "workflow": "codex-pro-guard", "round": 1,
                        "nonce": nonce, "learning_trace_digest": "a" * 64,
                        "client_request_id": "11111111-1111-4111-8111-111111111111"}
            url = "https://chatgpt.com/g/g-p-" + "1" * 32 + "/c/" + "2" * 8 + "-2222-2222-2222-" + "2" * 12
            artifact = {"id": "file-1", "kind": "file", "path": "/private/review.md",
                        "uri": "file:///private/review.md", "title": "review.md"}
            receipt = {"schema": "auracall.browser_attachment_ui_receipt.v1",
                       "attachmentPaths": [artifact["path"]], "uploadCompletion": "confirmed",
                       "sentUserTurnAttachments": "confirmed", "submittedUserId": "user-1"}
            record = {"runId": response_id, "bundle": {
                "run": {"id": response_id, "status": "failed", "sourceKind": "direct",
                        "initialInputs": {"requestInput": prompt, "metadata": metadata,
                                          "attachments": [{"id": "file-1", "uri": artifact["uri"],
                                                           "fileName": artifact["title"]}],
                                          "auracall": {"chatgptConversationUrl": url}}},
                "steps": [{"id": "step-1", "status": "failed", "service": "chatgpt",
                           "input": {"prompt": prompt, "artifacts": [artifact],
                                     "structuredData": {"metadata": metadata}},
                           "failure": {"code": "runner_execution_failed", "ownerStepId": "step-1",
                                       "details": {"phase": "after", "retryable": False}}}],
                "events": [{"payload": {"runtimeEvidence": {"evidenceRef": "chatgpt-prompt-submitted"}}},
                           {"payload": {"runtimeEvidence": {"details": {"attachmentUiReceipt": receipt}}}}]}}
            record_bytes = json.dumps(record).encode()
            (run_dir / "record.json").write_bytes(record_bytes)
            observation = {"schema": "auracall.response_recovery_observation.v1",
                           "response_id": response_id, "original_status": "failed",
                           "original_run_modified": False, "prompt_submitted": False,
                           "account_verdict": "match", "request_metadata": metadata,
                           "original_record_digest": hashlib.sha256(record_bytes).hexdigest(),
                           "logical_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                           "wire_prompt_sha256": "b" * 64,
                           "answer_text": answer, "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                           "user_message_id": "user-1", "assistant_message_id": "assistant-1",
                           "browser_process_id": 1234, "target_id": "target-1",
                           "conversation_url": url, "observed_at": "2026-09-24T10:00:00Z"}
            sidecar = run_dir / "recovery-observation.json"
            sidecar.write_text(json.dumps(observation))
            with self.private_store(root):
                self.assertEqual(outcome_model.import_recovery_observations(root / "runs")["imported"], 1)
                self.assertEqual(outcome_model.import_recovery_observations(root / "runs")["unchanged"], 1)
                status = outcome_model.status()
                self.assertEqual(status["recovery_observations"]["ungraded_failed_response_records"], 1)
                self.assertEqual(status["recovery_observations"]["attachment_ui_confirmed_records"], 1)
                self.assertFalse(status["forward_readiness"]["candidate_for_live_pilot"])
                raw = outcome_model.DB_PATH.read_bytes()
                self.assertNotIn(prompt.encode(), raw)
                self.assertNotIn(answer.encode(), raw)
                with outcome_model._connect() as connection:
                    self.assertEqual(connection.execute(
                        "SELECT COUNT(*) FROM episodes").fetchone()[0], 0)
                observation["answer_text"] = answer + " changed"
                observation["answer_sha256"] = hashlib.sha256(observation["answer_text"].encode()).hexdigest()
                sidecar.write_text(json.dumps(observation))
                self.assertEqual(outcome_model.import_recovery_observations(root / "runs")["conflict"], 1)
                observation["user_message_id"] = "wrong-user"
                sidecar.write_text(json.dumps(observation))
                self.assertEqual(outcome_model.import_recovery_observations(root / "runs")["rejected"], 1)
                observation["user_message_id"] = "user-1"
                sidecar.write_text(json.dumps(observation))
                record["bundle"]["events"].pop()
                (run_dir / "record.json").write_text(json.dumps(record))
                self.assertEqual(outcome_model.import_recovery_observations(root / "runs")["rejected"], 1)

    def test_external_browser_blocker_is_reported_but_never_imported_as_grade(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loop_dir = root / "loops" / "blocked"
            loop_dir.mkdir(parents=True)
            config = {"goal": "independent verification fixture"}
            state = {"run_id": "blocked", "config": config,
                     "config_digest": hashlib.sha256(json.dumps(config, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
                     "rounds": [{"round": 1, "iteration_prompt": "implement the fixture",
                                 "review": {"status": "external_blocker"}}]}
            (loop_dir / "loop.json").write_text(json.dumps(state))
            with self.private_store(root):
                result = outcome_model.import_bound_loops(root / "loops", root / "runs")
                self.assertEqual(result["external_blocker_excluded"], 1)
                self.assertEqual(result["imported"], 0)
                self.assertEqual(outcome_model.status()["bound_legacy_loop"]["rounds"], 0)

    def test_forward_readiness_requires_prospective_causal_and_browser_evidence(self):
        empty = outcome_model.forward_readiness({})
        self.assertFalse(empty["candidate_for_live_pilot"])
        self.assertEqual(len(empty["missing"]), len(empty["checks"]))
        evidence = {
            "review_model": {"validated_for_shadow": True},
            "review_score_model": {"validated_for_shadow": True},
            "iteration_model": {"validated_for_shadow": True,
                                "causal_prompt_comparison": True},
            "improvement_model": {"validated_for_shadow": True},
            "bound_loop_model": {"validated_for_shadow": True,
                                 "score_validated_for_shadow": True},
            "bound_legacy_loop": {"browser_ui_confirmed_rounds": 30,
                                  "linked_revisions": outcome_model.MIN_ITERATION_REVISION_GROUPS},
            "codex_model": {"validated_for_shadow": True,
                            "prospective_comparable_arms": True,
                            "causal_model_comparison": True},
            "local_prompt_model": {"validated_for_shadow": True},
            "managed_routing_policy": {"validated_for_shadow": True,
                                       "causal_model_comparison": True},
            "managed_prompt_policy": {"validated_for_shadow": True,
                                      "causal_prompt_comparison": True,
                                      "browser_provenance_verified": True},
            "browser_revision_product_outcomes": {
                "execution_verified_independent_products": 8,
                "execution_verified_product_wins": 8,
                "execution_verified_product_losses": 0},
        }
        self.assertTrue(outcome_model.forward_readiness(evidence)["candidate_for_live_pilot"])
        evidence["managed_prompt_policy"]["validated_for_shadow"] = False
        self.assertEqual(outcome_model.forward_readiness(evidence)["missing"],
                         ["prospective_managed_prompt_revision"])
        evidence["managed_prompt_policy"]["validated_for_shadow"] = True
        evidence["local_prompt_model"]["validated_for_shadow"] = False
        self.assertEqual(outcome_model.forward_readiness(evidence)["missing"],
                         ["prospective_local_prompt_product"])
        evidence["local_prompt_model"]["validated_for_shadow"] = True
        evidence["managed_routing_policy"]["causal_model_comparison"] = False
        self.assertEqual(outcome_model.forward_readiness(evidence)["missing"],
                         ["prospective_managed_model_effort"])
        evidence["managed_routing_policy"]["causal_model_comparison"] = True
        evidence["bound_legacy_loop"]["browser_ui_confirmed_rounds"] = 29
        self.assertEqual(outcome_model.forward_readiness(evidence)["missing"],
                         ["browser_attachment_receipts"])

    def test_managed_pair_evidence_requires_repeats_and_prefers_verified_efficiency(self):
        blocks = []
        left = ("gpt-6-luna", "low")
        right = ("gpt-6-sol", "medium")
        for product in ("product-a", "product-b"):
            for repeat in range(3):
                blocks.append({"block_key": f"{product}-{repeat}",
                               "product_key": product, "task_class": "routine",
                               "prompt_digest": "same-prompt",
                               "created_at_ms": 100 + repeat,
                               "arms": {
                                   left: {"product_pass": True, "quality_score": 100,
                                          "total_tokens": 80},
                                   right: {"product_pass": True, "quality_score": 100,
                                           "total_tokens": 120}}})
        reports, private = outcome_model._managed_pair_evidence(blocks)
        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertEqual((report["eligible_repeated_products"], report["left_wins"],
                          report["right_wins"], report["ties"]), (2, 2, 0, 0))
        self.assertEqual(len(private[report["comparison_id"]]), 2)
        reports, _private = outcome_model._managed_pair_evidence(blocks[:2])
        self.assertEqual(reports[0]["eligible_repeated_products"], 0)

    def test_managed_pair_evidence_never_pools_prompt_revisions(self):
        left = ("gpt-6-luna", "low")
        right = ("gpt-6-sol", "medium")
        blocks = []
        for repeat, prompt in enumerate(("base", "base", "revision")):
            blocks.append({"block_key": str(repeat), "product_key": "one-product",
                           "prompt_digest": prompt, "task_class": "routine",
                           "created_at_ms": repeat, "arms": {
                               left: {"product_pass": True, "quality_score": 100,
                                      "total_tokens": 80},
                               right: {"product_pass": True, "quality_score": 100,
                                       "total_tokens": 120}}})
        reports, _private = outcome_model._managed_pair_evidence(blocks)
        self.assertEqual(reports[0]["eligible_repeated_products"], 0)

    def test_managed_pair_evidence_counts_prompt_strata_as_one_product(self):
        left = ("gpt-6-luna", "low")
        right = ("gpt-6-sol", "medium")

        def blocks(*, conflicting=False):
            rows = []
            for prompt in ("base", "revision"):
                for repeat in range(3):
                    left_tokens, right_tokens = (80, 120)
                    if conflicting and prompt == "revision":
                        left_tokens, right_tokens = (120, 80)
                    rows.append({"block_key": f"{prompt}-{repeat}",
                                 "product_key": "one-product",
                                 "prompt_digest": prompt, "task_class": "routine",
                                 "created_at_ms": repeat, "arms": {
                                     left: {"product_pass": True, "quality_score": 100,
                                            "total_tokens": left_tokens},
                                     right: {"product_pass": True, "quality_score": 100,
                                             "total_tokens": right_tokens}}})
            return rows

        reports, private = outcome_model._managed_pair_evidence(blocks())
        report = reports[0]
        outcomes = private[report["comparison_id"]]
        self.assertEqual(report["eligible_repeated_products"], 1)
        self.assertEqual((report["left_wins"], report["right_wins"], report["ties"]),
                         (1, 0, 0))
        self.assertEqual(outcomes[0]["prompt_strata"], 2)

        reports, private = outcome_model._managed_pair_evidence(blocks(conflicting=True))
        report = reports[0]
        outcomes = private[report["comparison_id"]]
        self.assertEqual(report["eligible_repeated_products"], 1)
        self.assertEqual((report["left_wins"], report["right_wins"], report["ties"]),
                         (0, 0, 1))
        self.assertEqual(outcomes[0]["prompt_strata"], 2)

    def test_managed_prompt_experiment_counts_exact_precommits_and_completions(self):
        key = b"k" * 32
        arms = [("gpt-6-luna", "low"), ("gpt-6-sol", "medium")]
        product = {"files": {"verify.py": "pass\n"}, "protected_files": ["verify.py"],
                   "verify": ["python3", "verify.py"], "expected_final": "ready"}
        base = {**product, "id": "base", "prompt": "Build a complete tested parser.",
                "prompt_author": "benchmark"}
        candidate = {**product, "id": "candidate", "prompt": "Build an exact tested parser.",
                     "prompt_author": "chatgpt", "variant_of": "base",
                     "browser_origin_prompt": base["prompt"],
                     "browser_review": {"guard_id": "guard-1", "response_id": "resp_1"}}
        product_key = outcome_model._digest(key, stable_digest(product_definition(base)))
        entries = [
            {"scenario_key": outcome_model._digest(key, stable_digest(base)),
             "prompt_digest": outcome_model._digest(key, base["prompt"]),
             "prompt_author": "benchmark", "browser_guard_key": None,
             "browser_response_key": None},
            {"scenario_key": outcome_model._digest(key, stable_digest(candidate)),
             "prompt_digest": outcome_model._digest(key, candidate["prompt"]),
             "prompt_author": "chatgpt",
             "browser_guard_key": outcome_model._digest(key, "guard-1"),
             "browser_response_key": outcome_model._digest(key, "resp_1")},
        ]
        arm_order = json.dumps([list(arm) for arm in arms])
        database_rows = [
            ("set-complete", product_key, json.dumps(entries), arm_order),
            ("set-complete", product_key, json.dumps(entries), arm_order),
            ("set-incomplete", product_key, json.dumps(entries), None),
        ]
        prompt_arms = {
            arm: {"product_pass": True, "quality_score": 100, "total_tokens": 10}
            for arm in arms}
        complete_sets = [{"product_key": product_key, "prompts": [
            {"prompt_digest": entries[0]["prompt_digest"], "prompt_author": "benchmark",
             "browser_guard_key": None, "browser_response_key": None, "arms": prompt_arms},
            {"prompt_digest": entries[1]["prompt_digest"], "prompt_author": "chatgpt",
             "browser_guard_key": entries[1]["browser_guard_key"],
             "browser_response_key": entries[1]["browser_response_key"], "arms": prompt_arms},
        ]}]

        class Rows:
            def execute(self, _query):
                return self

            def fetchall(self):
                return database_rows

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        with patch.object(outcome_model, "_key", return_value=key), \
                patch.object(outcome_model, "_connect", return_value=Rows()), \
                patch.object(outcome_model, "_managed_complete_prompt_sets",
                             return_value=(complete_sets, {})):
            counts = outcome_model.managed_prompt_experiment_counts(base, candidate, arms)
        self.assertEqual(counts, {"attempts": 2, "complete": 1,
                                  "failed_or_incomplete": 1})

    def test_managed_routing_policy_publishes_only_prospectively_validated_arm(self):
        left = ("gpt-6-luna", "low")
        right = ("gpt-6-sol", "medium")

        def blocks(products, first_at):
            return [{"block_key": f"{product}-{repeat}", "product_key": product,
                     "prompt_digest": f"prompt-{product}", "task_class": "routine",
                     "created_at_ms": first_at + repeat, "baseline_arm": right,
                     "arms": {left: {"product_pass": True, "quality_score": 100,
                                      "total_tokens": 800},
                              right: {"product_pass": True, "quality_score": 100,
                                      "total_tokens": 1000}}}
                    for product in products for repeat in range(3)]

        development = blocks([f"dev-{index}" for index in range(8)], 100)
        prospective = blocks([f"future-{index}" for index in range(8)], 2_000_000)
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                with patch.object(outcome_model, "_managed_complete_blocks",
                                  return_value=(development, {})), \
                        patch.object(outcome_model.time, "time", return_value=1000):
                    initial = outcome_model.train_managed_routing_policy()
                self.assertEqual(initial["status"], "prospective_collection")
                artifact = json.loads(outcome_model.MANAGED_ROUTING_POLICY_PATH.read_text())
                self.assertEqual(artifact["validations"], {})
                with patch.object(outcome_model, "_managed_complete_blocks",
                                  return_value=(development + prospective, {})), \
                        patch.object(outcome_model.time, "time", return_value=3000):
                    validated = outcome_model.train_managed_routing_policy()
                self.assertTrue(validated["validated_for_shadow"])
                artifact = json.loads(outcome_model.MANAGED_ROUTING_POLICY_PATH.read_text())
                validation = next(iter(artifact["validations"].values()))
                self.assertEqual(validation["baseline_arm"], list(right))
                self.assertEqual(validation["recommended_arm"], list(left))
                self.assertEqual(validation["prospective_losses"], 0)

    def test_managed_prompt_set_precommits_browser_order_and_exact_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                product = {"files": {"verify.py": "pass\n"},
                           "protected_files": ["verify.py"],
                           "verify": ["python3", "verify.py"], "expected_final": "ready"}
                review = {"guard_id": "review-1", "response_id": "resp_123",
                          "origin_sha256": "a" * 64}
                base = {**product, "id": "base", "comparison_id": "prompt-pair",
                        "task_class": "routine", "prompt_author": "benchmark",
                        "prompt": "Build a complete tested command line utility from the supplied fixture and verify every required behavior."}
                candidate = {**product, "id": "candidate", "comparison_id": "prompt-pair",
                             "task_class": "routine", "prompt_author": "chatgpt",
                             "variant_of": "base", "browser_review": review,
                             "browser_origin_prompt": base["prompt"],
                             "prompt": "Build the supplied command line utility, preserve the verifier, check invalid input and run every required test before finishing."}
                suite = "11111111-1111-1111-1111-111111111111"
                prompt_set_id = "22222222-2222-2222-2222-222222222222"
                with patch("smoke_bench.verified_browser_revision", return_value=review):
                    prompt_set = outcome_model.create_managed_prompt_set(
                        suite, prompt_set_id, [base, candidate], 0)
                self.assertEqual(prompt_set["schema"], "modellabs.managed-prompt-set.v1")
                self.assertEqual({item["prompt_author"] for item in prompt_set["prompts"]},
                                 {"benchmark", "chatgpt"})
                by_digest = {stable_digest(item): item for item in (base, candidate)}
                for index, entry in enumerate(prompt_set["prompts"]):
                    block = outcome_model.create_managed_benchmark_block(
                        suite, f"{index + 3:08x}-3333-3333-3333-333333333333",
                        by_digest[entry["scenario_sha256"]],
                        [("gpt-6-luna", "low"), ("gpt-6-sol", "medium")], 0,
                        prompt_set_id=prompt_set_id, prompt_arm_index=index)
                    self.assertEqual(block["prompt_set_commitment"], prompt_set["commitment"])
                with self.assertRaisesRegex(ValueError, "differs from precommit"):
                    outcome_model.create_managed_benchmark_block(
                        suite, "99999999-9999-9999-9999-999999999999", base,
                        [("gpt-6-luna", "low"), ("gpt-6-sol", "medium")], 0,
                        prompt_set_id=prompt_set_id,
                        prompt_arm_index=next(index for index, entry in enumerate(
                            prompt_set["prompts"]) if entry["prompt_author"] == "chatgpt"))
                raw = outcome_model.DB_PATH.read_bytes()
                self.assertNotIn(base["prompt"].encode(), raw)
                self.assertNotIn(candidate["prompt"].encode(), raw)

    def test_managed_prompt_policy_requires_new_prospective_products(self):
        def prompt_sets(product_names, first_at):
            rows = []
            arm = ("gpt-6-sol", "medium")
            for product_index, product in enumerate(product_names):
                for repeat in range(3):
                    rows.append({"prompt_set_key": f"{product}-{repeat}",
                                 "suite_key": product, "created_at_ms": first_at + repeat,
                                 "comparison_key": product, "product_key": product,
                                 "task_class": "routine", "repetition_index": repeat,
                                 "prompts": [
                                     {"prompt_digest": f"base-{product}",
                                      "prompt_features": {"1": 0.1},
                                      "prompt_author": "benchmark", "browser_guard_key": None,
                                      "browser_response_key": None,
                                      "arms": {arm: {"product_pass": True,
                                                     "quality_score": 100,
                                                     "total_tokens": 1000}}},
                                     {"prompt_digest": f"candidate-{product}",
                                      "prompt_features": {"2": 0.2 + product_index / 100},
                                      "prompt_author": "chatgpt", "browser_guard_key": "guard",
                                      "browser_response_key": "response",
                                      "arms": {arm: {"product_pass": True,
                                                     "quality_score": 100,
                                                     "total_tokens": 800}}}]})
            return rows

        development = prompt_sets([f"dev-{index}" for index in range(8)], 100)
        prospective = prompt_sets([f"future-{index}" for index in range(8)], 2_000_000)
        counts = {"precommitted_prompt_sets": len(development)}
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                with patch.object(outcome_model, "_managed_complete_prompt_sets",
                                  return_value=(development, counts)), \
                        patch.object(outcome_model.time, "time", return_value=1000):
                    initial = outcome_model.train_managed_prompt_policy()
                self.assertEqual(initial["status"], "prospective_collection")
                self.assertTrue(initial["comparisons"][0]["development_checkpoint_created"])
                self.assertFalse(initial["validated_for_shadow"])
                with patch.object(outcome_model, "_managed_complete_prompt_sets",
                                  return_value=(development + prospective, counts)), \
                        patch.object(outcome_model.time, "time", return_value=3000):
                    validated = outcome_model.train_managed_prompt_policy()
                self.assertEqual(validated["status"], "prospective_validated")
                self.assertTrue(validated["validated_for_shadow"])
                self.assertEqual(validated["comparisons"][0]["prospective_wins"], 8)
                self.assertFalse(validated["authoritative_for_prompt_selection"])

    def private_store(self, root: Path) -> ExitStack:
        stack = ExitStack()
        private = root / "learning"
        stack.enter_context(patch.object(outcome_model, "LEARNING_ROOT", private))
        stack.enter_context(patch.object(outcome_model, "KEY_PATH", private / "feature-key"))
        stack.enter_context(patch.object(outcome_model, "DB_PATH", private / "episodes.sqlite3"))
        stack.enter_context(patch.object(outcome_model, "MODEL_PATH", private / "review-model.json"))
        stack.enter_context(patch.object(outcome_model, "SCORE_MODEL_PATH", private / "review-score-model.json"))
        stack.enter_context(patch.object(outcome_model, "CODEX_MODEL_PATH", private / "codex-model.json"))
        stack.enter_context(patch.object(outcome_model, "LOCAL_PROMPT_MODEL_PATH", private / "local-prompt-model.json"))
        stack.enter_context(patch.object(outcome_model, "LOCAL_PROMPT_EVAL_PATH", private / "local-prompt-eval.json"))
        stack.enter_context(patch.object(outcome_model, "MANAGED_ROUTING_POLICY_PATH", private / "managed-routing-policy.json"))
        stack.enter_context(patch.object(outcome_model, "MANAGED_PROMPT_POLICY_PATH", private / "managed-prompt-policy.json"))
        stack.enter_context(patch.object(outcome_model, "ITERATION_MODEL_PATH", private / "iteration-model.json"))
        stack.enter_context(patch.object(outcome_model, "IMPROVEMENT_MODEL_PATH", private / "improvement-model.json"))
        stack.enter_context(patch.object(outcome_model, "BOUND_LOOP_MODEL_PATH", private / "bound-loop-model.json"))
        stack.enter_context(patch.object(outcome_model, "BOUND_LOOP_EVAL_PATH", private / "bound-loop-eval.json"))
        stack.enter_context(patch.object(outcome_model, "BOUND_LOOP_SCORE_EVAL_PATH", private / "bound-loop-score-eval.json"))
        stack.enter_context(patch.object(outcome_model, "REVIEW_EVAL_PATH", private / "review-evaluation-checkpoint.json"))
        stack.enter_context(patch.object(outcome_model, "SCORE_EVAL_PATH", private / "review-score-evaluation-checkpoint.json"))
        stack.enter_context(patch.object(outcome_model, "ITERATION_EVAL_PATH", private / "iteration-evaluation-checkpoint.json"))
        stack.enter_context(patch.object(outcome_model, "IMPROVEMENT_EVAL_PATH", private / "improvement-evaluation-checkpoint.json"))
        stack.enter_context(patch.object(outcome_model, "CODEX_EVAL_PATH", private / "codex-evaluation-checkpoint.json"))
        return stack

    def test_accepted_steer_is_context_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                prompt = "private user correction during an active turn"
                prepared = outcome_model.prepare_codex_prompt(prompt)
                self.assertTrue(outcome_model.capture_codex_steer(
                    "thread-a", "turn-a", "event-a", prepared))
                self.assertTrue(outcome_model.capture_codex_steer(
                    "thread-a", "turn-a", "event-a", prepared))
                with self.assertRaisesRegex(ValueError, "identity conflict"):
                    outcome_model.capture_codex_steer(
                        "thread-a", "turn-a", "event-a",
                        outcome_model.prepare_codex_prompt("different correction"))
                with outcome_model._connect() as connection:
                    row = connection.execute("""SELECT prompt_digest,features,attribution
                                              FROM codex_steers""").fetchone()
                    self.assertEqual(connection.execute("SELECT COUNT(*) FROM codex_turns").fetchone()[0], 0)
                self.assertEqual(row[2], "context_only")
                self.assertNotIn(prompt, str(row))
                self.assertEqual(outcome_model.status()["codex"]["accepted_user_steers"], 1)

    def test_bound_loop_existing_database_adds_ui_column_without_promoting_history(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                outcome_model.LEARNING_ROOT.mkdir(parents=True)
                with sqlite3.connect(outcome_model.DB_PATH) as connection:
                    connection.execute("""CREATE TABLE bound_loop_rounds (
                        episode_id TEXT PRIMARY KEY, root_key TEXT NOT NULL,
                        round_number INTEGER NOT NULL, submitted_at TEXT NOT NULL,
                        origin_prompt_digest TEXT NOT NULL, generation_prompt_digest TEXT NOT NULL,
                        prompt_features TEXT NOT NULL, codex_result_features TEXT,
                        browser_result_features TEXT NOT NULL, feedback_features TEXT NOT NULL,
                        parent_episode_id TEXT, quality_score INTEGER NOT NULL,
                        passed INTEGER NOT NULL, source_response_id TEXT NOT NULL,
                        trace_digest TEXT NOT NULL, requested_model TEXT NOT NULL,
                        attachment_content_verified INTEGER NOT NULL DEFAULT 0,
                        browser_dispatch_bound INTEGER NOT NULL DEFAULT 0
                    )""")
                    connection.execute("INSERT INTO bound_loop_rounds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                       ("old-episode", "old-root", 1, "2026-09-20T00:00:00Z",
                                        "a" * 64, "b" * 64, "{}", None, "{}", "{}", None,
                                        92, 1, "resp_old", "c" * 64, "gpt-5.2", 0, 1))
                with outcome_model._connect() as connection:
                    self.assertEqual(connection.execute(
                        "SELECT browser_dispatch_bound,browser_ui_confirmed,browser_suggestion_supplied FROM bound_loop_rounds"
                    ).fetchone(), (1, 0, 0))

    def test_bound_loop_suggestion_is_a_separate_pre_browser_feature(self):
        prompt = {"0": 0.5}
        without = outcome_model._bound_loop_vector(prompt, None, None, False)
        with_suggestion = outcome_model._bound_loop_vector(prompt, None, None, True)
        added = set(with_suggestion) - set(without)
        self.assertEqual(len(added), 1)
        self.assertEqual(with_suggestion[next(iter(added))], 1.0)

    def test_bound_browser_suggestion_requires_parent_verdict_and_exact_prompt(self):
        suggestion = "Fix only the failing parser check."
        digest = hashlib.sha256(suggestion.encode()).hexdigest()
        parent = {"review": {"verdict": {"suggested_next_prompt": suggestion}}}
        adapter = {"verdict": parent["review"]["verdict"]}
        prompt = 'Original goal remains authoritative. ' + json.dumps(suggestion)
        trace = {"parent_browser_suggestion_sha256": digest,
                 "parent_browser_suggestion_supplied": True}
        self.assertEqual(outcome_model._bound_browser_suggestion_supplied(
            trace, 2, prompt, parent, adapter), 1)
        self.assertEqual(outcome_model._bound_browser_suggestion_supplied(
            {}, 2, prompt, parent, adapter), 0)
        with self.assertRaisesRegex(ValueError, "was not supplied"):
            outcome_model._bound_browser_suggestion_supplied(trace, 2, "different prompt", parent, adapter)
        with self.assertRaisesRegex(ValueError, "verdict differs"):
            outcome_model._bound_browser_suggestion_supplied(trace, 2, prompt, parent,
                                                             {"verdict": {"suggested_next_prompt": "other"}})
        self.assertEqual(outcome_model._bound_browser_suggestion_supplied(
            {"parent_browser_suggestion_sha256": None,
             "parent_browser_suggestion_supplied": False}, 1, "first prompt", None, None), 0)
        with self.assertRaisesRegex(ValueError, "was not supplied"):
            outcome_model._bound_browser_suggestion_supplied(trace, 1, "first prompt", None, None)

    def test_local_benchmark_capture_keeps_keyed_prompt_and_result_only(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                prompt = "private revised benchmark prompt"
                final = "private exact Codex result"
                scenario = {"id": "variant", "prompt": prompt, "prompt_author": "codex",
                            "comparison_id": "pair", "variant_of": "original",
                            "files": {"verify.py": "pass\n"}, "protected_files": ["verify.py"],
                            "verify": ["python3", "verify.py"], "expected_final": "ready"}
                result = {"schema": "modellabs.benchmark-result.v2",
                          "run_id": "11111111-1111-1111-1111-111111111111",
                          "suite_id": "22222222-2222-2222-2222-222222222222",
                          "scenario_id": "variant", "scenario_sha256": stable_digest(scenario),
                          "product_sha256": stable_digest(product_definition(scenario)),
                          "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                          "comparison_id": "pair", "variant_of": "original",
                          "prompt_author": "codex", "model": "gpt-6-luna", "effort": "low",
                          "fixture_integrity": True, "codex_exit_code": 0,
                          "verifier_exit_code": 0, "product_pass": True,
                          "satisfaction_score": 100, "total_tokens": 123,
                          "elapsed_ms": 250}
                self.assertTrue(outcome_model.capture_benchmark_prompt_run(result, scenario, final))
                self.assertTrue(outcome_model.capture_benchmark_prompt_run(result, scenario, final))
                self.assertEqual(outcome_model.status()["local_prompt_experiments"],
                                 {"verified_runs_with_prompt_and_result": 1, "suites": 1,
                                  "comparison_groups": 1, "distinct_prompts": 1,
                                  "product_passes": 1,
                                  "training_role": "separate_local_benchmark_observational_only"})
                self.assertEqual(outcome_model.analyze_local_prompt_experiments()["paired_prompt_arms"], 0)
                self.assertEqual(outcome_model.train_local_prompt_model()["status"],
                                 "insufficient_managed_independent_products")
                self.assertNotIn(prompt.encode(), outcome_model.DB_PATH.read_bytes())
                self.assertNotIn(final.encode(), outcome_model.DB_PATH.read_bytes())
                with self.assertRaisesRegex(ValueError, "identity conflict"):
                    outcome_model.capture_benchmark_prompt_run(result, scenario, "different final")
                tampered = {**result, "product_sha256": "0" * 64}
                self.assertFalse(outcome_model.capture_benchmark_prompt_run(tampered, scenario, final))
                forged_execution = {**result, "model_provenance": "observed_per_request",
                                    "observed_model": "gpt-6-luna", "observed_effort": "low"}
                self.assertFalse(outcome_model.capture_benchmark_prompt_run(
                    forged_execution, scenario, final))
                original = {**scenario, "id": "original", "prompt": "private original prompt",
                            "prompt_author": "benchmark"}
                original.pop("variant_of")
                original_result = {**result,
                                   "run_id": "33333333-3333-3333-3333-333333333333",
                                   "scenario_id": "original",
                                   "scenario_sha256": stable_digest(original),
                                   "prompt_sha256": hashlib.sha256(original["prompt"].encode()).hexdigest(),
                                   "variant_of": None, "prompt_author": "benchmark"}
                self.assertTrue(outcome_model.capture_benchmark_prompt_run(
                    original_result, original, "private original result"))
                analysis = outcome_model.analyze_local_prompt_experiments()
                self.assertEqual((analysis["paired_prompt_arms"],
                                  analysis["independent_paired_products"],
                                  analysis["repeated_prompt_arms"]), (1, 1, 0))
                self.assertEqual(analysis["status"], "insufficient_independent_product_groups")

    def test_managed_benchmark_requires_precommit_canonical_execution_and_live_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.private_store(root):
                from smoke_bench import materialize, product_definition, stable_digest

                scenario = {"id": "managed-product", "prompt": "Build the verified product.",
                            "prompt_author": "benchmark", "comparison_id": "managed-pair",
                            "task_class": "simple", "files": {"verify.py": "pass\n"},
                            "protected_files": ["verify.py"],
                            "verify": ["python3", "verify.py"], "expected_final": "ready"}
                suite_id = "11111111-1111-1111-1111-111111111111"
                block_id = "22222222-2222-2222-2222-222222222222"
                block = outcome_model.create_managed_benchmark_block(
                    suite_id, block_id, scenario,
                    [("gpt-6-luna", "low"), ("gpt-6-sol", "medium")], 0)
                self.assertEqual(block["schema"], "modellabs.managed-benchmark-block.v1")
                self.assertEqual(len(block["arms"]), 2)
                self.assertEqual(block["scenario_task_class"], "simple")
                self.assertEqual(block["routing_task_class"], "routine")
                self.assertEqual(block["baseline_arm"], ["gpt-6-sol", "medium"])
                with self.assertRaisesRegex(ValueError, "router baseline"):
                    outcome_model.create_managed_benchmark_block(
                        suite_id, "77777777-7777-7777-7777-777777777777", scenario,
                        [("gpt-6-luna", "low"), ("gpt-6-astra", "high")], 0)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    outcome_model.create_managed_benchmark_block(
                        suite_id, block_id, scenario,
                        [("gpt-6-luna", "low"), ("gpt-6-sol", "medium")], 0)

                model, effort = block["arms"][0]
                thread_id = "33333333-3333-3333-3333-333333333333"
                turn_id = "44444444-4444-4444-4444-444444444444"
                prepared = outcome_model.prepare_codex_prompt(scenario["prompt"])
                outcome_model.capture_codex_turn(thread_id, turn_id, prepared,
                                                 model, effort, "routine")
                self.assertTrue(outcome_model.capture_codex_result(
                    thread_id, turn_id, "ready"))
                with outcome_model._connect() as connection:
                    created_at = connection.execute("SELECT created_at_ms FROM managed_benchmark_blocks").fetchone()[0]

                metrics = root / "metrics.jsonl"
                receipt_dir = root / "receipts"
                receipt_dir.mkdir()
                usage = {"inputTokens": 8, "cachedInputTokens": 0,
                         "cacheWriteInputTokens": 0, "outputTokens": 2,
                         "reasoningOutputTokens": 0, "totalTokens": 10}
                rows = [
                    {"event": "route_accepted", "receipt_id": f"{thread_id}:{turn_id}:accepted",
                     "recorded_at_ms": created_at + 1, "thread_id": thread_id,
                     "turn_id": turn_id, "model": model, "effort": effort,
                     "task_class": "routine", "task_bucket": "routine:modelControl"},
                    {"event": "turn_completed", "receipt_id": f"{thread_id}:{turn_id}:terminal",
                     "recorded_at_ms": created_at + 2, "thread_id": thread_id,
                     "turn_id": turn_id, "model": model, "effort": effort,
                     "task_class": "routine", "status": "completed", "source": "live_event"},
                    {"event": "turn_usage", "receipt_id": f"{thread_id}:{turn_id}:usage",
                     "recorded_at_ms": created_at + 3, "thread_id": thread_id,
                     "turn_id": turn_id, "model": model, "effort": effort,
                     "source": "proxy_thread_usage_delta", "usage": usage},
                    {"event": "route_execution_observed",
                     "receipt_id": f"{thread_id}:{turn_id}:execution",
                     "recorded_at_ms": created_at + 4, "thread_id": thread_id,
                     "turn_id": turn_id, "model": model, "effort": effort,
                     "source": "host_thread_settings_updated_pre_admission",
                     "settings_confirmation": "exact"},
                    {"event": "quality_grade", "recorded_at_ms": created_at + 5,
                     "thread_id": thread_id, "source_turn_id": turn_id,
                     "source": "explicit", "model": model, "effort": effort,
                     "task_class": "routine", "quality_score": 100,
                     "verification": "passed", "outcome": "verified"},
                ]
                for row in rows[:4]:
                    receipt = receipt_dir / f"{hashlib.sha256(row['receipt_id'].encode()).hexdigest()}.json"
                    receipt.write_text(json.dumps(row) + "\n")
                metrics.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
                self.assertEqual(outcome_model.sync_codex_grades(metrics),
                                 {"graded": 1, "usage_matched": 1,
                                  "observed": 1, "rerouted": 0})

                workspace = root / "workspace"
                workspace.mkdir()
                materialize(workspace, scenario)
                result = {"schema": "modellabs.managed-benchmark-result.v1",
                          "run_id": "55555555-5555-5555-5555-555555555555",
                          "suite_id": suite_id, "block_id": block_id, "arm_index": 0,
                          "thread_id": thread_id, "turn_id": turn_id,
                          "scenario_id": scenario["id"],
                          "scenario_sha256": stable_digest(scenario),
                          "product_sha256": stable_digest(product_definition(scenario)),
                          "prompt_sha256": hashlib.sha256(scenario["prompt"].encode()).hexdigest(),
                          "comparison_id": scenario["comparison_id"], "variant_of": None,
                          "prompt_author": "benchmark", "scenario_task_class": "simple",
                          "task_class": "routine",
                          "model": model, "effort": effort, "fixture_integrity": True,
                          "codex_exit_code": 0, "verifier_exit_code": 0,
                          "product_pass": True, "exact_final_response": True,
                          "satisfaction_score": 100, "total_tokens": 10,
                          "elapsed_ms": 50, "repetition_index": 0}
                self.assertTrue(outcome_model.capture_managed_benchmark_prompt_run(
                    result, scenario, "ready", workspace, metrics))
                self.assertTrue(outcome_model.capture_managed_benchmark_prompt_run(
                    result, scenario, "ready", workspace, metrics))
                with outcome_model._connect() as connection:
                    binding = connection.execute("""SELECT observed_model,observed_effort,
                        exact_total_tokens FROM managed_benchmark_bindings""").fetchone()
                self.assertEqual(binding, (model, effort, 10))
                self.assertEqual(outcome_model.status()["managed_benchmarks"], {
                    "precommitted_blocks": 1, "bound_runs": 1,
                    "complete_randomized_blocks": 0, "incomplete_randomized_blocks": 0,
                    "class_unproven_blocks": 0, "class_mismatch_blocks": 1,
                    "invalid_randomized_blocks": 0,
                    "independent_products": 0,
                    "training_role": "causal_candidate_not_yet_trained"})
                self.assertFalse(outcome_model.capture_managed_benchmark_prompt_run(
                    {**result, "model": "gpt-6-astra"}, scenario, "ready", workspace, metrics))
                (workspace / "verify.py").write_text("raise AssertionError('tampered')\n")
                self.assertFalse(outcome_model.capture_managed_benchmark_prompt_run(
                    {**result, "run_id": "66666666-6666-6666-6666-666666666666"},
                    scenario, "ready", workspace, metrics))
                with metrics.open("a", encoding="utf-8") as destination:
                    destination.write(json.dumps({
                        "event": "model_rerouted", "thread_id": thread_id, "turn_id": turn_id,
                        "assigned_model": model, "from_model": model,
                        "to_model": "gpt-6-astra", "reason": "policy"}) + "\n")
                with self.assertRaisesRegex(ValueError, "conflicts"):
                    outcome_model.managed_turn_evidence(thread_id, turn_id, metrics)
                execution_receipt = receipt_dir / f"{hashlib.sha256(
                    f'{thread_id}:{turn_id}:execution'.encode()).hexdigest()}.json"
                execution_receipt.unlink()
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    outcome_model.managed_turn_evidence(thread_id, turn_id, metrics)

    def test_managed_attempt_budget_excludes_class_unproven_and_mismatched_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                scenario = {"id": "attempt-product", "prompt": "Build the verified product.",
                            "prompt_author": "benchmark", "comparison_id": "attempt-pair",
                            "task_class": "routine", "files": {"verify.py": "pass\n"},
                            "protected_files": ["verify.py"],
                            "verify": ["python3", "verify.py"], "expected_final": "ready"}
                arms = [("gpt-6-luna", "low"), ("gpt-6-sol", "medium")]
                suite_id = "11111111-1111-1111-1111-111111111111"
                block_ids = [
                    "22222222-2222-2222-2222-222222222222",
                    "33333333-3333-3333-3333-333333333333",
                    "44444444-4444-4444-4444-444444444444",
                ]
                for repetition, block_id in enumerate(block_ids):
                    outcome_model.create_managed_benchmark_block(
                        suite_id, block_id, scenario, arms, repetition)
                key = outcome_model._key()
                with outcome_model._connect() as connection:
                    connection.execute("""UPDATE managed_benchmark_blocks
                        SET scenario_task_class=NULL WHERE block_key=?""",
                                       (outcome_model._digest(key, block_ids[1]),))
                    connection.execute("""UPDATE managed_benchmark_blocks
                        SET scenario_task_class='difficult' WHERE block_key=?""",
                                       (outcome_model._digest(key, block_ids[2]),))
                self.assertEqual(
                    outcome_model.managed_scenario_attempts([scenario], arms),
                    {"attempt-product": 1})

    def test_browser_benchmark_capture_requires_bound_review(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                review = {"guard_id": "review-1", "response_id": "resp_123",
                          "origin_sha256": "a" * 64}
                scenario = {"id": "browser-variant", "prompt": "Browser revision",
                            "browser_origin_prompt": "Original", "browser_review": review,
                            "prompt_author": "chatgpt", "comparison_id": "pair",
                            "variant_of": "base", "files": {"verify.py": "pass\n"},
                            "protected_files": ["verify.py"], "verify": ["python3", "verify.py"],
                            "expected_final": "ready"}
                result = {"schema": "modellabs.benchmark-result.v2",
                          "run_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                          "suite_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                          "scenario_id": scenario["id"], "scenario_sha256": stable_digest(scenario),
                          "product_sha256": stable_digest(product_definition(scenario)),
                          "prompt_sha256": hashlib.sha256(scenario["prompt"].encode()).hexdigest(),
                          "comparison_id": "pair", "variant_of": "base", "prompt_author": "chatgpt",
                          "browser_review": review, "model": "gpt-6-luna", "effort": "low",
                          "fixture_integrity": True, "codex_exit_code": 0,
                          "verifier_exit_code": 0, "product_pass": True,
                          "satisfaction_score": 100, "total_tokens": 123, "elapsed_ms": 250}
                with patch("smoke_bench.verified_browser_revision", return_value=review):
                    self.assertTrue(outcome_model.capture_benchmark_prompt_run(
                        result, scenario, "verified final"))
                    self.assertFalse(outcome_model.capture_benchmark_prompt_run(
                        {**result, "browser_review": {**review, "response_id": "other"}},
                        scenario, "verified final"))
                with patch("smoke_bench.verified_browser_revision", side_effect=ValueError("stale")):
                    self.assertFalse(outcome_model.capture_benchmark_prompt_run(
                        result, scenario, "verified final"))
                with outcome_model._connect() as connection:
                    row = connection.execute("SELECT source_kind,browser_guard_key,browser_response_key "
                                             "FROM benchmark_prompt_runs").fetchone()
                self.assertEqual(row[0], "browser_review_local_sandbox_benchmark")
                self.assertTrue(row[1] and row[2])
                self.assertNotIn(b"Browser revision", outcome_model.DB_PATH.read_bytes())

    def test_three_prompt_arms_compare_each_revision_to_one_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                product = {"files": {"verify.py": "pass\n"},
                           "protected_files": ["verify.py"], "verify": ["python3", "verify.py"],
                           "expected_final": "ready"}
                review = {"guard_id": "review-1", "response_id": "resp_123",
                          "origin_sha256": "a" * 64}
                for arm, author, tokens in ((1, "benchmark", 1000), (2, "codex", 800),
                                            (3, "chatgpt", 700)):
                    prompts = {1: "Baseline request for normalization",
                               2: "Codex revision checks malformed input carefully",
                               3: "Browser revision verifies stdout and error behavior"}
                    scenario = {**product, "id": f"arm-{arm}", "comparison_id": "pair",
                                "prompt": prompts[arm], "prompt_author": author}
                    if arm > 1:
                        scenario["variant_of"] = "arm-1"
                    if author == "chatgpt":
                        scenario.update(browser_review=review,
                                        browser_origin_prompt=prompts[1])
                    for repeat in range(3):
                        result = {"schema": "modellabs.benchmark-result.v2",
                                  "run_id": f"{arm:08x}-0000-0000-0000-{repeat + 1:012x}",
                                  "suite_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                                  "scenario_id": scenario["id"],
                                  "scenario_sha256": stable_digest(scenario),
                                  "product_sha256": stable_digest(product_definition(scenario)),
                                  "prompt_sha256": hashlib.sha256(scenario["prompt"].encode()).hexdigest(),
                                  "comparison_id": "pair", "variant_of": scenario.get("variant_of"),
                                  "prompt_author": author, "model": "gpt-6-luna", "effort": "low",
                                  "fixture_integrity": True, "codex_exit_code": 0,
                                  "verifier_exit_code": 0, "product_pass": True,
                                  "satisfaction_score": 100, "total_tokens": tokens,
                                  "elapsed_ms": 100, "repetition_index": repeat}
                        if author == "chatgpt":
                            result["browser_review"] = review
                        with patch("smoke_bench.verified_browser_revision", return_value=review):
                            self.assertTrue(outcome_model.capture_benchmark_prompt_run(
                                result, scenario, f"result {arm} {repeat}"))
                analysis = outcome_model.analyze_local_prompt_experiments()
                self.assertEqual(analysis["paired_prompt_arms"], 2)
                self.assertEqual(analysis["repeated_prompt_arms"], 2)
                self.assertEqual(analysis["browser_authored_paired_arms"], 1)
                self.assertEqual(analysis["comparison_outcomes"]["conclusive_pairs"], 2)
                self.assertEqual(analysis["comparison_outcomes"]["chatgpt_candidate_wins"], 1)
                self.assertEqual(analysis["conclusive_independent_products"], 1)

    def test_observational_prompt_products_never_train_the_managed_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                for product_number in range(8):
                    product = {"files": {"verify.py": f"assert {product_number} == {product_number}\n"},
                               "protected_files": ["verify.py"],
                               "verify": ["python3", "verify.py"], "expected_final": "ready"}
                    for variant_number, author in enumerate(("benchmark", "codex")):
                        scenario = {**product, "id": f"product-{product_number}-{author}",
                                    "comparison_id": f"comparison-{product_number}",
                                    "prompt_author": author,
                                    "prompt": f"private {author} prompt for product {product_number}"}
                        if author == "codex":
                            scenario["variant_of"] = f"product-{product_number}-benchmark"
                        for repeat in range(3):
                            result = {"schema": "modellabs.benchmark-result.v2",
                                      "run_id": f"{product_number + 1:08x}-0000-0000-0000-{variant_number * 3 + repeat + 1:012x}",
                                      "suite_id": f"{product_number + 1:08x}-1111-1111-1111-111111111111",
                                      "scenario_id": scenario["id"],
                                      "scenario_sha256": stable_digest(scenario),
                                      "product_sha256": stable_digest(product_definition(scenario)),
                                      "prompt_sha256": hashlib.sha256(scenario["prompt"].encode()).hexdigest(),
                                      "comparison_id": scenario["comparison_id"],
                                      "variant_of": scenario.get("variant_of"),
                                      "prompt_author": author, "model": "gpt-6-luna", "effort": "low",
                                      "model_provenance": "cli_requested_only", "observed_model": None,
                                      "fixture_integrity": True, "codex_exit_code": 0,
                                      "verifier_exit_code": 0, "product_pass": True,
                                      "satisfaction_score": 100,
                                      "total_tokens": (800 if product_number % 2 == 0 else 1200)
                                      if author == "codex" else 1000,
                                      "elapsed_ms": 100, "repetition_index": repeat}
                            self.assertTrue(outcome_model.capture_benchmark_prompt_run(
                                result, scenario, f"private answer {product_number} {author} {repeat}"))
                report = outcome_model.train_local_prompt_model()
                self.assertEqual(report["status"], "insufficient_managed_independent_products")
                self.assertEqual(report["independent_products"], 0)
                self.assertEqual(report["observational_independent_products"], 8)
                self.assertEqual(report["comparison_counts"]["conclusive_pairs"], 8)
                self.assertFalse(report["authoritative_for_routing"])
                self.assertNotIn(b"private revised prompt", outcome_model.DB_PATH.read_bytes())
                self.assertFalse(outcome_model.LOCAL_PROMPT_MODEL_PATH.exists())

    def test_unblocked_legacy_token_medians_do_not_become_prompt_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                product = {"files": {"verify.py": "pass\n"}, "protected_files": ["verify.py"],
                           "verify": ["python3", "verify.py"], "expected_final": "ready"}
                for author, tokens in (("benchmark", 1000), ("codex", 700)):
                    scenario = {**product, "id": author, "comparison_id": "legacy",
                                "prompt_author": author, "prompt": f"private {author} prompt"}
                    if author == "codex":
                        scenario["variant_of"] = "benchmark"
                    for repeat in range(3):
                        result = {"schema": "modellabs.benchmark-result.v2",
                                  "run_id": f"{repeat + 1:08x}-0000-0000-0000-{(author == 'codex') + 1:012x}",
                                  "suite_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                                  "scenario_id": author, "scenario_sha256": stable_digest(scenario),
                                  "product_sha256": stable_digest(product_definition(scenario)),
                                  "prompt_sha256": hashlib.sha256(scenario["prompt"].encode()).hexdigest(),
                                  "comparison_id": "legacy", "variant_of": scenario.get("variant_of"),
                                  "prompt_author": author, "model": "gpt-6-luna", "effort": "low",
                                  "fixture_integrity": True, "codex_exit_code": 0,
                                  "verifier_exit_code": 0, "product_pass": True,
                                  "satisfaction_score": 100, "total_tokens": tokens,
                                  "elapsed_ms": 100}
                        self.assertTrue(outcome_model.capture_benchmark_prompt_run(
                            result, scenario, f"private result {author} {repeat}"))
                _groups, _proofs, counts = outcome_model._local_prompt_comparisons()
                self.assertEqual(counts["unblocked_pairs"], 1)
                self.assertNotIn("conclusive_pairs", counts)

    def test_adopted_browser_review_and_product_outcomes_stay_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                key = outcome_model._key()
                parent_id = outcome_model._digest(key, "resp_parent")
                child_id = outcome_model._digest(key, "resp_child")
                candidate_prompt = "private browser revision"
                candidate_digest = outcome_model._digest(key, candidate_prompt)
                with outcome_model._connect() as connection:
                    for episode_id, round_number, score, passed in (
                            (parent_id, 1, 87, 0), (child_id, 2, 100, 1)):
                        connection.execute("""INSERT INTO episodes
                            (episode_id,group_id,source,submitted_at,round_number,prompt_role,
                             prompt_digest,revision_digest,features,quality_score,passed,
                             source_response_id,feature_version)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (episode_id, "task-group", "auracall_pro_guard",
                             f"2026-09-24T00:0{round_number}:00Z", round_number,
                             "review_goal", "origin-digest", "revision-digest", "{}",
                             score, passed, f"resp_{round_number}", outcome_model.FEATURE_VERSION))
                    connection.execute("""INSERT INTO iteration_traces
                        (episode_id,root_key,parent_episode_id,origin_prompt_digest,
                         generation_prompt_digest,prompt_author,prompt_features,
                         submitted_at,passed,adopted_parent_suggestion)
                        VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (child_id, "root", parent_id, "origin-digest", candidate_digest,
                         "chatgpt", "{}", "2026-09-24T00:02:00Z", 1, 1))
                product = {"files": {"verify.py": "pass\n"}, "protected_files": ["verify.py"],
                           "verify": ["python3", "verify.py"], "expected_final": "ready"}
                review = {"guard_id": "guard-parent", "response_id": "resp_parent",
                          "origin_sha256": "a" * 64}
                base = {**product, "id": "base", "prompt": "private original",
                        "prompt_author": "benchmark", "comparison_id": "pair"}
                candidate = {**product, "id": "candidate", "prompt": candidate_prompt,
                             "prompt_author": "chatgpt", "comparison_id": "pair",
                             "variant_of": "base", "browser_origin_prompt": base["prompt"],
                             "browser_review": review}
                with patch("smoke_bench.verified_browser_revision", return_value=review):
                    for block in range(3):
                        for ordinal, scenario, tokens in ((1, base, 1000), (2, candidate, 1200)):
                            result = {"schema": "modellabs.benchmark-result.v2",
                                      "run_id": f"{block + 1:08x}-0000-0000-0000-{ordinal:012x}",
                                      "suite_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                                      "scenario_id": scenario["id"],
                                      "scenario_sha256": stable_digest(scenario),
                                      "product_sha256": stable_digest(product_definition(scenario)),
                                      "prompt_sha256": hashlib.sha256(scenario["prompt"].encode()).hexdigest(),
                                      "comparison_id": "pair", "variant_of": scenario.get("variant_of"),
                                      "prompt_author": scenario["prompt_author"],
                                      "browser_review": scenario.get("browser_review"),
                                      "model": "gpt-6-sol", "effort": "medium",
                                      "fixture_integrity": True, "codex_exit_code": 0,
                                      "verifier_exit_code": 0, "product_pass": True,
                                      "satisfaction_score": 100, "total_tokens": tokens,
                                      "elapsed_ms": 100, "repetition_index": block}
                            self.assertTrue(outcome_model.capture_benchmark_prompt_run(
                                result, scenario, f"private answer {block} {ordinal}"))
                report = outcome_model.analyze_browser_revision_product_outcomes()
                self.assertEqual(report["exact_matched_blocks"], 3)
                self.assertEqual(report["repeated_linked_groups"], 1)
                self.assertEqual(report["review_score_improved_groups"], 1)
                self.assertEqual(report["both_product_pass_groups"], 1)
                self.assertEqual(report["candidate_more_tokens_every_block_groups"], 1)
                self.assertEqual(report["review_improved_without_token_gain_groups"], 1)
                self.assertEqual(report["execution_verified_groups"], 0)
                self.assertEqual(report["execution_verified_independent_products"], 0)
                with outcome_model._connect() as connection:
                    connection.execute("UPDATE benchmark_prompt_runs SET model_provenance='observed_per_request'")
                label_only = outcome_model.analyze_browser_revision_product_outcomes()
                self.assertEqual(label_only["execution_verified_groups"], 0)
                managed_sets = []
                for repeat in range(3):
                    managed_sets.append({
                        "prompt_set_key": f"prompt-set-{repeat}",
                        "suite_key": outcome_model._digest(
                            key, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                        "created_at_ms": 100 + repeat,
                        "comparison_key": outcome_model._digest(key, "pair"),
                        "product_key": outcome_model._digest(
                            key, stable_digest(product_definition(base))),
                        "task_class": "routine", "repetition_index": repeat,
                        "prompts": [
                            {"prompt_digest": outcome_model._digest(key, base["prompt"]),
                             "prompt_features": {}, "prompt_author": "benchmark",
                             "browser_guard_key": None, "browser_response_key": None,
                             "arms": {("gpt-6-sol", "medium"): {
                                 "product_pass": True, "quality_score": 100,
                                 "total_tokens": 1000}}},
                            {"prompt_digest": candidate_digest,
                             "prompt_features": {}, "prompt_author": "chatgpt",
                             "browser_guard_key": outcome_model._digest(key, "guard-parent"),
                             "browser_response_key": parent_id,
                             "arms": {("gpt-6-sol", "medium"): {
                                 "product_pass": True, "quality_score": 100,
                                 "total_tokens": 1200}}},
                        ],
                    })
                with patch.object(outcome_model, "_managed_complete_prompt_sets",
                                  return_value=(managed_sets, {})):
                    managed = outcome_model.analyze_browser_revision_product_outcomes()
                    self.assertEqual(managed["execution_verified_groups"], 1)
                    self.assertEqual(managed["execution_verified_product_losses"], 1)
                    forged_sets = [{**item, "prompts": [
                        {**item["prompts"][0], "prompt_digest": "forged"},
                        item["prompts"][1],
                    ]} for item in managed_sets]
                    with patch.object(outcome_model, "_managed_complete_prompt_sets",
                                      return_value=(forged_sets, {})):
                        forged = outcome_model.analyze_browser_revision_product_outcomes()
                    self.assertEqual(forged["execution_verified_groups"], 0)
                with outcome_model._connect() as connection:
                    connection.execute("""UPDATE benchmark_prompt_runs
                                       SET observed_model='gpt-6-sol', observed_effort='medium'""")
                verified_loss = outcome_model.analyze_browser_revision_product_outcomes()
                self.assertEqual(verified_loss["execution_verified_independent_products"], 1)
                self.assertEqual(verified_loss["execution_verified_product_losses"], 1)
                self.assertEqual(verified_loss["execution_verified_product_wins"], 0)
                with outcome_model._connect() as connection:
                    connection.execute("""UPDATE benchmark_prompt_runs SET observed_model='gpt-6-luna'
                                       WHERE prompt_author='chatgpt'""")
                mismatch = outcome_model.analyze_browser_revision_product_outcomes()
                self.assertEqual(mismatch["execution_verified_groups"], 0)
                with outcome_model._connect() as connection:
                    connection.execute("""UPDATE benchmark_prompt_runs
                                       SET total_tokens=800, observed_model='gpt-6-sol'
                                       WHERE prompt_author='chatgpt'""")
                verified_win = outcome_model.analyze_browser_revision_product_outcomes()
                self.assertEqual(verified_win["execution_verified_product_wins"], 1)
                self.assertEqual(verified_win["execution_verified_product_losses"], 0)
                self.assertFalse(report["authoritative_for_routing"])
                with patch("smoke_bench.verified_browser_revision", return_value=review):
                    duplicate = {**result, "run_id": "dddddddd-0000-0000-0000-000000000000",
                                 "repetition_index": 0}
                    self.assertTrue(outcome_model.capture_benchmark_prompt_run(
                        duplicate, candidate, "private duplicate answer"))
                ambiguous = outcome_model.analyze_browser_revision_product_outcomes()
                self.assertEqual(ambiguous["ambiguous_matched_blocks"], 1)
                self.assertEqual(ambiguous["repeated_linked_groups"], 0)

    def test_export_codex_result_requires_exact_completed_prompt_and_private_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = root / "sessions"
            sessions.mkdir()
            private = root / "private"
            private.mkdir(mode=0o700)
            thread = "11111111-1111-1111-1111-111111111111"
            turn = "22222222-2222-2222-2222-222222222222"
            prompt = private / "generation.txt"
            prompt.write_text("exact generation prompt\n", encoding="utf-8")
            result = "private Codex final answer"
            records = [
                {"type": "session_meta", "payload": {"id": thread}},
                {"type": "event_msg", "payload": {"type": "task_started", "turn_id": turn}},
                {"type": "response_item", "payload": {"type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": "exact generation prompt"}]}},
                {"type": "response_item", "payload": {"type": "message", "role": "assistant"}},
                {"type": "event_msg", "payload": {"type": "task_complete", "turn_id": turn,
                    "last_agent_message": result}},
            ]
            (sessions / f"rollout-{thread}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
            output = private / "answer.txt"
            staged = outcome_model.export_codex_result(thread, turn, prompt, output, sessions)
            self.assertEqual(output.read_text(encoding="utf-8"), result)
            self.assertEqual(staged["result_sha256"], hashlib.sha256(result.encode()).hexdigest())
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            turn_dir = private / "turn"
            turn_dir.mkdir(mode=0o700)
            pair = outcome_model.export_codex_turn(thread, turn, turn_dir, sessions)
            self.assertEqual((turn_dir / "generation-prompt.txt").read_text(), "exact generation prompt")
            self.assertEqual((turn_dir / "codex-final.txt").read_text(), result)
            self.assertEqual(pair["generation_prompt_sha256"],
                             hashlib.sha256(b"exact generation prompt").hexdigest())
            self.assertEqual(pair["codex_result_sha256"], hashlib.sha256(result.encode()).hexdigest())
            self.assertEqual((turn_dir / "generation-prompt.txt").stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                outcome_model.export_codex_turn(thread, turn, turn_dir, sessions)
            blocked_dir = private / "blocked"
            blocked_dir.mkdir(mode=0o700)
            (blocked_dir / "codex-final.txt").write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                outcome_model.export_codex_turn(thread, turn, blocked_dir, sessions)
            self.assertFalse((blocked_dir / "generation-prompt.txt").exists())
            with self.assertRaises(FileExistsError):
                outcome_model.export_codex_result(thread, turn, prompt, output, sessions)
            prompt.write_text("different prompt\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact single-prompt"):
                outcome_model.export_codex_result(thread, turn, prompt, private / "wrong.txt", sessions)

    def test_managed_backfill_requires_exact_completed_session_and_never_invents_grade(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.private_store(root):
                key = outcome_model._key()
                prompt = "private original prompt"
                result = "private completed result"
                prepared = outcome_model.prepare_codex_prompt(prompt)
                with outcome_model._connect() as connection:
                    for turn_id, model in (("good", "gpt-6-sol"), ("mismatch", "gpt-6-luna")):
                        connection.execute("""INSERT INTO session_turns
                            (thread_key,turn_key,prompt_digest,prompt_features,result_digest,
                             result_features,model,effort,completed_at,source_path_digest,input_scope)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                            (outcome_model._digest(key, "thread"), outcome_model._digest(key, turn_id),
                             prepared["prompt_digest"], prepared["features"],
                             outcome_model._digest(key, result),
                             json.dumps(outcome_model._features(result, "", 1, key)),
                             model, "medium", "2026-09-01T00:00:00+00:00", "source",
                             "single_pre_inference_message"))
                metrics = root / "metrics.jsonl"
                rows = []
                for turn_id in ("good", "mismatch"):
                    rows.extend((
                        {"event": "route_accepted", "thread_id": "thread", "turn_id": turn_id,
                         "model": "gpt-6-sol", "effort": "medium", "task_class": "routine",
                         "recorded_at_ms": 1790000000000},
                        {"event": "turn_completed", "thread_id": "thread", "turn_id": turn_id,
                         "model": "gpt-6-sol", "effort": "medium", "status": "completed"},
                    ))
                metrics.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
                self.assertEqual(outcome_model.backfill_managed_turns(metrics),
                                 {"imported": 1, "unchanged": 0, "conflict": 0, "ineligible": 1})
                self.assertEqual(outcome_model.backfill_managed_turns(metrics),
                                 {"imported": 0, "unchanged": 1, "conflict": 0, "ineligible": 1})
                codex = outcome_model.status()["codex"]
                self.assertEqual(codex["accepted_turns"], 1)
                self.assertEqual(codex["result_captured_turns"], 1)
                self.assertEqual(codex["graded_turns"], 0)
                self.assertNotIn(prompt.encode(), outcome_model.DB_PATH.read_bytes())
                self.assertNotIn(result.encode(), outcome_model.DB_PATH.read_bytes())

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
                             result_digest,observed_model,observed_effort,model_provenance)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (f"thread-{number}", f"turn-{number}", f"group-{number}", observed_at_ms,
                             f"prompt-{number}", features, model, "low", "simple",
                             95 if number % 2 else 40, "passed" if number % 2 else "failed",
                             100 + number, f"result-{number}", model, "low", "observed_per_request"))

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

    def test_review_score_model_uses_numeric_grades_and_frozen_future_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                key = outcome_model._key()

                def add_episode(number, submitted_at):
                    score = (10, 30, 50, 70, 90)[number % 5]
                    features = json.dumps(outcome_model._features(
                        "original task", f"candidate score {number}", 1, key))
                    with outcome_model._connect() as connection:
                        connection.execute("""INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                           (f"score-episode-{number}", f"score-group-{number}",
                                            "auracall_pro_guard", submitted_at, 1, "review_goal",
                                            f"score-prompt-{number}", f"score-revision-{number}", features,
                                            None, None, None, score, int(score >= 90),
                                            "nonce_bound_pro_guard", f"score-response-{number}",
                                            f"score-guard-{number}", outcome_model.FEATURE_VERSION))

                for number in range(40):
                    add_episode(number, f"2026-09-01T00:{number:02d}:00+00:00")
                outcome_model.train_review_model()
                first = outcome_model.train_review_score_model()
                self.assertEqual(first["trained_episodes"], 40)
                self.assertEqual(first["prospective_holdout_episodes"], 0)
                self.assertFalse(first["validated_for_shadow"])
                prediction = outcome_model.predict_review("original task", "candidate score 41", 1)
                self.assertGreaterEqual(prediction["predicted_review_score"], 0)
                self.assertLessEqual(prediction["predicted_review_score"], 100)
                frozen = outcome_model.SCORE_EVAL_PATH.read_bytes()
                checkpoint = json.loads(frozen)
                start = datetime.fromisoformat(checkpoint["created_at"])
                for number in range(40, 60):
                    add_episode(number, (start + timedelta(seconds=number)).isoformat())
                later = outcome_model.train_review_score_model()
                self.assertEqual(later["prospective_holdout_episodes"], 20)
                self.assertEqual(later["prospective_holdout_task_groups"], 20)
                self.assertEqual(outcome_model.SCORE_EVAL_PATH.read_bytes(), frozen)

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
                record["bundle"]["run"]["initialInputs"]["model"] = "changed-model"
                (run_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")
                changed = outcome_model.import_auracall(guard_root, runs_root, "fixture")
                self.assertEqual(changed["conflict"], 1)
                self.assertEqual(changed["unchanged"], 0)
                record["bundle"]["run"]["initialInputs"]["model"] = "gpt-5.2"
                (run_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")
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

    def test_bound_codex_result_links_a_review_packet_without_equating_packet_to_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guards, runs = root / "guards", root / "runs"
            guards.mkdir()
            (runs / "resp_bound").mkdir(parents=True)
            thread = "019f9999-0000-7000-8000-000000000021"
            turn = "019f9999-0000-7000-8000-000000000022"
            prompt, answer = "implement the bounded change", "final answer\n"
            sources = {}
            for label, value in (("origin_prompt", "original user request"),
                                 ("generation_prompt", prompt), ("codex_result", answer)):
                path = root / label
                path.write_text(value, encoding="utf-8")
                sources[label] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            trace = {"root_guard_id": "bound", "parent_guard_id": None,
                     "parent_response_id": None, "prompt_author": "user",
                     "origin_prompt_sha256": sources["origin_prompt"]["sha256"],
                     "generation_prompt_sha256": sources["generation_prompt"]["sha256"],
                     "codex_thread_id": thread, "codex_turn_id": turn,
                     "codex_result_sha256": sources["codex_result"]["sha256"]}
            digest = hashlib.sha256(json.dumps(trace, sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest()
            guard = {"status": "completed", "response_id": "resp_bound", "guard_id": "bound",
                     "nonce": "nonce-bound", "round": 1, "submitted_at": "2026-09-23T00:00:00Z",
                     "submission_fingerprint": "fingerprint-bound", "source_files": sources,
                     "learning_trace": trace, "learning_trace_digest": digest,
                     "verdict": {"summary": "reviewed packet", "blocking_findings": [],
                                 "tests_or_checks_required": []},
                     "evaluation": {"valid": True, "nonce_matched": True,
                                    "passed": True, "score": 96}}
            record = {"runId": "resp_bound", "bundle": {"run": {
                "id": "resp_bound", "status": "succeeded", "initialInputs": {
                    "model": "gpt-5.2", "instructions":
                        "Review the supplied artifact against this goal:\noriginal user request\n\nFRESHNESS_NONCE: nonce-bound",
                    "requestInput": "review packet with checks plus final answer",
                    "metadata": {"workflow": "codex-pro-guard", "guard_id": "bound",
                                 "guard_nonce": "nonce-bound", "round": 1,
                                 "submission_fingerprint": "fingerprint-bound",
                                 "learning_trace_digest": digest}}}}}
            record["bundle"]["sharedState"] = {"structuredOutputs": [{
                "key": "response.output", "value": [{"role": "assistant", "content": [{
                    "type": "output_text", "text": "nonce-bound Browser answer with graded findings"}]}]}]}
            guard_path = guards / "bound.json"
            guard_path.write_text(json.dumps(guard), encoding="utf-8")
            (runs / "resp_bound" / "record.json").write_text(json.dumps(record), encoding="utf-8")
            with self.private_store(root):
                prepared = outcome_model.prepare_codex_prompt(prompt)
                outcome_model.capture_codex_turn(thread, turn, prepared,
                                                 "gpt-6-sol", "medium", "routine")
                outcome_model.capture_codex_result(thread, turn, answer)
                with outcome_model._connect() as connection:
                    key = outcome_model._key()
                    wrong = (outcome_model._digest(key, thread),
                             outcome_model._digest(key, turn),
                             outcome_model._digest(key, "different answer"))
                    self.assertEqual(outcome_model._link_review_to_codex(
                        connection, "not-an-episode", prepared["prompt_digest"],
                        outcome_model._digest(key, "review packet"), wrong), "no_match")
                imported = outcome_model.import_auracall(guards, runs)
                self.assertEqual(imported["codex_link_imported"], 1)
                self.assertEqual(imported["browser_result_imported"], 1)
                self.assertEqual(outcome_model.status()["browser_results"]["nonce_bound_feature_records"], 1)
                self.assertEqual(outcome_model.status()["browser_codex_links"]
                                 ["exact_prompt_result_review_links"], 1)
                self.assertEqual(outcome_model.status()["browser_codex_links"]
                                 ["unlinked_review_traces"], 0)
                with outcome_model._connect() as connection:
                    episode = outcome_model._digest(outcome_model._key(), "resp_bound")
                    thread_key = outcome_model._digest(outcome_model._key(), thread)
                    turn_key = outcome_model._digest(outcome_model._key(), turn)
                    answer_digest = outcome_model._digest(outcome_model._key(), answer)
                    connection.execute("""UPDATE codex_turns SET result_digest=?
                                          WHERE thread_key=? AND turn_key=?""",
                                       ("changed-result", thread_key, turn_key))
                    self.assertEqual(outcome_model._link_review_to_codex(
                        connection, episode, prepared["prompt_digest"],
                        outcome_model._digest(outcome_model._key(), "review packet"),
                        (thread_key, turn_key, answer_digest)), "no_match")
                    self.assertNotIn(episode, outcome_model._linked_codex_outcomes(connection))
                    connection.commit()
                    self.assertEqual(outcome_model.status()["browser_codex_links"]
                                     ["exact_prompt_result_review_links"], 0)
                    self.assertEqual(outcome_model.status()["browser_codex_links"]
                                     ["unlinked_prompt_seen_managed"], 1)
                    self.assertEqual(outcome_model.status()["browser_codex_links"]
                                     ["unlinked_prompt_absent_both"], 0)
                    connection.execute("""UPDATE codex_turns SET result_digest=?
                                          WHERE thread_key=? AND turn_key=?""",
                                       (answer_digest, thread_key, turn_key))
                    self.assertEqual(outcome_model._link_review_to_codex(
                        connection, episode, prepared["prompt_digest"],
                        outcome_model._digest(outcome_model._key(), "review packet"),
                        (thread_key, turn_key, answer_digest)), "unchanged")
                    linked = outcome_model._linked_codex_outcomes(connection)
                    self.assertEqual(linked[episode][1:], (None, None))
                    connection.execute("UPDATE codex_turns SET total_tokens=123 WHERE thread_key=? AND turn_key=?",
                                       (outcome_model._digest(outcome_model._key(), thread),
                                        outcome_model._digest(outcome_model._key(), turn)))
                    linked = outcome_model._linked_codex_outcomes(connection)
                    self.assertEqual(linked[episode][1:], (None, None))
                    connection.execute("""UPDATE codex_turns SET observed_model=?,observed_effort=?,
                                       model_provenance='observed_per_request'
                                       WHERE thread_key=? AND turn_key=?""",
                                       ("gpt-6-sol", "medium",
                                        outcome_model._digest(outcome_model._key(), thread),
                                        outcome_model._digest(outcome_model._key(), turn)))
                    linked = outcome_model._linked_codex_outcomes(connection)
                    self.assertEqual(linked[episode][1:], ("gpt-6-sol", "medium"))
                    connection.execute("UPDATE review_codex_links SET link_status='ambiguous' WHERE episode_id=?",
                                       (episode,))
                    self.assertNotIn(episode, outcome_model._linked_codex_outcomes(connection))
                self.assertNotIn(b"final answer", outcome_model.DB_PATH.read_bytes())
                self.assertNotIn(b"Browser answer", outcome_model.DB_PATH.read_bytes())
                sources["codex_result"]["sha256"] = "0" * 64
                guard_path.write_text(json.dumps(guard), encoding="utf-8")
                self.assertEqual(outcome_model.import_auracall(guards, runs)["trace_rejected"], 1)

    def test_bound_legacy_loop_import_requires_record_and_trace_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loop_root, runs_root = root / "loops", root / "runs"
            loop_dir = loop_root / "fixture-loop"
            round_dir = loop_dir / "round-1"
            round_dir.mkdir(parents=True)
            response_id, nonce = "resp_bound_loop", "codex-pro-guard-1-fixture"
            goal, prompt, final = "private original goal", "private generated prompt", "private Codex answer"
            review_prompt = "Review the private artifact."
            browser_answer = json.dumps({"nonce": nonce, "pass": True, "score": 95,
                                         "summary": "private browser feedback",
                                         "blocking_findings": [], "tests_or_checks_required": []})
            verdict = json.loads(browser_answer)
            conversation = "https://chatgpt.com/c/fixture-loop"
            manifest = [{"path": str(root / "artifact.txt"), "sha256": "a" * 64}]
            trace = {"schema": "modellabs.auracall_loop_trace.v1", "run_id": "fixture-loop",
                     "round": 1, "origin_prompt_source": "loop_goal_argument",
                     "origin_prompt_sha256": hashlib.sha256(goal.encode()).hexdigest(),
                     "generation_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "prompt_author": "mixed",
                     "review_prompt_sha256": hashlib.sha256(review_prompt.encode()).hexdigest(),
                     "attachment_manifest_sha256": hashlib.sha256(json.dumps(
                         manifest, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()).hexdigest(),
                     "codex_result_sha256": hashlib.sha256(final.encode()).hexdigest(),
                     "codex_thread_id": "thread-fixture", "parent_response_id": None}
            trace_path = round_dir / "learning-trace.json"
            trace_path.write_text(json.dumps(trace, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False) + "\n")
            trace_digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            adapter_path = round_dir / "auracall-pro-guard-state.json"
            adapter = {"schema": "codex.auracall_pro_guard_review.v1", "status": "completed",
                       "created_at": "2026-09-24T00:00:00Z", "nonce": nonce,
                       "client_request_id": "client-fixture", "response_id": response_id,
                       "active_response_id": response_id, "correction_response_id": None,
                       "response_read": {"id": response_id}, "conversation_url": conversation,
                       "verdict": verdict,
                       "request_material": {"prompt": review_prompt, "files": manifest,
                                            "model": "gpt-5.2", "runtime_profile": "agent-browser-chatgpt",
                                            "schema": "guard", "round": 1,
                                            "conversation_url": conversation,
                                            "learning_trace_digest": trace_digest}}
            adapter_path.write_text(json.dumps(adapter))
            config = {"goal": goal}
            loop_state = {"run_id": "fixture-loop", "config": config,
                          "config_digest": hashlib.sha256(json.dumps(config, sort_keys=True,
                               separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
                          "rounds": [{"round": 1, "iteration_prompt": prompt,
                                      "verification": {"status": "passed", "returncode": 0},
                                      "work": {"status": "completed", "final_message": final,
                                               "thread_id": "thread-fixture"},
                                      "review": {"status": "completed", "verdict": verdict,
                                                 "learning_trace_status": "bound_at_submission",
                                                 "learning_trace_file": str(trace_path),
                                                 "adapter_state_path": str(adapter_path)}}]}
            (loop_dir / "loop.json").write_text(json.dumps(loop_state))
            record = {"runId": response_id, "bundle": {
                "run": {"id": response_id, "status": "succeeded", "initialInputs": {
                    "model": "gpt-5.2", "requestInput":
                        f"FRESHNESS_NONCE: {nonce}\nYour JSON object must include exactly this nonce in its nonce field.\n\n{review_prompt}",
                    "attachments": [{"id": "codex-pro-guard-1", "fileName": "artifact.txt",
                                     "uri": (root / "artifact.txt").as_uri()}],
                    "auracall": {"runtimeProfile": "agent-browser-chatgpt", "service": "chatgpt",
                                 "chatgptConversationUrl": conversation},
                    "metadata": {"workflow": "codex-pro-guard", "schema": "guard",
                                 "nonce": nonce, "round": 1,
                                 "client_request_id": "client-fixture",
                                 "learning_trace_digest": trace_digest}}},
                "steps": [{"status": "succeeded", "output": {"structuredData": {"browserRun": {
                    "service": "chatgpt", "runtimeProfileId": "agent-browser-chatgpt",
                    "tabUrl": conversation, "promptTransport": {
                        "attachments": [{"path": str(root / "artifact.txt"),
                                         "displayPath": str(root / "artifact.txt")}],
                        "metadata": {"mode": "inline"}},
                    "attachmentUiReceipt": {
                        "schema": "auracall.browser_attachment_ui_receipt.v1",
                        "attachmentPaths": [str(root / "artifact.txt")],
                        "uploadCompletion": "confirmed",
                        "sentUserTurnAttachments": "confirmed",
                        "submittedUserId": "user-fixture"}}}}}],
                "sharedState": {"structuredOutputs": [{"key": "response.output",
                    "value": [{"role": "assistant", "content": [{"type": "output_text",
                        "text": browser_answer}]}]}]}}}
            record_dir = runs_root / response_id
            record_dir.mkdir(parents=True)
            record_path = record_dir / "record.json"
            record_path.write_text(json.dumps(record))
            with self.private_store(root):
                first = outcome_model.import_bound_loops(loop_root, runs_root)
                self.assertEqual(first["imported"], 1)
                self.assertEqual(outcome_model.import_bound_loops(loop_root, runs_root)["unchanged"], 1)
                status = outcome_model.status()["bound_legacy_loop"]
                self.assertEqual(status["rounds"], 1)
                self.assertEqual(status["codex_result_feature_records"], 1)
                self.assertEqual(status["provider_verified_attachment_contents"], 0)
                self.assertEqual(status["browser_dispatch_bound_rounds"], 1)
                self.assertEqual(status["browser_ui_confirmed_rounds"], 1)
                self.assertEqual(status["browser_suggestion_supplied_rounds"], 0)
                raw = outcome_model.DB_PATH.read_bytes()
                for secret in (goal, prompt, final, browser_answer):
                    self.assertNotIn(secret.encode(), raw)
                record["bundle"]["run"]["initialInputs"]["attachments"][0]["uri"] = "file:///wrong"
                record_path.write_text(json.dumps(record))
                self.assertEqual(outcome_model.import_bound_loops(loop_root, runs_root)["rejected"], 1)
                record["bundle"]["run"]["initialInputs"]["attachments"][0]["uri"] = (
                    root / "artifact.txt").as_uri()
                record["bundle"]["steps"][0]["output"]["structuredData"]["browserRun"][
                    "promptTransport"]["attachments"][0]["path"] = "/wrong"
                record_path.write_text(json.dumps(record))
                self.assertEqual(outcome_model.import_bound_loops(loop_root, runs_root)["conflict"], 1)
                record["bundle"]["steps"][0]["output"]["structuredData"]["browserRun"][
                    "promptTransport"]["attachments"][0]["path"] = str(root / "artifact.txt")
                record["bundle"]["steps"][0]["output"]["structuredData"]["browserRun"][
                    "attachmentUiReceipt"]["uploadCompletion"] = "timed_out"
                record_path.write_text(json.dumps(record))
                self.assertEqual(outcome_model.import_bound_loops(loop_root, runs_root)["conflict"], 1)
                record["bundle"]["steps"][0]["output"]["structuredData"]["browserRun"][
                    "attachmentUiReceipt"]["uploadCompletion"] = "confirmed"
                record["bundle"]["run"]["initialInputs"]["metadata"]["learning_trace_digest"] = "0" * 64
                record_path.write_text(json.dumps(record))
                self.assertEqual(outcome_model.import_bound_loops(loop_root, runs_root)["rejected"], 1)
            record["bundle"]["run"]["initialInputs"]["metadata"]["learning_trace_digest"] = trace_digest
            del record["bundle"]["steps"][0]["output"]["structuredData"]["browserRun"][
                "attachmentUiReceipt"]
            record_path.write_text(json.dumps(record))
            with self.private_store(root / "legacy"):
                self.assertEqual(outcome_model.import_bound_loops(loop_root, runs_root)["imported"], 1)
                self.assertEqual(outcome_model.status()["bound_legacy_loop"]["browser_ui_confirmed_rounds"], 0)
                self.assertEqual(outcome_model.train_bound_loop_model()["rounds"], 0)

    def test_bound_loop_model_uses_post_codex_inputs_and_forward_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                self.assertEqual(outcome_model.train_bound_loop_model()["status"],
                                 "insufficient_bound_loop_rounds")
                key = outcome_model._key()
                def insert_roots(first: int, last: int, started_at: datetime) -> None:
                    with outcome_model._connect() as connection:
                        for root in range(first, last):
                            previous = None
                            for number in range(1, 4):
                                episode = f"episode-{root}-{number}"
                                prompt = f"private generation prompt {root} {number}"
                                codex = f"private Codex result {root} {number}"
                                browser = f"private browser result {root} {number}"
                                passed = int((root + number) % 2 == 0)
                                submitted_at = (started_at + timedelta(
                                    days=root - first, minutes=number)).isoformat()
                                connection.execute("INSERT INTO bound_loop_rounds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                    (episode, f"root-{root}", number,
                                     submitted_at,
                                     outcome_model._digest(key, f"goal-{root}"),
                                     outcome_model._digest(key, prompt),
                                     json.dumps(outcome_model._features(f"goal-{root}", prompt, number, key)),
                                     json.dumps(outcome_model._features(codex, "", 1, key)),
                                     json.dumps(outcome_model._features(browser, "", 1, key)),
                                     json.dumps(outcome_model._features("private feedback", "", 1, key)),
                                     previous, 95 if passed else 65, passed,
                                     f"resp_{root}_{number}", "a" * 64, "gpt-5.2", 0, 1, 1, 0))
                                previous = episode
                now = datetime.now().astimezone()
                insert_roots(0, 12, now - timedelta(days=20))
                trained = outcome_model.train_bound_loop_model()
                self.assertEqual(trained["rounds"], 36)
                self.assertEqual(trained["input_stage"], "post_codex_pre_browser")
                self.assertFalse(trained["authoritative_for_routing"])
                self.assertEqual(trained["prospective_holdout_rounds"], 0)
                checkpoint = outcome_model.BOUND_LOOP_EVAL_PATH.read_bytes()
                score_checkpoint = outcome_model.BOUND_LOOP_SCORE_EVAL_PATH.read_bytes()
                insert_roots(12, 14, now + timedelta(days=2))
                later = outcome_model.train_bound_loop_model()
                self.assertEqual(later["prospective_holdout_rounds"], 6)
                self.assertEqual(later["prospective_score_holdout_rounds"], 6)
                self.assertFalse(later["validated_for_shadow"])
                self.assertFalse(later["score_validated_for_shadow"])
                self.assertEqual(checkpoint, outcome_model.BOUND_LOOP_EVAL_PATH.read_bytes())
                self.assertEqual(score_checkpoint, outcome_model.BOUND_LOOP_SCORE_EVAL_PATH.read_bytes())
                with outcome_model._connect() as connection:
                    connection.execute("UPDATE bound_loop_rounds SET browser_dispatch_bound=0 WHERE episode_id=?",
                                       ("episode-13-2",))
                excluded = outcome_model.train_bound_loop_model()
                self.assertEqual(excluded["rounds"], 40)
                self.assertEqual(excluded["observed_rounds"], 42)
                self.assertEqual(excluded["status"], "insufficient_bound_loop_split")
                with outcome_model._connect() as connection:
                    connection.execute("UPDATE bound_loop_rounds SET browser_ui_confirmed=0 WHERE episode_id=?",
                                       ("episode-13-1",))
                excluded_ui = outcome_model.train_bound_loop_model()
                self.assertEqual(excluded_ui["rounds"], 39)
                raw = outcome_model.DB_PATH.read_bytes()
                for secret in ("private generation prompt", "private Codex result", "private browser result"):
                    self.assertNotIn(secret.encode(), raw)

    def test_bound_loop_score_can_validate_only_on_later_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.private_store(Path(directory)):
                now = datetime.now().astimezone()

                def insert_roots(first: int, last: int, offset_days: int) -> None:
                    with outcome_model._connect() as connection:
                        for root in range(first, last):
                            previous = None
                            score = 95 if root % 2 else 65
                            for number in range(1, 4):
                                episode = f"score-{root}-{number}"
                                submitted = (now + timedelta(days=offset_days, hours=root,
                                                             minutes=number)).isoformat()
                                connection.execute(
                                    "INSERT INTO bound_loop_rounds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                    (episode, f"score-root-{root}", number, submitted,
                                     "a" * 64, "b" * 64,
                                     json.dumps({"0": 1.0 if root % 2 else -1.0}),
                                     None, json.dumps({}), json.dumps({}), previous, score,
                                     int(score >= 90), f"resp_score_{root}_{number}",
                                     "c" * 64, "gpt-5.2", 0, 1, 1, 0))
                                previous = episode

                insert_roots(0, 12, -3)
                first = outcome_model.train_bound_loop_model()
                self.assertFalse(first["score_validated_for_shadow"])
                score_checkpoint = outcome_model.BOUND_LOOP_SCORE_EVAL_PATH.read_bytes()
                insert_roots(12, 22, 1)
                later = outcome_model.train_bound_loop_model()
                self.assertEqual(later["prospective_score_holdout_rounds"], 30)
                self.assertEqual(later["prospective_score_holdout_task_groups"], 10)
                self.assertTrue(later["score_validated_for_shadow"])
                self.assertLess(later["prospective_score_model_mae_points"],
                                later["prospective_score_baseline_mae_points"])
                self.assertEqual(score_checkpoint, outcome_model.BOUND_LOOP_SCORE_EVAL_PATH.read_bytes())

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
                self.assertEqual(outcome_model.sync_codex_grades(metrics),
                                 {"graded": 1, "usage_matched": 1, "observed": 0, "rerouted": 0})
                self.assertEqual(outcome_model.sync_codex_grades(metrics),
                                 {"graded": 0, "usage_matched": 0, "observed": 0, "rerouted": 0})
                self.assertEqual(outcome_model.train_codex_model()["graded_with_usage"], 0)
                self.assertTrue(outcome_model.capture_codex_result("thread-1", "turn-1",
                                                                  "private Codex final result"))
                self.assertEqual(outcome_model.status()["codex"],
                                 {"accepted_turns": 1, "thread_groups": 1,
                                  "graded_turns": 1, "exact_usage_turns": 1,
                                  "result_captured_turns": 1,
                                  "complete_prompt_result_usage_grade_turns": 1,
                                  "observed_single_arm_turns": 0,
                                  "server_rerouted_turns": 0,
                                  "accepted_user_steers": 0,
                                  "steered_thread_groups": 0,
                                  "steered_turns": 0,
                                  "durable_user_messages": 0,
                                  "durable_after_prior_output_messages": 0,
                                  "durable_user_message_turns": 0,
                                  "steer_training_role": "context_only_not_independent_outcome"})
                self.assertNotIn(b"private user prompt", outcome_model.DB_PATH.read_bytes())
                self.assertNotIn(b"private Codex final result", outcome_model.DB_PATH.read_bytes())
                report = outcome_model.train_codex_model()
                self.assertEqual(report["status"], "insufficient_comparable_outcomes")
                self.assertEqual(report["verified_model_effort_outcomes"], 0)
                self.assertEqual(report["graded_with_usage"], 1)

    def test_server_reroute_disqualifies_a_claimed_single_model_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.private_store(root):
                prepared = outcome_model.prepare_codex_prompt("private user prompt")
                outcome_model.capture_codex_turn("thread-r", "turn-r", prepared,
                                                 "gpt-6-luna", "low", "routine")
                with outcome_model._connect() as connection:
                    connection.execute("""UPDATE codex_turns SET observed_model='gpt-6-luna',
                        observed_effort='low',model_provenance='observed_per_request'
                        WHERE thread_key=? AND turn_key=?""",
                        (outcome_model._digest(outcome_model._key(), "thread-r"),
                         outcome_model._digest(outcome_model._key(), "turn-r")))
                metrics = root / "metrics.jsonl"
                metrics.write_text(json.dumps({"event": "model_rerouted", "thread_id": "thread-r",
                    "turn_id": "turn-r", "assigned_model": "gpt-6-luna",
                    "from_model": "gpt-6-luna", "to_model": "gpt-6-sol",
                    "reason": "highRiskCyberActivity"}) + "\n")
                self.assertEqual(outcome_model.sync_codex_grades(metrics)["rerouted"], 1)
                self.assertEqual(outcome_model.sync_codex_grades(metrics)["rerouted"], 0)
                self.assertEqual(outcome_model.status()["codex"]["server_rerouted_turns"], 1)
                with outcome_model._connect() as connection:
                    self.assertEqual(connection.execute("""SELECT model_provenance,reroute_seen
                        FROM codex_turns""").fetchone(), ("server_rerouted", 1))

    def test_positive_host_settings_evidence_promotes_only_completed_exact_single_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.private_store(root):
                prepared = outcome_model.prepare_codex_prompt("private user prompt")
                outcome_model.capture_codex_turn("thread-o", "turn-o", prepared,
                                                 "gpt-6-luna", "low", "routine")
                metrics = root / "metrics.jsonl"
                rows = [
                    {"event": "turn_completed", "thread_id": "thread-o", "turn_id": "turn-o",
                     "status": "completed"},
                    {"event": "turn_usage", "thread_id": "thread-o", "turn_id": "turn-o",
                     "model": "gpt-6-luna", "effort": "low",
                     "source": "proxy_thread_usage_delta", "usage": {"totalTokens": 123}},
                    {"event": "route_execution_observed", "thread_id": "thread-o",
                     "turn_id": "turn-o", "model": "gpt-6-luna", "effort": "low",
                     "source": "host_thread_settings_updated_pre_admission",
                     "settings_confirmation": "exact"},
                ]
                metrics.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
                self.assertEqual(outcome_model.sync_codex_grades(metrics)["observed"], 1)
                self.assertEqual(outcome_model.sync_codex_grades(metrics)["observed"], 0)
                with outcome_model._connect() as connection:
                    self.assertEqual(connection.execute("""SELECT observed_model,observed_effort,
                        model_provenance,reroute_seen FROM codex_turns""").fetchone(),
                        ("gpt-6-luna", "low", "observed_per_request", 0))

                outcome_model.capture_codex_turn("thread-r", "turn-r", prepared,
                                                 "gpt-6-luna", "low", "routine")
                rows.extend([
                    {"event": "turn_completed", "thread_id": "thread-r", "turn_id": "turn-r",
                     "status": "completed"},
                    {"event": "turn_usage", "thread_id": "thread-r", "turn_id": "turn-r",
                     "model": "gpt-6-luna", "effort": "low",
                     "source": "proxy_thread_usage_delta", "usage": {"totalTokens": 123}},
                    {"event": "route_execution_observed", "thread_id": "thread-r",
                     "turn_id": "turn-r", "model": "gpt-6-luna", "effort": "low",
                     "source": "host_thread_settings_updated_pre_admission",
                     "settings_confirmation": "exact"},
                    {"event": "model_rerouted", "thread_id": "thread-r", "turn_id": "turn-r",
                     "assigned_model": "gpt-6-luna", "from_model": "gpt-6-luna",
                     "to_model": "gpt-6-sol", "reason": "highRiskCyberActivity"},
                ])
                metrics.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
                self.assertEqual(outcome_model.sync_codex_grades(metrics)["observed"], 0)
                with outcome_model._connect() as connection:
                    self.assertEqual(connection.execute("""SELECT model_provenance,reroute_seen
                        FROM codex_turns WHERE selected_model='gpt-6-luna'
                        ORDER BY accepted_at_ms DESC LIMIT 1""").fetchone(),
                        ("server_rerouted", 1))

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

            def make_round(guard_id, response_id, round_number, generation, parent=None,
                           root_id="first", inline=False):
                sources = {"origin_prompt": source(origin), "generation_prompt": source(generation)}
                trace = {"root_guard_id": root_id, "parent_guard_id": parent,
                         "parent_response_id": f"resp_{parent.replace('-', '_')}" if parent else None,
                         "prompt_author": "chatgpt" if parent else "user",
                         "origin_prompt_sha256": sources["origin_prompt"]["sha256"],
                         "generation_prompt_sha256": sources["generation_prompt"]["sha256"]}
                digest = hashlib.sha256(json.dumps(trace, sort_keys=True,
                                                   separators=(",", ":")).encode()).hexdigest()
                nonce = f"nonce-{guard_id}"
                summary = ("A complete revised prompt would be:\n\n“revised browser prompt”"
                           if inline and not parent else f"feedback from {guard_id}")
                state = {"status": "completed", "response_id": response_id, "guard_id": guard_id,
                         "nonce": nonce, "round": round_number,
                         "review_format": "inline" if inline else None,
                         "submitted_at": f"2026-09-23T00:0{round_number}:00Z",
                         "submission_fingerprint": f"fingerprint-{guard_id}",
                         "learning_trace": trace, "learning_trace_digest": digest,
                         "source_files": sources,
                         "verdict": {"summary": summary, "nonce": nonce, "pass": bool(parent),
                                     "suggested_next_prompt": ("revised browser prompt"
                                                               if not parent and not inline else None),
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
                self.assertEqual(outcome_model.status()["browser_codex_links"]
                                 ["unlinked_prompt_absent_both"], 1)
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
                make_round("inline-first", "resp_inline_first", 1, first_prompt,
                           root_id="inline-first", inline=True)
                make_round("inline-second", "resp_inline_second", 2, second_prompt,
                           "inline-first", root_id="inline-first", inline=True)
                inline_import = outcome_model.import_auracall(guards, runs, "inline-first")
                self.assertEqual(inline_import["trace_imported"], 1)
                child_import = outcome_model.import_auracall(guards, runs, "inline-second")
                self.assertEqual(child_import["trace_imported"], 1)
                self.assertEqual(outcome_model.status()["iterations"]["adopted_explicit_inline_revisions"], 1)
                self.assertEqual(outcome_model.status()["iterations"]["adopted_browser_suggestions"], 1)
                self.assertEqual(outcome_model.import_auracall(guards, runs, "inline-second")
                                 ["trace_unchanged"], 1)
                with outcome_model._connect() as connection:
                    connection.execute("""UPDATE iteration_traces
                                        SET explicit_inline_revision_digest=NULL,
                                            adopted_parent_inline_revision=0
                                        WHERE root_key=?""",
                                       (outcome_model._digest(outcome_model._key(), "inline-first"),))
                self.assertEqual(outcome_model.import_auracall(guards, runs, "inline-first")
                                 ["trace_unchanged"], 1)
                self.assertEqual(outcome_model.import_auracall(guards, runs, "inline-second")
                                 ["trace_unchanged"], 1)
                self.assertEqual(outcome_model.status()["iterations"]["adopted_explicit_inline_revisions"], 1)

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
                    connection.execute("INSERT INTO browser_results VALUES (?,?)",
                                       ("episode-0", json.dumps(outcome_model._features(
                                           "prior browser answer", "", 1, key))))
                    connection.execute("""UPDATE codex_turns SET total_tokens=123,observed_model='gpt-6-sol',
                                       observed_effort='medium',model_provenance='observed_per_request'
                                       WHERE thread_key=? AND turn_key=?""",
                                       (outcome_model._digest(key, "linked-thread"),
                                        outcome_model._digest(key, "linked-turn")))
                    self.assertEqual(outcome_model._link_review_to_codex(
                        connection, "episode-0", prepared["prompt_digest"],
                        outcome_model._digest(key, "actual Codex final answer")), "imported")
                report = outcome_model.train_iteration_model()
                self.assertEqual(report["trained_rounds"], 60)
                self.assertEqual(report["context_complete_revision_task_groups"], 20)
                self.assertEqual(report["linked_parent_codex_results"], 1)
                self.assertEqual(report["linked_parent_codex_arms"], 1)
                self.assertEqual(report["linked_parent_browser_results"], 1)
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
                with_codex_arm = outcome_model.predict_iteration(
                    "task context", "candidate 42", "chatgpt", 2, "reviewer fixes",
                    "reviewed evidence packet", 65, "candidate 42",
                    "actual Codex final answer", "gpt-6-sol", "medium")
                self.assertTrue(with_codex_arm["parent_codex_arm_supplied"])
                self.assertEqual(with_codex_arm["linked_parent_codex_arms_in_training"], 1)
                with_browser_result = outcome_model.predict_iteration(
                    "task context", "candidate 42", "chatgpt", 2,
                    parent_browser_result="prior browser answer")
                self.assertTrue(with_browser_result["parent_browser_result_supplied"])
                self.assertEqual(with_browser_result["linked_parent_browser_results_in_training"], 1)
                with self.assertRaises(ValueError):
                    outcome_model.predict_iteration("task context", "candidate 42", "chatgpt", 2,
                                                    parent_codex_model="gpt-6-sol",
                                                    parent_codex_effort="medium")
                with self.assertRaises(ValueError):
                    outcome_model.predict_iteration("task context", "candidate 42", "chatgpt", 2,
                                                    parent_codex_result="actual Codex final answer",
                                                    parent_codex_model="gpt-6-sol")
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
                    connection.execute("INSERT INTO browser_results VALUES (?,?)",
                                       ("episode-0-first", json.dumps(outcome_model._features(
                                           "earlier browser answer", "", 1, key))))
                    connection.execute("""UPDATE codex_turns SET total_tokens=123,observed_model='gpt-6-sol',
                                       observed_effort='medium',model_provenance='observed_per_request'
                                       WHERE thread_key=? AND turn_key=?""",
                                       (outcome_model._digest(key, "improvement-thread"),
                                        outcome_model._digest(key, "improvement-turn")))
                    self.assertEqual(outcome_model._link_review_to_codex(
                        connection, "episode-0-first", prepared["prompt_digest"],
                        outcome_model._digest(key, "actual earlier Codex answer")), "imported")
                self.assertEqual(outcome_model.train_iteration_model()["trained_rounds"], 48)
                first = outcome_model.train_improvement_model()
                self.assertEqual(first["scored_revisions"], 24)
                self.assertEqual(first["linked_parent_codex_results"], 1)
                self.assertEqual(first["linked_parent_codex_arms"], 1)
                self.assertEqual(first["linked_parent_browser_results"], 1)
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

    def test_durable_mid_turn_user_messages_are_context_not_result_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = root / "sessions"
            sessions.mkdir()
            thread = "01234567-89ab-cdef-0123-456789abcdef"
            turn = "11111111-2222-3333-4444-555555555555"
            def user(message_id, text, *, bound_turn=turn, image=False):
                content = [{"type": "input_text", "text": text}]
                if image:
                    content.append({"type": "input_image", "image_url": "private"})
                return {"type": "response_item", "payload": {
                    "type": "message", "role": "user", "id": message_id,
                    "internal_chat_message_metadata_passthrough": {"turn_id": bound_turn},
                    "content": content}}
            rows = [
                {"type": "session_meta", "payload": {"id": thread}},
                {"type": "event_msg", "payload": {"type": "task_started", "turn_id": turn}},
                user("msg-initial", "private original user prompt"),
                {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "partial work"}]}},
                user("msg-steer", "private user correction"),
                user("msg-wrong-turn", "not bound", bound_turn="other-turn"),
                user("msg-image", "text with image", image=True),
                {"type": "event_msg", "timestamp": "2026-09-24T00:00:00Z", "payload": {
                    "type": "task_complete", "turn_id": turn,
                    "last_agent_message": "private final answer"}},
            ]
            (sessions / f"rollout-{thread}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            with self.private_store(root):
                first = outcome_model.import_codex_sessions(sessions, thread)
                self.assertEqual(first["context_imported"], 2)
                self.assertEqual(first["turn_ineligible"], 1)
                second = outcome_model.import_codex_sessions(sessions, thread)
                self.assertEqual(second["context_unchanged"], 2)
                summary = outcome_model.status()["codex"]
                self.assertEqual(summary["durable_user_messages"], 2)
                self.assertEqual(summary["durable_after_prior_output_messages"], 1)
                self.assertEqual(summary["durable_user_message_turns"], 1)
                self.assertEqual(outcome_model.status()["standalone_codex"]["completed_prompt_result_pairs"], 0)
                raw = outcome_model.DB_PATH.read_bytes()
                self.assertNotIn(b"private original user prompt", raw)
                self.assertNotIn(b"private user correction", raw)
                rows.insert(2, rows.pop(3))
                (sessions / f"rollout-{thread}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
                replay = outcome_model.import_codex_sessions(sessions, thread)
                self.assertEqual(replay["context_phase_disagreement"], 1)
                self.assertEqual(replay["context_conflict"], 0)
                rows[3]["payload"]["content"][0]["text"] = "changed under same message ID"
                (sessions / f"rollout-{thread}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
                self.assertEqual(outcome_model.import_codex_sessions(sessions, thread)["context_conflict"], 1)

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
