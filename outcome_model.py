"""Local, prompt-free-at-rest learning from provenance-bound graded episodes.

The operational metrics log never stores text or text features. This module
reads private source records, keeps only keyed feature hashes, and trains a
small regularized outcome model in a separate private directory.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
import stat
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from paths import ROOT


LEARNING_ROOT = ROOT / "learning"
KEY_PATH = LEARNING_ROOT / "feature-key"
DB_PATH = LEARNING_ROOT / "episodes.sqlite3"
MODEL_PATH = LEARNING_ROOT / "review-model.json"
SCORE_MODEL_PATH = LEARNING_ROOT / "review-score-model-v1.json"
CODEX_MODEL_PATH = LEARNING_ROOT / "codex-model.json"
LOCAL_PROMPT_MODEL_PATH = LEARNING_ROOT / "local-prompt-preference-model-v1.json"
LOCAL_PROMPT_EVAL_PATH = LEARNING_ROOT / "local-prompt-evaluation-checkpoint-v1.json"
MANAGED_ROUTING_POLICY_PATH = LEARNING_ROOT / "managed-routing-policy-v2.json"
MANAGED_PROMPT_POLICY_PATH = LEARNING_ROOT / "managed-prompt-policy-v1.json"
ITERATION_MODEL_PATH = LEARNING_ROOT / "iteration-model-v7.json"
IMPROVEMENT_MODEL_PATH = LEARNING_ROOT / "revision-improvement-model-v4.json"
BOUND_LOOP_MODEL_PATH = LEARNING_ROOT / "bound-loop-post-result-model-v2.json"
BOUND_LOOP_EVAL_PATH = LEARNING_ROOT / "bound-loop-evaluation-checkpoint-v2.json"
BOUND_LOOP_SCORE_EVAL_PATH = LEARNING_ROOT / "bound-loop-score-evaluation-checkpoint-v1.json"
REVIEW_EVAL_PATH = LEARNING_ROOT / "review-evaluation-checkpoint.json"
SCORE_EVAL_PATH = LEARNING_ROOT / "review-score-evaluation-checkpoint-v1.json"
ITERATION_EVAL_PATH = LEARNING_ROOT / "iteration-evaluation-checkpoint-v7.json"
IMPROVEMENT_EVAL_PATH = LEARNING_ROOT / "revision-improvement-evaluation-checkpoint-v4.json"
CODEX_EVAL_PATH = LEARNING_ROOT / "codex-evaluation-checkpoint.json"
FEATURE_COUNT = 512
FEATURE_VERSION = 1
ITERATION_FEATURE_VERSION = 6
BOUND_LOOP_FEATURE_VERSION = 2
MIN_ITERATION_REVISION_GROUPS = 8
MODEL_IDS = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
             "gpt-6-luna", "gpt-6-sol", "gpt-6-astra")
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max", "ultra")
TASK_CLASSES = ("simple", "routine", "difficult", "consequential")
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_SESSION_BYTES = 256 * 1024 * 1024
MAX_SESSION_LINE_BYTES = 4 * 1024 * 1024
MIN_REVIEW_HOLDOUT = 20
MIN_REVIEW_BRIER_GAIN = 0.005
MIN_REVIEW_RELATIVE_GAIN = 0.05
MIN_SCORE_MAE_GAIN = 0.03
MIN_MANAGED_REPEATS_PER_PRODUCT = 3
MIN_MANAGED_DEVELOPMENT_PRODUCTS = 8
MIN_MANAGED_PROSPECTIVE_PRODUCTS = 8
MIN_MANAGED_WIN_PRODUCTS = 7
MAX_MANAGED_TOKEN_RATIO = 0.9
MAX_MANAGED_FAILURES_PER_PRODUCT = 2
GOAL_START = "Review the supplied artifact against this goal:\n"
GOAL_END = "\n\nFRESHNESS_NONCE:"
TOKEN = re.compile(r"[a-z][a-z0-9_]{1,31}", re.I)


def _private_root() -> None:
    if LEARNING_ROOT.is_symlink():
        raise ValueError("learning directory may not be a symlink")
    LEARNING_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    LEARNING_ROOT.chmod(0o700)


def _key() -> bytes:
    _private_root()
    try:
        descriptor = os.open(KEY_PATH, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        try:
            descriptor = os.open(KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            descriptor = os.open(KEY_PATH, os.O_RDONLY | os.O_NOFOLLOW)
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(secrets.token_bytes(32))
                handle.flush()
                os.fsync(handle.fileno())
            descriptor = os.open(KEY_PATH, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("learning key must be a regular file")
        value = handle.read(33)
    if len(value) != 32:
        raise ValueError("invalid learning key")
    return value


def _digest(key: bytes, value: str) -> str:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()


def _features(context: str, revision: str, round_number: int, key: bytes) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for prefix, text in (("c", context), ("r", revision)):
        words = TOKEN.findall(text.lower())[:8000]
        for word in words:
            index = int.from_bytes(hmac.new(key, f"{prefix}:{word}".encode(), hashlib.sha256).digest()[:4], "big") % FEATURE_COUNT
            counts[str(index)] += 1
        for left, right in zip(words, words[1:]):
            index = int.from_bytes(hmac.new(key, f"{prefix}:{left} {right}".encode(), hashlib.sha256).digest()[:4], "big") % FEATURE_COUNT
            counts[str(index)] += 1
    vector = {index: math.log1p(count) for index, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
    vector = {index: value / norm for index, value in vector.items()}
    vector[str(FEATURE_COUNT)] = min(max(round_number, 1), 20) / 20.0
    vector[str(FEATURE_COUNT + 1)] = min(len(context.split()), 2000) / 2000.0
    vector[str(FEATURE_COUNT + 2)] = min(len(revision.split()), 8000) / 8000.0
    return vector


def _connect() -> sqlite3.Connection:
    _private_root()
    if DB_PATH.is_symlink():
        raise ValueError("learning database may not be a symlink")
    connection = sqlite3.connect(DB_PATH, timeout=10)
    os.chmod(DB_PATH, 0o600)
    connection.execute("PRAGMA trusted_schema=OFF")
    connection.execute("PRAGMA busy_timeout=10000")
    connection.execute("""CREATE TABLE IF NOT EXISTS episodes (
        episode_id TEXT PRIMARY KEY, group_id TEXT NOT NULL, source TEXT NOT NULL,
        submitted_at TEXT NOT NULL, round_number INTEGER NOT NULL,
        prompt_role TEXT NOT NULL, prompt_digest TEXT NOT NULL,
        revision_digest TEXT NOT NULL, features TEXT NOT NULL,
        requested_model TEXT, observed_model TEXT, effort TEXT,
        quality_score INTEGER, passed INTEGER, verifier TEXT,
        source_response_id TEXT NOT NULL, source_guard_id TEXT,
        feature_version INTEGER NOT NULL
    )""")
    connection.execute("CREATE INDEX IF NOT EXISTS episodes_group ON episodes(group_id, submitted_at)")
    connection.execute("""CREATE TABLE IF NOT EXISTS iteration_traces (
        episode_id TEXT PRIMARY KEY, root_key TEXT NOT NULL, parent_episode_id TEXT,
        origin_prompt_digest TEXT NOT NULL, generation_prompt_digest TEXT NOT NULL,
        prompt_author TEXT NOT NULL, prompt_features TEXT NOT NULL,
        parent_feedback_features TEXT, result_feedback_features TEXT,
        submitted_at TEXT NOT NULL, passed INTEGER NOT NULL,
        suggested_prompt_digest TEXT, adopted_parent_suggestion INTEGER NOT NULL DEFAULT 0,
        parent_result_features TEXT, parent_quality_score INTEGER,
        explicit_inline_revision_digest TEXT,
        adopted_parent_inline_revision INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
    )""")
    trace_columns = {row[1] for row in connection.execute("PRAGMA table_info(iteration_traces)")}
    if "suggested_prompt_digest" not in trace_columns:
        connection.execute("ALTER TABLE iteration_traces ADD COLUMN suggested_prompt_digest TEXT")
    if "adopted_parent_suggestion" not in trace_columns:
        connection.execute("ALTER TABLE iteration_traces ADD COLUMN adopted_parent_suggestion INTEGER NOT NULL DEFAULT 0")
    if "parent_result_features" not in trace_columns:
        connection.execute("ALTER TABLE iteration_traces ADD COLUMN parent_result_features TEXT")
    if "parent_quality_score" not in trace_columns:
        connection.execute("ALTER TABLE iteration_traces ADD COLUMN parent_quality_score INTEGER")
    if "explicit_inline_revision_digest" not in trace_columns:
        connection.execute("ALTER TABLE iteration_traces ADD COLUMN explicit_inline_revision_digest TEXT")
    if "adopted_parent_inline_revision" not in trace_columns:
        connection.execute("ALTER TABLE iteration_traces ADD COLUMN adopted_parent_inline_revision INTEGER NOT NULL DEFAULT 0")
    connection.execute("""CREATE TABLE IF NOT EXISTS codex_turns (
        thread_key TEXT NOT NULL, turn_key TEXT NOT NULL, group_id TEXT NOT NULL,
        accepted_at_ms INTEGER NOT NULL, prompt_digest TEXT NOT NULL,
        features TEXT NOT NULL, selected_model TEXT NOT NULL, effort TEXT NOT NULL,
        task_class TEXT NOT NULL, quality_score INTEGER, verification TEXT,
        total_tokens INTEGER, PRIMARY KEY(thread_key, turn_key)
    )""")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(codex_turns)")}
    if "result_digest" not in columns:
        connection.execute("ALTER TABLE codex_turns ADD COLUMN result_digest TEXT")
    if "result_features" not in columns:
        connection.execute("ALTER TABLE codex_turns ADD COLUMN result_features TEXT")
    for name, definition in (("observed_model", "TEXT"), ("observed_effort", "TEXT"),
                             ("model_provenance", "TEXT NOT NULL DEFAULT 'proxy_selected_only'"),
                             ("reroute_seen", "INTEGER NOT NULL DEFAULT 0")):
        if name not in columns:
            connection.execute(f"ALTER TABLE codex_turns ADD COLUMN {name} {definition}")
    connection.execute("""CREATE TABLE IF NOT EXISTS codex_steers (
        event_key TEXT PRIMARY KEY, thread_key TEXT NOT NULL, turn_key TEXT NOT NULL,
        accepted_at_ms INTEGER NOT NULL, prompt_digest TEXT NOT NULL,
        features TEXT NOT NULL, attribution TEXT NOT NULL
            CHECK(attribution='context_only')
    )""")
    connection.execute("CREATE INDEX IF NOT EXISTS codex_steers_turn ON codex_steers(thread_key,turn_key)")
    connection.execute("""CREATE TABLE IF NOT EXISTS session_user_messages (
        message_key TEXT PRIMARY KEY, thread_key TEXT NOT NULL, turn_key TEXT NOT NULL,
        prompt_digest TEXT NOT NULL, features TEXT NOT NULL,
        phase TEXT NOT NULL CHECK(phase IN ('pre_inference','after_prior_output')),
        attribution TEXT NOT NULL CHECK(attribution='context_only')
    )""")
    connection.execute("""CREATE INDEX IF NOT EXISTS session_user_messages_turn
                       ON session_user_messages(thread_key,turn_key)""")
    connection.execute("""CREATE TABLE IF NOT EXISTS session_turns (
        thread_key TEXT NOT NULL, turn_key TEXT NOT NULL,
        prompt_digest TEXT NOT NULL, prompt_features TEXT NOT NULL,
        result_digest TEXT NOT NULL, result_features TEXT NOT NULL,
        model TEXT, effort TEXT, reported_total_tokens INTEGER,
        completed_at TEXT NOT NULL, source_path_digest TEXT NOT NULL,
        grade_status TEXT NOT NULL DEFAULT 'ungraded',
        PRIMARY KEY(thread_key,turn_key)
    )""")
    session_columns = {row[1] for row in connection.execute("PRAGMA table_info(session_turns)")}
    for name, definition in (("task_class", "TEXT"), ("exact_total_tokens", "INTEGER"),
                             ("quality_score", "INTEGER"), ("verification", "TEXT"),
                             ("input_scope", "TEXT NOT NULL DEFAULT 'legacy_unverified'"),
                             ("observed_model", "TEXT"), ("observed_effort", "TEXT"),
                             ("model_provenance", "TEXT NOT NULL DEFAULT 'session_metadata_only'")):
        if name not in session_columns:
            connection.execute(f"ALTER TABLE session_turns ADD COLUMN {name} {definition}")
    connection.execute("""CREATE TABLE IF NOT EXISTS review_codex_links (
        episode_id TEXT PRIMARY KEY, thread_key TEXT NOT NULL, turn_key TEXT NOT NULL,
        source_kind TEXT NOT NULL, linked_at TEXT NOT NULL,
        link_status TEXT NOT NULL DEFAULT 'valid',
        FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
    )""")
    link_columns = {row[1] for row in connection.execute("PRAGMA table_info(review_codex_links)")}
    if "link_status" not in link_columns:
        connection.execute("ALTER TABLE review_codex_links ADD COLUMN link_status TEXT NOT NULL DEFAULT 'valid'")
    connection.execute("""CREATE TABLE IF NOT EXISTS browser_results (
        episode_id TEXT PRIMARY KEY, result_features TEXT NOT NULL,
        FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS recovery_observations (
        observation_id TEXT PRIMARY KEY, source_response_id TEXT NOT NULL,
        original_record_digest TEXT NOT NULL, observation_digest TEXT NOT NULL,
        observed_at TEXT NOT NULL, prompt_digest TEXT NOT NULL,
        prompt_features TEXT NOT NULL, result_digest TEXT NOT NULL,
        result_features TEXT NOT NULL, attachment_ui_confirmed INTEGER NOT NULL,
        evidence_role TEXT NOT NULL DEFAULT 'ungraded_failed_response'
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS benchmark_prompt_runs (
        run_key TEXT PRIMARY KEY, suite_key TEXT NOT NULL,
        comparison_key TEXT, product_key TEXT NOT NULL, scenario_key TEXT NOT NULL,
        prompt_digest TEXT NOT NULL, prompt_features TEXT NOT NULL,
        result_digest TEXT NOT NULL, result_features TEXT NOT NULL,
        prompt_author TEXT NOT NULL, model TEXT NOT NULL, effort TEXT NOT NULL,
        product_pass INTEGER NOT NULL, satisfaction_score INTEGER NOT NULL,
        total_tokens INTEGER NOT NULL, elapsed_ms INTEGER NOT NULL,
        observed_at_ms INTEGER NOT NULL, source_kind TEXT NOT NULL
    )""")
    benchmark_columns = {row[1] for row in connection.execute("PRAGMA table_info(benchmark_prompt_runs)")}
    for name, definition in (("model_provenance", "TEXT NOT NULL DEFAULT 'cli_requested_only'"),
                             ("observed_model", "TEXT"), ("observed_effort", "TEXT"),
                             ("browser_guard_key", "TEXT"), ("browser_response_key", "TEXT"),
                             ("repetition_index", "INTEGER")):
        if name not in benchmark_columns:
            connection.execute(f"ALTER TABLE benchmark_prompt_runs ADD COLUMN {name} {definition}")
    connection.execute("""CREATE TABLE IF NOT EXISTS managed_prompt_sets (
        prompt_set_key TEXT PRIMARY KEY, suite_key TEXT NOT NULL,
        created_at_ms INTEGER NOT NULL, comparison_key TEXT NOT NULL,
        product_key TEXT NOT NULL, task_class TEXT NOT NULL,
        repetition_index INTEGER NOT NULL, prompt_order TEXT NOT NULL,
        order_position_probability REAL NOT NULL,
        randomization_protocol TEXT NOT NULL, randomization_nonce TEXT NOT NULL,
        commitment TEXT NOT NULL
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS managed_benchmark_blocks (
        block_key TEXT PRIMARY KEY, suite_key TEXT NOT NULL,
        created_at_ms INTEGER NOT NULL, scenario_key TEXT NOT NULL,
        product_key TEXT NOT NULL, prompt_digest TEXT NOT NULL,
        task_class TEXT NOT NULL, repetition_index INTEGER NOT NULL,
        arm_order TEXT NOT NULL, assignment_probability REAL NOT NULL,
        randomization_protocol TEXT NOT NULL, randomization_nonce TEXT NOT NULL,
        commitment TEXT NOT NULL
    )""")
    managed_block_columns = {row[1] for row in connection.execute(
        "PRAGMA table_info(managed_benchmark_blocks)")}
    for name in ("baseline_model", "baseline_effort", "scenario_task_class",
                 "prompt_set_key"):
        if name not in managed_block_columns:
            connection.execute(f"ALTER TABLE managed_benchmark_blocks ADD COLUMN {name} TEXT")
    if "prompt_arm_index" not in managed_block_columns:
        connection.execute("ALTER TABLE managed_benchmark_blocks ADD COLUMN prompt_arm_index INTEGER")
    connection.execute("""CREATE TABLE IF NOT EXISTS managed_benchmark_bindings (
        run_key TEXT PRIMARY KEY, block_key TEXT NOT NULL, arm_index INTEGER NOT NULL,
        thread_key TEXT NOT NULL, turn_key TEXT NOT NULL,
        observed_model TEXT NOT NULL, observed_effort TEXT NOT NULL,
        exact_total_tokens INTEGER NOT NULL, bound_at_ms INTEGER NOT NULL,
        UNIQUE(block_key,arm_index), UNIQUE(thread_key,turn_key),
        FOREIGN KEY(run_key) REFERENCES benchmark_prompt_runs(run_key),
        FOREIGN KEY(block_key) REFERENCES managed_benchmark_blocks(block_key)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS bound_loop_rounds (
        episode_id TEXT PRIMARY KEY, root_key TEXT NOT NULL,
        round_number INTEGER NOT NULL, submitted_at TEXT NOT NULL,
        origin_prompt_digest TEXT NOT NULL, generation_prompt_digest TEXT NOT NULL,
        prompt_features TEXT NOT NULL, codex_result_features TEXT,
        browser_result_features TEXT NOT NULL, feedback_features TEXT NOT NULL,
        parent_episode_id TEXT, quality_score INTEGER NOT NULL,
        passed INTEGER NOT NULL, source_response_id TEXT NOT NULL,
        trace_digest TEXT NOT NULL, requested_model TEXT NOT NULL,
        attachment_content_verified INTEGER NOT NULL DEFAULT 0,
        browser_dispatch_bound INTEGER NOT NULL DEFAULT 0,
        browser_ui_confirmed INTEGER NOT NULL DEFAULT 0,
        browser_suggestion_supplied INTEGER NOT NULL DEFAULT 0
    )""")
    bound_columns = {row[1] for row in connection.execute("PRAGMA table_info(bound_loop_rounds)")}
    if "browser_dispatch_bound" not in bound_columns:
        connection.execute("ALTER TABLE bound_loop_rounds ADD COLUMN browser_dispatch_bound INTEGER NOT NULL DEFAULT 0")
    if "browser_ui_confirmed" not in bound_columns:
        connection.execute("ALTER TABLE bound_loop_rounds ADD COLUMN browser_ui_confirmed INTEGER NOT NULL DEFAULT 0")
    if "browser_suggestion_supplied" not in bound_columns:
        connection.execute("ALTER TABLE bound_loop_rounds ADD COLUMN browser_suggestion_supplied INTEGER NOT NULL DEFAULT 0")
    connection.execute("CREATE INDEX IF NOT EXISTS bound_loop_rounds_root ON bound_loop_rounds(root_key, round_number)")
    return connection


def prepare_codex_prompt(prompt: str) -> dict[str, str]:
    """Extract private keyed features before a managed turn is admitted."""
    if not prompt.strip():
        raise ValueError("Codex prompt is empty")
    key = _key()
    return {"prompt_digest": _digest(key, prompt),
            "features": json.dumps(_features(prompt, "", 1, key), sort_keys=True)}


def capture_codex_turn(thread_id: str, turn_id: str, prepared: dict[str, str],
                       model: str, effort: str, task_class: str) -> None:
    """Store the accepted choice and prompt features, never the prompt text."""
    key = _key()
    if not prepared.get("prompt_digest") or not prepared.get("features"):
        raise ValueError("missing prepared prompt features")
    json.loads(prepared["features"])
    with _connect() as connection:
        values = (_digest(key, thread_id), _digest(key, turn_id), _digest(key, thread_id),
                  int(time.time() * 1000), prepared["prompt_digest"], prepared["features"],
                  model, effort, task_class)
        existing = connection.execute("""SELECT prompt_digest,selected_model,effort,task_class
                                         FROM codex_turns WHERE thread_key=? AND turn_key=?""",
                                      values[:2]).fetchone()
        if existing:
            if existing != (prepared["prompt_digest"], model, effort, task_class):
                raise ValueError("Codex turn learning identity conflict")
            return
        connection.execute("""INSERT INTO codex_turns
                            (thread_key,turn_key,group_id,accepted_at_ms,prompt_digest,
                             features,selected_model,effort,task_class)
                            VALUES (?,?,?,?,?,?,?,?,?)""", values)


def capture_codex_result(thread_id: str, turn_id: str, result_text: str) -> bool:
    """Keep only keyed features from the last completed agent message."""
    if not result_text.strip() or len(result_text) > 256_000:
        return False
    key = _key()
    identity = (_digest(key, thread_id), _digest(key, turn_id))
    digest = _digest(key, result_text)
    features = json.dumps(_features(result_text, "", 1, key), sort_keys=True)
    with _connect() as connection:
        row = connection.execute("SELECT result_digest FROM codex_turns WHERE thread_key=? AND turn_key=?",
                                 identity).fetchone()
        if row is None:
            return False
        if row[0] is not None:
            if row[0] != digest:
                raise ValueError("Codex result learning identity conflict")
            return True
        connection.execute("""UPDATE codex_turns SET result_digest=?,result_features=?
                              WHERE thread_key=? AND turn_key=?""", (digest, features, *identity))
    return True


def capture_codex_steer(thread_id: str, turn_id: str, event_id: str,
                        prepared: dict[str, str]) -> bool:
    """Keep an accepted steer as context, never as an independently graded turn."""
    if not all(isinstance(value, str) and value for value in (thread_id, turn_id, event_id)):
        raise ValueError("missing accepted steer identity")
    if not isinstance(prepared, dict) or not prepared.get("prompt_digest") or not prepared.get("features"):
        raise ValueError("missing steer prompt features")
    json.loads(prepared["features"])
    key = _key()
    values = (_digest(key, event_id), _digest(key, thread_id), _digest(key, turn_id),
              int(time.time() * 1000), prepared["prompt_digest"], prepared["features"], "context_only")
    with _connect() as connection:
        existing = connection.execute("""SELECT thread_key,turn_key,prompt_digest,features,attribution
                                         FROM codex_steers WHERE event_key=?""", (values[0],)).fetchone()
        if existing is not None:
            if existing != (*values[1:3], *values[4:]):
                raise ValueError("Codex steer learning identity conflict")
            return True
        connection.execute("INSERT INTO codex_steers VALUES (?,?,?,?,?,?,?)", values)
    return True


def capture_benchmark_prompt_run(result: dict[str, Any], scenario: dict[str, Any],
                                 final_text: str) -> bool:
    """Keep keyed prompt/result features from one verified local benchmark run."""
    from smoke_bench import product_definition, stable_digest, verified_browser_revision

    uuid = r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
    run_id, suite_id = result.get("run_id"), result.get("suite_id")
    author = result.get("prompt_author")
    prompt = scenario.get("prompt")
    passed = result.get("product_pass")
    total, elapsed = result.get("total_tokens"), result.get("elapsed_ms")
    score = result.get("satisfaction_score")
    model_provenance = result.get("model_provenance", "cli_requested_only")
    observed_model = result.get("observed_model")
    observed_effort = result.get("observed_effort")
    repetition_index = result.get("repetition_index")
    if (result.get("schema") != "modellabs.benchmark-result.v2"
            or not isinstance(run_id, str) or not re.fullmatch(uuid, run_id)
            or not isinstance(suite_id, str) or not re.fullmatch(uuid, suite_id)
            or not isinstance(prompt, str) or not prompt.strip()
            or not isinstance(final_text, str) or not final_text.strip()
            or len(prompt) > 256_000 or len(final_text) > 256_000
            or author not in {"benchmark", "codex", "chatgpt"}
            or author != scenario.get("prompt_author", "benchmark")
            or result.get("model") not in MODEL_IDS or result.get("effort") not in EFFORTS
            # This capture API receives a caller-supplied benchmark result. The
            # current CLI runner has no independent per-request execution-model
            # receipt, so matching claimed IDs cannot upgrade its provenance.
            or model_provenance != "cli_requested_only"
            or observed_model is not None or observed_effort is not None
            or (repetition_index is not None
                and (type(repetition_index) is not int or repetition_index < 0))
            or result.get("scenario_id") != scenario.get("id")
            or result.get("scenario_sha256") != stable_digest(scenario)
            or result.get("product_sha256") != stable_digest(product_definition(scenario))
            or result.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest()
            or result.get("comparison_id") != scenario.get("comparison_id")
            or result.get("variant_of") != scenario.get("variant_of")
            or result.get("fixture_integrity") is not True
            or result.get("codex_exit_code") != 0
            or not isinstance(passed, bool)
            or (passed and result.get("verifier_exit_code") != 0)
            or (not passed and not isinstance(result.get("verifier_exit_code"), int))
            or not isinstance(total, int) or total <= 0
            or not isinstance(elapsed, int) or elapsed < 0
            or not isinstance(score, int) or not 0 <= score <= 100):
        return False
    browser_review = scenario.get("browser_review")
    if author == "chatgpt":
        if (not isinstance(browser_review, dict)
                or result.get("browser_review") != browser_review
                or not isinstance(browser_review.get("origin_sha256"), str)
                or not isinstance(scenario.get("browser_origin_prompt"), str)
                or scenario.get("variant_of") is None):
            return False
        try:
            verified = verified_browser_revision(browser_review.get("guard_id"), prompt,
                                                 scenario["browser_origin_prompt"])
        except (ValueError, OSError, json.JSONDecodeError, TypeError):
            return False
        if verified != browser_review:
            return False
    elif browser_review is not None or result.get("browser_review") is not None:
        return False
    key = _key()
    guard_key = _digest(key, browser_review["guard_id"]) if browser_review else None
    response_key = _digest(key, browser_review["response_id"]) if browser_review else None
    source_kind = ("browser_review_local_sandbox_benchmark" if browser_review
                   else "local_sandbox_benchmark")
    values = (_digest(key, run_id), _digest(key, suite_id),
              _digest(key, scenario["comparison_id"]) if scenario.get("comparison_id") else None,
              _digest(key, result["product_sha256"]), _digest(key, result["scenario_sha256"]),
              _digest(key, prompt), json.dumps(_features(prompt, "", 1, key), sort_keys=True),
              _digest(key, final_text), json.dumps(_features(final_text, "", 1, key), sort_keys=True),
              author, result["model"], result["effort"], int(passed), score,
              total, elapsed, int(time.time() * 1000), source_kind,
              model_provenance, observed_model, observed_effort, guard_key, response_key,
              repetition_index)
    with _connect() as connection:
        existing = connection.execute("""SELECT suite_key,comparison_key,product_key,scenario_key,
                                       prompt_digest,prompt_features,result_digest,result_features,
                                       prompt_author,model,effort,product_pass,satisfaction_score,
                                       total_tokens,elapsed_ms,source_kind,
                                       model_provenance,observed_model,observed_effort,
                                       browser_guard_key,browser_response_key,repetition_index
                                       FROM benchmark_prompt_runs WHERE run_key=?""", (values[0],)).fetchone()
        if existing is not None:
            if existing != (*values[1:16], *values[17:]):
                raise ValueError("benchmark run learning identity conflict")
            return True
        connection.execute("""INSERT INTO benchmark_prompt_runs
                            (run_key,suite_key,comparison_key,product_key,scenario_key,
                             prompt_digest,prompt_features,result_digest,result_features,
                             prompt_author,model,effort,product_pass,satisfaction_score,
                             total_tokens,elapsed_ms,observed_at_ms,source_kind,
                             model_provenance,observed_model,observed_effort,
                             browser_guard_key,browser_response_key,repetition_index)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)
    return True


def create_managed_prompt_set(suite_id: str, prompt_set_id: str,
                              scenarios: list[dict[str, Any]],
                              repetition_index: int) -> dict[str, Any]:
    """Precommit a randomized complete prompt set before any product inference."""
    from modellabs import route
    from smoke_bench import (product_definition, stable_digest,
                             verified_browser_revision)

    uuid = r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
    if (not re.fullmatch(uuid, suite_id) or not re.fullmatch(uuid, prompt_set_id)
            or type(repetition_index) is not int or repetition_index < 0
            or not isinstance(scenarios, list) or len(scenarios) < 2):
        raise ValueError("invalid managed prompt set")
    comparisons = {scenario.get("comparison_id") for scenario in scenarios}
    products = {stable_digest(product_definition(scenario)) for scenario in scenarios}
    scenario_classes = {scenario.get("task_class") for scenario in scenarios}
    routed_classes = {route(str(scenario.get("prompt", "")))["class"]
                      for scenario in scenarios}
    prompts = [scenario.get("prompt") for scenario in scenarios]
    authors = [scenario.get("prompt_author", "benchmark") for scenario in scenarios]
    baselines = [scenario for scenario, author in zip(scenarios, authors)
                 if author == "benchmark"]
    if (len(comparisons) != 1 or not isinstance(next(iter(comparisons)), str)
            or not next(iter(comparisons)).strip() or len(products) != 1
            or len(scenario_classes) != 1 or len(routed_classes) != 1
            or scenario_classes != routed_classes
            or len(baselines) != 1 or "chatgpt" not in authors
            or any(author not in {"benchmark", "chatgpt"} for author in authors)
            or any(not isinstance(prompt, str) or not prompt.strip()
                   or len(prompt) > 256_000 for prompt in prompts)
            or len(set(prompts)) != len(prompts)):
        raise ValueError("managed prompt arms must be one matched browser-revision product")
    baseline = baselines[0]
    for scenario, author in zip(scenarios, authors):
        if author != "chatgpt":
            continue
        review = scenario.get("browser_review") or {}
        observed = verified_browser_revision(
            str(review.get("guard_id", "")), str(scenario.get("prompt", "")),
            str(scenario.get("browser_origin_prompt", "")))
        if (observed != review or scenario.get("variant_of") != baseline.get("id")
                or scenario.get("browser_origin_prompt") != baseline.get("prompt")):
            raise ValueError("managed prompt revision lacks exact browser provenance")
    key = _key()
    private_entries = []
    public_by_private: dict[str, dict[str, Any]] = {}
    for scenario, author in zip(scenarios, authors):
        review = scenario.get("browser_review") or {}
        scenario_sha = stable_digest(scenario)
        entry = {"scenario_key": _digest(key, scenario_sha),
                 "prompt_digest": _digest(key, scenario["prompt"]),
                 "prompt_author": author,
                 "browser_guard_key": (_digest(key, review["guard_id"])
                                       if author == "chatgpt" else None),
                 "browser_response_key": (_digest(key, review["response_id"])
                                          if author == "chatgpt" else None)}
        private_entries.append(entry)
        public_by_private[entry["scenario_key"]] = {
            "scenario_sha256": scenario_sha,
            "prompt_sha256": hashlib.sha256(scenario["prompt"].encode()).hexdigest(),
            "prompt_author": author}
    nonce = secrets.token_bytes(32)
    ordered = sorted(private_entries, key=lambda entry: hmac.new(
        nonce, json.dumps(entry, sort_keys=True, separators=(",", ":")).encode(),
        hashlib.sha256).digest())
    prompt_order = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
    protocol = "hmac-sha256-complete-prompt-order-v1"
    comparison_id = next(iter(comparisons))
    product_sha = next(iter(products))
    task_class = next(iter(routed_classes))
    commitment_payload = json.dumps({
        "suite_id": suite_id, "prompt_set_id": prompt_set_id,
        "comparison_sha256": hashlib.sha256(comparison_id.encode()).hexdigest(),
        "product_sha256": product_sha, "task_class": task_class,
        "repetition_index": repetition_index, "prompt_order": ordered,
        "protocol": protocol, "nonce": nonce.hex(),
    }, sort_keys=True, separators=(",", ":"))
    commitment = hashlib.sha256(commitment_payload.encode()).hexdigest()
    values = (_digest(key, prompt_set_id), _digest(key, suite_id),
              int(time.time() * 1000), _digest(key, comparison_id),
              _digest(key, product_sha), task_class, repetition_index,
              prompt_order, 1.0 / len(ordered), protocol, nonce.hex(), commitment)
    with _connect() as connection:
        if connection.execute("SELECT 1 FROM managed_prompt_sets WHERE prompt_set_key=?",
                              (values[0],)).fetchone() is not None:
            raise ValueError("managed prompt set already exists")
        connection.execute("""INSERT INTO managed_prompt_sets
            (prompt_set_key,suite_key,created_at_ms,comparison_key,product_key,
             task_class,repetition_index,prompt_order,order_position_probability,
             randomization_protocol,randomization_nonce,commitment)
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", values)
    return {"schema": "modellabs.managed-prompt-set.v1", "suite_id": suite_id,
            "prompt_set_id": prompt_set_id, "comparison_id": comparison_id,
            "product_sha256": product_sha, "task_class": task_class,
            "repetition_index": repetition_index,
            "prompts": [public_by_private[entry["scenario_key"]] for entry in ordered],
            "order_position_probability": 1.0 / len(ordered),
            "randomization_protocol": protocol, "commitment": commitment}


def create_managed_benchmark_block(suite_id: str, block_id: str,
                                   scenario: dict[str, Any],
                                   arms: list[tuple[str, str]],
                                   repetition_index: int, *,
                                   prompt_set_id: str | None = None,
                                   prompt_arm_index: int | None = None) -> dict[str, Any]:
    """Precommit one private randomized complete block before any inference."""
    from modellabs import route
    from smoke_bench import product_definition, stable_digest

    uuid = r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
    prompt, scenario_task_class = scenario.get("prompt"), scenario.get("task_class")
    if (not re.fullmatch(uuid, suite_id) or not re.fullmatch(uuid, block_id)
            or not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 256_000
            or scenario_task_class not in TASK_CLASSES
            or type(repetition_index) is not int or repetition_index < 0
            or not isinstance(arms, list) or len(arms) < 2
            or len(set(arms)) != len(arms)
            or any(not isinstance(arm, tuple) or len(arm) != 2
                   or arm[0] not in MODEL_IDS or arm[1] not in EFFORTS for arm in arms)):
        raise ValueError("invalid managed benchmark block")
    # Product difficulty is a benchmark annotation.  The execution receipt
    # contains the router's independently derived class, so precommit both and
    # never assume that the two taxonomies happen to agree.
    baseline_route = route(prompt)
    routing_task_class = baseline_route["class"]
    baseline_arm = (baseline_route["model"], baseline_route["effort"])
    if baseline_arm not in arms:
        raise ValueError("managed benchmark arms must include the current router baseline")
    nonce = secrets.token_bytes(32)
    ordered = sorted(arms, key=lambda arm: hmac.new(
        nonce, json.dumps(arm, separators=(",", ":")).encode(), hashlib.sha256).digest())
    arm_order = json.dumps([list(arm) for arm in ordered], separators=(",", ":"))
    protocol = "hmac-sha256-uniform-order-v1"
    commitment_payload = json.dumps({
        "suite_id": suite_id, "block_id": block_id,
        "scenario_sha256": stable_digest(scenario),
        "product_sha256": stable_digest(product_definition(scenario)),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "scenario_task_class": scenario_task_class,
        "routing_task_class": routing_task_class,
        "baseline_arm": list(baseline_arm),
        "repetition_index": repetition_index,
        "arm_order": json.loads(arm_order), "protocol": protocol,
        "nonce": nonce.hex(),
    }, sort_keys=True, separators=(",", ":"))
    commitment = hashlib.sha256(commitment_payload.encode()).hexdigest()
    key = _key()
    prompt_set_key = None
    prompt_set_commitment = None
    if (prompt_set_id is None) != (prompt_arm_index is None):
        raise ValueError("managed prompt set identity is incomplete")
    if prompt_set_id is not None:
        if (not re.fullmatch(uuid, prompt_set_id) or type(prompt_arm_index) is not int
                or prompt_arm_index < 0):
            raise ValueError("invalid managed prompt arm identity")
        prompt_set_key = _digest(key, prompt_set_id)
        with _connect() as connection:
            prompt_set = connection.execute("""SELECT suite_key,product_key,task_class,
                repetition_index,prompt_order,commitment FROM managed_prompt_sets
                WHERE prompt_set_key=?""", (prompt_set_key,)).fetchone()
        try:
            prompt_entry = json.loads(prompt_set[4])[prompt_arm_index]
        except (TypeError, IndexError, json.JSONDecodeError):
            raise ValueError("managed prompt arm is not precommitted") from None
        expected_prompt_entry = {
            "scenario_key": _digest(key, stable_digest(scenario)),
            "prompt_digest": _digest(key, prompt),
            "prompt_author": scenario.get("prompt_author", "benchmark"),
            "browser_guard_key": (_digest(key, scenario["browser_review"]["guard_id"])
                                  if scenario.get("prompt_author") == "chatgpt" else None),
            "browser_response_key": (_digest(key, scenario["browser_review"]["response_id"])
                                     if scenario.get("prompt_author") == "chatgpt" else None)}
        if (prompt_set[0] != _digest(key, suite_id)
                or prompt_set[1] != _digest(key, stable_digest(product_definition(scenario)))
                or prompt_set[2] != routing_task_class
                or prompt_set[3] != repetition_index
                or prompt_entry != expected_prompt_entry):
            raise ValueError("managed prompt arm differs from precommit")
        prompt_set_commitment = prompt_set[5]
    values = (_digest(key, block_id), _digest(key, suite_id), int(time.time() * 1000),
              _digest(key, stable_digest(scenario)),
              _digest(key, stable_digest(product_definition(scenario))),
              _digest(key, prompt), routing_task_class, repetition_index, arm_order,
              1.0 / len(ordered), protocol, nonce.hex(), commitment,
              baseline_arm[0], baseline_arm[1], scenario_task_class,
              prompt_set_key, prompt_arm_index)
    with _connect() as connection:
        if connection.execute("SELECT 1 FROM managed_benchmark_blocks WHERE block_key=?",
                              (values[0],)).fetchone() is not None:
            raise ValueError("managed benchmark block already exists")
        connection.execute("""INSERT INTO managed_benchmark_blocks
            (block_key,suite_key,created_at_ms,scenario_key,product_key,prompt_digest,
             task_class,repetition_index,arm_order,assignment_probability,
             randomization_protocol,randomization_nonce,commitment,
             baseline_model,baseline_effort,scenario_task_class,prompt_set_key,prompt_arm_index)
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)
    return {"schema": "modellabs.managed-benchmark-block.v1",
            "suite_id": suite_id, "block_id": block_id,
            "scenario_sha256": stable_digest(scenario),
            "product_sha256": stable_digest(product_definition(scenario)),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "scenario_task_class": scenario_task_class,
            "routing_task_class": routing_task_class,
            "baseline_arm": list(baseline_arm),
            "repetition_index": repetition_index,
            "arms": [list(arm) for arm in ordered],
            "assignment_probability": 1.0 / len(ordered),
            "order_position_probability": 1.0 / len(ordered),
            "randomization_estimand": "complete-block-order-only-all-arms-run",
            "randomization_protocol": protocol, "commitment": commitment,
            "prompt_set_id": prompt_set_id, "prompt_arm_index": prompt_arm_index,
            "prompt_set_commitment": prompt_set_commitment}


def _metric_receipt_at(metrics_path: Path, receipt_id: str) -> dict[str, Any]:
    path = metrics_path.parent / "receipts" / f"{hashlib.sha256(receipt_id.encode()).hexdigest()}.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("managed benchmark canonical receipt is unavailable")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict) or value.get("receipt_id") != receipt_id:
        raise ValueError("managed benchmark canonical receipt is invalid")
    return value


def managed_turn_evidence(thread_id: str, turn_id: str,
                          metrics_path: Path | None = None,
                          *, require_grade: bool = True) -> dict[str, Any]:
    """Derive one execution fact from canonical proxy receipts, never caller labels."""
    from telemetry import METRICS_PATH

    uuid = r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
    if not re.fullmatch(uuid, thread_id) or not re.fullmatch(uuid, turn_id):
        raise ValueError("managed benchmark requires exact thread and turn UUIDs")
    path = metrics_path or METRICS_PATH
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SESSION_BYTES:
        raise ValueError("managed benchmark metrics are unavailable")
    stages = {stage: _metric_receipt_at(path, f"{thread_id}:{turn_id}:{stage}")
              for stage in ("accepted", "terminal", "usage", "execution")}
    expected_events = {"accepted": "route_accepted", "terminal": "turn_completed",
                       "usage": "turn_usage", "execution": "route_execution_observed"}
    if any(value.get("event") != expected_events[stage]
           or value.get("thread_id") != thread_id or value.get("turn_id") != turn_id
           for stage, value in stages.items()):
        raise ValueError("managed benchmark receipt stages do not match")
    accepted, terminal, usage, execution = (stages[name] for name in
                                             ("accepted", "terminal", "usage", "execution"))
    model, effort = accepted.get("model"), accepted.get("effort")
    total_tokens = (usage.get("usage") or {}).get("totalTokens")
    if (model not in MODEL_IDS or effort not in EFFORTS
            or accepted.get("task_class") not in TASK_CLASSES
            or type(accepted.get("recorded_at_ms")) is not int
            or terminal.get("status") != "completed" or terminal.get("model") != model
            or terminal.get("effort") != effort
            or usage.get("model") != model or usage.get("effort") != effort
            or usage.get("source") != "proxy_thread_usage_delta"
            or type(total_tokens) is not int or total_tokens <= 0
            or execution.get("model") != model or execution.get("effort") != effort
            or execution.get("source") != "host_thread_settings_updated_pre_admission"
            or execution.get("settings_confirmation") != "exact"):
        raise ValueError("managed benchmark execution evidence is incomplete")
    grades: list[dict[str, Any]] = []
    receipt_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in expected_events.values()}
    reroute_seen = False
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if len(line) > MAX_SESSION_LINE_BYTES:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("thread_id") != thread_id:
                continue
            source_turn = row.get("source_turn_id") if row.get("event") == "quality_grade" else row.get("turn_id")
            if source_turn != turn_id:
                continue
            if row.get("event") in receipt_rows:
                receipt_rows[row["event"]].append(row)
            elif row.get("event") == "quality_grade":
                grades.append(row)
            elif row.get("event") == "model_rerouted":
                reroute_seen = True
    if reroute_seen or any(rows != [stages[stage]] for stage, event in expected_events.items()
                           for rows in [receipt_rows[event]]):
        raise ValueError("managed benchmark telemetry conflicts with canonical receipts")
    if require_grade:
        if (len(grades) != 1 or grades[0].get("source") != "explicit"
                or grades[0].get("model") != model or grades[0].get("effort") != effort
                or type(grades[0].get("quality_score")) is not int
                or grades[0].get("verification") not in {"passed", "failed"}):
            raise ValueError("managed benchmark requires one exact verifier grade")
        grade = grades[0]
    else:
        if grades:
            raise ValueError("managed benchmark pre-grade evidence already has a grade")
        grade = None
    return {"model": model, "effort": effort, "task_class": accepted["task_class"],
            "total_tokens": total_tokens, "accepted_at_ms": accepted.get("recorded_at_ms"),
            "usage": usage["usage"],
            "quality_score": grade.get("quality_score") if grade else None,
            "verification": grade.get("verification") if grade else None}


def capture_managed_benchmark_prompt_run(result: dict[str, Any], scenario: dict[str, Any],
                                         final_text: str, workspace: Path,
                                         metrics_path: Path | None = None) -> bool:
    """Bind a verified product run to a precommitted arm and proxy-owned turn."""
    from smoke_bench import grade, product_definition, stable_digest

    block_id, thread_id, turn_id = (result.get(name) for name in
                                    ("block_id", "thread_id", "turn_id"))
    arm_index = result.get("arm_index")
    uuid = r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
    if (result.get("schema") != "modellabs.managed-benchmark-result.v1"
            or not all(isinstance(value, str) and re.fullmatch(uuid, value)
                       for value in (block_id, thread_id, turn_id))
            or type(arm_index) is not int or arm_index < 0):
        return False
    evidence = managed_turn_evidence(thread_id, turn_id, metrics_path)
    usage = evidence["usage"]
    verified = grade(workspace, scenario, final_text, {
        "input_tokens": usage["inputTokens"],
        "cached_input_tokens": usage.get("cachedInputTokens", 0),
        "output_tokens": usage["outputTokens"],
        "reasoning_output_tokens": usage.get("reasoningOutputTokens", 0),
    }, 0)
    verified_fields = ("fixture_integrity", "product_pass", "exact_final_response",
                       "satisfaction_score", "total_tokens", "verifier_exit_code")
    if any(result.get(name) != verified.get(name) for name in verified_fields):
        return False
    key = _key()
    block_key, run_key = _digest(key, block_id), _digest(key, result.get("run_id", ""))
    with _connect() as connection:
        block = connection.execute("""SELECT suite_key,created_at_ms,scenario_key,product_key,
            prompt_digest,task_class,repetition_index,arm_order,scenario_task_class
            FROM managed_benchmark_blocks WHERE block_key=?""", (block_key,)).fetchone()
    try:
        arms = json.loads(block[7]) if block is not None else None
        assigned = arms[arm_index]
    except (TypeError, ValueError, IndexError, json.JSONDecodeError):
        return False
    from modellabs import route
    routing_task_class = route(scenario.get("prompt", ""))["class"]
    expected_block = (_digest(key, result.get("suite_id", "")),
                      _digest(key, stable_digest(scenario)),
                      _digest(key, stable_digest(product_definition(scenario))),
                      _digest(key, scenario.get("prompt", "")), routing_task_class,
                      result.get("repetition_index"))
    product_pass, score = result.get("product_pass"), result.get("satisfaction_score")
    verification = "passed" if product_pass is True else "failed"
    if (block[0] != expected_block[0] or block[2:7] != expected_block[1:]
            or block[8] != scenario.get("task_class")
            or block[1] > evidence.get("accepted_at_ms", -1)
            or assigned != [evidence["model"], evidence["effort"]]
            or result.get("model") != evidence["model"]
            or result.get("effort") != evidence["effort"]
            or result.get("scenario_task_class") != scenario.get("task_class")
            or result.get("task_class") != evidence["task_class"]
            or result.get("total_tokens") != evidence["total_tokens"]
            or score != evidence["quality_score"] or verification != evidence["verification"]):
        return False
    identity = (_digest(key, thread_id), _digest(key, turn_id))
    with _connect() as connection:
        turn = connection.execute("""SELECT prompt_digest,result_digest,selected_model,effort,
            task_class,quality_score,verification,total_tokens,observed_model,observed_effort,
            model_provenance,reroute_seen FROM codex_turns
            WHERE thread_key=? AND turn_key=?""", identity).fetchone()
    expected_turn = (_digest(key, scenario["prompt"]), _digest(key, final_text),
                     evidence["model"], evidence["effort"], evidence["task_class"],
                     score, verification, evidence["total_tokens"], evidence["model"],
                     evidence["effort"], "observed_per_request", 0)
    if turn != expected_turn:
        return False
    legacy_result = {**result, "schema": "modellabs.benchmark-result.v2",
                     "model_provenance": "cli_requested_only",
                     "observed_model": None, "observed_effort": None}
    if not capture_benchmark_prompt_run(legacy_result, scenario, final_text):
        return False
    binding = (run_key, block_key, arm_index, *identity, evidence["model"], evidence["effort"],
               evidence["total_tokens"], int(time.time() * 1000))
    with _connect() as connection:
        existing = connection.execute("""SELECT block_key,arm_index,thread_key,turn_key,
            observed_model,observed_effort,exact_total_tokens
            FROM managed_benchmark_bindings WHERE run_key=?""", (run_key,)).fetchone()
        if existing is not None:
            if existing != binding[1:-1]:
                raise ValueError("managed benchmark binding identity conflict")
            return True
        connection.execute("""INSERT INTO managed_benchmark_bindings
            (run_key,block_key,arm_index,thread_key,turn_key,observed_model,observed_effort,
             exact_total_tokens,bound_at_ms) VALUES (?,?,?,?,?,?,?,?,?)""", binding)
    return True


def _managed_complete_blocks() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Reconstruct only complete receipt-bound randomized blocks."""
    with _connect() as connection:
        rows = connection.execute("""SELECT b.block_key,b.product_key,b.task_class,
            b.created_at_ms,b.arm_order,b.baseline_model,b.baseline_effort,
            b.prompt_digest,b.scenario_task_class,
            r.arm_index,r.observed_model,r.observed_effort,
            r.exact_total_tokens,p.product_pass,p.satisfaction_score,p.total_tokens
            FROM managed_benchmark_blocks AS b
            LEFT JOIN managed_benchmark_bindings AS r ON r.block_key=b.block_key
            LEFT JOIN benchmark_prompt_runs AS p ON p.run_key=r.run_key
            ORDER BY b.created_at_ms,b.block_key,r.arm_index""").fetchall()
    grouped: dict[str, list[tuple[Any, ...]]] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)
    complete: list[dict[str, Any]] = []
    counts = Counter(precommitted_blocks=len(grouped))
    for block_rows in grouped.values():
        first = block_rows[0]
        try:
            order = json.loads(first[4])
        except (TypeError, json.JSONDecodeError):
            counts["invalid_blocks"] += 1
            continue
        if first[8] is None:
            counts["class_unproven_blocks"] += 1
            continue
        if first[8] != first[2]:
            counts["class_mismatch_blocks"] += 1
            continue
        bound = [row for row in block_rows if row[9] is not None]
        if (not isinstance(order, list) or len(order) < 2
                or len(bound) != len(order)
                or sorted(row[9] for row in bound) != list(range(len(order)))):
            counts["incomplete_blocks"] += 1
            continue
        arms: dict[tuple[str, str], dict[str, Any]] = {}
        valid = True
        for row in bound:
            index, model, effort, exact_tokens, passed, score, stored_tokens = row[9:16]
            assigned = order[index] if 0 <= index < len(order) else None
            arm = (model, effort)
            if (assigned != [model, effort] or model not in MODEL_IDS or effort not in EFFORTS
                    or arm in arms or type(exact_tokens) is not int or exact_tokens <= 0
                    or exact_tokens != stored_tokens or passed not in (0, 1)
                    or type(score) is not int or not 0 <= score <= 100):
                valid = False
                break
            arms[arm] = {"product_pass": bool(passed), "quality_score": score,
                         "total_tokens": exact_tokens}
        if not valid:
            counts["invalid_blocks"] += 1
            continue
        baseline_arm = ((first[5], first[6])
                        if first[5] in MODEL_IDS and first[6] in EFFORTS else None)
        baseline_source = "precommitted"
        if baseline_arm is None:
            legacy_defaults = {"simple": ("gpt-6-luna", "low"),
                               "routine": ("gpt-6-sol", "medium"),
                               "difficult": ("gpt-6-sol", "high")}
            candidate = legacy_defaults.get(first[2])
            baseline_arm = candidate if candidate in arms else None
            baseline_source = "legacy_task_default" if baseline_arm else "unavailable"
        if baseline_arm is not None and baseline_arm not in arms:
            counts["invalid_blocks"] += 1
            continue
        complete.append({"block_key": first[0], "product_key": first[1],
                         "task_class": first[2], "created_at_ms": first[3],
                         "prompt_digest": first[7],
                         "baseline_arm": baseline_arm,
                         "baseline_source": baseline_source, "arms": arms})
    counts["complete_blocks"] = len(complete)
    return complete, dict(counts)


def _managed_pair_evidence(blocks: list[dict[str, Any]],
                           minimum_repeats: int = MIN_MANAGED_REPEATS_PER_PRODUCT
                           ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Reduce randomized blocks to conservative same-product pair outcomes."""
    grouped: dict[tuple[str, tuple[str, str], tuple[str, str]],
                  dict[str, list[dict[str, Any]]]] = {}
    for block in blocks:
        arms = sorted(block["arms"])
        for left_index, left in enumerate(arms):
            for right in arms[left_index + 1:]:
                key = (block["task_class"], left, right)
                stratum = hashlib.sha256(json.dumps(
                    [block["product_key"], block["prompt_digest"]],
                    separators=(",", ":")).encode()).hexdigest()
                grouped.setdefault(key, {}).setdefault(stratum, []).append(block)
    reports: list[dict[str, Any]] = []
    private: dict[str, list[dict[str, Any]]] = {}
    for (task_class, left, right), products in sorted(grouped.items()):
        stratum_outcomes: list[dict[str, Any]] = []
        for stratum_key, product_blocks in products.items():
            if len(product_blocks) < minimum_repeats:
                continue
            product_key = product_blocks[0]["product_key"]
            prompt_digest = product_blocks[0]["prompt_digest"]
            left_rows = [block["arms"][left] for block in product_blocks]
            right_rows = [block["arms"][right] for block in product_blocks]
            left_passes = sum(row["product_pass"] for row in left_rows)
            right_passes = sum(row["product_pass"] for row in right_rows)
            winner = "tie"
            quality_deltas = [right_row["quality_score"] - left_row["quality_score"]
                              for left_row, right_row in zip(left_rows, right_rows)]
            token_ratios = [right_row["total_tokens"] / left_row["total_tokens"]
                            for left_row, right_row in zip(left_rows, right_rows)]
            if right_passes == len(product_blocks) and left_passes <= len(product_blocks) - 2:
                winner = "right"
            elif left_passes == len(product_blocks) and right_passes <= len(product_blocks) - 2:
                winner = "left"
            elif left_passes == right_passes == len(product_blocks):
                quality_delta = median(quality_deltas)
                if abs(quality_delta) >= 10 and all(delta * quality_delta > 0
                                                     for delta in quality_deltas):
                    winner = "right" if quality_delta > 0 else "left"
                elif all(delta == 0 for delta in quality_deltas):
                    ratio = median(token_ratios)
                    if ratio <= MAX_MANAGED_TOKEN_RATIO and all(value < 1 for value in token_ratios):
                        winner = "right"
                    elif ratio >= 1 / MAX_MANAGED_TOKEN_RATIO and all(value > 1
                                                                       for value in token_ratios):
                        winner = "left"
            stratum_outcomes.append({"product_key": product_key,
                                     "prompt_digest": prompt_digest,
                                     "stratum_key": stratum_key, "winner": winner,
                                     "repeats": len(product_blocks),
                                     "first_at_ms": min(block["created_at_ms"]
                                                        for block in product_blocks),
                                     "left_all_pass": left_passes == len(product_blocks),
                                     "right_all_pass": right_passes == len(product_blocks),
                                     "median_right_to_left_token_ratio": median(token_ratios),
                                     "median_quality_delta": median(quality_deltas)})
        # Different fixed prompts for the same product are robustness strata,
        # not independent products. Count the product once, and only award a
        # winner when every eligible prompt stratum agrees.
        product_groups: dict[str, list[dict[str, Any]]] = {}
        for row in stratum_outcomes:
            product_groups.setdefault(row["product_key"], []).append(row)
        outcomes: list[dict[str, Any]] = []
        for product_key, rows in product_groups.items():
            directions = {row["winner"] for row in rows}
            winner = next(iter(directions)) if len(directions) == 1 else "tie"
            product_stratum_key = hashlib.sha256(json.dumps(sorted(
                row["stratum_key"] for row in rows), separators=(",", ":")).encode()).hexdigest()
            outcomes.append({"product_key": product_key,
                             "stratum_key": product_stratum_key,
                             "prompt_strata": len(rows), "winner": winner,
                             "repeats": min(row["repeats"] for row in rows),
                             "first_at_ms": min(row["first_at_ms"] for row in rows),
                             "left_all_pass": all(row["left_all_pass"] for row in rows),
                             "right_all_pass": all(row["right_all_pass"] for row in rows),
                             "median_right_to_left_token_ratio": median(
                                 row["median_right_to_left_token_ratio"] for row in rows),
                             "median_quality_delta": median(
                                 row["median_quality_delta"] for row in rows)})
        comparison_id = hashlib.sha256(json.dumps(
            [task_class, list(left), list(right)], separators=(",", ":")).encode()).hexdigest()
        wins = Counter(row["winner"] for row in outcomes)
        comparison_blocks = [block for product_blocks in products.values()
                             for block in product_blocks]
        baselines = {block.get("baseline_arm") for block in comparison_blocks}
        baseline_arm = next(iter(baselines)) if len(baselines) == 1 else None
        baseline_bound = baseline_arm in {left, right}
        report = {"comparison_id": comparison_id, "task_class": task_class,
                  "left_arm": list(left), "right_arm": list(right),
                  "baseline_arm": list(baseline_arm) if baseline_bound else None,
                  "baseline_bound": baseline_bound,
                  "eligible_repeated_products": len(outcomes),
                  "left_wins": wins["left"], "right_wins": wins["right"],
                  "ties": wins["tie"],
                  "minimum_repeats_per_product": minimum_repeats,
                  "causal_model_comparison": True,
                  "authoritative_for_routing": False}
        reports.append(report)
        private[comparison_id] = outcomes
    return reports, private


def train_managed_routing_policy(*, create_checkpoint: bool = True) -> dict[str, Any]:
    """Checkpoint receipt-bound model comparisons and score only future products."""
    blocks, counts = _managed_complete_blocks()
    comparisons, outcomes = _managed_pair_evidence(blocks)
    artifact = {"schema": "modellabs.managed_routing_policy.v2",
                "checkpoints": {}, "validations": {}}
    if MANAGED_ROUTING_POLICY_PATH.exists():
        artifact = _read_json(MANAGED_ROUTING_POLICY_PATH)
        if (artifact.get("schema") != "modellabs.managed_routing_policy.v2"
                or not isinstance(artifact.get("checkpoints"), dict)
                or not isinstance(artifact.get("validations"), dict)):
            raise ValueError("managed routing policy version mismatch")
    checkpoints = artifact["checkpoints"]
    changed = False
    public: list[dict[str, Any]] = []
    now_ms = int(time.time() * 1000)
    for report in comparisons:
        comparison_id = report["comparison_id"]
        rows = outcomes[comparison_id]
        checkpoint = checkpoints.get(comparison_id)
        if (create_checkpoint and report["baseline_bound"] and checkpoint is None
                and len(rows) >= MIN_MANAGED_DEVELOPMENT_PRODUCTS):
            wins = {"left": report["left_wins"], "right": report["right_wins"]}
            leader = max(wins, key=wins.get)
            other = "right" if leader == "left" else "left"
            if (wins[leader] >= MIN_MANAGED_WIN_PRODUCTS and wins[other] == 0
                    and all(row[f"{leader}_all_pass"] for row in rows)):
                checkpoint = {"created_at_ms": now_ms,
                              "development_product_keys": sorted(row["product_key"] for row in rows),
                              "task_class": report["task_class"],
                              "left_arm": report["left_arm"], "right_arm": report["right_arm"],
                              "recommended_side": leader,
                              "minimum_repeats_per_product": MIN_MANAGED_REPEATS_PER_PRODUCT}
                checkpoint["checkpoint_sha256"] = hashlib.sha256(json.dumps(
                    checkpoint, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                checkpoints[comparison_id] = checkpoint
                changed = True
        prospective: list[dict[str, Any]] = []
        valid_checkpoint = False
        if checkpoint is not None:
            digest_fields = {key: value for key, value in checkpoint.items()
                             if key != "checkpoint_sha256"}
            valid_checkpoint = (checkpoint.get("checkpoint_sha256") == hashlib.sha256(json.dumps(
                digest_fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                and checkpoint.get("task_class") == report["task_class"]
                and checkpoint.get("left_arm") == report["left_arm"]
                and checkpoint.get("right_arm") == report["right_arm"])
            if not valid_checkpoint:
                raise ValueError("managed routing checkpoint integrity mismatch")
            development = set(checkpoint.get("development_product_keys", ()))
            prospective = [row for row in rows
                           if row["product_key"] not in development
                           and row["first_at_ms"] > checkpoint["created_at_ms"]]
        side = checkpoint.get("recommended_side") if valid_checkpoint else None
        other_side = "right" if side == "left" else "left" if side == "right" else None
        prospective_wins = sum(row["winner"] == side for row in prospective) if side else 0
        prospective_losses = sum(row["winner"] == other_side for row in prospective) if side else 0
        validated = bool(side and len(prospective) >= MIN_MANAGED_PROSPECTIVE_PRODUCTS
                         and prospective_wins >= MIN_MANAGED_WIN_PRODUCTS
                         and prospective_losses == 0
                         and all(row[f"{side}_all_pass"] for row in prospective))
        public.append({**report, "development_checkpoint_created": valid_checkpoint,
                       "prospective_products": len(prospective),
                       "prospective_wins": prospective_wins,
                       "prospective_losses": prospective_losses,
                       "validated_for_shadow": validated})
    desired_validations: dict[str, dict[str, Any]] = {}
    for report in public:
        if not report["validated_for_shadow"]:
            continue
        checkpoint = checkpoints[report["comparison_id"]]
        side = checkpoint["recommended_side"]
        arm = report[f"{side}_arm"]
        evidence = {"comparison_id": report["comparison_id"],
                    "task_class": report["task_class"],
                    "baseline_arm": report["baseline_arm"],
                    "recommended_arm": arm,
                    "prospective_products": report["prospective_products"],
                    "prospective_wins": report["prospective_wins"],
                    "prospective_losses": report["prospective_losses"],
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"]}
        evidence_sha = hashlib.sha256(json.dumps(
            evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        previous = artifact["validations"].get(report["comparison_id"], {})
        desired_validations[report["comparison_id"]] = {
            **evidence, "evidence_sha256": evidence_sha,
            "validated_at_ms": (previous.get("validated_at_ms")
                                if previous.get("evidence_sha256") == evidence_sha
                                else now_ms)}
    if artifact["validations"] != desired_validations:
        artifact["validations"] = desired_validations
        changed = True
    if changed:
        _private_root()
        temporary = MANAGED_ROUTING_POLICY_PATH.with_name(
            f".{MANAGED_ROUTING_POLICY_PATH.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                       "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, MANAGED_ROUTING_POLICY_PATH)
    eligible_products = sum(row["eligible_repeated_products"] for row in public)
    any_checkpoint = any(row["development_checkpoint_created"] for row in public)
    any_validated = any(row["validated_for_shadow"] for row in public)
    status_name = ("prospective_validated" if any_validated else
                   "prospective_collection" if any_checkpoint else
                   "development_collection" if eligible_products else
                   "insufficient_repeated_products" if comparisons else
                   "insufficient_complete_blocks")
    return {"status": status_name,
            **counts, "independent_products": len({block["product_key"] for block in blocks}),
            "comparisons": public,
            "validated_task_classes": sorted({row["task_class"] for row in public
                                                if row["validated_for_shadow"]}),
            "validated_for_shadow": any_validated,
            "causal_model_comparison": bool(comparisons),
            "authoritative_for_routing": False,
            "minimum_development_products": MIN_MANAGED_DEVELOPMENT_PRODUCTS,
            "minimum_prospective_products": MIN_MANAGED_PROSPECTIVE_PRODUCTS}


def _managed_complete_prompt_sets() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Reconstruct randomized prompt sets only from complete canonical model blocks."""
    eligible_blocks, block_counts = _managed_complete_blocks()
    by_block = {block["block_key"]: block for block in eligible_blocks}
    with _connect() as connection:
        rows = connection.execute("""SELECT b.block_key,b.prompt_set_key,b.prompt_arm_index,
            b.scenario_key,p.prompt_digest,p.prompt_features,p.prompt_author,
            p.browser_guard_key,p.browser_response_key
            FROM managed_benchmark_blocks AS b
            JOIN managed_benchmark_bindings AS r ON r.block_key=b.block_key
            JOIN benchmark_prompt_runs AS p ON p.run_key=r.run_key
            WHERE b.prompt_set_key IS NOT NULL
            ORDER BY b.block_key,r.arm_index""").fetchall()
        prompt_sets = connection.execute("""SELECT prompt_set_key,suite_key,created_at_ms,
            comparison_key,product_key,task_class,repetition_index,prompt_order
            FROM managed_prompt_sets ORDER BY created_at_ms,prompt_set_key""").fetchall()
    block_metadata: dict[str, tuple[Any, ...]] = {}
    invalid_block_metadata: set[str] = set()
    for row in rows:
        metadata = row[1:]
        if row[0] in block_metadata and block_metadata[row[0]] != metadata:
            invalid_block_metadata.add(row[0])
        else:
            block_metadata[row[0]] = metadata
    blocks_by_set: dict[str, list[tuple[int, str, tuple[Any, ...], dict[str, Any]]]] = {}
    for block_key, metadata in block_metadata.items():
        if block_key in invalid_block_metadata or block_key not in by_block:
            continue
        prompt_set_key, prompt_arm_index = metadata[:2]
        if type(prompt_arm_index) is int:
            blocks_by_set.setdefault(prompt_set_key, []).append(
                (prompt_arm_index, block_key, metadata, by_block[block_key]))
    counts = Counter(precommitted_prompt_sets=len(prompt_sets),
                     eligible_model_blocks=len(eligible_blocks), **block_counts)
    complete: list[dict[str, Any]] = []
    for row in prompt_sets:
        set_key, suite_key, created_at_ms, comparison_key, product_key, task_class, repetition, raw_order = row
        try:
            order = json.loads(raw_order)
        except (TypeError, json.JSONDecodeError):
            counts["invalid_prompt_sets"] += 1
            continue
        blocks = blocks_by_set.get(set_key, [])
        if (not isinstance(order, list) or len(order) < 2
                or len(blocks) != len(order)
                or sorted(item[0] for item in blocks) != list(range(len(order)))):
            counts["incomplete_prompt_sets"] += 1
            continue
        prompts: list[dict[str, Any]] = []
        arm_sets: set[tuple[tuple[str, str], ...]] = set()
        valid = True
        for prompt_arm_index, _block_key, metadata, block in sorted(blocks):
            (_prompt_set_key, _stored_index, scenario_key, prompt_digest, features, author,
             browser_guard_key, browser_response_key) = metadata
            expected = order[prompt_arm_index]
            observed = {"scenario_key": scenario_key, "prompt_digest": prompt_digest,
                        "prompt_author": author, "browser_guard_key": browser_guard_key,
                        "browser_response_key": browser_response_key}
            if (any(expected.get(name) != observed[name] for name in observed)
                    or block["product_key"] != product_key
                    or block["task_class"] != task_class):
                valid = False
                break
            try:
                vector = json.loads(features)
            except (TypeError, json.JSONDecodeError):
                valid = False
                break
            if not isinstance(vector, dict):
                valid = False
                break
            arm_sets.add(tuple(sorted(block["arms"])))
            prompts.append({"prompt_digest": prompt_digest, "prompt_features": vector,
                            "prompt_author": author, "browser_guard_key": browser_guard_key,
                            "browser_response_key": browser_response_key,
                            "arms": block["arms"]})
        if (not valid or len(arm_sets) != 1
                or sum(item["prompt_author"] == "benchmark" for item in prompts) != 1
                or not any(item["prompt_author"] == "chatgpt"
                           and item["browser_guard_key"] and item["browser_response_key"]
                           for item in prompts)):
            counts["invalid_prompt_sets"] += 1
            continue
        complete.append({"prompt_set_key": set_key, "suite_key": suite_key,
                         "created_at_ms": created_at_ms, "comparison_key": comparison_key,
                         "product_key": product_key, "task_class": task_class,
                         "repetition_index": repetition, "prompts": prompts})
    counts["complete_prompt_sets"] = len(complete)
    return complete, dict(counts)


def _managed_prompt_pair_evidence(
        prompt_sets: list[dict[str, Any]],
        minimum_repeats: int = MIN_MANAGED_REPEATS_PER_PRODUCT,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Reduce randomized prompt sets to same-model, same-product prompt outcomes."""
    grouped: dict[tuple[str, str, str, str], dict[str, list[tuple[Any, ...]]]] = {}
    for prompt_set in prompt_sets:
        baseline = next(item for item in prompt_set["prompts"]
                        if item["prompt_author"] == "benchmark")
        for candidate in (item for item in prompt_set["prompts"]
                          if item["prompt_author"] == "chatgpt"):
            common_arms = sorted(set(baseline["arms"]) & set(candidate["arms"]))
            for model, effort in common_arms:
                key = (prompt_set["task_class"], model, effort, "chatgpt")
                stratum = hashlib.sha256(json.dumps([
                    prompt_set["product_key"], baseline["prompt_digest"],
                    candidate["prompt_digest"], model, effort],
                    separators=(",", ":")).encode()).hexdigest()
                grouped.setdefault(key, {}).setdefault(stratum, []).append(
                    (prompt_set, baseline, candidate))
    reports: list[dict[str, Any]] = []
    private: dict[str, list[dict[str, Any]]] = {}
    for (task_class, model, effort, author), strata in sorted(grouped.items()):
        outcomes: list[dict[str, Any]] = []
        for stratum_key, rows in strata.items():
            if len(rows) < minimum_repeats:
                continue
            base_rows = [base["arms"][(model, effort)] for _set, base, _candidate in rows]
            candidate_rows = [candidate["arms"][(model, effort)]
                              for _set, _base, candidate in rows]
            base_passes = sum(item["product_pass"] for item in base_rows)
            candidate_passes = sum(item["product_pass"] for item in candidate_rows)
            winner = "tie"
            score_deltas = [right["quality_score"] - left["quality_score"]
                            for left, right in zip(base_rows, candidate_rows)]
            token_ratios = [right["total_tokens"] / left["total_tokens"]
                            for left, right in zip(base_rows, candidate_rows)]
            if candidate_passes == len(rows) and base_passes <= len(rows) - 2:
                winner = "candidate"
            elif base_passes == len(rows) and candidate_passes <= len(rows) - 2:
                winner = "base"
            elif base_passes == candidate_passes == len(rows):
                score_delta = median(score_deltas)
                if abs(score_delta) >= 10 and all(delta * score_delta > 0
                                                   for delta in score_deltas):
                    winner = "candidate" if score_delta > 0 else "base"
                elif all(delta == 0 for delta in score_deltas):
                    ratio = median(token_ratios)
                    if ratio <= MAX_MANAGED_TOKEN_RATIO and all(value < 1 for value in token_ratios):
                        winner = "candidate"
                    elif ratio >= 1 / MAX_MANAGED_TOKEN_RATIO and all(value > 1 for value in token_ratios):
                        winner = "base"
            first_set, base, candidate = rows[0]
            vector = {index: candidate["prompt_features"].get(index, 0.0)
                      - base["prompt_features"].get(index, 0.0)
                      for index in (base["prompt_features"].keys()
                                    | candidate["prompt_features"].keys())}
            vector = {index: value for index, value in vector.items() if abs(value) > 1e-12}
            outcomes.append({"stratum_key": stratum_key,
                             "product_key": first_set["product_key"],
                             "winner": winner, "repeats": len(rows),
                             "first_at_ms": min(item[0]["created_at_ms"] for item in rows),
                             "base_all_pass": base_passes == len(rows),
                             "candidate_all_pass": candidate_passes == len(rows),
                             "median_candidate_to_base_token_ratio": median(token_ratios),
                             "median_score_delta": median(score_deltas),
                             "feature_delta": vector})
        comparison_id = hashlib.sha256(json.dumps(
            [task_class, model, effort, author], separators=(",", ":")).encode()).hexdigest()
        wins = Counter(item["winner"] for item in outcomes)
        reports.append({"comparison_id": comparison_id, "task_class": task_class,
                        "model": model, "effort": effort,
                        "candidate_author": author,
                        "eligible_repeated_products": len(outcomes),
                        "base_wins": wins["base"], "candidate_wins": wins["candidate"],
                        "ties": wins["tie"],
                        "minimum_repeats_per_product": minimum_repeats,
                        "browser_provenance_verified": True,
                        "causal_prompt_comparison": True,
                        "authoritative_for_prompt_selection": False})
        private[comparison_id] = outcomes
    return reports, private


def train_managed_prompt_policy(*, create_checkpoint: bool = True) -> dict[str, Any]:
    """Checkpoint randomized browser-prompt comparisons and score future products."""
    prompt_sets, counts = _managed_complete_prompt_sets()
    comparisons, outcomes = _managed_prompt_pair_evidence(prompt_sets)
    artifact = {"schema": "modellabs.managed_prompt_policy.v1", "checkpoints": {}}
    if MANAGED_PROMPT_POLICY_PATH.exists():
        artifact = _read_json(MANAGED_PROMPT_POLICY_PATH)
        if (artifact.get("schema") != "modellabs.managed_prompt_policy.v1"
                or not isinstance(artifact.get("checkpoints"), dict)):
            raise ValueError("managed prompt policy version mismatch")
    checkpoints = artifact["checkpoints"]
    changed = False
    public: list[dict[str, Any]] = []
    now_ms = int(time.time() * 1000)
    for report in comparisons:
        comparison_id = report["comparison_id"]
        rows = outcomes[comparison_id]
        checkpoint = checkpoints.get(comparison_id)
        if (create_checkpoint and checkpoint is None
                and len(rows) >= MIN_MANAGED_DEVELOPMENT_PRODUCTS):
            wins = {"base": report["base_wins"], "candidate": report["candidate_wins"]}
            leader = max(wins, key=wins.get)
            other = "candidate" if leader == "base" else "base"
            if (wins[leader] >= MIN_MANAGED_WIN_PRODUCTS and wins[other] == 0
                    and all(row[f"{leader}_all_pass"] for row in rows)):
                checkpoint = {"created_at_ms": now_ms,
                              "development_stratum_keys": sorted(
                                  row["stratum_key"] for row in rows),
                              "task_class": report["task_class"],
                              "model": report["model"], "effort": report["effort"],
                              "candidate_author": report["candidate_author"],
                              "recommended_side": leader,
                              "minimum_repeats_per_product": MIN_MANAGED_REPEATS_PER_PRODUCT}
                checkpoint["checkpoint_sha256"] = hashlib.sha256(json.dumps(
                    checkpoint, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                checkpoints[comparison_id] = checkpoint
                changed = True
        prospective: list[dict[str, Any]] = []
        valid_checkpoint = False
        if checkpoint is not None:
            digest_fields = {name: value for name, value in checkpoint.items()
                             if name != "checkpoint_sha256"}
            valid_checkpoint = (checkpoint.get("checkpoint_sha256") == hashlib.sha256(
                json.dumps(digest_fields, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
                and checkpoint.get("task_class") == report["task_class"]
                and checkpoint.get("model") == report["model"]
                and checkpoint.get("effort") == report["effort"]
                and checkpoint.get("candidate_author") == report["candidate_author"])
            if not valid_checkpoint:
                raise ValueError("managed prompt checkpoint integrity mismatch")
            development = set(checkpoint["development_stratum_keys"])
            prospective = [row for row in rows
                           if row["stratum_key"] not in development
                           and row["first_at_ms"] > checkpoint["created_at_ms"]]
        side = checkpoint.get("recommended_side") if valid_checkpoint else None
        other_side = "candidate" if side == "base" else "base" if side == "candidate" else None
        prospective_wins = sum(row["winner"] == side for row in prospective) if side else 0
        prospective_losses = sum(row["winner"] == other_side for row in prospective) if side else 0
        validated = bool(side and len(prospective) >= MIN_MANAGED_PROSPECTIVE_PRODUCTS
                         and prospective_wins >= MIN_MANAGED_WIN_PRODUCTS
                         and prospective_losses == 0
                         and all(row[f"{side}_all_pass"] for row in prospective))
        public.append({**report, "development_checkpoint_created": valid_checkpoint,
                       "prospective_products": len(prospective),
                       "prospective_wins": prospective_wins,
                       "prospective_losses": prospective_losses,
                       "recommended_side": side,
                       "validated_for_shadow": validated})
    if changed:
        _private_root()
        temporary = MANAGED_PROMPT_POLICY_PATH.with_name(
            f".{MANAGED_PROMPT_POLICY_PATH.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                       "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, MANAGED_PROMPT_POLICY_PATH)
    any_checkpoint = any(row["development_checkpoint_created"] for row in public)
    any_validated = any(row["validated_for_shadow"] for row in public)
    eligible_products = sum(row["eligible_repeated_products"] for row in public)
    status_name = ("prospective_validated" if any_validated else
                   "prospective_collection" if any_checkpoint else
                   "development_collection" if eligible_products else
                   "insufficient_repeated_products" if comparisons else
                   "insufficient_complete_prompt_sets")
    return {"status": status_name, **counts,
            "independent_products": len({item["product_key"] for item in prompt_sets}),
            "comparisons": public,
            "validated_task_model_efforts": sorted({
                f"{row['task_class']}:{row['model']}:{row['effort']}" for row in public
                if row["validated_for_shadow"]}),
            "validated_for_shadow": any_validated,
            "causal_prompt_comparison": bool(comparisons),
            "browser_provenance_verified": bool(prompt_sets),
            "authoritative_for_prompt_selection": False,
            "minimum_development_products": MIN_MANAGED_DEVELOPMENT_PRODUCTS,
            "minimum_prospective_products": MIN_MANAGED_PROSPECTIVE_PRODUCTS}


def managed_scenario_progress(scenarios: list[dict[str, Any]],
                              arms: list[tuple[str, str]]) -> dict[str, int]:
    """Count complete exact-arm blocks for known local benchmark products."""
    from smoke_bench import product_definition, stable_digest

    expected = set(arms)
    if len(expected) < 2 or len(expected) != len(arms):
        raise ValueError("managed scenario progress requires distinct arms")
    blocks, _counts = _managed_complete_blocks()
    key = _key()
    product_counts = Counter(
        block["product_key"] for block in blocks if set(block["arms"]) == expected)
    return {scenario["id"]: product_counts[_digest(
        key, stable_digest(product_definition(scenario)))] for scenario in scenarios}


def managed_scenario_attempts(scenarios: list[dict[str, Any]],
                              arms: list[tuple[str, str]]) -> dict[str, int]:
    """Count every precommitted exact-arm block, including failed/incomplete attempts."""
    from smoke_bench import product_definition, stable_digest

    expected = set(arms)
    if len(expected) < 2 or len(expected) != len(arms):
        raise ValueError("managed scenario attempts require distinct arms")
    with _connect() as connection:
        rows = connection.execute("""SELECT product_key,arm_order
            FROM managed_benchmark_blocks
            WHERE scenario_task_class IS NOT NULL
              AND scenario_task_class=task_class""").fetchall()
    counts: Counter[str] = Counter()
    for product_key, arm_order in rows:
        try:
            observed = {tuple(arm) for arm in json.loads(arm_order)}
        except (TypeError, json.JSONDecodeError):
            continue
        if observed == expected:
            counts[product_key] += 1
    key = _key()
    return {scenario["id"]: counts[_digest(
        key, stable_digest(product_definition(scenario)))] for scenario in scenarios}


def managed_prompt_experiment_counts(base: dict[str, Any], candidate: dict[str, Any],
                                     arms: list[tuple[str, str]]) -> dict[str, int]:
    """Count exact precommitted and complete runs for one reviewed prompt pair."""
    from smoke_bench import product_definition, stable_digest

    expected_arms = set(arms)
    review = candidate.get("browser_review") or {}
    if (len(expected_arms) < 2 or len(expected_arms) != len(arms)
            or base.get("prompt_author", "benchmark") != "benchmark"
            or candidate.get("prompt_author") != "chatgpt"
            or candidate.get("variant_of") != base.get("id")
            or candidate.get("browser_origin_prompt") != base.get("prompt")
            or product_definition(base) != product_definition(candidate)
            or not isinstance(review.get("guard_id"), str)
            or not isinstance(review.get("response_id"), str)):
        raise ValueError("managed prompt experiment identity is incomplete")
    key = _key()
    product_key = _digest(key, stable_digest(product_definition(base)))
    expected = {
        (_digest(key, stable_digest(base)), _digest(key, base["prompt"]),
         "benchmark", None, None),
        (_digest(key, stable_digest(candidate)), _digest(key, candidate["prompt"]),
         "chatgpt", _digest(key, review["guard_id"]),
         _digest(key, review["response_id"])),
    }

    def prompt_identity(raw_order: str) -> set[tuple[Any, ...]] | None:
        try:
            order = json.loads(raw_order)
        except (TypeError, json.JSONDecodeError):
            return None
        if (not isinstance(order, list) or len(order) != 2
                or any(not isinstance(item, dict) for item in order)):
            return None
        try:
            return {(item["scenario_key"], item["prompt_digest"], item["prompt_author"],
                     item.get("browser_guard_key"), item.get("browser_response_key"))
                    for item in order}
        except (KeyError, TypeError):
            return None

    with _connect() as connection:
        rows = connection.execute("""SELECT p.prompt_set_key,p.product_key,p.prompt_order,
            b.arm_order FROM managed_prompt_sets AS p
            LEFT JOIN managed_benchmark_blocks AS b ON b.prompt_set_key=p.prompt_set_key
            ORDER BY p.prompt_set_key,b.block_key""").fetchall()
    attempts_by_set: dict[str, dict[str, Any]] = {}
    for prompt_set_key, observed_product, raw_order, arm_order in rows:
        entry = attempts_by_set.setdefault(prompt_set_key, {
            "product_key": observed_product, "prompt_order": raw_order, "arm_orders": []})
        if arm_order is not None:
            entry["arm_orders"].append(arm_order)
    attempts = 0
    for item in attempts_by_set.values():
        if item["product_key"] != product_key or prompt_identity(item["prompt_order"]) != expected:
            continue
        parsed_arms = []
        for raw_arms in item["arm_orders"]:
            try:
                parsed_arms.append({tuple(arm) for arm in json.loads(raw_arms)})
            except (TypeError, json.JSONDecodeError):
                parsed_arms.append(set())
        # A crash after the prompt-set precommit but before block precommit is
        # still one failed attempt. Blocks for a different arm set are not.
        if parsed_arms and any(observed != expected_arms for observed in parsed_arms):
            continue
        attempts += 1

    complete_sets, _counts = _managed_complete_prompt_sets()
    complete = 0
    for prompt_set in complete_sets:
        if prompt_set["product_key"] != product_key or len(prompt_set["prompts"]) != 2:
            continue
        observed = {(item["prompt_digest"], item["prompt_author"],
                     item.get("browser_guard_key"), item.get("browser_response_key"))
                    for item in prompt_set["prompts"]}
        expected_without_scenario = {(item[1], item[2], item[3], item[4]) for item in expected}
        if (observed == expected_without_scenario
                and all(set(item["arms"]) == expected_arms for item in prompt_set["prompts"])):
            complete += 1
    return {"attempts": attempts, "complete": complete,
            "failed_or_incomplete": max(0, attempts - complete)}


def analyze_local_prompt_experiments() -> dict[str, Any]:
    """Audit paired local prompt evidence without mixing it with browser grades."""
    with _connect() as connection:
        rows = connection.execute("""SELECT comparison_key,product_key,model,effort,
                                   prompt_digest,prompt_author,product_pass
                                   FROM benchmark_prompt_runs WHERE comparison_key IS NOT NULL""").fetchall()
    groups: dict[tuple[str, str, str, str], dict[str, Counter[str]]] = {}
    for comparison, product, model, effort, prompt, author, passed in rows:
        group = groups.setdefault((comparison, product, model, effort), {})
        counts = group.setdefault(prompt, Counter())
        counts["runs"] += 1
        counts["passes"] += int(passed)
        counts[f"author:{author}"] += 1
    paired = []
    for key, group in groups.items():
        baselines = [(digest, counts) for digest, counts in group.items()
                     if counts["author:benchmark"] == counts["runs"]]
        if len(baselines) != 1:
            continue
        for digest, counts in group.items():
            if digest == baselines[0][0]:
                continue
            if (counts["author:codex"] == counts["runs"]
                    or counts["author:chatgpt"] == counts["runs"]):
                paired.append((key, baselines[0], (digest, counts)))
    repeated = [(key, base, candidate) for key, base, candidate in paired
                if min(base[1]["runs"], candidate[1]["runs"]) >= 3]
    paired_products = {key[1] for key, _base, _candidate in paired}
    browser_pairs = sum(candidate[1]["author:chatgpt"] > 0 for _key, _base, candidate in paired)
    conclusive, execution_proofs, outcome_counts = _local_prompt_comparisons()
    return {"status": ("insufficient_independent_product_groups"
                       if len(paired_products) < 8 else
                       "insufficient_repeated_prompt_groups" if len(repeated) < 8 else
                       "insufficient_conclusive_products" if len(conclusive) < 8 else
                       "eligible_for_advisory_training"),
            "captured_comparison_runs": len(rows),
            "paired_prompt_arms": len(paired),
            "repeated_prompt_arms": len(repeated),
            "browser_authored_paired_arms": browser_pairs,
            "conclusive_independent_products": len(conclusive),
            "execution_verified_independent_products": sum(execution_proofs.values()),
            "comparison_outcomes": outcome_counts,
            "independent_paired_products": len(paired_products),
            "minimum_independent_products": 8,
            "minimum_runs_per_prompt": 3,
            "browser_grades_included": False,
            "authoritative_for_routing": False}


def analyze_browser_revision_product_outcomes() -> dict[str, Any]:
    """Contrast exact adopted browser revisions with matched local products.

    Review grades and sandbox product grades remain separate outcomes. This
    is a post-result audit, never a feature for predicting an earlier review.
    """
    with _connect() as connection:
        eligible = connection.execute("""SELECT COUNT(*) FROM benchmark_prompt_runs
            WHERE prompt_author='chatgpt' AND browser_response_key IS NOT NULL
            AND repetition_index IS NOT NULL""").fetchone()[0]
        rows = connection.execute("""SELECT candidate.browser_response_key,child.episode_id,
            candidate.product_key,candidate.model,candidate.effort,candidate.suite_key,
            candidate.repetition_index,parent.quality_score,parent.passed,
            child.quality_score,child.passed,base.product_pass,candidate.product_pass,
            base.satisfaction_score,candidate.satisfaction_score,
            base.total_tokens,candidate.total_tokens,
            base.model_provenance,candidate.model_provenance,
            base.observed_model,candidate.observed_model,
            base.observed_effort,candidate.observed_effort,
            base.prompt_digest,candidate.prompt_digest
            FROM benchmark_prompt_runs AS candidate
            JOIN episodes AS parent ON parent.episode_id=candidate.browser_response_key
            JOIN iteration_traces AS trace ON trace.parent_episode_id=parent.episode_id
                AND trace.generation_prompt_digest=candidate.prompt_digest
                AND (trace.adopted_parent_suggestion=1
                     OR trace.adopted_parent_inline_revision=1)
            JOIN episodes AS child ON child.episode_id=trace.episode_id
                AND child.group_id=parent.group_id
            JOIN benchmark_prompt_runs AS base ON base.suite_key=candidate.suite_key
                AND base.comparison_key=candidate.comparison_key
                AND base.product_key=candidate.product_key
                AND base.model=candidate.model AND base.effort=candidate.effort
                AND base.repetition_index=candidate.repetition_index
                AND base.prompt_author='benchmark'
            WHERE candidate.prompt_author='chatgpt'
                AND candidate.source_kind='browser_review_local_sandbox_benchmark'
                AND candidate.browser_response_key IS NOT NULL
                AND candidate.repetition_index IS NOT NULL""").fetchall()
    complete_prompt_sets, _managed_counts = _managed_complete_prompt_sets()
    managed_execution_counts: Counter[tuple[Any, ...]] = Counter()
    for prompt_set in complete_prompt_sets:
        baseline = next(item for item in prompt_set["prompts"]
                        if item["prompt_author"] == "benchmark")
        for candidate in (item for item in prompt_set["prompts"]
                          if item["prompt_author"] == "chatgpt"):
            for model, effort in set(baseline["arms"]) & set(candidate["arms"]):
                managed_execution_counts[(
                    candidate["browser_response_key"], prompt_set["product_key"],
                    model, effort, prompt_set["suite_key"],
                    prompt_set["repetition_index"], baseline["prompt_digest"],
                    candidate["prompt_digest"],
                )] += 1
    blocks: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
    for row in rows:
        blocks.setdefault((row[0], *row[2:7]), []).append(row)
    unique = [matches[0] for matches in blocks.values() if len(matches) == 1]
    groups: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
    for row in unique:
        groups.setdefault((row[0], row[1], *row[2:5]), []).append(row)
    counts = Counter()
    verified_products: set[str] = set()
    verified_wins: set[str] = set()
    verified_losses: set[str] = set()
    counts["eligible_browser_runs"] = eligible
    counts["ambiguous_matched_blocks"] = sum(len(matches) != 1 for matches in blocks.values())
    counts["exact_matched_blocks"] = len(unique)
    for matched in groups.values():
        if len(matched) < 3:
            counts["insufficient_block_groups"] += 1
            continue
        parent_scores = {row[7] for row in matched}
        child_scores = {row[9] for row in matched}
        review_states = {(row[8], row[10]) for row in matched}
        if (len(parent_scores) != 1 or len(child_scores) != 1 or len(review_states) != 1
                or any(type(score) is not int or not 0 <= score <= 100
                       for score in parent_scores | child_scores)):
            counts["conflicting_review_groups"] += 1
            continue
        counts["repeated_linked_groups"] += 1
        if next(iter(child_scores)) > next(iter(parent_scores)):
            counts["review_score_improved_groups"] += 1
        both_pass = all(row[11] and row[12] for row in matched)
        if both_pass:
            counts["both_product_pass_groups"] += 1
        if both_pass and all(row[16] > row[15] > 0 for row in matched):
            counts["candidate_more_tokens_every_block_groups"] += 1
            if next(iter(child_scores)) > next(iter(parent_scores)):
                counts["review_improved_without_token_gain_groups"] += 1
        if all((row[17] == row[18] == "observed_per_request"
                and row[19] == row[20] == row[3]
                and row[21] == row[22] == row[4])
               or managed_execution_counts[(row[0], row[2], row[3], row[4],
                                             row[5], row[6], row[23], row[24])] == 1
               for row in matched):
            counts["execution_verified_groups"] += 1
            product = matched[0][2]
            verified_products.add(product)
            base_passes = sum(bool(row[11]) for row in matched)
            candidate_passes = sum(bool(row[12]) for row in matched)
            winner = None
            if candidate_passes == len(matched) and base_passes <= len(matched) - 2:
                winner = "candidate"
            elif base_passes == len(matched) and candidate_passes <= len(matched) - 2:
                winner = "base"
            elif base_passes == candidate_passes == len(matched):
                score_deltas = [row[14] - row[13] for row in matched]
                score_delta = median(score_deltas)
                if abs(score_delta) >= 10 and all(delta * score_delta > 0 for delta in score_deltas):
                    winner = "candidate" if score_delta > 0 else "base"
                elif all(delta == 0 for delta in score_deltas):
                    ratios = [row[16] / row[15] for row in matched if row[15] > 0 and row[16] > 0]
                    if len(ratios) == len(matched) and median(ratios) <= 0.9 and all(r < 1 for r in ratios):
                        winner = "candidate"
                    elif len(ratios) == len(matched) and median(ratios) >= 1 / 0.9 and all(r > 1 for r in ratios):
                        winner = "base"
            if winner == "candidate":
                verified_wins.add(product)
            elif winner == "base":
                verified_losses.add(product)
    return {**{name: counts[name] for name in (
                "eligible_browser_runs", "exact_matched_blocks", "ambiguous_matched_blocks",
                "repeated_linked_groups", "insufficient_block_groups", "conflicting_review_groups",
                "review_score_improved_groups", "both_product_pass_groups",
                "candidate_more_tokens_every_block_groups",
                "review_improved_without_token_gain_groups", "execution_verified_groups")},
            "execution_verified_independent_products": len(verified_products),
            "execution_verified_product_wins": len(verified_wins - verified_losses),
            "execution_verified_product_losses": len(verified_losses),
            "browser_grades_are_product_grades": False,
            "authoritative_for_routing": False}


def _local_prompt_comparisons() -> tuple[dict[str, list[tuple[str, dict[str, float], int]]],
                                         dict[str, bool], dict[str, int]]:
    """Build conservative prompt-only preferences from repeated, same-product runs."""
    with _connect() as connection:
        rows = connection.execute("""SELECT p.comparison_key,p.product_key,p.model,p.effort,
                                   p.prompt_digest,p.prompt_features,p.prompt_author,p.product_pass,
                                   p.satisfaction_score,p.total_tokens,p.observed_at_ms,
                                   p.suite_key,p.repetition_index,b.prompt_set_key,b.prompt_arm_index,
                                   r.observed_model,r.observed_effort,r.exact_total_tokens
                                   FROM benchmark_prompt_runs AS p
                                   LEFT JOIN managed_benchmark_bindings AS r ON r.run_key=p.run_key
                                   LEFT JOIN managed_benchmark_blocks AS b ON b.block_key=r.block_key
                                   WHERE p.comparison_key IS NOT NULL""").fetchall()
    arms: dict[tuple[str, str, str, str], dict[str, list[tuple[Any, ...]]]] = {}
    for (comparison, product, model, effort, prompt, features, author, passed, score,
         tokens, at, suite, repetition, prompt_set, prompt_arm, observed_model,
         observed_effort, exact_tokens) in rows:
        managed_execution = (prompt_set is not None and type(prompt_arm) is int
                             and observed_model == model and observed_effort == effort
                             and exact_tokens == tokens)
        arms.setdefault((comparison, product, model, effort), {}).setdefault(prompt, []).append(
            (json.loads(features), author, bool(passed), score, tokens, at,
             managed_execution, prompt_set or suite, repetition, managed_execution))
    grouped: dict[str, list[tuple[str, dict[str, float], int]]] = {}
    proofs: dict[str, bool] = {}
    counts = Counter()
    for (_comparison, product, _model, _effort), prompts in arms.items():
        bases = [runs for runs in prompts.values()
                 if all(row[1] == "benchmark" for row in runs)]
        if len(bases) != 1:
            counts["missing_or_ambiguous_baseline"] += 1
            continue
        base = bases[0]
        candidates = [runs for runs in prompts.values()
                      if all(row[1] == "codex" for row in runs)
                      or all(row[1] == "chatgpt" for row in runs)]
        for candidate in candidates:
            author = candidate[0][1]
            if min(len(base), len(candidate)) < 3:
                counts["insufficient_repeats_or_authorship"] += 1
                continue
            def keyed(runs: list[tuple[Any, ...]]) -> dict[tuple[str, int], tuple[Any, ...]]:
                eligible = [row for row in runs if row[8] is not None]
                result = {(row[7], row[8]): row for row in eligible}
                return result if len(result) == len(eligible) else {}
            base_blocks, candidate_blocks = keyed(base), keyed(candidate)
            common = sorted(base_blocks.keys() & candidate_blocks.keys())
            if len(common) < 3:
                counts["unblocked_pairs"] += 1
                continue
            paired = [(base_blocks[block], candidate_blocks[block]) for block in common]
            counts["repeated_pairs"] += 1
            counts[f"repeated_{author}_pairs"] += 1
            if all(row[9] for pair in paired for row in pair):
                counts["managed_randomized_pairs"] += 1
            base_passes = sum(left[2] for left, _right in paired)
            candidate_passes = sum(right[2] for _left, right in paired)
            winner = None
            if candidate_passes == len(paired) and base_passes <= len(paired) - 2:
                winner = "candidate"
            elif base_passes == len(paired) and candidate_passes <= len(paired) - 2:
                winner = "base"
            elif base_passes == len(paired) and candidate_passes == len(paired):
                score_deltas = [right[3] - left[3] for left, right in paired]
                score_delta = median(score_deltas)
                if abs(score_delta) >= 10 and all(delta * score_delta > 0 for delta in score_deltas):
                    winner = "candidate" if score_delta > 0 else "base"
                elif all(delta == 0 for delta in score_deltas):
                    ratios = [right[4] / left[4] for left, right in paired if left[4] > 0 and right[4] > 0]
                    if len(ratios) != len(paired):
                        counts["invalid_token_pairs"] += 1
                    elif median(ratios) <= 0.9 and all(ratio < 1 for ratio in ratios):
                        winner = "candidate"
                    elif median(ratios) >= 1 / 0.9 and all(ratio > 1 for ratio in ratios):
                        winner = "base"
            if winner is None:
                counts["inconclusive_pairs"] += 1
                continue
            counts["conclusive_pairs"] += 1
            counts[f"{winner}_wins"] += 1
            counts[f"{author}_{winner}_wins"] += 1
            base_vector, candidate_vector = paired[0][0][0], paired[0][1][0]
            vector = {index: candidate_vector.get(index, 0.0) - base_vector.get(index, 0.0)
                      for index in base_vector.keys() | candidate_vector.keys()}
            vector = {index: value for index, value in vector.items() if abs(value) > 1e-12}
            if not vector:
                counts["identical_feature_pairs"] += 1
                continue
            first_at = datetime.fromtimestamp(min(row[5] for pair in paired for row in pair) / 1000,
                                              timezone.utc).isoformat()
            label = int(winner == "candidate")
            grouped.setdefault(product, []).extend(((first_at, vector, label),
                                                      (first_at, {index: -value for index, value in vector.items()},
                                                       1 - label)))
            proofs[product] = proofs.get(product, True) and all(row[6] for pair in paired for row in pair)
    return grouped, proofs, dict(counts)


def train_local_prompt_model() -> dict[str, Any]:
    """Fit an advisory preference only from randomized, receipt-bound prompt pairs."""
    observed_grouped, proofs, counts = _local_prompt_comparisons()
    grouped = {product: rows for product, rows in observed_grouped.items()
               if proofs.get(product) is True}
    products = sorted(grouped, key=lambda product: min(row[0] for row in grouped[product]))
    report = {"independent_products": len(products), "comparison_counts": counts,
              "observational_independent_products": len(observed_grouped),
              "source": "managed_randomized_local_sandbox_benchmark",
              "browser_grades_included": False,
              "authoritative_for_routing": False}
    if len(products) < 8:
        return {**report, "status": "insufficient_managed_independent_products",
                "minimum_products": 8}
    holdout_count = max(2, math.ceil(len(products) * 0.2))
    train_products, test_products = products[:-holdout_count], products[-holdout_count:]
    training = [(vector, label) for product in train_products
                for _at, vector, label in grouped[product]]
    test = [(vector, label) for product in test_products
            for _at, vector, label in grouped[product]]
    weights, bias = _fit(training)
    baseline_brier = 0.25
    model_brier = _brier(test, weights, bias)
    checkpoint = _evaluation_checkpoint(LOCAL_PROMPT_EVAL_PATH,
                                        "modellabs.local_prompt_evaluation_checkpoint.v1",
                                        products, grouped,
                                        {"development_execution_verified": all(
                                            proofs[product] for product in products)})
    prospective_products, prospective = _prospective_rows(products, grouped, checkpoint)
    prospective_baseline = (sum((checkpoint["baseline_rate"] - label) ** 2
                                for _vector, label in prospective) / len(prospective)
                            if prospective else None)
    prospective_model = (_brier(prospective, checkpoint["weights"], checkpoint["bias"])
                         if prospective else None)
    gain = (prospective_baseline - prospective_model) if prospective else 0.0
    validated = (checkpoint["development_execution_verified"]
                 and all(proofs[product] for product in products)
                 and len(prospective_products) >= 10
                 and all(proofs[product] for product in prospective_products)
                 and gain >= MIN_REVIEW_BRIER_GAIN
                 and gain / max(prospective_baseline or 0.0, 1e-9) >= MIN_REVIEW_RELATIVE_GAIN)
    final_rows = [(vector, label) for product in products
                  for _at, vector, label in grouped[product]]
    final_weights, final_bias = _fit(final_rows)
    artifact = {"schema": "modellabs.local_prompt_preference_model.v1",
                "feature_version": FEATURE_VERSION,
                "trained_comparisons": sum(len(rows) // 2 for rows in grouped.values()),
                "independent_products": len(products),
                "holdout_products": len(test_products),
                "baseline_brier": round(baseline_brier, 6),
                "model_brier": round(model_brier, 6),
                "retrospective_holdout_is_diagnostic_only": True,
                "prospective_holdout_products": len(prospective_products),
                "prospective_baseline_brier": round(prospective_baseline, 6) if prospective else None,
                "prospective_model_brier": round(prospective_model, 6) if prospective else None,
                "execution_model_verified_for_development": checkpoint["development_execution_verified"],
                "current_training_execution_verified": all(proofs[product]
                                                              for product in products),
                "validated_for_shadow": validated,
                "authoritative_for_routing": False,
                "browser_grades_included": False,
                "training_randomized": True,
                "causal_prompt_comparison": True,
                "weights": final_weights, "bias": final_bias}
    temporary = LOCAL_PROMPT_MODEL_PATH.with_name(
        f".{LOCAL_PROMPT_MODEL_PATH.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        json.dump(artifact, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, LOCAL_PROMPT_MODEL_PATH)
    return {**report, "status": "trained", **{key: value for key, value in artifact.items()
                                             if key not in {"weights", "bias"}}}


def predict_local_prompt(base_prompt: str, candidate_prompt: str) -> dict[str, Any]:
    model = _read_json(LOCAL_PROMPT_MODEL_PATH)
    if (model.get("schema") != "modellabs.local_prompt_preference_model.v1"
            or model.get("feature_version") != FEATURE_VERSION):
        raise ValueError("local prompt model version mismatch")
    key = _key()
    base = _features(base_prompt, "", 1, key)
    candidate = _features(candidate_prompt, "", 1, key)
    vector = {index: candidate.get(index, 0.0) - base.get(index, 0.0)
              for index in base.keys() | candidate.keys()}
    probability = _sigmoid(model["bias"] + sum(
        model["weights"][int(index)] * value for index, value in vector.items()
        if int(index) < len(model["weights"])))
    return {"candidate_preference_probability": round(probability, 4),
            "validated_for_shadow": model["validated_for_shadow"],
            "authoritative_for_routing": False,
            "browser_grades_included": False,
            "independent_training_products": model["independent_products"]}


def _session_message_text(message: dict[str, Any]) -> str | None:
    if message.get("type") != "message" or message.get("role") != "user":
        return None
    parts = [item.get("text") for item in message.get("content") or []
             if isinstance(item, dict) and item.get("type") == "input_text"
             and isinstance(item.get("text"), str)]
    value = "\n".join(parts).strip()
    if (not value or value.startswith(("<codex_internal_context", "# AGENTS.md instructions",
                                       "<environment_context"))):
        return None
    return value


def _session_context_row(message: dict[str, Any], source_thread: str,
                         turn_id: str, inference_seen: bool, key: bytes) -> tuple[str, ...] | None:
    """Bind one text-only durable user message to its exact recorded turn."""
    content = message.get("content")
    metadata = message.get("internal_chat_message_metadata_passthrough")
    message_id = message.get("id")
    if (not isinstance(message_id, str) or not 0 < len(message_id) <= 128
            or not isinstance(metadata, dict) or metadata.get("turn_id") != turn_id
            or not isinstance(content, list) or not content
            or any(not isinstance(item, dict) or item.get("type") != "input_text"
                   or not isinstance(item.get("text"), str) for item in content)):
        return None
    prompt = _session_message_text(message)
    if prompt is None or len(prompt) > 256_000:
        return None
    return (_digest(key, f"{source_thread}:{message_id}"), _digest(key, source_thread),
            _digest(key, turn_id), _digest(key, prompt),
            json.dumps(_features(prompt, "", 1, key), sort_keys=True),
            "after_prior_output" if inference_seen else "pre_inference", "context_only")


def import_codex_sessions(sessions_root: Path | None = None,
                          thread_id: str | None = None) -> dict[str, int]:
    """Capture completed standalone Codex prompts and answers without labels."""
    root = sessions_root or Path.home() / ".codex/sessions"
    if root.is_symlink():
        raise ValueError("sessions root may not be a symlink")
    if thread_id is not None and not re.fullmatch(r"[0-9a-fA-F-]{36}", thread_id):
        raise ValueError("invalid thread id")
    key = _key()
    counts = Counter()
    paths = root.rglob(f"*{thread_id}.jsonl") if thread_id else root.rglob("rollout-*.jsonl")
    with _connect() as connection:
        for path in paths:
            try:
                if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SESSION_BYTES:
                    counts["file_skipped"] += 1
                    continue
                current: dict[str, Any] | None = None
                source_thread = None
                with path.open("r", encoding="utf-8") as source:
                    for line in source:
                        if len(line) > MAX_SESSION_LINE_BYTES:
                            counts["line_skipped"] += 1
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        payload = item.get("payload") or {}
                        kind = item.get("type")
                        if kind == "session_meta":
                            source_thread = payload.get("id")
                            if thread_id is not None and source_thread != thread_id:
                                raise ValueError("session thread identity mismatch")
                        elif kind == "event_msg" and payload.get("type") == "task_started":
                            current = {"turn_id": payload.get("turn_id"), "prompts": [],
                                       "model": None, "effort": None, "tokens": None,
                                       "inference_seen": False, "late_user": False}
                        elif current is not None and kind == "turn_context":
                            if payload.get("turn_id") == current["turn_id"]:
                                current["model"], current["effort"] = payload.get("model"), payload.get("effort")
                        elif current is not None and kind == "response_item":
                            value = _session_message_text(payload)
                            if value is not None:
                                if isinstance(source_thread, str) and isinstance(current["turn_id"], str):
                                    context_row = _session_context_row(
                                        payload, source_thread, current["turn_id"],
                                        current["inference_seen"], key)
                                    if context_row is not None:
                                        previous_context = connection.execute("""SELECT thread_key,turn_key,
                                            prompt_digest,features,phase,attribution
                                            FROM session_user_messages WHERE message_key=?""",
                                            (context_row[0],)).fetchone()
                                        if previous_context is None:
                                            connection.execute("""INSERT INTO session_user_messages
                                                VALUES (?,?,?,?,?,?,?)""", context_row)
                                            counts["context_imported"] += 1
                                        elif previous_context == context_row[1:]:
                                            counts["context_unchanged"] += 1
                                        elif (previous_context[:4] == context_row[1:5]
                                              and previous_context[5] == context_row[6]):
                                            # A replay can place the same durable message
                                            # on either side of prior output. Keep the
                                            # original phase; it is not an outcome label.
                                            counts["context_phase_disagreement"] += 1
                                        else:
                                            counts["context_conflict"] += 1
                                if current["inference_seen"]:
                                    current["late_user"] = True
                                elif len(current["prompts"]) < 8:
                                    current["prompts"].append(value)
                            elif payload.get("role") == "assistant" or payload.get("type") != "message":
                                current["inference_seen"] = True
                        elif current is not None and kind == "token_usage_record":
                            if payload.get("turn_id") == current["turn_id"]:
                                tokens = (payload.get("turn_token_usage") or {}).get("total_tokens")
                                if isinstance(tokens, int) and tokens > 0:
                                    current["tokens"] = tokens
                        elif (current is not None and kind == "event_msg"
                              and payload.get("type") == "task_complete"):
                            if payload.get("turn_id") != current["turn_id"]:
                                current = None
                                continue
                            prompt = (current["prompts"][0].strip()
                                      if len(current["prompts"]) == 1 and not current["late_user"] else "")
                            result = payload.get("last_agent_message")
                            if (not isinstance(source_thread, str) or not isinstance(current["turn_id"], str)
                                    or not prompt or not isinstance(result, str) or not result.strip()
                                    or len(prompt) > 256_000 or len(result) > 256_000):
                                counts["turn_ineligible"] += 1
                                current = None
                                continue
                            completed_at = payload.get("completed_at")
                            if isinstance(completed_at, (int, float)):
                                completed_at = datetime.fromtimestamp(completed_at, timezone.utc).isoformat()
                            elif not isinstance(completed_at, str):
                                completed_at = item.get("timestamp")
                            if not isinstance(completed_at, str):
                                counts["turn_ineligible"] += 1
                                current = None
                                continue
                            identity = (_digest(key, source_thread), _digest(key, current["turn_id"]))
                            prompt_digest, result_digest = _digest(key, prompt), _digest(key, result)
                            previous = connection.execute("""SELECT prompt_digest,result_digest,input_scope FROM session_turns
                                                             WHERE thread_key=? AND turn_key=?""", identity).fetchone()
                            if previous:
                                if previous[:2] == (prompt_digest, result_digest):
                                    if previous[2] != "single_pre_inference_message":
                                        connection.execute("""UPDATE session_turns SET input_scope='single_pre_inference_message'
                                                              WHERE thread_key=? AND turn_key=?""", identity)
                                    counts["unchanged"] += 1
                                else:
                                    counts["conflict"] += 1
                            else:
                                connection.execute("""INSERT INTO session_turns
                                    (thread_key,turn_key,prompt_digest,prompt_features,result_digest,
                                     result_features,model,effort,reported_total_tokens,completed_at,
                                     source_path_digest,input_scope)
                                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                                    (*identity, prompt_digest,
                                     json.dumps(_features(prompt, "", 1, key), sort_keys=True),
                                     result_digest, json.dumps(_features(result, "", 1, key), sort_keys=True),
                                     current["model"], current["effort"], current["tokens"],
                                     completed_at, _digest(key, str(path)), "single_pre_inference_message"))
                                counts["imported"] += 1
                            current = None
            except (OSError, ValueError, TypeError):
                counts["file_skipped"] += 1
    return {name: counts[name] for name in ("imported", "unchanged", "conflict", "turn_ineligible",
                                           "file_skipped", "line_skipped", "context_imported", "context_unchanged",
                                           "context_phase_disagreement", "context_conflict")}


def _exact_completed_codex_turn(thread_id: str, turn_id: str,
                                sessions_root: Path | None = None) -> tuple[str, str]:
    """Read one completed pair, normalizing prompt outer whitespace as session import does."""
    uuid = r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
    if not re.fullmatch(uuid, thread_id) or not re.fullmatch(uuid, turn_id):
        raise ValueError("exact thread and turn UUIDs are required")
    root = sessions_root or Path.home() / ".codex/sessions"
    if root.is_symlink() or not root.is_dir():
        raise ValueError("sessions root must be a real directory")
    matches: list[tuple[str, str]] = []
    for path in root.rglob(f"*{thread_id}.jsonl"):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SESSION_BYTES:
            continue
        source_thread = None
        current: dict[str, Any] | None = None
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if len(line) > MAX_SESSION_LINE_BYTES:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = item.get("payload") or {}
                kind = item.get("type")
                if kind == "session_meta":
                    source_thread = payload.get("id")
                elif kind == "event_msg" and payload.get("type") == "task_started":
                    current = ({"prompts": [], "inference_seen": False, "late_user": False}
                               if payload.get("turn_id") == turn_id else None)
                elif current is not None and kind == "response_item":
                    value = _session_message_text(payload)
                    if value is not None:
                        if current["inference_seen"]:
                            current["late_user"] = True
                        elif len(current["prompts"]) < 8:
                            current["prompts"].append(value)
                    elif payload.get("role") == "assistant" or payload.get("type") != "message":
                        current["inference_seen"] = True
                elif (current is not None and kind == "event_msg"
                      and payload.get("type") == "task_complete"):
                    if source_thread != thread_id or payload.get("turn_id") != turn_id:
                        current = None
                        continue
                    result = payload.get("last_agent_message")
                    if (len(current["prompts"]) != 1 or current["late_user"]
                            or not isinstance(result, str) or not result.strip()
                            or len(current["prompts"][0]) > 256_000
                            or len(result) > 256_000):
                        raise ValueError("completed turn is not an exact single-prompt result match")
                    matches.append((current["prompts"][0], result))
                    current = None
    if len(matches) != 1:
        raise ValueError("exactly one completed matching Codex turn is required")
    return matches[0]


def _private_export_parent(output: Path) -> None:
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink() or parent.stat().st_mode & 0o077:
        raise ValueError("result output requires an existing private directory")


def _write_exclusive_private_text(output: Path, value: str) -> bytes:
    payload = value.encode("utf-8")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return payload


def export_codex_result(thread_id: str, turn_id: str, generation_prompt_file: Path,
                        output_file: Path, sessions_root: Path | None = None) -> dict[str, str]:
    """Stage one exact completed final answer for a digest-bound browser review."""
    prompt_path = generation_prompt_file.expanduser()
    if prompt_path.is_symlink() or not prompt_path.is_file() or prompt_path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("generation prompt must be a bounded regular file")
    expected_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not expected_prompt:
        raise ValueError("generation prompt is empty")
    prompt, result = _exact_completed_codex_turn(thread_id, turn_id, sessions_root)
    if prompt != expected_prompt:
        raise ValueError("completed turn is not an exact single-prompt result match")
    output = output_file.expanduser()
    _private_export_parent(output)
    payload = _write_exclusive_private_text(output, result)
    return {"thread_id": thread_id, "turn_id": turn_id,
            "output_file": str(output.resolve()),
            "result_sha256": hashlib.sha256(payload).hexdigest()}


def export_codex_turn(thread_id: str, turn_id: str, output_dir: Path,
                      sessions_root: Path | None = None) -> dict[str, str]:
    """Stage one completed user message and exact final answer for traced review."""
    prompt, result = _exact_completed_codex_turn(thread_id, turn_id, sessions_root)
    directory = output_dir.expanduser()
    prompt_path = directory / "generation-prompt.txt"
    result_path = directory / "codex-final.txt"
    _private_export_parent(prompt_path)
    prompt_payload = _write_exclusive_private_text(prompt_path, prompt)
    try:
        result_payload = _write_exclusive_private_text(result_path, result)
    except Exception:
        prompt_path.unlink()
        raise
    return {"thread_id": thread_id, "turn_id": turn_id,
            "generation_prompt_file": str(prompt_path.resolve()),
            "generation_prompt_sha256": hashlib.sha256(prompt_payload).hexdigest(),
            "codex_result_file": str(result_path.resolve()),
            "codex_result_sha256": hashlib.sha256(result_payload).hexdigest()}


def sync_session_grades(metrics_path: Path | None = None) -> dict[str, int]:
    """Join complete retrospective sessions to exact proxy usage and explicit grades."""
    from telemetry import METRICS_PATH

    path = metrics_path or METRICS_PATH
    if path.is_symlink() or not path.is_file():
        return {"joined": 0, "unchanged": 0, "ineligible": 0}
    key = _key()
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("event") not in {
                    "route_accepted", "turn_completed", "turn_usage", "quality_grade",
                    "model_rerouted"}:
                continue
            thread_id = row.get("thread_id")
            turn_id = row.get("source_turn_id") if row["event"] == "quality_grade" else row.get("turn_id")
            if not isinstance(thread_id, str) or not isinstance(turn_id, str):
                continue
            identity = (_digest(key, thread_id), _digest(key, turn_id))
            grouped.setdefault(identity, {}).setdefault(row["event"], []).append(row)
    counts = Counter()
    with _connect() as connection:
        for identity, events in grouped.items():
            session = connection.execute("""SELECT model,effort,reported_total_tokens,grade_status,
                                        exact_total_tokens,quality_score,verification,input_scope
                                        FROM session_turns WHERE thread_key=? AND turn_key=?""",
                                         identity).fetchone()
            if session is None:
                continue
            if events.get("model_rerouted"):
                counts["ineligible"] += 1
                continue
            if any(len(events.get(name, [])) != 1 for name in
                   ("route_accepted", "turn_completed", "turn_usage", "quality_grade")):
                counts["ineligible"] += 1
                continue
            accepted, completed, usage, grade = (events[name][0] for name in
                                                  ("route_accepted", "turn_completed", "turn_usage", "quality_grade"))
            total = (usage.get("usage") or {}).get("totalTokens")
            score = grade.get("quality_score")
            if (session[7] != "single_pre_inference_message"
                    or completed.get("status") != "completed"
                    or usage.get("source") not in {"proxy_thread_usage", "proxy_thread_usage_delta"}
                    or grade.get("source") != "explicit"
                    or grade.get("verification") not in {"passed", "failed"}
                    or not isinstance(score, int) or not 0 <= score <= 100
                    or not isinstance(total, int) or total <= 0
                    or session[0] not in MODEL_IDS or session[1] not in EFFORTS
                    or session[2] != total
                    or any(event.get("model") != session[0] or event.get("effort") != session[1]
                           for event in (accepted, completed, usage, grade))):
                counts["ineligible"] += 1
                continue
            expected = (total, score, grade["verification"])
            if session[3] == "explicit" and session[4:7] == expected:
                counts["unchanged"] += 1
                continue
            if session[3] != "ungraded":
                counts["ineligible"] += 1
                continue
            connection.execute("""UPDATE session_turns SET grade_status='explicit',task_class=?,
                                exact_total_tokens=?,quality_score=?,verification=?
                                WHERE thread_key=? AND turn_key=? AND grade_status='ungraded'""",
                               (accepted.get("task_class"), total, score, grade["verification"], *identity))
            counts["joined"] += 1
    return {name: counts[name] for name in ("joined", "unchanged", "ineligible")}


def backfill_managed_turns(metrics_path: Path | None = None) -> dict[str, int]:
    """Recover exact managed prompt/result pairs from completed local sessions.

    This repairs historical capture gaps only. It does not manufacture grades or
    usage, and it never replaces a turn observed directly by the live proxy.
    """
    from telemetry import METRICS_PATH

    path = metrics_path or METRICS_PATH
    counts = Counter()
    if path.is_symlink() or not path.is_file():
        return {name: 0 for name in ("imported", "unchanged", "conflict", "ineligible")}
    events: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("event") not in {"route_accepted", "turn_completed"}:
                continue
            thread_id, turn_id = row.get("thread_id"), row.get("turn_id")
            if isinstance(thread_id, str) and isinstance(turn_id, str):
                events.setdefault((thread_id, turn_id), {}).setdefault(row["event"], []).append(row)
    key = _key()
    with _connect() as connection:
        for (thread_id, turn_id), pair in events.items():
            if len(pair.get("route_accepted", [])) != 1 or len(pair.get("turn_completed", [])) != 1:
                counts["ineligible"] += 1
                continue
            accepted, completed = pair["route_accepted"][0], pair["turn_completed"][0]
            identity = (_digest(key, thread_id), _digest(key, turn_id))
            session = connection.execute("""SELECT prompt_digest,prompt_features,result_digest,
                                          result_features,model,effort,input_scope FROM session_turns
                                          WHERE thread_key=? AND turn_key=?""", identity).fetchone()
            at_ms = accepted.get("recorded_at_ms")
            task_class = accepted.get("task_class")
            if (session is None or session[6] != "single_pre_inference_message"
                    or completed.get("status") != "completed"
                    or not isinstance(at_ms, int) or at_ms <= 0
                    or task_class not in TASK_CLASSES
                    or session[4] not in MODEL_IDS or session[5] not in EFFORTS
                    or any(event.get("model") != session[4] or event.get("effort") != session[5]
                           for event in (accepted, completed))):
                counts["ineligible"] += 1
                continue
            existing = connection.execute("""SELECT prompt_digest,features,result_digest,
                                           result_features,selected_model,effort,task_class
                                           FROM codex_turns WHERE thread_key=? AND turn_key=?""",
                                          identity).fetchone()
            expected = (*session[:4], session[4], session[5], task_class)
            if existing is not None:
                counts["unchanged" if existing == expected else "conflict"] += 1
                continue
            connection.execute("""INSERT INTO codex_turns
                               (thread_key,turn_key,group_id,accepted_at_ms,prompt_digest,features,
                                result_digest,result_features,selected_model,effort,task_class)
                               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                               (*identity, identity[0], at_ms, *expected))
            counts["imported"] += 1
    return {name: counts[name] for name in ("imported", "unchanged", "conflict", "ineligible")}


def sync_codex_grades(metrics_path: Path | None = None) -> dict[str, int]:
    """Join exact usage, grades, and positive execution evidence onto turns."""
    from telemetry import METRICS_PATH

    path = metrics_path or METRICS_PATH
    key = _key()
    counts = Counter()
    if not path.exists():
        return {"graded": 0, "usage_matched": 0, "observed": 0, "rerouted": 0}
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or not isinstance(row.get("thread_id"), str):
                continue
            records.append(row)
    completed = {(row.get("thread_id"), row.get("turn_id")) for row in records
                 if row.get("event") == "turn_completed" and row.get("status") == "completed"}
    exact_usage = {(row.get("thread_id"), row.get("turn_id"), row.get("model"), row.get("effort"))
                   for row in records
                   if row.get("event") == "turn_usage"
                   and row.get("source") == "proxy_thread_usage_delta"
                   and isinstance((row.get("usage") or {}).get("totalTokens"), int)
                   and (row.get("usage") or {}).get("totalTokens") > 0}
    rerouted_turns = {(row.get("thread_id"), row.get("turn_id")) for row in records
                      if row.get("event") == "model_rerouted"}
    with _connect() as connection:
        for row in records:
            event = row.get("event")
            turn_id = row.get("source_turn_id") if event == "quality_grade" else row.get("turn_id")
            if not isinstance(turn_id, str):
                continue
            identity = (_digest(key, row["thread_id"]), _digest(key, turn_id))
            if event == "quality_grade":
                score, verification = row.get("quality_score"), row.get("verification")
                if (not isinstance(score, int) or not 0 <= score <= 100
                        or verification not in {"passed", "failed"}
                        or row.get("source") != "explicit"):
                    continue
                cursor = connection.execute("""UPDATE codex_turns SET quality_score=?,verification=?
                                               WHERE thread_key=? AND turn_key=? AND quality_score IS NULL""",
                                            (score, verification, *identity))
                counts["graded"] += cursor.rowcount
            elif event == "turn_usage":
                tokens = (row.get("usage") or {}).get("totalTokens")
                if (not isinstance(tokens, int) or tokens <= 0
                        or row.get("source") != "proxy_thread_usage_delta"
                        or (row["thread_id"], turn_id) not in completed):
                    continue
                cursor = connection.execute("""UPDATE codex_turns SET total_tokens=?
                                               WHERE thread_key=? AND turn_key=? AND total_tokens IS NULL
                                               AND selected_model=? AND effort=?""",
                                            (tokens, *identity, row.get("model"), row.get("effort")))
                counts["usage_matched"] += cursor.rowcount
            elif event == "model_rerouted":
                if (not isinstance(row.get("assigned_model"), str)
                        or not isinstance(row.get("from_model"), str)
                        or not isinstance(row.get("to_model"), str)
                        or row["from_model"] == row["to_model"]):
                    continue
                cursor = connection.execute("""UPDATE codex_turns SET reroute_seen=1,
                    model_provenance='server_rerouted' WHERE thread_key=? AND turn_key=?
                    AND selected_model=? AND reroute_seen=0""",
                    (*identity, row["assigned_model"]))
                counts["rerouted"] += cursor.rowcount
            elif event == "route_execution_observed":
                model, effort = row.get("model"), row.get("effort")
                if (model not in MODEL_IDS or effort not in EFFORTS
                        or row.get("source") != "host_thread_settings_updated_pre_admission"
                        or row.get("settings_confirmation") != "exact"
                        or (row["thread_id"], turn_id) not in completed
                        or (row["thread_id"], turn_id, model, effort) not in exact_usage
                        or (row["thread_id"], turn_id) in rerouted_turns):
                    continue
                cursor = connection.execute("""UPDATE codex_turns SET observed_model=?,
                    observed_effort=?,model_provenance='observed_per_request'
                    WHERE thread_key=? AND turn_key=? AND selected_model=? AND effort=?
                    AND reroute_seen=0 AND model_provenance='proxy_selected_only'
                    AND observed_model IS NULL AND observed_effort IS NULL""",
                    (model, effort, *identity, model, effort))
                counts["observed"] += cursor.rowcount
    return {"graded": counts["graded"], "usage_matched": counts["usage_matched"],
            "observed": counts["observed"], "rerouted": counts["rerouted"]}


def _read_json(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_SOURCE_BYTES:
            raise ValueError("source is not a bounded regular file")
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("source is not a JSON object")
    return value


def _recovery_observation_row(run_dir: Path, key: bytes) -> tuple[Any, ...]:
    """Read a failed-response sidecar as ungraded evidence, never a review label."""
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("recovery run directory is not a real directory")
    response_id = run_dir.name
    if not re.fullmatch(r"resp_[A-Za-z0-9_-]+", response_id):
        raise ValueError("invalid recovery response identity")
    record_path = run_dir / "record.json"
    sidecar_path = run_dir / "recovery-observation.json"
    if record_path.is_symlink() or sidecar_path.is_symlink():
        raise ValueError("recovery evidence path is a symlink")
    with os.fdopen(os.open(record_path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_SOURCE_BYTES:
            raise ValueError("recovery record is not a bounded regular file")
        record_bytes = handle.read(MAX_SOURCE_BYTES + 1)
    with os.fdopen(os.open(sidecar_path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_SOURCE_BYTES:
            raise ValueError("recovery sidecar is not a bounded regular file")
        sidecar_bytes = handle.read(MAX_SOURCE_BYTES + 1)
    record = json.loads(record_bytes)
    observation = json.loads(sidecar_bytes)
    if not isinstance(record, dict) or not isinstance(observation, dict):
        raise ValueError("recovery evidence is not a JSON object")
    bundle = record.get("bundle") or {}
    run = bundle.get("run") or {}
    steps = bundle.get("steps") or []
    if (record.get("runId") != response_id or run.get("id") != response_id
            or run.get("status") != "failed" or run.get("sourceKind") != "direct"
            or not isinstance(steps, list) or len(steps) != 1
            or not isinstance(steps[0], dict) or steps[0].get("status") != "failed"
            or steps[0].get("service") != "chatgpt"):
        raise ValueError("recovery source is not one failed direct ChatGPT run")
    inputs = run.get("initialInputs") or {}
    step_input = steps[0].get("input") or {}
    metadata = inputs.get("metadata") or {}
    prompt = inputs.get("requestInput")
    nonce = metadata.get("nonce") if isinstance(metadata, dict) else None
    round_number = metadata.get("round") if isinstance(metadata, dict) else None
    failure = steps[0].get("failure") or {}
    failure_details = failure.get("details") or {}
    if (not isinstance(metadata, dict) or not isinstance(prompt, str) or not prompt.strip()
            or step_input.get("prompt") != prompt
            or (step_input.get("structuredData") or {}).get("metadata") != metadata
            or failure.get("code") != "runner_execution_failed"
            or failure.get("ownerStepId") != steps[0].get("id")
            or failure_details.get("phase") != "after"
            or failure_details.get("retryable") is not False
            or metadata.get("schema") != "guard" or metadata.get("workflow") != "codex-pro-guard"
            or type(round_number) is not int or not 1 <= round_number <= 20
            or not isinstance(nonce, str) or not 16 <= len(nonce) <= 128
            or nonce not in prompt
            or not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("learning_trace_digest")))
            or not re.fullmatch(r"[0-9a-fA-F-]{36}", str(metadata.get("client_request_id")))):
        raise ValueError("recovery source lacks exact guard request correlation")
    answer = observation.get("answer_text")
    if (observation.get("schema") != "auracall.response_recovery_observation.v1"
            or observation.get("response_id") != response_id
            or observation.get("original_status") != "failed"
            or observation.get("original_run_modified") is not False
            or observation.get("prompt_submitted") is not False
            or observation.get("account_verdict") != "match"
            or observation.get("request_metadata") != metadata
            or observation.get("original_record_digest") != hashlib.sha256(record_bytes).hexdigest()
            or observation.get("logical_prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest()
            or not re.fullmatch(r"[0-9a-f]{64}", str(observation.get("wire_prompt_sha256")))
            or not isinstance(answer, str) or not answer.strip() or len(answer) > 256_000
            or nonce not in answer
            or observation.get("answer_sha256") != hashlib.sha256(answer.encode()).hexdigest()
            or not isinstance(observation.get("user_message_id"), str)
            or not observation["user_message_id"].strip()
            or not isinstance(observation.get("assistant_message_id"), str)
            or not observation["assistant_message_id"].strip()
            or type(observation.get("browser_process_id")) is not int
            or observation["browser_process_id"] < 1
            or not isinstance(observation.get("target_id"), str)
            or not observation["target_id"].strip()
            or observation.get("conversation_url") != (inputs.get("auracall") or {}).get("chatgptConversationUrl")):
        raise ValueError("recovery observation does not match the failed source")
    observed_at = observation.get("observed_at")
    if not isinstance(observed_at, str):
        raise ValueError("recovery observation timestamp is missing")
    _utc_timestamp(observed_at)
    artifacts = step_input.get("artifacts") or []
    requested = inputs.get("attachments") or []
    if not isinstance(artifacts, list) or not isinstance(requested, list) or len(artifacts) != len(requested):
        raise ValueError("recovery attachment lists differ")
    confirmed = 0
    if artifacts:
        events = bundle.get("events") or []
        submitted = next((index for index, event in enumerate(events)
                          if (event.get("payload") or {}).get("runtimeEvidence", {}).get("evidenceRef")
                          == "chatgpt-prompt-submitted"), -1)
        receipts = [(index, ((event.get("payload") or {}).get("runtimeEvidence", {}).get("details") or {}).get("attachmentUiReceipt"))
                    for index, event in enumerate(events) if isinstance(event, dict)]
        receipts = [(index, receipt) for index, receipt in receipts if isinstance(receipt, dict)]
        paths = [artifact.get("path") for artifact in artifacts if isinstance(artifact, dict)]
        if (len(paths) != len(artifacts) or len(paths) > 10
                or any(not isinstance(item, str) or not item for item in paths)
                or len(set(paths)) != len(paths)
                or any(not isinstance(item, dict) or artifact.get("id") != item.get("id")
                       or artifact.get("uri") != item.get("uri")
                       or artifact.get("title") != item.get("fileName")
                       for artifact, item in zip(artifacts, requested))
                or submitted < 0 or not receipts or receipts[-1][0] <= submitted):
            raise ValueError("recovery attachment provenance is incomplete")
        receipt = receipts[-1][1]
        if (receipt.get("schema") != "auracall.browser_attachment_ui_receipt.v1"
                or receipt.get("attachmentPaths") != paths
                or receipt.get("uploadCompletion") != "confirmed"
                or receipt.get("sentUserTurnAttachments") != "confirmed"
                or receipt.get("submittedUserId") != observation["user_message_id"]):
            raise ValueError("recovery attachment receipt differs from the observed turn")
        confirmed = 1
    return (_digest(key, response_id), response_id,
            observation["original_record_digest"], hashlib.sha256(sidecar_bytes).hexdigest(),
            observed_at, _digest(key, prompt),
            json.dumps(_features(prompt, "", round_number, key), sort_keys=True),
            _digest(key, answer), json.dumps(_features(answer, "", 1, key), sort_keys=True),
            confirmed, "ungraded_failed_response")


def import_recovery_observations(runs_root: Path | None = None) -> dict[str, int]:
    """Import private keyed features from eligible failed-run observations."""
    runs_root = runs_root or Path.home() / ".auracall/runtime/runs"
    if runs_root.is_symlink():
        raise ValueError("AuraCall runs root may not be a symlink")
    counts = Counter()
    key = _key()
    with _connect() as connection:
        for sidecar in sorted(runs_root.glob("resp_*/recovery-observation.json")):
            try:
                row = _recovery_observation_row(sidecar.parent, key)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
                counts["rejected"] += 1
                continue
            existing = connection.execute("SELECT * FROM recovery_observations WHERE observation_id=?",
                                          (row[0],)).fetchone()
            if existing is None:
                connection.execute("INSERT INTO recovery_observations VALUES (?,?,?,?,?,?,?,?,?,?,?)", row)
                counts["imported"] += 1
            else:
                counts["unchanged" if existing == row else "conflict"] += 1
    return {name: counts[name] for name in ("imported", "unchanged", "rejected", "conflict")}


def _guard_reference_manifest(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Verify binary evidence without decoding documents or putting their contents in metrics."""
    if not isinstance(sources, list) or len(sources) > 8:
        raise ValueError("invalid guard reference count")
    mimes = {".md": "text/markdown", ".pdf": "application/pdf",
             ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    reserved = {"review-goal.md", "candidate.md", "review-guide.md", "review-prompt.md",
                "review-record.md", "review-chat.md"}
    names, manifest, total = set(), [], 0
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("invalid guard reference identity")
        name = source.get("fileName")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,159}", name)
                or name.casefold() in names | reserved or Path(name).suffix.lower() not in mimes
                or source.get("mimeType") != mimes[Path(name).suffix.lower()]
                or type(source.get("size")) is not int or not 0 < source["size"] <= 20 * 1024 * 1024
                or not isinstance(source.get("path"), str) or not Path(source["path"]).is_absolute()):
            raise ValueError("invalid guard reference identity")
        descriptor = os.open(source["path"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != source["size"]:
                raise ValueError("guard reference is not a bounded regular file")
            raw = handle.read(20 * 1024 * 1024 + 1)
        if len(raw) != source["size"] or hashlib.sha256(raw).hexdigest() != source.get("sha256"):
            raise ValueError("guard reference bytes changed")
        names.add(name.casefold())
        total += len(raw)
        manifest.append({key: source[key] for key in ("fileName", "mimeType", "size", "sha256")})
    if total > 50 * 1024 * 1024:
        raise ValueError("guard references exceed combined bound")
    return manifest


def _guard_review_texts(state: dict[str, Any], record: dict[str, Any]) -> tuple[str, str, str | None]:
    """Resolve actual reviewed bytes, never mistake a file-handoff prompt for the candidate."""
    inputs = record["bundle"]["run"]["initialInputs"]
    if state.get("review_format") not in {None, "inline", "file"}:
        raise ValueError("unsupported guard review format")
    if state.get("review_format") != "file":
        instructions, revision = inputs.get("instructions"), inputs.get("requestInput")
        if not isinstance(instructions, str) or not isinstance(revision, str) or not revision.strip():
            raise ValueError("missing legacy review text")
        start = instructions.find(GOAL_START)
        end = instructions.find(GOAL_END, start + len(GOAL_START))
        if start < 0 or end <= start + len(GOAL_START):
            raise ValueError("missing legacy review goal")
        return instructions[start + len(GOAL_START):end].strip(), revision, None
    handoff = state.get("file_review") or {}
    files = handoff.get("files") or {}
    if (handoff.get("schema") != "codex.pro_guard_file_handoff.v1"
            or handoff.get("prompt_author") != "codex" or set(files) != {"goal", "artifact", "guide", "prompt"}):
        raise ValueError("invalid file review handoff")
    expected = {"schema": handoff["schema"], "prompt_author": "codex",
                "files": {name: {"fileName": source["fileName"], "sha256": source["sha256"]}
                          for name, source in files.items()}}
    references = handoff.get("references") or []
    reference_manifest = _guard_reference_manifest(references)
    if reference_manifest != _guard_reference_manifest(state.get("reference_sources") or []):
        raise ValueError("guard reference snapshots differ from supplied documents")
    if references:
        expected["references"] = reference_manifest
    if (inputs.get("metadata") or {}).get("guardFileReview") != expected:
        raise ValueError("file review is not bound to the submitted request")
    contents = {name: _bound_text(source, preserve_whitespace=True) for name, source in files.items()}
    if inputs.get("requestInput") != contents["prompt"] or inputs.get("instructions") not in {None, ""}:
        raise ValueError("file review generation prompt changed")
    sources = state.get("source_files") or {}
    for name in ("goal", "artifact"):
        if (sources[name]["sha256"] != files[name]["sha256"]
                or _bound_text(sources[name], preserve_whitespace=True) != contents[name]):
            raise ValueError("file review differs from its original source bytes")
    attachments = inputs.get("attachments") or []
    if len(attachments) != 3 + len(references):
        raise ValueError("unexpected review attachments")
    for name in ("goal", "artifact", "guide"):
        matches = [item for item in attachments if isinstance(item, dict) and item.get("id") == f"guard-{name}"]
        if (len(matches) != 1 or matches[0].get("fileName") != files[name]["fileName"]
                or matches[0].get("uri") != Path(files[name]["path"]).as_uri()
                or matches[0].get("mimeType") != "text/markdown"):
            raise ValueError("reviewed file was not attached to the bound request")
    for index, source in enumerate(references):
        matches = [item for item in attachments if isinstance(item, dict)
                   and item.get("id") == f"guard-reference-{index}"]
        if (len(matches) != 1 or matches[0].get("fileName") != source["fileName"]
                or matches[0].get("uri") != Path(source["path"]).as_uri()
                or matches[0].get("mimeType") != source["mimeType"]):
            raise ValueError("reference file was not attached to the bound request")
    raw = _bound_text(state["review_artifact"], preserve_whitespace=True)
    match = re.fullmatch(r"\s*(?:# [^\n]+\n\s*)?```json\s*\n(.*?)\n```\s*", raw, re.S)
    if not match:
        raise ValueError("invalid downloaded review record")
    verdict = json.loads(match.group(1))
    if verdict != state.get("verdict"):
        raise ValueError("saved verdict differs from the browser file")
    for name, expected_value in {"schema": "codex.pro_guard_review_file.v1", "nonce": state["nonce"],
            "submission_fingerprint": state["submission_fingerprint"],
            "goal_sha256": files["goal"]["sha256"], "artifact_sha256": files["artifact"]["sha256"]}.items():
        if verdict.get(name) != expected_value:
            raise ValueError("browser review file identity changed")
    if (type(verdict.get("score")) is not int or not 0 <= verdict["score"] <= 100
            or type(verdict.get("pass")) is not bool
            or not isinstance(verdict.get("blocking_findings"), list)
            or not isinstance(verdict.get("tests_or_checks_required"), list)):
        raise ValueError("invalid file review score or gates")
    passed = verdict["pass"] and verdict["score"] >= 90 and not verdict["blocking_findings"] and not verdict["tests_or_checks_required"]
    if state["evaluation"]["score"] != verdict["score"] or state["evaluation"]["passed"] is not bool(passed):
        raise ValueError("file review evaluation differs from the browser verdict")
    outputs = record.get("bundle", {}).get("sharedState", {}).get("structuredOutputs") or []
    matching = [item.get("value") for item in outputs if isinstance(item, dict) and item.get("key") == "response.output"]
    if len(matching) != 1 or not isinstance(matching[0], list):
        raise ValueError("missing file review response output")
    chat = "\n".join(part["text"] for item in matching[0] if isinstance(item, dict)
                     and item.get("type") == "message" and item.get("role") == "assistant"
                     for part in item.get("content") or [] if isinstance(part, dict)
                     and part.get("type") == "output_text" and isinstance(part.get("text"), str)).strip()
    if (not chat or chat != state.get("review_text")
            or chat != _bound_text(state["review_chat"], preserve_whitespace=True)):
        raise ValueError("file review chat differs from the response")
    # AuraCall must have returned the review as an artifact, not merely linked it in prose.
    found = False
    artifacts = record.get("bundle", {}).get("sharedState", {}).get("artifacts") or []
    for item in [*matching[0], *artifacts]:
        if not isinstance(item, dict) or not (item.get("type") == "artifact" or item.get("kind") == "file"):
            continue
        metadata = item.get("metadata") or {}
        value = metadata.get("localPath") or metadata.get("path") or item.get("path") or item.get("uri")
        if not isinstance(value, str):
            continue
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme:
            if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
                continue
            value = urllib.request.url2pathname(parsed.path)
        if not value.startswith("/") or Path(value).name != "review-record.md":
            continue
        candidate = _bound_text({"path": value, "sha256": state["review_artifact"]["sha256"]}, preserve_whitespace=True)
        found = found or candidate == raw
    if not found:
        raise ValueError("review file lacks materialized provider artifact evidence")
    return contents["goal"], contents["artifact"], chat


def _guard_row(path: Path, runs_root: Path, key: bytes) -> tuple[Any, ...] | None:
    state = _read_json(path)
    evaluation = state.get("evaluation") or {}
    if (state.get("status") != "completed" or evaluation.get("valid") is not True
            or evaluation.get("nonce_matched") is not True
            or not isinstance(evaluation.get("passed"), bool)
            or not isinstance(evaluation.get("score"), int)
            or not 0 <= evaluation["score"] <= 100):
        return None
    response_id, guard_id = state.get("response_id"), state.get("guard_id")
    if not isinstance(response_id, str) or not re.fullmatch(r"resp_[A-Za-z0-9_]+", response_id):
        return None
    if not isinstance(guard_id, str) or not guard_id:
        return None
    record_path = runs_root / response_id / "record.json"
    if record_path.is_symlink() or record_path.parent.is_symlink():
        return None
    record = _read_json(record_path)
    run = (record.get("bundle") or {}).get("run") or {}
    inputs = run.get("initialInputs") or {}
    metadata = inputs.get("metadata") or {}
    round_number = state.get("round")
    if (record.get("runId") != response_id or run.get("id") != response_id
            or run.get("status") != "succeeded"
            or metadata.get("workflow") != "codex-pro-guard"
            or metadata.get("guard_id") != guard_id
            or metadata.get("guard_nonce") != state.get("nonce")
            or metadata.get("round") != round_number
            or metadata.get("submission_fingerprint") != state.get("submission_fingerprint")
            or not isinstance(round_number, int) or round_number < 1):
        return None
    goal, revision, _chat = _guard_review_texts(state, record)
    if not goal or not revision.strip():
        return None
    submitted_at = state.get("submitted_at")
    if not isinstance(submitted_at, str):
        return None
    datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
    context_digest = _digest(key, goal)
    # A changed PDF/DOCX is a distinct candidate even when its Markdown overview is unchanged.
    revision_identity = revision
    references = (state.get("file_review") or {}).get("references") or []
    if references:
        revision_identity = json.dumps({"candidate": revision,
            "references": [{name: source[name] for name in ("fileName", "mimeType", "size", "sha256")}
                           for source in references]}, sort_keys=True, separators=(",", ":"))
    return (_digest(key, response_id), context_digest, "auracall_pro_guard", submitted_at,
            round_number, "review_goal", context_digest, _digest(key, revision_identity),
            json.dumps(_features(goal, revision, round_number, key), sort_keys=True),
            str(inputs.get("model") or ""), None, None,
            evaluation["score"], int(evaluation["passed"]), "nonce_bound_pro_guard",
            response_id, guard_id, FEATURE_VERSION)


def _browser_result_features(path: Path, runs_root: Path, key: bytes) -> str | None:
    """Extract only keyed features from one nonce-bound assistant result."""
    state = _read_json(path)
    record = _read_json(runs_root / state["response_id"] / "record.json")
    if state.get("review_format") == "file":
        _goal, _artifact, chat = _guard_review_texts(state, record)
        return json.dumps(_features(chat, "", 1, key), sort_keys=True)
    outputs = ((record.get("bundle") or {}).get("sharedState") or {}).get("structuredOutputs")
    if not isinstance(outputs, list):
        return None
    matching = [item.get("value") for item in outputs
                if isinstance(item, dict) and item.get("key") == "response.output"]
    if len(matching) != 1 or not isinstance(matching[0], list) or len(matching[0]) != 1:
        return None
    message = matching[0][0]
    if (not isinstance(message, dict) or message.get("role") != "assistant"
            or not isinstance(message.get("content"), list) or len(message["content"]) != 1):
        return None
    content = message["content"][0]
    if not isinstance(content, dict) or content.get("type") != "output_text":
        return None
    answer = content.get("text")
    if (not isinstance(answer, str) or not answer.strip() or len(answer) > 256_000
            or not isinstance(state.get("nonce"), str) or state["nonce"] not in answer):
        return None
    return json.dumps(_features(answer, "", 1, key), sort_keys=True)


def _bound_text(source: Any, *, preserve_whitespace: bool = False) -> str:
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        raise ValueError("missing bound prompt source")
    path = Path(source["path"])
    if not path.is_absolute() or not isinstance(source.get("sha256"), str):
        raise ValueError("invalid bound prompt source")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_SOURCE_BYTES:
            raise ValueError("prompt source is not a bounded regular file")
        content = handle.read(MAX_SOURCE_BYTES + 1)
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise ValueError("bound prompt source changed")
    value = content.decode("utf-8")
    if not value.strip():
        raise ValueError("bound prompt source is empty")
    return value if preserve_whitespace else value.strip()


def _verdict_feedback(state: dict[str, Any]) -> str:
    verdict = state.get("verdict") or {}
    if not isinstance(verdict, dict):
        raise ValueError("invalid reviewer verdict")
    fragments = [verdict.get("summary")]
    for field in ("blocking_findings", "nonblocking_findings"):
        for finding in verdict.get(field) or []:
            if isinstance(finding, dict):
                fragments.extend((finding.get("issue"), finding.get("required_fix")))
    fragments.extend(verdict.get("tests_or_checks_required") or [])
    fragments.append(verdict.get("suggested_next_prompt"))
    return " ".join(value[:2000] for value in fragments if isinstance(value, str))[:12000]


def _guard_trace(path: Path, runs_root: Path, key: bytes,
                 row: tuple[Any, ...]) -> tuple[Any, ...] | None:
    from smoke_bench import explicit_inline_revision

    state = _read_json(path)
    trace = state.get("learning_trace")
    if trace is None:
        return None
    if not isinstance(trace, dict) or trace.get("prompt_author") not in {"user", "chatgpt", "codex", "mixed"}:
        raise ValueError("invalid learning trace")
    sources = state.get("source_files") or {}
    trace_payload = {"root_guard_id": trace.get("root_guard_id"),
                     "parent_guard_id": trace.get("parent_guard_id"),
                     "parent_response_id": trace.get("parent_response_id"),
                     "prompt_author": trace.get("prompt_author"),
                     "origin_prompt_sha256": (sources.get("origin_prompt") or {}).get("sha256"),
                     "generation_prompt_sha256": (sources.get("generation_prompt") or {}).get("sha256")}
    binding_fields = ("codex_thread_id", "codex_turn_id", "codex_result_sha256")
    if any(field in trace for field in binding_fields):
        if (not all(isinstance(trace.get(field), str) for field in binding_fields)
                or not isinstance(sources.get("codex_result"), dict)
                or sources["codex_result"].get("sha256") != trace["codex_result_sha256"]):
            raise ValueError("incomplete bound Codex result identity")
        uuid = r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
        if (not re.fullmatch(uuid, trace["codex_thread_id"])
                or not re.fullmatch(uuid, trace["codex_turn_id"])):
            raise ValueError("invalid bound Codex turn identity")
        _bound_text(sources["codex_result"], preserve_whitespace=True)
        trace_payload.update({field: trace[field] for field in binding_fields})
    trace_digest = hashlib.sha256(json.dumps(trace_payload, sort_keys=True,
                                              separators=(",", ":")).encode()).hexdigest()
    response_id = state.get("response_id")
    record = _read_json(runs_root / response_id / "record.json")
    metadata = (((record.get("bundle") or {}).get("run") or {}).get("initialInputs") or {}).get("metadata") or {}
    if (state.get("learning_trace_digest") != trace_digest
            or metadata.get("learning_trace_digest") != trace_digest):
        raise ValueError("learning trace is not bound to the AuraCall submission")
    origin = _bound_text(sources.get("origin_prompt"))
    generation = _bound_text(sources.get("generation_prompt"))
    root_guard_id = trace.get("root_guard_id")
    if not isinstance(root_guard_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}", root_guard_id):
        raise ValueError("invalid trace root")
    parent_episode_id = None
    parent_feedback = None
    parent_result_features = None
    parent_quality_score = None
    adopted_parent_suggestion = 0
    adopted_parent_inline_revision = 0
    parent_id = trace.get("parent_guard_id")
    if parent_id is not None:
        if not isinstance(parent_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}", parent_id):
            raise ValueError("invalid parent guard id")
        parent_path = path.parent / f"{parent_id}.json"
        parent = _read_json(parent_path)
        parent_row = _guard_row(parent_path, runs_root, key)
        parent_trace = (_guard_trace(parent_path, runs_root, key, parent_row)
                        if parent_row is not None else None)
        if (parent_row is None or parent.get("response_id") != trace.get("parent_response_id")
                or parent_trace is None
                or parent.get("round") != state.get("round") - 1
                or (parent.get("learning_trace") or {}).get("root_guard_id") != root_guard_id
                or parent_trace[3] != _digest(key, origin)):
            raise ValueError("parent trace identity mismatch")
        parent_episode_id = parent_row[0]
        parent_feedback = json.dumps(_features(_verdict_feedback(parent), "", 1, key), sort_keys=True)
        parent_record = _read_json(runs_root / parent["response_id"] / "record.json")
        _parent_goal, parent_artifact, _parent_chat = _guard_review_texts(parent, parent_record)
        if not isinstance(parent_artifact, str) or not parent_artifact.strip():
            raise ValueError("parent review artifact is missing")
        # The legacy column name says "result", but this is reviewer input,
        # not a verified Codex final answer. Exact answers are linked separately.
        parent_result_features = json.dumps(_features(parent_artifact, "", 1, key), sort_keys=True)
        parent_quality_score = parent_row[12]
        parent_suggestion = (parent.get("verdict") or {}).get("suggested_next_prompt")
        if isinstance(parent_suggestion, str) and parent_suggestion.strip() == generation:
            adopted_parent_suggestion = 1
        parent_inline_revision = explicit_inline_revision(parent)
        if parent_inline_revision is not None and parent_inline_revision == generation:
            adopted_parent_inline_revision = 1
    elif root_guard_id != state.get("guard_id") or state.get("round") != 1:
        raise ValueError("invalid first trace round")
    feedback = _verdict_feedback(state)
    suggestion = (state.get("verdict") or {}).get("suggested_next_prompt")
    if suggestion is not None and (not isinstance(suggestion, str) or len(suggestion) > 12000):
        raise ValueError("invalid browser suggested prompt")
    inline_revision = explicit_inline_revision(state)
    return (row[0], _digest(key, root_guard_id), parent_episode_id,
            _digest(key, origin), _digest(key, generation), trace["prompt_author"],
            json.dumps(_features(origin, generation, state["round"], key), sort_keys=True),
            parent_feedback, json.dumps(_features(feedback, "", 1, key), sort_keys=True),
            row[3], row[13], _digest(key, suggestion) if isinstance(suggestion, str) and suggestion else None,
            adopted_parent_suggestion, parent_result_features, parent_quality_score,
            _digest(key, inline_revision) if inline_revision is not None else None,
            adopted_parent_inline_revision)


def _link_review_to_codex(connection: sqlite3.Connection, episode_id: str,
                          generation_digest: str, artifact_digest: str,
                          binding: tuple[str, str, str] | None = None) -> str:
    """Link only a completed, exact prompt/result pair; a packet may contain more."""
    if binding is None:
        live = connection.execute("""SELECT thread_key,turn_key FROM codex_turns
                                     WHERE prompt_digest=? AND result_digest IS NOT NULL
                                     AND result_digest=?""",
                                  (generation_digest, artifact_digest)).fetchall()
        completed_sessions = connection.execute("""SELECT thread_key,turn_key FROM session_turns
                                                   WHERE prompt_digest=? AND result_digest=?
                                                   AND input_scope='single_pre_inference_message'""",
                                                (generation_digest, artifact_digest)).fetchall()
    else:
        thread_key, turn_key, result_digest = binding
        live = connection.execute("""SELECT thread_key,turn_key FROM codex_turns
                                     WHERE thread_key=? AND turn_key=? AND prompt_digest=?
                                     AND result_digest=?""",
                                  (thread_key, turn_key, generation_digest, result_digest)).fetchall()
        completed_sessions = connection.execute("""SELECT thread_key,turn_key FROM session_turns
                                                   WHERE thread_key=? AND turn_key=? AND prompt_digest=?
                                                   AND result_digest=? AND input_scope='single_pre_inference_message'""",
                                                (thread_key, turn_key, generation_digest, result_digest)).fetchall()
    identities = set(live) | set(completed_sessions)
    existing = connection.execute("""SELECT thread_key,turn_key,link_status FROM review_codex_links
                                     WHERE episode_id=?""", (episode_id,)).fetchone()
    if not identities:
        if existing and existing[2] != "no_match":
            connection.execute("""UPDATE review_codex_links SET link_status='no_match'
                                  WHERE episode_id=?""", (episode_id,))
        return "no_match"
    if len(identities) != 1:
        if existing and existing[2] != "ambiguous":
            connection.execute("""UPDATE review_codex_links SET link_status='ambiguous'
                                  WHERE episode_id=?""", (episode_id,))
        return "ambiguous"
    identity = next(iter(identities))
    if existing:
        if existing[:2] != identity:
            return "conflict"
        if existing[2] != "valid":
            connection.execute("""UPDATE review_codex_links SET link_status='valid'
                                  WHERE episode_id=?""", (episode_id,))
        return "unchanged"
    source_kind = "managed" if identity in live else "completed_session"
    connection.execute("""INSERT INTO review_codex_links
                         (episode_id,thread_key,turn_key,source_kind,linked_at)
                         VALUES (?,?,?,?,?)""",
                       (episode_id, *identity, source_kind, datetime.now(timezone.utc).isoformat()))
    return "imported"


def import_auracall(guard_root: Path | None = None, runs_root: Path | None = None,
                    guard_id: str | None = None) -> dict[str, int]:
    guard_root = guard_root or Path.home() / ".auracall/pro-guard/runs"
    runs_root = runs_root or Path.home() / ".auracall/runtime/runs"
    if guard_root.is_symlink() or runs_root.is_symlink():
        raise ValueError("AuraCall source roots may not be symlinks")
    if guard_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}", guard_id):
        raise ValueError("invalid guard id")
    key = _key()
    counts = Counter()
    paths = [guard_root / f"{guard_id}.json"] if guard_id else sorted(guard_root.glob("*.json"))
    with _connect() as connection:
        for path in paths:
            try:
                row = _guard_row(path, runs_root, key)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                counts["rejected"] += 1
                continue
            if row is None:
                counts["ineligible"] += 1
                continue
            existing = connection.execute("SELECT * FROM episodes WHERE episode_id=?", (row[0],)).fetchone()
            if existing:
                # Every source identity and label is immutable once admitted.
                # A matching feature vector and grade do not prove that the
                # reviewed prompt, result, response, or provenance is unchanged.
                if existing != row:
                    counts["conflict"] += 1
                    continue
                else:
                    counts["unchanged"] += 1
            else:
                connection.execute("INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
                counts["imported"] += 1
            try:
                browser_features = _browser_result_features(path, runs_root, key)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                browser_features = None
            if browser_features is not None:
                existing_browser = connection.execute(
                    "SELECT result_features FROM browser_results WHERE episode_id=?", (row[0],)).fetchone()
                if existing_browser is None:
                    connection.execute("INSERT INTO browser_results VALUES (?,?)", (row[0], browser_features))
                    counts["browser_result_imported"] += 1
                elif existing_browser[0] != browser_features:
                    counts["browser_result_conflict"] += 1
                    continue
            try:
                trace = _guard_trace(path, runs_root, key, row)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                counts["trace_rejected"] += 1
                continue
            if trace is None:
                counts["trace_missing"] += 1
                continue
            if trace is not None:
                existing_trace = connection.execute(
                    "SELECT root_key,parent_episode_id,origin_prompt_digest,generation_prompt_digest,prompt_author,prompt_features,parent_feedback_features,result_feedback_features,submitted_at,passed,suggested_prompt_digest,adopted_parent_suggestion,parent_result_features,parent_quality_score,explicit_inline_revision_digest,adopted_parent_inline_revision FROM iteration_traces WHERE episode_id=?",
                    (trace[0],)).fetchone()
                if existing_trace:
                    old_parent = existing_trace[12:14]
                    new_parent = trace[13:15]
                    old_inline_digest, old_adopted_inline = existing_trace[14:16]
                    new_inline_digest, new_adopted_inline = trace[15:17]
                    if (existing_trace[:12] != trace[1:13]
                            or any(old is not None and old != new for old, new in zip(old_parent, new_parent))
                            or (old_inline_digest is not None and old_inline_digest != new_inline_digest)
                            or (old_adopted_inline not in (0, new_adopted_inline))):
                        counts["trace_conflict"] += 1
                        continue
                    if existing_trace[12:] != trace[13:]:
                        connection.execute("""UPDATE iteration_traces
                                           SET parent_result_features=?,parent_quality_score=?,
                                           explicit_inline_revision_digest=?,adopted_parent_inline_revision=?
                                           WHERE episode_id=?""", (*trace[13:], trace[0]))
                    counts["trace_unchanged"] += 1
                else:
                    connection.execute("INSERT INTO iteration_traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", trace)
                    counts["trace_imported"] += 1
                state = _read_json(path)
                bound = state.get("learning_trace") or {}
                binding = None
                if bound.get("codex_thread_id"):
                    result = _bound_text(state["source_files"]["codex_result"], preserve_whitespace=True)
                    binding = (_digest(key, bound["codex_thread_id"]),
                               _digest(key, bound["codex_turn_id"]), _digest(key, result))
                link_result = _link_review_to_codex(connection, row[0], trace[4], row[7], binding)
                counts[f"codex_link_{link_result}"] += 1
                if link_result == "conflict":
                    counts["conflict"] += 1
    return {name: counts[name] for name in ("imported", "unchanged", "ineligible", "rejected", "conflict",
                                            "browser_result_imported", "browser_result_conflict",
                                            "trace_imported", "trace_unchanged", "trace_missing",
                                            "trace_rejected", "trace_conflict",
                                            "codex_link_imported", "codex_link_unchanged",
                                            "codex_link_no_match", "codex_link_ambiguous", "codex_link_conflict")}


def _bound_browser_suggestion_supplied(trace: dict[str, Any], number: int, prompt: str,
                                       previous: dict[str, Any] | None,
                                       previous_adapter: dict[str, Any] | None) -> int:
    fields = ("parent_browser_suggestion_sha256", "parent_browser_suggestion_supplied")
    if not any(field in trace for field in fields):
        return 0  # An older trace cannot gain provenance after submission.
    if not all(field in trace for field in fields):
        raise ValueError("loop suggestion provenance is incomplete")
    if number == 1:
        expected_digest, expected_supplied = None, False
    else:
        parent_verdict = (previous.get("review") or {}).get("verdict") if previous else None
        if (not isinstance(parent_verdict, dict) or previous_adapter is None
                or previous_adapter.get("verdict") != parent_verdict):
            raise ValueError("loop parent reviewer verdict differs")
        suggestion = parent_verdict.get("suggested_next_prompt")
        suggestion = suggestion.strip() if isinstance(suggestion, str) else ""
        expected_digest = hashlib.sha256(suggestion.encode()).hexdigest() if suggestion else None
        expected_supplied = bool(suggestion) and json.dumps(
            suggestion[:4000], ensure_ascii=False) in prompt
    if (trace.get("parent_browser_suggestion_sha256") != expected_digest
            or trace.get("parent_browser_suggestion_supplied") is not expected_supplied):
        raise ValueError("loop browser suggestion was not supplied as claimed")
    return int(expected_supplied)


def _bound_loop_row(loop_path: Path, round_record: dict[str, Any],
                    runs_root: Path, key: bytes) -> tuple[Any, ...] | None:
    """Verify one legacy-loop review without treating attachment hashes as browser proof."""
    state = _read_json(loop_path)
    config = state.get("config") or {}
    if (not isinstance(config, dict) or not isinstance(state.get("rounds"), list)
            or hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False).encode()).hexdigest() != state.get("config_digest")):
        raise ValueError("loop configuration is not digest-bound")
    number = round_record.get("round")
    goal, prompt = config.get("goal"), round_record.get("iteration_prompt")
    if (not isinstance(number, int) or number < 1 or not isinstance(goal, str)
            or not goal.strip() or not isinstance(prompt, str) or not prompt.strip()
            or len(goal) > 256_000 or len(prompt) > 256_000):
        raise ValueError("invalid loop prompt lineage")
    review = round_record.get("review") or {}
    if review.get("status") != "completed" or review.get("learning_trace_status") != "bound_at_submission":
        return None
    verification = round_record.get("verification") or {}
    if verification.get("status") != "passed" or verification.get("returncode") != 0:
        raise ValueError("loop review lacks passed local verification")
    round_dir = loop_path.parent / f"round-{number}"
    trace_path = round_dir / "learning-trace.json"
    adapter_path = round_dir / "auracall-pro-guard-state.json"
    if (review.get("learning_trace_file") != str(trace_path)
            or review.get("adapter_state_path") != str(adapter_path)
            or trace_path.is_symlink() or adapter_path.is_symlink()):
        raise ValueError("loop review paths are not the bound round paths")
    trace_bytes = trace_path.read_bytes()
    if len(trace_bytes) > 65536:
        raise ValueError("loop trace is too large")
    trace = json.loads(trace_bytes)
    adapter = _read_json(adapter_path)
    material = adapter.get("request_material") or {}
    trace_digest = hashlib.sha256(trace_bytes).hexdigest()
    work = round_record.get("work") or {}
    final = work.get("final_message") if work.get("status") == "completed" else None
    final_digest = (hashlib.sha256(final.encode()).hexdigest()
                    if isinstance(final, str) and final.strip() else None)
    manifest = material.get("files")
    if (not isinstance(trace, dict) or trace.get("schema") != "modellabs.auracall_loop_trace.v1"
            or trace.get("run_id") != state.get("run_id")
            or trace.get("round") != number
            or trace.get("origin_prompt_source") != "loop_goal_argument"
            or trace.get("prompt_author") != "mixed"
            or trace.get("origin_prompt_sha256") != hashlib.sha256(goal.encode()).hexdigest()
            or trace.get("generation_prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest()
            or trace.get("codex_result_sha256") != final_digest
            or trace.get("codex_thread_id") != work.get("thread_id")
            or not isinstance(manifest, list)
            or trace.get("attachment_manifest_sha256") != hashlib.sha256(json.dumps(
                manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            or trace.get("review_prompt_sha256") != hashlib.sha256(
                str(material.get("prompt") or "").encode()).hexdigest()
            or material.get("learning_trace_digest") != trace_digest
            or material.get("round") != number
            or material.get("schema") != "guard"
            or adapter.get("schema") != "codex.auracall_pro_guard_review.v1"
            or adapter.get("status") != "completed"
            or adapter.get("correction_response_id") is not None):
        raise ValueError("loop trace does not match the submitted review")
    response_id = adapter.get("active_response_id") or adapter.get("response_id")
    if (not isinstance(response_id, str) or not re.fullmatch(r"resp_[A-Za-z0-9_]+", response_id)
            or adapter.get("response_id") != response_id
            or (adapter.get("response_read") or {}).get("id") != response_id):
        raise ValueError("loop response identity is ambiguous")
    nonce = adapter.get("nonce")
    if (not isinstance(nonce, str) or not re.fullmatch(r"codex-pro-guard-[A-Za-z0-9_-]+", nonce)
            or not isinstance(material.get("prompt"), str) or not material["prompt"].strip()):
        raise ValueError("loop nonce or review prompt is invalid")
    expected_input = (f"FRESHNESS_NONCE: {nonce}\n"
                      f"Your JSON object must include exactly this nonce in its nonce field.\n\n"
                      f"{material['prompt'].strip()}")
    record_path = runs_root / response_id / "record.json"
    if record_path.is_symlink() or record_path.parent.is_symlink():
        raise ValueError("loop response record path is a symlink")
    record = _read_json(record_path)
    run = (record.get("bundle") or {}).get("run") or {}
    inputs = run.get("initialInputs") or {}
    metadata = inputs.get("metadata") or {}
    target = inputs.get("auracall") or {}
    if (record.get("runId") != response_id or run.get("id") != response_id
            or run.get("status") != "succeeded"
            or metadata.get("workflow") != "codex-pro-guard"
            or metadata.get("schema") != "guard"
            or metadata.get("nonce") != nonce
            or metadata.get("round") != number
            or metadata.get("client_request_id") != adapter.get("client_request_id")
            or metadata.get("learning_trace_digest") != trace_digest
            or inputs.get("requestInput") != expected_input
            or inputs.get("model") != material.get("model")
            or target.get("runtimeProfile") != material.get("runtime_profile")
            or target.get("service") != "chatgpt"
            or target.get("chatgptConversationUrl") != material.get("conversation_url")
            or adapter.get("conversation_url") != material.get("conversation_url")):
        raise ValueError("AuraCall record does not match the loop submission")
    expected_attachments = []
    for ordinal, item in enumerate(manifest, start=1):
        if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
                or not Path(item["path"]).is_absolute()
                or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256")))):
            raise ValueError("loop attachment manifest is malformed")
        path = Path(item["path"])
        expected_attachments.append((f"codex-pro-guard-{ordinal}", path.name, path.as_uri()))
    requested_attachments = inputs.get("attachments") or []
    if (not isinstance(requested_attachments, list)
            or [(item.get("id"), item.get("fileName"), item.get("uri"))
                for item in requested_attachments if isinstance(item, dict)] != expected_attachments
            or len(requested_attachments) != len(expected_attachments)):
        raise ValueError("AuraCall request attachments differ from the bound manifest")
    steps = (record.get("bundle") or {}).get("steps") or []
    browser_runs = [(step.get("output") or {}).get("structuredData", {}).get("browserRun")
                    for step in steps if isinstance(step, dict) and step.get("status") == "succeeded"
                    and isinstance((step.get("output") or {}).get("structuredData"), dict)
                    and isinstance(((step.get("output") or {}).get("structuredData") or {}).get("browserRun"), dict)]
    if (len(browser_runs) != 1 or browser_runs[0].get("service") != "chatgpt"
            or browser_runs[0].get("runtimeProfileId") != material.get("runtime_profile")
            or browser_runs[0].get("tabUrl") != material.get("conversation_url")):
        raise ValueError("AuraCall record lacks the exact succeeded browser run")
    transport = browser_runs[0].get("promptTransport") or {}
    transported = transport.get("attachments") or []
    direct_paths = [item.get("path") for item in transported
                    if isinstance(item, dict)] if isinstance(transported, list) else []
    dispatch_bound = int(
        len(direct_paths) == len(transported)
        and direct_paths[:len(manifest)] == [item["path"] for item in manifest]
        and (len(direct_paths) == len(manifest)
             or (len(direct_paths) == len(manifest) + 1
                 and transport.get("metadata", {}).get("mode") == "request_attachment"
                 and transported[-1].get("displayPath") == "auracall-request.txt")))
    ui_receipt = browser_runs[0].get("attachmentUiReceipt")
    ui_confirmed = int(
        dispatch_bound == 1
        and isinstance(ui_receipt, dict)
        and ui_receipt.get("schema") == "auracall.browser_attachment_ui_receipt.v1"
        and ui_receipt.get("attachmentPaths") == direct_paths
        and ui_receipt.get("uploadCompletion") == "confirmed"
        and ui_receipt.get("sentUserTurnAttachments") == "confirmed"
        and isinstance(ui_receipt.get("submittedUserId"), str)
        and bool(ui_receipt["submittedUserId"].strip()))
    outputs = ((record.get("bundle") or {}).get("sharedState") or {}).get("structuredOutputs")
    matches = [item.get("value") for item in outputs or []
               if isinstance(item, dict) and item.get("key") == "response.output"]
    if len(matches) != 1 or not isinstance(matches[0], list) or len(matches[0]) != 1:
        raise ValueError("loop browser output is not one assistant message")
    message = matches[0][0]
    content = message.get("content") if isinstance(message, dict) else None
    if (not isinstance(message, dict) or message.get("role") != "assistant"
            or not isinstance(content, list) or len(content) != 1
            or content[0].get("type") != "output_text"
            or not isinstance(content[0].get("text"), str)):
        raise ValueError("loop browser output is malformed")
    answer = content[0]["text"]
    verdict = json.loads(answer)
    if (not isinstance(verdict, dict) or verdict != adapter.get("verdict")
            or verdict != review.get("verdict") or verdict.get("nonce") != nonce
            or type(verdict.get("score")) is not int or not 0 <= verdict["score"] <= 100
            or not isinstance(verdict.get("pass"), bool)):
        raise ValueError("loop verdict is not nonce-bound to the browser output")
    parent_response = trace.get("parent_response_id")
    previous = None
    previous_adapter = None
    if number == 1 and parent_response is not None:
        raise ValueError("first loop round has a parent response")
    if number > 1:
        previous = next((item for item in state["rounds"] if item.get("round") == number - 1), None)
        if previous is None:
            raise ValueError("loop parent round is missing")
        previous_adapter_path = (previous.get("review") or {}).get("adapter_state_path")
        if previous_adapter_path != str(loop_path.parent / f"round-{number - 1}" /
                                        "auracall-pro-guard-state.json"):
            raise ValueError("loop parent review is missing")
        previous_adapter = _read_json(Path(previous_adapter_path))
        if parent_response != (previous_adapter.get("active_response_id")
                               or previous_adapter.get("correction_response_id")
                               or previous_adapter.get("response_id")):
            raise ValueError("loop parent response identity differs")
    suggestion_supplied = _bound_browser_suggestion_supplied(
        trace, number, prompt, previous, previous_adapter)
    submitted_at = adapter.get("created_at")
    if not isinstance(submitted_at, str):
        raise ValueError("loop submission timestamp is missing")
    _utc_timestamp(submitted_at)
    passed = int(verdict["pass"] and verdict["score"] >= 90
                 and not verdict.get("blocking_findings")
                 and not verdict.get("tests_or_checks_required"))
    return (_digest(key, response_id), _digest(key, str(state["run_id"])), number,
            submitted_at, _digest(key, goal), _digest(key, prompt),
            json.dumps(_features(goal, prompt, number, key), sort_keys=True),
            json.dumps(_features(final, "", 1, key), sort_keys=True) if final_digest else None,
            json.dumps(_features(answer, "", 1, key), sort_keys=True),
            json.dumps(_features(_verdict_feedback({"verdict": verdict}), "", 1, key), sort_keys=True),
            _digest(key, parent_response) if parent_response else None,
            verdict["score"], passed, response_id, trace_digest,
            str(material.get("model") or ""), 0, dispatch_bound, ui_confirmed,
            suggestion_supplied)


def import_bound_loops(loop_root: Path | None = None,
                       runs_root: Path | None = None) -> dict[str, int]:
    """Import only submission-bound legacy-loop grades into a separate store."""
    loop_root = loop_root or Path(os.environ.get(
        "CODEX_RESEARCH_HOME", Path.home() / "codex-research")) / "runs/codex-auracall-pro-guard"
    runs_root = runs_root or Path.home() / ".auracall/runtime/runs"
    if loop_root.is_symlink() or runs_root.is_symlink():
        raise ValueError("loop source roots may not be symlinks")
    counts = Counter()
    key = _key()
    with _connect() as connection:
        for loop_path in sorted(loop_root.glob("*/loop.json")):
            try:
                if loop_path.is_symlink() or loop_path.parent.is_symlink():
                    raise ValueError("loop state path is a symlink")
                state = _read_json(loop_path)
                rounds = state.get("rounds")
                if not isinstance(rounds, list):
                    raise ValueError("loop rounds are missing")
                if any(not isinstance(item, dict) or not isinstance(item.get("round"), int)
                       for item in rounds):
                    raise ValueError("loop round numbers are malformed")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                counts["rejected"] += 1
                continue
            for round_record in sorted(rounds, key=lambda item: item.get("round", 0)):
                try:
                    row = _bound_loop_row(loop_path, round_record, runs_root, key)
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, AttributeError):
                    counts["rejected"] += 1
                    continue
                if row is None:
                    if (round_record.get("review") or {}).get("status") == "external_blocker":
                        counts["external_blocker_excluded"] += 1
                    else:
                        counts["ineligible"] += 1
                    continue
                if row[10] is not None:
                    parent = connection.execute("""SELECT root_key,round_number FROM bound_loop_rounds
                                                 WHERE episode_id=?""", (row[10],)).fetchone()
                    if parent != (row[1], row[2] - 1):
                        counts["parent_unavailable"] += 1
                        continue
                existing = connection.execute("SELECT * FROM bound_loop_rounds WHERE episode_id=?",
                                              (row[0],)).fetchone()
                if existing is not None:
                    counts["unchanged" if existing == row else "conflict"] += 1
                    continue
                connection.execute("INSERT INTO bound_loop_rounds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
                counts["imported"] += 1
    return {name: counts[name] for name in ("imported", "unchanged", "ineligible", "rejected",
                                            "external_blocker_excluded", "parent_unavailable", "conflict")}


def _sigmoid(value: float) -> float:
    value = min(max(value, -30), 30)
    return 1.0 / (1.0 + math.exp(-value))


def _fit(rows: list[tuple[dict[str, float], int]]) -> tuple[list[float], float]:
    feature_slots = max(2 * (FEATURE_COUNT + 3) + 4, 1 + max(int(index) for vector, _label in rows
                                                   for index in vector))
    weights = [0.0] * feature_slots
    positives = sum(label for _features_row, label in rows)
    bias = math.log((positives + 1) / (len(rows) - positives + 1))
    for epoch in range(180):
        rate = 0.25 / (1.0 + epoch / 35.0)
        for vector, label in rows:
            probability = _sigmoid(bias + sum(weights[int(index)] * value for index, value in vector.items()))
            error = label - probability
            bias += rate * error / len(rows)
            for index, value in vector.items():
                slot = int(index)
                weights[slot] += rate * (error * value - 0.08 * weights[slot]) / len(rows)
    return weights, bias


def _brier(rows: list[tuple[dict[str, float], int]], weights: list[float], bias: float) -> float:
    return sum((_sigmoid(bias + sum(weights[int(index)] * value for index, value in vector.items())) - label) ** 2
               for vector, label in rows) / len(rows)


def _fit_score(rows: list[tuple[dict[str, float], float]]) -> tuple[list[float], float]:
    """Fit a bounded-score predictor; labels are review scores divided by 100."""
    slots = 1 + max(int(index) for vector, _label in rows for index in vector)
    weights = [0.0] * slots
    bias = sum(label for _vector, label in rows) / len(rows)
    for epoch in range(180):
        rate = 0.2 / (1.0 + epoch / 35.0)
        for vector, label in rows:
            estimate = bias + sum(weights[int(index)] * value for index, value in vector.items())
            error = label - estimate
            bias += rate * error / len(rows)
            for index, value in vector.items():
                slot = int(index)
                weights[slot] += rate * (error * value - 0.08 * weights[slot]) / len(rows)
    return weights, bias


def _score_prediction(vector: dict[str, float], weights: list[float], bias: float) -> float:
    return min(1.0, max(0.0, bias + sum(weights[int(index)] * value
                                        for index, value in vector.items()
                                        if int(index) < len(weights))))


def _score_mae(rows: list[tuple[dict[str, float], float]], weights: list[float], bias: float) -> float:
    return sum(abs(_score_prediction(vector, weights, bias) - label)
               for vector, label in rows) / len(rows)


def _utc_timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp is missing timezone")
    return parsed.timestamp()


def _evaluation_checkpoint(path: Path, schema: str, groups: list[str],
                           grouped: dict[str, list[tuple[str, dict[str, float], int]]],
                           extra: dict[str, Any] | None = None,
                           feature_version: int = FEATURE_VERSION) -> dict[str, Any]:
    """Freeze one predictor before future independent task groups arrive."""
    if path.exists():
        checkpoint = _read_json(path)
        if (checkpoint.get("schema") != schema
                or checkpoint.get("feature_version") != feature_version
                or not isinstance(checkpoint.get("development_groups"), list)
                or not isinstance(checkpoint.get("weights"), list)
                or not isinstance(checkpoint.get("bias"), (int, float))
                or not isinstance(checkpoint.get("baseline_rate"), (int, float))):
            raise ValueError("review evaluation checkpoint is incompatible")
        if extra and any(name not in checkpoint for name in extra):
            raise ValueError("evaluation checkpoint lacks eligibility metadata")
        return checkpoint
    training = [(vector, label) for group in groups for _at, vector, label in grouped[group]]
    weights, bias = _fit(training)
    checkpoint = {
        "schema": schema,
        "feature_version": feature_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_groups": groups,
        "development_episodes": len(training),
        "baseline_rate": sum(label for _vector, label in training) / len(training),
        "weights": weights,
        "bias": bias,
    }
    if extra:
        checkpoint.update(extra)
    _private_root()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        return _evaluation_checkpoint(path, schema, groups, grouped, extra, feature_version)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return checkpoint


def _prospective_rows(groups: list[str], grouped: dict[str, list[tuple[str, dict[str, float], int]]],
                      checkpoint: dict[str, Any]) -> tuple[list[str], list[tuple[dict[str, float], int]]]:
    development_groups = set(checkpoint["development_groups"])
    selected = []
    cutoff = _utc_timestamp(checkpoint["created_at"])
    for group in groups:
        if group in development_groups:
            continue
        try:
            first_submitted = min(_utc_timestamp(row[0]) for row in grouped[group])
        except (TypeError, ValueError, OverflowError):
            continue
        if first_submitted > cutoff:
            selected.append(group)
    return selected, [(vector, label) for group in selected
                      for _at, vector, label in grouped[group]]


def _codex_features(prompt_features: dict[str, float], model: str,
                    effort: str, task_class: str) -> dict[str, float]:
    if model not in MODEL_IDS or effort not in EFFORTS or task_class not in TASK_CLASSES:
        raise ValueError("unknown Codex model, effort, or task class")
    vector = dict(prompt_features)
    vector[str(FEATURE_COUNT + 3 + MODEL_IDS.index(model))] = 1.0
    vector[str(FEATURE_COUNT + 3 + len(MODEL_IDS) + EFFORTS.index(effort))] = 1.0
    vector[str(FEATURE_COUNT + 3 + len(MODEL_IDS) + len(EFFORTS) + TASK_CLASSES.index(task_class))] = 1.0
    return vector


def train_codex_model() -> dict[str, Any]:
    """Fit only within supported arms; observational predictions stay shadow-only."""
    sync_codex_grades()
    sync_session_grades()
    with _connect() as connection:
        raw = connection.execute("""SELECT group_id,accepted_at_ms,features,selected_model,
                                   effort,task_class,quality_score,verification,total_tokens,
                                   observed_model,observed_effort,model_provenance
                                   FROM codex_turns WHERE quality_score IS NOT NULL
                                   AND verification IS NOT NULL AND total_tokens IS NOT NULL
                                   AND result_digest IS NOT NULL
                                   ORDER BY accepted_at_ms,turn_key""").fetchall()
        live_identities = set(connection.execute("""SELECT thread_key,turn_key FROM codex_turns
                                                   WHERE quality_score IS NOT NULL
                                                   AND verification IS NOT NULL
                                                   AND total_tokens IS NOT NULL
                                                   AND result_digest IS NOT NULL""").fetchall())
        historical = connection.execute("""SELECT thread_key,turn_key,completed_at,prompt_features,
                                         model,effort,task_class,quality_score,verification,exact_total_tokens,
                                         observed_model,observed_effort,model_provenance
                                         FROM session_turns WHERE grade_status='explicit'
                                         AND input_scope='single_pre_inference_message'
                                         AND quality_score IS NOT NULL AND verification IS NOT NULL
                                         AND exact_total_tokens IS NOT NULL""").fetchall()
    for (thread_key, turn_key, completed_at, features, model, effort, task_class,
         score, verification, tokens, observed_model, observed_effort, provenance) in historical:
        if (thread_key, turn_key) in live_identities:
            continue
        try:
            observed_at_ms = int(datetime.fromisoformat(completed_at.replace("Z", "+00:00")).timestamp() * 1000)
        except (ValueError, OverflowError):
            continue
        raw.append((thread_key, observed_at_ms, features, model, effort, task_class,
                    score, verification, tokens, observed_model, observed_effort, provenance))
    graded_with_usage = len(raw)
    first_any_group_ms = {group: min(row[1] for row in raw if row[0] == group)
                          for group in {row[0] for row in raw}}
    # A selected model and exact token count do not establish which model
    # executed inference. Only independently observed, single-arm turns train
    # the model/effort comparison.
    raw = [(group, at, features, observed_model, observed_effort, task_class,
            score, verification, tokens)
           for (group, at, features, _selected_model, _selected_effort,
                task_class, score, verification, tokens, observed_model,
                observed_effort, provenance) in raw
           if provenance == "observed_per_request" and observed_model in MODEL_IDS
           and observed_effort in EFFORTS and task_class in TASK_CLASSES]
    arms = Counter((row[5], row[3], row[4]) for row in raw)
    supported = {arm for arm, count in arms.items() if count >= 10}
    comparable_classes = {task_class for task_class in TASK_CLASSES
                          if sum(arm[0] == task_class for arm in supported) >= 2}
    selected = [row for row in raw if (row[5], row[3], row[4]) in supported
                and row[5] in comparable_classes]
    groups = sorted({row[0] for row in selected},
                    key=lambda group: min(row[1] for row in selected if row[0] == group))
    if len(selected) < 40 or len(groups) < 8:
        return {"status": "insufficient_comparable_outcomes", "graded_with_usage": graded_with_usage,
                "verified_model_effort_outcomes": len(raw),
                "supported_arms": len(supported), "comparable_task_classes": len(comparable_classes)}
    mapped = [(row[0], row[1], _codex_features(json.loads(row[2]), row[3], row[4], row[5]),
               int(row[7] == "passed" and row[6] >= 90)) for row in selected]
    holdout_count = max(2, math.ceil(len(groups) * 0.2))
    train_groups, test_groups = set(groups[:-holdout_count]), set(groups[-holdout_count:])
    training = [(vector, label) for group, _at, vector, label in mapped if group in train_groups]
    test = [(vector, label) for group, _at, vector, label in mapped if group in test_groups]
    if len({label for _vector, label in training}) < 2 or len(test) < 8:
        return {"status": "insufficient_holdout_diversity", "graded_with_usage": graded_with_usage,
                "verified_model_effort_outcomes": len(raw)}
    weights, bias = _fit(training)
    baseline_rate = sum(label for _vector, label in training) / len(training)
    baseline_brier = sum((baseline_rate - label) ** 2 for _vector, label in test) / len(test)
    model_brier = _brier(test, weights, bias)
    checkpoint_groups: dict[str, list[tuple[str, dict[str, float], int]]] = {}
    for group, at_ms, vector, label in mapped:
        observed_at = datetime.fromtimestamp(at_ms / 1000, timezone.utc).isoformat()
        checkpoint_groups.setdefault(group, []).append((observed_at, vector, label))
    eligible_arms = sorted(f"{task_class}|{model}|{effort}" for task_class, model, effort in supported
                           if task_class in comparable_classes)
    checkpoint = _evaluation_checkpoint(CODEX_EVAL_PATH,
                                        "modellabs.codex_evaluation_checkpoint.v1", groups,
                                        checkpoint_groups, {"eligible_arms": eligible_arms})
    frozen_arms = set(checkpoint["eligible_arms"])
    prospective_candidates = [row for row in selected
                              if f"{row[5]}|{row[3]}|{row[4]}" in frozen_arms]
    candidate_groups: dict[str, list[tuple[str, dict[str, float], int]]] = {}
    for row in prospective_candidates:
        observed_at = datetime.fromtimestamp(row[1] / 1000, timezone.utc).isoformat()
        vector = _codex_features(json.loads(row[2]), row[3], row[4], row[5])
        label = int(row[7] == "passed" and row[6] >= 90)
        candidate_groups.setdefault(row[0], []).append((observed_at, vector, label))
    future_groups, _ignored = _prospective_rows(list(candidate_groups), candidate_groups, checkpoint)
    cutoff_ms = _utc_timestamp(checkpoint["created_at"]) * 1000
    future_groups = [group for group in future_groups
                     if first_any_group_ms[group] > cutoff_ms]
    prospective = [(vector, label) for group in future_groups
                   for _at, vector, label in candidate_groups[group]]
    prospective_arms = Counter((row[5], row[3], row[4]) for row in prospective_candidates
                               if row[0] in future_groups)
    comparable_prospective = any(
        sum(arm[0] == task_class and count >= 5 for arm, count in prospective_arms.items()) >= 2
        for task_class in TASK_CLASSES)
    prospective_baseline = (sum((checkpoint["baseline_rate"] - label) ** 2
                                for _vector, label in prospective) / len(prospective)
                            if prospective else None)
    prospective_model = (_brier(prospective, checkpoint["weights"], checkpoint["bias"])
                         if prospective else None)
    gain = (prospective_baseline - prospective_model) if prospective else 0.0
    validated = (len(prospective) >= MIN_REVIEW_HOLDOUT
                 and len(future_groups) >= 10 and comparable_prospective
                 and gain >= MIN_REVIEW_BRIER_GAIN
                 and gain / max(prospective_baseline or 0.0, 1e-9) >= MIN_REVIEW_RELATIVE_GAIN)
    weights, bias = _fit([(vector, label) for _group, _at, vector, label in mapped])
    arm_rows: dict[str, dict[str, Any]] = {}
    for task_class, model, effort in sorted(supported):
        if task_class not in comparable_classes:
            continue
        matches = [row for row in selected if (row[5], row[3], row[4]) == (task_class, model, effort)]
        arm_rows[f"{task_class}|{model}|{effort}"] = {
            "samples": len(matches), "median_turn_tokens": int(median(row[8] for row in matches))}
    artifact = {"schema": "modellabs.codex_outcome_model.v1", "feature_version": FEATURE_VERSION,
                "graded_with_usage": graded_with_usage, "verified_model_effort_outcomes": len(raw),
                "trained_turns": len(selected), "task_groups": len(groups), "holdout_turns": len(test),
                "baseline_brier": round(baseline_brier, 6), "model_brier": round(model_brier, 6),
                "retrospective_holdout_is_diagnostic_only": True,
                "prospective_checkpoint_created_at": checkpoint["created_at"],
                "prospective_holdout_turns": len(prospective),
                "prospective_holdout_task_groups": len(future_groups),
                "prospective_comparable_arms": comparable_prospective,
                "prospective_baseline_brier": round(prospective_baseline, 6) if prospective else None,
                "prospective_model_brier": round(prospective_model, 6) if prospective else None,
                "validated_for_shadow": validated,
                "causal_model_comparison": False, "supported_arms": arm_rows,
                "weights": weights, "bias": bias}
    temporary = CODEX_MODEL_PATH.with_name(f".{CODEX_MODEL_PATH.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        json.dump(artifact, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, CODEX_MODEL_PATH)
    return {key: value for key, value in artifact.items() if key not in {"weights", "bias"}}


def predict_codex(prompt: str, task_class: str) -> dict[str, Any]:
    model = _read_json(CODEX_MODEL_PATH)
    if model.get("schema") != "modellabs.codex_outcome_model.v1" or model.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Codex model version mismatch")
    vector = json.loads(prepare_codex_prompt(prompt)["features"])
    predictions = []
    for arm, evidence in model["supported_arms"].items():
        arm_class, model_id, effort = arm.split("|")
        if arm_class != task_class:
            continue
        features = _codex_features(vector, model_id, effort, task_class)
        probability = _sigmoid(model["bias"] + sum(model["weights"][int(index)] * value
                                                       for index, value in features.items()))
        predictions.append({"model": model_id, "effort": effort,
                            "observed_arm_samples": evidence["samples"],
                            "predicted_verified_pass_probability": round(probability, 4),
                            "observed_median_turn_tokens": evidence["median_turn_tokens"]})
    return {"validated_for_shadow": model["validated_for_shadow"],
            "causal_model_comparison": False, "predictions": predictions}


def _group_review_episodes() -> dict[str, list[tuple[str, dict[str, float], int, int]]]:
    """Keep pass and numeric score models on the same task-group split."""
    with _connect() as connection:
        raw = connection.execute("""SELECT e.group_id,t.root_key,e.submitted_at,e.features,e.passed,e.quality_score
                                    FROM episodes e LEFT JOIN iteration_traces t
                                    ON t.episode_id=e.episode_id
                                    WHERE e.source='auracall_pro_guard'
                                    AND e.verifier='nonce_bound_pro_guard'
                                    AND e.quality_score BETWEEN 0 AND 100
                                    ORDER BY e.submitted_at,e.episode_id""").fetchall()
    # Equal goals are conservatively one task; a verified trace additionally
    # joins rounds whose review-goal wording changed within the same task.
    parents: dict[str, str] = {}

    def find(node: str) -> str:
        parents.setdefault(node, node)
        path = []
        while parents[node] != node:
            path.append(node)
            node = parents[node]
        for member in path:
            parents[member] = node
        return node

    for goal_key, root_key, _at, _features_json, _passed, _score in raw:
        goal_node = f"goal:{goal_key}"
        if root_key is not None:
            root_node = f"root:{root_key}"
            parents[find(root_node)] = find(goal_node)
    component_keys: dict[str, str] = {}
    for goal_key, _root_key, _at, _features_json, _passed, _score in raw:
        component = find(f"goal:{goal_key}")
        component_keys[component] = min(component_keys.get(component, goal_key), goal_key)
    grouped: dict[str, list[tuple[str, dict[str, float], int, int]]] = {}
    for goal_key, _root_key, submitted_at, features, passed, score in raw:
        group_id = component_keys[find(f"goal:{goal_key}")]
        grouped.setdefault(group_id, []).append((submitted_at, json.loads(features), passed, score))
    return grouped


def train_review_model() -> dict[str, Any]:
    grouped = {group: [(at, vector, passed) for at, vector, passed, _score in rows]
               for group, rows in _group_review_episodes().items()}
    episode_count = sum(len(rows) for rows in grouped.values())
    groups = sorted(grouped, key=lambda item: min(row[0] for row in grouped[item]))
    if episode_count < 20 or len(groups) < 6:
        return {"status": "insufficient_data", "episodes": episode_count, "task_groups": len(groups)}
    holdout_count = max(2, math.ceil(len(groups) * 0.2))
    train_groups, test_groups = groups[:-holdout_count], groups[-holdout_count:]
    training = [(vector, label) for group in train_groups for _at, vector, label in grouped[group]]
    test = [(vector, label) for group in test_groups for _at, vector, label in grouped[group]]
    if len({label for _vector, label in training}) < 2:
        return {"status": "insufficient_label_diversity", "episodes": episode_count, "task_groups": len(groups)}
    weights, bias = _fit(training)
    baseline_rate = sum(label for _vector, label in training) / len(training)
    baseline_brier = sum((baseline_rate - label) ** 2 for _vector, label in test) / len(test)
    model_brier = _brier(test, weights, bias)
    checkpoint = _evaluation_checkpoint(REVIEW_EVAL_PATH,
                                        "modellabs.review_evaluation_checkpoint.v1", groups, grouped)
    prospective_groups, prospective = _prospective_rows(groups, grouped, checkpoint)
    prospective_baseline = (sum((checkpoint["baseline_rate"] - label) ** 2
                                for _vector, label in prospective) / len(prospective)
                            if prospective else None)
    prospective_model = (_brier(prospective, checkpoint["weights"], checkpoint["bias"])
                         if prospective else None)
    gain = (prospective_baseline - prospective_model) if prospective else 0.0
    validated = (len(prospective) >= MIN_REVIEW_HOLDOUT
                 and len(prospective_groups) >= 10
                 and gain >= MIN_REVIEW_BRIER_GAIN
                 and gain / max(prospective_baseline or 0.0, 1e-9) >= MIN_REVIEW_RELATIVE_GAIN)
    final_weights, final_bias = _fit([(vector, label) for group in groups
                                      for _at, vector, label in grouped[group]])
    artifact = {"schema": "modellabs.review_outcome_model.v1", "feature_version": FEATURE_VERSION,
                "task_group_policy": "connected_review_goal_and_verified_iteration_root",
                "trained_episodes": episode_count, "task_groups": len(groups),
                "holdout_episodes": len(test), "holdout_task_groups": len(test_groups),
                "baseline_brier": round(baseline_brier, 6), "model_brier": round(model_brier, 6),
                "retrospective_holdout_is_diagnostic_only": True,
                "prospective_checkpoint_created_at": checkpoint["created_at"],
                "prospective_checkpoint_development_episodes": checkpoint["development_episodes"],
                "prospective_holdout_episodes": len(prospective),
                "prospective_holdout_task_groups": len(prospective_groups),
                "prospective_baseline_brier": round(prospective_baseline, 6) if prospective else None,
                "prospective_model_brier": round(prospective_model, 6) if prospective else None,
                "minimum_holdout": MIN_REVIEW_HOLDOUT,
                "minimum_brier_gain": MIN_REVIEW_BRIER_GAIN,
                "minimum_relative_gain": MIN_REVIEW_RELATIVE_GAIN,
                "validated_for_shadow": validated, "weights": final_weights, "bias": final_bias}
    _private_root()
    temporary = MODEL_PATH.with_name(f".{MODEL_PATH.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        json.dump(artifact, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, MODEL_PATH)
    return {key: value for key, value in artifact.items() if key not in {"weights", "bias"}}


def _score_checkpoint(groups: list[str],
                      grouped: dict[str, list[tuple[str, dict[str, float], float]]]) -> dict[str, Any]:
    schema = "modellabs.review_score_evaluation_checkpoint.v1"
    if SCORE_EVAL_PATH.exists():
        checkpoint = _read_json(SCORE_EVAL_PATH)
        if (checkpoint.get("schema") != schema or checkpoint.get("feature_version") != FEATURE_VERSION
                or not isinstance(checkpoint.get("development_groups"), list)
                or not isinstance(checkpoint.get("weights"), list)
                or not isinstance(checkpoint.get("bias"), (int, float))
                or not isinstance(checkpoint.get("baseline_median"), (int, float))):
            raise ValueError("review score evaluation checkpoint is incompatible")
        return checkpoint
    training = [(vector, label) for group in groups for _at, vector, label in grouped[group]]
    weights, bias = _fit_score(training)
    checkpoint = {"schema": schema, "feature_version": FEATURE_VERSION,
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "development_groups": groups, "development_episodes": len(training),
                  "baseline_median": median(label for _vector, label in training),
                  "weights": weights, "bias": bias}
    _private_root()
    try:
        descriptor = os.open(SCORE_EVAL_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        return _score_checkpoint(groups, grouped)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return checkpoint


def train_review_score_model() -> dict[str, Any]:
    """Predict nonce-bound browser scores; never authorize prompt or model changes."""
    grouped = {group: [(at, vector, score / 100.0) for at, vector, _passed, score in rows]
               for group, rows in _group_review_episodes().items()}
    groups = sorted(grouped, key=lambda item: min(row[0] for row in grouped[item]))
    count = sum(len(rows) for rows in grouped.values())
    if count < 20 or len(groups) < 6:
        return {"status": "insufficient_data", "episodes": count, "task_groups": len(groups)}
    holdout_count = max(2, math.ceil(len(groups) * 0.2))
    train_groups, test_groups = groups[:-holdout_count], groups[-holdout_count:]
    training = [(vector, label) for group in train_groups for _at, vector, label in grouped[group]]
    test = [(vector, label) for group in test_groups for _at, vector, label in grouped[group]]
    if len({label for _vector, label in training}) < 3:
        return {"status": "insufficient_score_diversity", "episodes": count, "task_groups": len(groups)}
    weights, bias = _fit_score(training)
    baseline = median(label for _vector, label in training)
    baseline_mae = sum(abs(baseline - label) for _vector, label in test) / len(test)
    model_mae = _score_mae(test, weights, bias)
    checkpoint = _score_checkpoint(groups, grouped)
    prospective_groups, prospective = _prospective_rows(groups, grouped, checkpoint)
    prospective_baseline = (sum(abs(checkpoint["baseline_median"] - label)
                                for _vector, label in prospective) / len(prospective)
                            if prospective else None)
    prospective_model = (_score_mae(prospective, checkpoint["weights"], checkpoint["bias"])
                         if prospective else None)
    gain = (prospective_baseline - prospective_model) if prospective else 0.0
    validated = (len(prospective) >= MIN_REVIEW_HOLDOUT
                 and len(prospective_groups) >= 10
                 and gain >= MIN_SCORE_MAE_GAIN
                 and gain / max(prospective_baseline or 0.0, 1e-9) >= MIN_REVIEW_RELATIVE_GAIN)
    final_weights, final_bias = _fit_score([(vector, label) for group in groups
                                            for _at, vector, label in grouped[group]])
    artifact = {"schema": "modellabs.review_score_model.v1", "feature_version": FEATURE_VERSION,
                "trained_episodes": count, "task_groups": len(groups),
                "task_group_policy": "connected_review_goal_and_verified_iteration_root",
                "holdout_episodes": len(test), "holdout_task_groups": len(test_groups),
                "baseline_mae_points": round(100 * baseline_mae, 4),
                "model_mae_points": round(100 * model_mae, 4),
                "retrospective_holdout_is_diagnostic_only": True,
                "prospective_checkpoint_created_at": checkpoint["created_at"],
                "prospective_checkpoint_development_episodes": checkpoint["development_episodes"],
                "prospective_holdout_episodes": len(prospective),
                "prospective_holdout_task_groups": len(prospective_groups),
                "prospective_baseline_mae_points": round(100 * prospective_baseline, 4) if prospective else None,
                "prospective_model_mae_points": round(100 * prospective_model, 4) if prospective else None,
                "minimum_holdout": MIN_REVIEW_HOLDOUT,
                "minimum_mae_gain_points": 100 * MIN_SCORE_MAE_GAIN,
                "minimum_relative_gain": MIN_REVIEW_RELATIVE_GAIN,
                "validated_for_shadow": validated, "causal_prompt_comparison": False,
                "weights": final_weights, "bias": final_bias}
    _private_root()
    temporary = SCORE_MODEL_PATH.with_name(f".{SCORE_MODEL_PATH.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        json.dump(artifact, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, SCORE_MODEL_PATH)
    return {key: value for key, value in artifact.items() if key not in {"weights", "bias"}}


def predict_review(goal: str, revision: str, round_number: int) -> dict[str, Any]:
    model = _read_json(MODEL_PATH)
    if model.get("schema") != "modellabs.review_outcome_model.v1" or model.get("feature_version") != FEATURE_VERSION:
        raise ValueError("review model version mismatch")
    vector = _features(goal, revision, round_number, _key())
    probability = _sigmoid(model["bias"] + sum(model["weights"][int(index)] * value
                                                   for index, value in vector.items()
                                                   if int(index) < len(model["weights"])))
    result = {"predicted_review_pass_probability": round(probability, 4),
            "validated_for_shadow": model["validated_for_shadow"],
            "training_episodes": model["trained_episodes"]}
    if SCORE_MODEL_PATH.exists():
        score = _read_json(SCORE_MODEL_PATH)
        if (score.get("schema") != "modellabs.review_score_model.v1"
                or score.get("feature_version") != FEATURE_VERSION):
            raise ValueError("review score model version mismatch")
        result["predicted_review_score"] = round(100 * _score_prediction(
            vector, score["weights"], score["bias"]), 1)
        result["score_validated_for_shadow"] = score["validated_for_shadow"]
    return result


def _iteration_vector(prompt_features: dict[str, float],
                      feedback_features: dict[str, float] | None,
                      author: str,
                      parent_result_features: dict[str, float] | None = None,
                      parent_quality_score: int | None = None,
                      adopted_parent_suggestion: bool = False,
                      parent_codex_final_features: dict[str, float] | None = None,
                      parent_codex_model: str | None = None,
                      parent_codex_effort: str | None = None,
                      parent_browser_result_features: dict[str, float] | None = None) -> dict[str, float]:
    authors = ("user", "chatgpt", "codex", "mixed")
    if author not in authors:
        raise ValueError("unknown prompt author")
    vector = dict(prompt_features)
    if feedback_features is not None:
        vector.update({str(FEATURE_COUNT + 3 + int(index)): value
                       for index, value in feedback_features.items()})
    vector[str(2 * (FEATURE_COUNT + 3) + authors.index(author))] = 1.0
    if parent_result_features is not None:
        vector.update({str(2 * (FEATURE_COUNT + 3) + len(authors) + int(index)): value
                       for index, value in parent_result_features.items()})
    if parent_quality_score is not None:
        if not 0 <= parent_quality_score <= 100:
            raise ValueError("parent quality score is out of range")
        vector[str(3 * (FEATURE_COUNT + 3) + len(authors))] = parent_quality_score / 100.0
    if adopted_parent_suggestion:
        vector[str(3 * (FEATURE_COUNT + 3) + len(authors) + 1)] = 1.0
    if parent_codex_final_features is not None:
        base = 3 * (FEATURE_COUNT + 3) + len(authors) + 2
        vector.update({str(base + int(index)): value
                       for index, value in parent_codex_final_features.items()})
        vector[str(base + FEATURE_COUNT + 3)] = 1.0
    if parent_codex_model is not None or parent_codex_effort is not None:
        if (parent_codex_final_features is None or parent_codex_model not in MODEL_IDS
                or parent_codex_effort not in EFFORTS):
            raise ValueError("parent Codex arm requires a linked final answer and supported model/effort")
        arm_base = 4 * (FEATURE_COUNT + 3) + len(authors) + 3
        vector[str(arm_base + MODEL_IDS.index(parent_codex_model))] = 1.0
        vector[str(arm_base + len(MODEL_IDS) + EFFORTS.index(parent_codex_effort))] = 1.0
    if parent_browser_result_features is not None:
        browser_base = 4 * (FEATURE_COUNT + 3) + len(authors) + 3 + len(MODEL_IDS) + len(EFFORTS)
        vector.update({str(browser_base + int(index)): value
                       for index, value in parent_browser_result_features.items()})
        vector[str(browser_base + FEATURE_COUNT + 3)] = 1.0
    return vector


def _linked_codex_outcomes(connection: sqlite3.Connection) -> dict[str, tuple[dict[str, float], str | None, str | None]]:
    """Return final answers and arms only from valid exact browser-to-Codex links."""
    rows = connection.execute("""SELECT l.episode_id,
                                CASE WHEN l.source_kind='managed' THEN c.result_features
                                     WHEN l.source_kind='completed_session' THEN s.result_features END,
                                CASE WHEN l.source_kind='managed' AND c.total_tokens IS NOT NULL
                                          AND c.model_provenance='observed_per_request' THEN c.observed_model
                                     WHEN l.source_kind='completed_session'
                                          AND s.model_provenance='observed_per_request' THEN s.observed_model END,
                                CASE WHEN l.source_kind='managed' AND c.total_tokens IS NOT NULL
                                          AND c.model_provenance='observed_per_request' THEN c.observed_effort
                                     WHEN l.source_kind='completed_session'
                                          AND s.model_provenance='observed_per_request' THEN s.observed_effort END
                                FROM review_codex_links l
                                LEFT JOIN codex_turns c ON l.source_kind='managed'
                                    AND c.thread_key=l.thread_key AND c.turn_key=l.turn_key
                                LEFT JOIN session_turns s ON l.source_kind='completed_session'
                                    AND s.thread_key=l.thread_key AND s.turn_key=l.turn_key
                                WHERE l.link_status='valid'""").fetchall()
    return {episode_id: (json.loads(features), model if model in MODEL_IDS else None,
                         effort if effort in EFFORTS else None)
            for episode_id, features, model, effort in rows if features}


def _bound_loop_vector(prompt: dict[str, float], codex_result: dict[str, float] | None,
                       parent: tuple[Any, ...] | None, browser_suggestion_supplied: bool) -> dict[str, float]:
    """Use only inputs available after Codex and before this round's browser grade."""
    vector = _iteration_vector(
        prompt, json.loads(parent[9]) if parent else None, "mixed",
        None, parent[11] if parent else None, False,
        json.loads(parent[7]) if parent and parent[7] else None,
        None, None, json.loads(parent[8]) if parent else None)
    if codex_result is not None:
        base = 6 * (FEATURE_COUNT + 3) + len(MODEL_IDS) + len(EFFORTS) + 16
        vector.update({str(base + int(index)): value for index, value in codex_result.items()})
        vector[str(base + FEATURE_COUNT + 3)] = 1.0
    if browser_suggestion_supplied:
        vector[str(8 * (FEATURE_COUNT + 3) + len(MODEL_IDS) + len(EFFORTS) + 32)] = 1.0
    return vector


def _bound_score_checkpoint(groups: list[str],
                            grouped: dict[str, list[tuple[str, dict[str, float], float]]]) -> dict[str, Any]:
    """Freeze score predictions before any prospective bound task group arrives."""
    schema = "modellabs.bound_loop_score_evaluation_checkpoint.v1"
    if BOUND_LOOP_SCORE_EVAL_PATH.exists():
        checkpoint = _read_json(BOUND_LOOP_SCORE_EVAL_PATH)
        if (checkpoint.get("schema") != schema
                or checkpoint.get("feature_version") != BOUND_LOOP_FEATURE_VERSION
                or not isinstance(checkpoint.get("development_groups"), list)
                or not isinstance(checkpoint.get("weights"), list)
                or not isinstance(checkpoint.get("bias"), (int, float))
                or not isinstance(checkpoint.get("baseline_median"), (int, float))):
            raise ValueError("bound loop score evaluation checkpoint is incompatible")
        return checkpoint
    training = [(vector, label) for group in groups for _at, vector, label in grouped[group]]
    weights, bias = _fit_score(training)
    checkpoint = {"schema": schema, "feature_version": BOUND_LOOP_FEATURE_VERSION,
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "development_groups": groups, "development_episodes": len(training),
                  "baseline_median": median(label for _vector, label in training),
                  "weights": weights, "bias": bias}
    _private_root()
    try:
        descriptor = os.open(BOUND_LOOP_SCORE_EVAL_PATH,
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        return _bound_score_checkpoint(groups, grouped)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return checkpoint


def train_bound_loop_model() -> dict[str, Any]:
    """Fit a separate observational post-result model with forward validation."""
    with _connect() as connection:
        all_rows = connection.execute("SELECT * FROM bound_loop_rounds ORDER BY submitted_at,root_key,round_number").fetchall()
    eligible_ids = {row[0] for row in all_rows if row[17] == 1 and row[18] == 1}
    while True:
        removed = {row[0] for row in all_rows
                   if row[0] in eligible_ids and row[10] and row[10] not in eligible_ids}
        if not removed:
            break
        eligible_ids.difference_update(removed)
    raw = [row for row in all_rows if row[0] in eligible_ids]
    by_id = {row[0]: row for row in raw}
    groups: dict[str, list[tuple[str, dict[str, float], int]]] = {}
    score_groups: dict[str, list[tuple[str, dict[str, float], float]]] = {}
    revision_groups: set[str] = set()
    for row in raw:
        parent = by_id.get(row[10]) if row[10] else None
        if row[10] and (parent is None or parent[1] != row[1] or parent[2] != row[2] - 1):
            raise ValueError("bound loop model found broken parent lineage")
        if parent:
            revision_groups.add(row[1])
        vector = _bound_loop_vector(json.loads(row[6]), json.loads(row[7]) if row[7] else None,
                                    parent, bool(row[19]))
        groups.setdefault(row[1], []).append((row[3], vector, row[12]))
        score_groups.setdefault(row[1], []).append((row[3], vector, row[11] / 100.0))
    base = {"rounds": len(raw), "task_groups": len(groups),
            "observed_rounds": len(all_rows), "dispatch_bound_rounds": sum(row[17] for row in all_rows),
            "browser_ui_confirmed_rounds": sum(row[18] for row in all_rows),
            "browser_suggestion_supplied_rounds": sum(row[19] for row in all_rows),
            "revision_task_groups": len(revision_groups),
            "codex_result_rounds": sum(row[7] is not None for row in raw),
            "provider_verified_attachment_contents": sum(row[16] for row in raw),
            "input_stage": "post_codex_pre_browser", "authoritative_for_routing": False}
    if len(raw) < 30 or len(groups) < 10 or len(revision_groups) < MIN_ITERATION_REVISION_GROUPS:
        return {"status": "insufficient_bound_loop_rounds", **base}
    ordered = sorted(groups, key=lambda root: min(row[0] for row in groups[root]))
    holdout_count = max(2, math.ceil(len(ordered) * 0.2))
    training_groups, holdout_groups = ordered[:-holdout_count], ordered[-holdout_count:]
    training = [(vector, label) for root in training_groups for _at, vector, label in groups[root]]
    holdout = [(vector, label) for root in holdout_groups for _at, vector, label in groups[root]]
    if (len({label for _vector, label in training}) < 2 or len(holdout) < 8
            or len(set(training_groups) & revision_groups) < MIN_ITERATION_REVISION_GROUPS
            or len(set(holdout_groups) & revision_groups) < 2):
        return {"status": "insufficient_bound_loop_split", **base, "holdout_rounds": len(holdout)}
    weights, bias = _fit(training)
    baseline_rate = sum(label for _vector, label in training) / len(training)
    baseline_brier = sum((baseline_rate - label) ** 2 for _vector, label in holdout) / len(holdout)
    model_brier = _brier(holdout, weights, bias)
    checkpoint = _evaluation_checkpoint(BOUND_LOOP_EVAL_PATH,
                                        "modellabs.bound_loop_evaluation_checkpoint.v2",
                                        ordered, groups, feature_version=BOUND_LOOP_FEATURE_VERSION)
    prospective_groups, prospective = _prospective_rows(ordered, groups, checkpoint)
    prospective_baseline = (sum((checkpoint["baseline_rate"] - label) ** 2
                                for _vector, label in prospective) / len(prospective)
                            if prospective else None)
    prospective_model = (_brier(prospective, checkpoint["weights"], checkpoint["bias"])
                         if prospective else None)
    gain = (prospective_baseline - prospective_model) if prospective else 0.0
    validated = (len(prospective) >= MIN_REVIEW_HOLDOUT and len(prospective_groups) >= 10
                 and len(set(prospective_groups) & revision_groups) >= MIN_ITERATION_REVISION_GROUPS
                 and gain >= MIN_REVIEW_BRIER_GAIN
                 and gain / max(prospective_baseline or 0.0, 1e-9) >= MIN_REVIEW_RELATIVE_GAIN)
    score_training = [(vector, label) for root in training_groups
                      for _at, vector, label in score_groups[root]]
    score_holdout = [(vector, label) for root in holdout_groups
                     for _at, vector, label in score_groups[root]]
    score_weights, score_bias = _fit_score(score_training)
    baseline_score = median(label for _vector, label in score_training)
    baseline_mae = sum(abs(baseline_score - label) for _vector, label in score_holdout) / len(score_holdout)
    score_checkpoint = _bound_score_checkpoint(ordered, score_groups)
    prospective_score_groups, prospective_scores = _prospective_rows(
        ordered, score_groups, score_checkpoint)
    prospective_score_baseline = (
        sum(abs(score_checkpoint["baseline_median"] - label)
            for _vector, label in prospective_scores) / len(prospective_scores)
        if prospective_scores else None)
    prospective_score_model = (
        _score_mae(prospective_scores, score_checkpoint["weights"], score_checkpoint["bias"])
        if prospective_scores else None)
    score_gain = (prospective_score_baseline - prospective_score_model
                  if prospective_scores else 0.0)
    score_validated = (len(prospective_scores) >= MIN_REVIEW_HOLDOUT
                       and len(prospective_score_groups) >= 10
                       and len(set(prospective_score_groups) & revision_groups)
                       >= MIN_ITERATION_REVISION_GROUPS
                       and score_gain >= MIN_SCORE_MAE_GAIN
                       and score_gain / max(prospective_score_baseline or 0.0, 1e-9)
                       >= MIN_REVIEW_RELATIVE_GAIN)
    artifact = {"schema": "modellabs.bound_loop_post_result_model.v2",
                "feature_version": BOUND_LOOP_FEATURE_VERSION, **base,
                "holdout_rounds": len(holdout),
                "baseline_brier": round(baseline_brier, 6), "model_brier": round(model_brier, 6),
                "baseline_mae_points": round(100 * baseline_mae, 4),
                "model_mae_points": round(100 * _score_mae(score_holdout, score_weights, score_bias), 4),
                "retrospective_holdout_is_diagnostic_only": True,
                "prospective_checkpoint_created_at": checkpoint["created_at"],
                "prospective_score_checkpoint_created_at": score_checkpoint["created_at"],
                "prospective_holdout_rounds": len(prospective),
                "prospective_holdout_task_groups": len(prospective_groups),
                "prospective_baseline_brier": round(prospective_baseline, 6) if prospective else None,
                "prospective_model_brier": round(prospective_model, 6) if prospective else None,
                "prospective_score_holdout_rounds": len(prospective_scores),
                "prospective_score_holdout_task_groups": len(prospective_score_groups),
                "prospective_score_baseline_mae_points": (
                    round(100 * prospective_score_baseline, 4) if prospective_scores else None),
                "prospective_score_model_mae_points": (
                    round(100 * prospective_score_model, 4) if prospective_scores else None),
                "validated_for_shadow": validated, "score_validated_for_shadow": score_validated,
                "causal_prompt_comparison": False,
                "weights": None, "bias": None, "score_weights": None, "score_bias": None}
    artifact["weights"], artifact["bias"] = _fit(
        [(vector, label) for root in ordered for _at, vector, label in groups[root]])
    artifact["score_weights"], artifact["score_bias"] = _fit_score(
        [(vector, label) for root in ordered for _at, vector, label in score_groups[root]])
    _private_root()
    temporary = BOUND_LOOP_MODEL_PATH.with_name(
        f".{BOUND_LOOP_MODEL_PATH.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        json.dump(artifact, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, BOUND_LOOP_MODEL_PATH)
    return {key: value for key, value in artifact.items()
            if key not in {"weights", "bias", "score_weights", "score_bias"}}


def train_iteration_model() -> dict[str, Any]:
    """Learn prompt-revision outcomes only from explicitly linked review rounds."""
    with _connect() as connection:
        linked_results = _linked_codex_outcomes(connection)
        browser_results = {episode_id: json.loads(features) for episode_id, features in
                           connection.execute("SELECT episode_id,result_features FROM browser_results")}
        raw = connection.execute("""SELECT root_key,submitted_at,prompt_features,
                                   parent_feedback_features,prompt_author,passed,
                                   parent_result_features,parent_quality_score,
                                   CASE WHEN adopted_parent_suggestion=1
                                             OR adopted_parent_inline_revision=1 THEN 1 ELSE 0 END,
                                   parent_episode_id
                                   FROM iteration_traces ORDER BY submitted_at,episode_id""").fetchall()
    groups: dict[str, list[tuple[str, dict[str, float], int]]] = {}
    revision_groups: set[str] = set()
    revision_rounds = 0
    linked_parent_codex_results = 0
    linked_parent_codex_arms = 0
    linked_parent_browser_results = 0
    for root_key, submitted_at, prompt, feedback, author, passed, parent_result, parent_score, adopted, parent_id in raw:
        if parent_id is not None and parent_result is not None and parent_score is not None:
            revision_groups.add(root_key)
            revision_rounds += 1
        codex_outcome = linked_results.get(parent_id)
        if codex_outcome is not None:
            linked_parent_codex_results += 1
        codex_result, codex_model, codex_effort = codex_outcome or (None, None, None)
        if codex_model is not None and codex_effort is not None:
            linked_parent_codex_arms += 1
        browser_result = browser_results.get(parent_id)
        if browser_result is not None:
            linked_parent_browser_results += 1
        vector = _iteration_vector(json.loads(prompt), json.loads(feedback) if feedback else None,
                                   author, json.loads(parent_result) if parent_result else None,
                                   parent_score, bool(adopted), codex_result,
                                   codex_model if codex_effort is not None else None,
                                   codex_effort if codex_model is not None else None,
                                   browser_result)
        groups.setdefault(root_key, []).append((submitted_at, vector, passed))
    ordered = sorted(groups, key=lambda root: min(row[0] for row in groups[root]))
    if len(raw) < 30 or len(groups) < 10:
        return {"status": "insufficient_linked_rounds", "rounds": len(raw), "task_groups": len(groups),
                "context_complete_revision_rounds": revision_rounds,
                "context_complete_revision_task_groups": len(revision_groups),
                "linked_parent_codex_results": linked_parent_codex_results,
                "linked_parent_codex_arms": linked_parent_codex_arms,
                "linked_parent_browser_results": linked_parent_browser_results}
    if len(revision_groups) < MIN_ITERATION_REVISION_GROUPS:
        return {"status": "insufficient_context_complete_revisions", "rounds": len(raw),
                "task_groups": len(groups), "context_complete_revision_rounds": revision_rounds,
                "context_complete_revision_task_groups": len(revision_groups),
                "linked_parent_codex_results": linked_parent_codex_results,
                "linked_parent_codex_arms": linked_parent_codex_arms,
                "linked_parent_browser_results": linked_parent_browser_results,
                "minimum_revision_task_groups": MIN_ITERATION_REVISION_GROUPS}
    holdout_count = max(2, math.ceil(len(ordered) * 0.2))
    training_revision_groups = set(ordered[:-holdout_count]) & revision_groups
    test_revision_groups = set(ordered[-holdout_count:]) & revision_groups
    if (len(training_revision_groups) < MIN_ITERATION_REVISION_GROUPS
            or len(test_revision_groups) < 2) and not ITERATION_EVAL_PATH.exists():
        return {"status": "insufficient_revision_split", "rounds": len(raw),
                "task_groups": len(groups),
                "training_context_complete_revision_task_groups": len(training_revision_groups),
                "holdout_context_complete_revision_task_groups": len(test_revision_groups)}
    training = [(vector, label) for root in ordered[:-holdout_count]
                for _at, vector, label in groups[root]]
    test = [(vector, label) for root in ordered[-holdout_count:]
            for _at, vector, label in groups[root]]
    if len({label for _vector, label in training}) < 2 or len(test) < 8:
        return {"status": "insufficient_holdout_diversity", "rounds": len(raw),
                "task_groups": len(groups), "holdout_rounds": len(test)}
    weights, bias = _fit(training)
    baseline_rate = sum(label for _vector, label in training) / len(training)
    baseline_brier = sum((baseline_rate - label) ** 2 for _vector, label in test) / len(test)
    model_brier = _brier(test, weights, bias)
    checkpoint = _evaluation_checkpoint(ITERATION_EVAL_PATH,
                                        "modellabs.iteration_evaluation_checkpoint.v7", ordered, groups,
                                        feature_version=ITERATION_FEATURE_VERSION)
    prospective_groups, prospective = _prospective_rows(ordered, groups, checkpoint)
    prospective_revision_groups = set(prospective_groups) & revision_groups
    prospective_baseline = (sum((checkpoint["baseline_rate"] - label) ** 2
                                for _vector, label in prospective) / len(prospective)
                            if prospective else None)
    prospective_model = (_brier(prospective, checkpoint["weights"], checkpoint["bias"])
                         if prospective else None)
    gain = (prospective_baseline - prospective_model) if prospective else 0.0
    validated = (len(prospective) >= MIN_REVIEW_HOLDOUT
                 and len(prospective_groups) >= 10
                 and len(prospective_revision_groups) >= MIN_ITERATION_REVISION_GROUPS
                 and gain >= MIN_REVIEW_BRIER_GAIN
                 and gain / max(prospective_baseline or 0.0, 1e-9) >= MIN_REVIEW_RELATIVE_GAIN)
    final_weights, final_bias = _fit([(vector, label) for root in ordered
                                      for _at, vector, label in groups[root]])
    artifact = {"schema": "modellabs.iteration_outcome_model.v7", "feature_version": ITERATION_FEATURE_VERSION,
                "trained_rounds": len(raw), "task_groups": len(groups), "holdout_rounds": len(test),
                "context_complete_revision_rounds": revision_rounds,
                "context_complete_revision_task_groups": len(revision_groups),
                "linked_parent_codex_results": linked_parent_codex_results,
                "linked_parent_codex_arms": linked_parent_codex_arms,
                "linked_parent_browser_results": linked_parent_browser_results,
                "minimum_revision_task_groups": MIN_ITERATION_REVISION_GROUPS,
                "training_context_complete_revision_task_groups": len(training_revision_groups),
                "holdout_context_complete_revision_task_groups": len(test_revision_groups),
                "baseline_brier": round(baseline_brier, 6), "model_brier": round(model_brier, 6),
                "retrospective_holdout_is_diagnostic_only": True,
                "prospective_checkpoint_created_at": checkpoint["created_at"],
                "prospective_holdout_rounds": len(prospective),
                "prospective_holdout_task_groups": len(prospective_groups),
                "prospective_context_complete_revision_task_groups": len(prospective_revision_groups),
                "prospective_baseline_brier": round(prospective_baseline, 6) if prospective else None,
                "prospective_model_brier": round(prospective_model, 6) if prospective else None,
                "validated_for_shadow": validated,
                "causal_prompt_comparison": False, "weights": final_weights, "bias": final_bias}
    temporary = ITERATION_MODEL_PATH.with_name(f".{ITERATION_MODEL_PATH.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        json.dump(artifact, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, ITERATION_MODEL_PATH)
    return {key: value for key, value in artifact.items() if key not in {"weights", "bias"}}


def train_improvement_model() -> dict[str, Any]:
    """Predict whether a linked revision improves its parent's browser score."""
    with _connect() as connection:
        linked_results = _linked_codex_outcomes(connection)
        browser_results = {episode_id: json.loads(features) for episode_id, features in
                           connection.execute("SELECT episode_id,result_features FROM browser_results")}
        raw = connection.execute("""SELECT t.root_key,
                                   (SELECT MIN(first.submitted_at) FROM iteration_traces first
                                    WHERE first.root_key=t.root_key),t.prompt_features,
                                   t.parent_feedback_features,t.prompt_author,
                                   t.parent_result_features,t.parent_quality_score,
                                   CASE WHEN t.adopted_parent_suggestion=1
                                             OR t.adopted_parent_inline_revision=1 THEN 1 ELSE 0 END,
                                   e.quality_score,t.parent_episode_id
                                   FROM iteration_traces t JOIN episodes e ON e.episode_id=t.episode_id
                                   WHERE t.parent_episode_id IS NOT NULL
                                   AND t.parent_result_features IS NOT NULL
                                   AND t.parent_quality_score IS NOT NULL
                                   AND e.quality_score IS NOT NULL
                                   ORDER BY t.submitted_at,t.episode_id""").fetchall()
    groups: dict[str, list[tuple[str, dict[str, float], int]]] = {}
    linked_parent_codex_results = 0
    linked_parent_codex_arms = 0
    linked_parent_browser_results = 0
    for root, submitted_at, prompt, feedback, author, parent_result, parent_score, adopted, score, parent_id in raw:
        if not (0 <= parent_score <= 100 and 0 <= score <= 100):
            continue
        codex_outcome = linked_results.get(parent_id)
        if codex_outcome is not None:
            linked_parent_codex_results += 1
        codex_result, codex_model, codex_effort = codex_outcome or (None, None, None)
        if codex_model is not None and codex_effort is not None:
            linked_parent_codex_arms += 1
        browser_result = browser_results.get(parent_id)
        if browser_result is not None:
            linked_parent_browser_results += 1
        vector = _iteration_vector(json.loads(prompt), json.loads(feedback) if feedback else None,
                                   author, json.loads(parent_result), parent_score, bool(adopted), codex_result,
                                   codex_model if codex_effort is not None else None,
                                   codex_effort if codex_model is not None else None,
                                   browser_result)
        groups.setdefault(root, []).append((submitted_at, vector, int(score > parent_score)))
    ordered = sorted(groups, key=lambda root: min(row[0] for row in groups[root]))
    revisions = sum(len(rows) for rows in groups.values())
    if revisions < 20 or len(groups) < 10:
        return {"status": "insufficient_scored_revisions", "scored_revisions": revisions,
                "task_groups": len(groups),
                "linked_parent_codex_results": linked_parent_codex_results,
                "linked_parent_codex_arms": linked_parent_codex_arms,
                "linked_parent_browser_results": linked_parent_browser_results}
    holdout_count = max(2, math.ceil(len(ordered) * 0.2))
    training = [(vector, label) for root in ordered[:-holdout_count]
                for _at, vector, label in groups[root]]
    test = [(vector, label) for root in ordered[-holdout_count:]
            for _at, vector, label in groups[root]]
    if len({label for _vector, label in training}) < 2 or len(test) < 4:
        return {"status": "insufficient_improvement_diversity", "scored_revisions": revisions,
                "task_groups": len(groups), "holdout_revisions": len(test)}
    weights, bias = _fit(training)
    baseline_rate = sum(label for _vector, label in training) / len(training)
    baseline_brier = sum((baseline_rate - label) ** 2 for _vector, label in test) / len(test)
    model_brier = _brier(test, weights, bias)
    checkpoint = _evaluation_checkpoint(IMPROVEMENT_EVAL_PATH,
                                        "modellabs.revision_improvement_evaluation_checkpoint.v4",
                                        ordered, groups, feature_version=ITERATION_FEATURE_VERSION)
    prospective_groups, prospective = _prospective_rows(ordered, groups, checkpoint)
    prospective_baseline = (sum((checkpoint["baseline_rate"] - label) ** 2
                                for _vector, label in prospective) / len(prospective)
                            if prospective else None)
    prospective_model = (_brier(prospective, checkpoint["weights"], checkpoint["bias"])
                         if prospective else None)
    gain = (prospective_baseline - prospective_model) if prospective else 0.0
    validated = (len(prospective) >= MIN_REVIEW_HOLDOUT
                 and len(prospective_groups) >= 10
                 and gain >= MIN_REVIEW_BRIER_GAIN
                 and gain / max(prospective_baseline or 0.0, 1e-9) >= MIN_REVIEW_RELATIVE_GAIN)
    final_weights, final_bias = _fit([(vector, label) for root in ordered
                                      for _at, vector, label in groups[root]])
    artifact = {"schema": "modellabs.revision_improvement_model.v4",
                "feature_version": ITERATION_FEATURE_VERSION,
                "scored_revisions": revisions, "task_groups": len(groups),
                "linked_parent_codex_results": linked_parent_codex_results,
                "linked_parent_codex_arms": linked_parent_codex_arms,
                "linked_parent_browser_results": linked_parent_browser_results,
                "holdout_revisions": len(test),
                "baseline_brier": round(baseline_brier, 6), "model_brier": round(model_brier, 6),
                "retrospective_holdout_is_diagnostic_only": True,
                "prospective_checkpoint_created_at": checkpoint["created_at"],
                "prospective_holdout_revisions": len(prospective),
                "prospective_holdout_task_groups": len(prospective_groups),
                "prospective_baseline_brier": round(prospective_baseline, 6) if prospective else None,
                "prospective_model_brier": round(prospective_model, 6) if prospective else None,
                "validated_for_shadow": validated,
                "causal_prompt_comparison": False,
                "weights": final_weights, "bias": final_bias}
    temporary = IMPROVEMENT_MODEL_PATH.with_name(
        f".{IMPROVEMENT_MODEL_PATH.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        json.dump(artifact, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, IMPROVEMENT_MODEL_PATH)
    return {key: value for key, value in artifact.items() if key not in {"weights", "bias"}}


def predict_iteration(origin_prompt: str, generation_prompt: str, author: str,
                      round_number: int, parent_feedback: str = "",
                      parent_result: str = "", parent_quality_score: int | None = None,
                      parent_suggestion: str = "", parent_codex_result: str = "",
                      parent_codex_model: str | None = None,
                      parent_codex_effort: str | None = None,
                      parent_browser_result: str = "") -> dict[str, Any]:
    model = _read_json(ITERATION_MODEL_PATH)
    if (model.get("schema") != "modellabs.iteration_outcome_model.v7"
            or model.get("feature_version") != ITERATION_FEATURE_VERSION):
        raise ValueError("iteration model version mismatch")
    if (parent_result or parent_quality_score is not None or parent_suggestion
            or parent_codex_result or parent_codex_model or parent_codex_effort
            or parent_browser_result) and round_number < 2:
        raise ValueError("first-round prediction cannot include parent evidence")
    adopted = bool(parent_suggestion.strip()) and parent_suggestion.strip() == generation_prompt.strip()
    key = _key()
    vector = _iteration_vector(_features(origin_prompt, generation_prompt, round_number, key),
                               _features(parent_feedback, "", 1, key) if parent_feedback else None,
                               author, _features(parent_result, "", 1, key) if parent_result else None,
                               parent_quality_score, adopted,
                               _features(parent_codex_result, "", 1, key) if parent_codex_result else None,
                               parent_codex_model, parent_codex_effort,
                               _features(parent_browser_result, "", 1, key) if parent_browser_result else None)
    probability = _sigmoid(model["bias"] + sum(model["weights"][int(index)] * value
                                                   for index, value in vector.items()
                                                   if int(index) < len(model["weights"])))
    improvement_probability = None
    improvement_validated = False
    if (IMPROVEMENT_MODEL_PATH.exists() and round_number >= 2
            and parent_result and parent_quality_score is not None):
        improvement = _read_json(IMPROVEMENT_MODEL_PATH)
        if (improvement.get("schema") != "modellabs.revision_improvement_model.v4"
                or improvement.get("feature_version") != ITERATION_FEATURE_VERSION):
            raise ValueError("revision improvement model version mismatch")
        improvement_probability = round(_sigmoid(
            improvement["bias"] + sum(improvement["weights"][int(index)] * value
                                      for index, value in vector.items()
                                      if int(index) < len(improvement["weights"]))), 4)
        improvement_validated = improvement["validated_for_shadow"]
    return {"predicted_review_pass_probability": round(probability, 4),
            "predicted_score_improvement_probability": improvement_probability,
            "improvement_validated_for_shadow": improvement_validated,
            "validated_for_shadow": model["validated_for_shadow"],
            "causal_prompt_comparison": False, "training_rounds": model["trained_rounds"],
            "parent_result_supplied": bool(parent_result),
            "parent_review_artifact_supplied": bool(parent_result),
            "parent_codex_result_supplied": bool(parent_codex_result),
            "linked_parent_codex_results_in_training": model.get("linked_parent_codex_results", 0),
            "parent_codex_arm_supplied": bool(parent_codex_model and parent_codex_effort),
            "linked_parent_codex_arms_in_training": model.get("linked_parent_codex_arms", 0),
            "parent_browser_result_supplied": bool(parent_browser_result),
            "linked_parent_browser_results_in_training": model.get("linked_parent_browser_results", 0),
            "parent_quality_score_supplied": parent_quality_score is not None,
            "adopted_parent_suggestion": adopted}


def forward_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    """Describe evidence still needed before a live adaptive-routing pilot.

    This is a diagnostic gate, not an authorization or an automatic mode switch.
    A pilot still needs a separately verified rollback and live canary.
    """
    review = summary.get("review_model") or {}
    score = summary.get("review_score_model") or {}
    iteration = summary.get("iteration_model") or {}
    improvement = summary.get("improvement_model") or {}
    bound = summary.get("bound_loop_model") or {}
    codex = summary.get("codex_model") or {}
    local_prompt = summary.get("local_prompt_model") or {}
    managed_routing = summary.get("managed_routing_policy") or {}
    managed_prompt = summary.get("managed_prompt_policy") or {}
    bound_rounds = summary.get("bound_legacy_loop") or {}
    checks = {
        "prospective_review_pass": review.get("validated_for_shadow") is True,
        "prospective_review_score": score.get("validated_for_shadow") is True,
        "prospective_prompt_iteration": iteration.get("validated_for_shadow") is True,
        "prospective_revision_improvement": improvement.get("validated_for_shadow") is True,
        "prospective_bound_codex_browser": (bound.get("validated_for_shadow") is True
                                            and bound.get("score_validated_for_shadow") is True),
        "browser_attachment_receipts": (bound_rounds.get("browser_ui_confirmed_rounds", 0) >= 30
                                        and bound_rounds.get("linked_revisions", 0) >= MIN_ITERATION_REVISION_GROUPS),
        "prospective_managed_model_effort": (
            managed_routing.get("validated_for_shadow") is True
            and managed_routing.get("causal_model_comparison") is True),
        "prospective_managed_prompt_revision": (
            managed_prompt.get("validated_for_shadow") is True
            and managed_prompt.get("causal_prompt_comparison") is True
            and managed_prompt.get("browser_provenance_verified") is True),
        "prospective_local_prompt_product": local_prompt.get("validated_for_shadow") is True,
    }
    return {"candidate_for_live_pilot": all(checks.values()),
            "checks": checks,
            "missing": [name for name, passed in checks.items() if not passed],
            "role": "diagnostic_only_requires_separate_canary_and_rollback"}


def status() -> dict[str, Any]:
    with _connect() as connection:
        rows = connection.execute("SELECT source,COUNT(*),COUNT(DISTINCT group_id),SUM(passed) FROM episodes GROUP BY source").fetchall()
        codex = connection.execute("""SELECT COUNT(*),COUNT(DISTINCT group_id),
                                     SUM(quality_score IS NOT NULL),SUM(total_tokens IS NOT NULL),
                                     SUM(result_digest IS NOT NULL),
                                     SUM(quality_score IS NOT NULL AND total_tokens IS NOT NULL
                                         AND result_digest IS NOT NULL),
                                     SUM(model_provenance='observed_per_request'
                                         AND observed_model IS NOT NULL AND observed_effort IS NOT NULL
                                         AND reroute_seen=0),
                                     SUM(reroute_seen)
                                     FROM codex_turns""").fetchone()
        steers = connection.execute("""SELECT COUNT(*),COUNT(DISTINCT thread_key),
                                    COUNT(DISTINCT thread_key || ':' || turn_key)
                                    FROM codex_steers""").fetchone()
        context_messages = connection.execute("""SELECT COUNT(*),
                                    SUM(phase='after_prior_output'),
                                    COUNT(DISTINCT thread_key || ':' || turn_key)
                                    FROM session_user_messages""").fetchone()
        traces = connection.execute("""SELECT COUNT(*),COUNT(DISTINCT root_key),
                                     SUM(parent_episode_id IS NOT NULL),
                                     SUM(adopted_parent_suggestion),
                                     SUM(adopted_parent_inline_revision),
                                     SUM(parent_episode_id IS NOT NULL AND parent_result_features IS NOT NULL
                                         AND parent_quality_score IS NOT NULL),
                                     COUNT(DISTINCT CASE WHEN parent_episode_id IS NOT NULL
                                         AND parent_result_features IS NOT NULL
                                         AND parent_quality_score IS NOT NULL THEN root_key END)
                                     FROM iteration_traces""").fetchone()
        sessions = connection.execute("""SELECT COUNT(*),COUNT(DISTINCT thread_key),
                                       SUM(reported_total_tokens IS NOT NULL),
                                       SUM(grade_status='explicit' AND exact_total_tokens IS NOT NULL),
                                       SUM(input_scope='single_pre_inference_message'),
                                       SUM(model_provenance='observed_per_request'
                                           AND observed_model IS NOT NULL AND observed_effort IS NOT NULL)
                                       FROM session_turns""").fetchone()
        cross_source = connection.execute("""SELECT SUM(link_status='valid'),
                                           SUM(source_kind='managed' AND link_status='valid'),
                                           SUM(source_kind='completed_session' AND link_status='valid'),
                                           SUM(link_status='ambiguous')
                                           FROM review_codex_links""").fetchone()
        unlinked_reviews = connection.execute("""SELECT COUNT(*),
            SUM(EXISTS(SELECT 1 FROM codex_turns AS c
                       WHERE c.prompt_digest=t.generation_prompt_digest)),
            SUM(EXISTS(SELECT 1 FROM session_turns AS s
                       WHERE s.prompt_digest=t.generation_prompt_digest
                       AND s.input_scope='single_pre_inference_message')),
            SUM(NOT EXISTS(SELECT 1 FROM codex_turns AS c
                           WHERE c.prompt_digest=t.generation_prompt_digest)
                AND NOT EXISTS(SELECT 1 FROM session_turns AS s
                               WHERE s.prompt_digest=t.generation_prompt_digest
                               AND s.input_scope='single_pre_inference_message'))
            FROM iteration_traces AS t
            WHERE NOT EXISTS(SELECT 1 FROM review_codex_links AS l
                             WHERE l.episode_id=t.episode_id AND l.link_status='valid')""").fetchone()
        browser_result_count = connection.execute("SELECT COUNT(*) FROM browser_results").fetchone()[0]
        recovered = connection.execute("SELECT COUNT(*),SUM(attachment_ui_confirmed) FROM recovery_observations").fetchone()
        benchmark = connection.execute("""SELECT COUNT(*),COUNT(DISTINCT suite_key),
                                       COUNT(DISTINCT comparison_key),COUNT(DISTINCT prompt_digest),
                                       SUM(product_pass)
                                       FROM benchmark_prompt_runs""").fetchone()
        managed_blocks = connection.execute("""SELECT b.block_key,b.product_key,b.arm_order,
            COUNT(r.run_key) FROM managed_benchmark_blocks AS b
            LEFT JOIN managed_benchmark_bindings AS r ON r.block_key=b.block_key
            GROUP BY b.block_key,b.product_key,b.arm_order""").fetchall()
        loop_rounds = connection.execute("""SELECT COUNT(*),COUNT(DISTINCT root_key),
                                         SUM(parent_episode_id IS NOT NULL),SUM(passed),
                                         SUM(codex_result_features IS NOT NULL),
                                         SUM(attachment_content_verified),SUM(browser_dispatch_bound),
                                         SUM(browser_ui_confirmed),SUM(browser_suggestion_supplied)
                                         FROM bound_loop_rounds""").fetchone()
    summary: dict[str, Any] = {"sources": {source: {"episodes": count, "task_groups": groups,
                                                       "passing": passing or 0}
                                            for source, count, groups, passing in rows}}
    summary["codex"] = {"accepted_turns": codex[0], "thread_groups": codex[1],
                        "graded_turns": codex[2] or 0, "exact_usage_turns": codex[3] or 0,
                        "result_captured_turns": codex[4] or 0,
                        "complete_prompt_result_usage_grade_turns": codex[5] or 0,
                        "observed_single_arm_turns": codex[6] or 0,
                        "server_rerouted_turns": codex[7] or 0,
                        "accepted_user_steers": steers[0],
                        "steered_thread_groups": steers[1],
                        "steered_turns": steers[2],
                        "durable_user_messages": context_messages[0],
                        "durable_after_prior_output_messages": context_messages[1] or 0,
                        "durable_user_message_turns": context_messages[2],
                        "steer_training_role": "context_only_not_independent_outcome"}
    summary["local_prompt_experiments"] = {
        "verified_runs_with_prompt_and_result": benchmark[0],
        "suites": benchmark[1], "comparison_groups": benchmark[2],
        "distinct_prompts": benchmark[3], "product_passes": benchmark[4] or 0,
        "training_role": "separate_local_benchmark_observational_only"}
    valid_managed, managed_counts = _managed_complete_blocks()
    summary["managed_benchmarks"] = {
        "precommitted_blocks": len(managed_blocks),
        "bound_runs": sum(row[3] for row in managed_blocks),
        "complete_randomized_blocks": len(valid_managed),
        "incomplete_randomized_blocks": managed_counts.get("incomplete_blocks", 0),
        "class_unproven_blocks": managed_counts.get("class_unproven_blocks", 0),
        "class_mismatch_blocks": managed_counts.get("class_mismatch_blocks", 0),
        "invalid_randomized_blocks": managed_counts.get("invalid_blocks", 0),
        "independent_products": len({row["product_key"] for row in valid_managed}),
        "training_role": "causal_candidate_not_yet_trained"}
    summary["managed_routing_policy"] = train_managed_routing_policy(create_checkpoint=False)
    summary["managed_prompt_policy"] = train_managed_prompt_policy(create_checkpoint=False)
    summary["browser_revision_product_outcomes"] = analyze_browser_revision_product_outcomes()
    summary["bound_legacy_loop"] = {
        "rounds": loop_rounds[0], "task_groups": loop_rounds[1],
        "linked_revisions": loop_rounds[2] or 0,
        "passing_reviews": loop_rounds[3] or 0,
        "codex_result_feature_records": loop_rounds[4] or 0,
        "provider_verified_attachment_contents": loop_rounds[5] or 0,
        "browser_dispatch_bound_rounds": loop_rounds[6] or 0,
        "browser_ui_confirmed_rounds": loop_rounds[7] or 0,
        "browser_suggestion_supplied_rounds": loop_rounds[8] or 0,
        "training_role": "separate_bound_observational_not_authoritative"}
    if BOUND_LOOP_MODEL_PATH.exists():
        bound_model = _read_json(BOUND_LOOP_MODEL_PATH)
        summary["bound_loop_model"] = {key: bound_model.get(key) for key in
                                       ("rounds", "task_groups", "revision_task_groups",
                                        "codex_result_rounds", "input_stage",
                                        "baseline_brier", "model_brier",
                                        "baseline_mae_points", "model_mae_points",
                                        "prospective_holdout_rounds", "prospective_holdout_task_groups",
                                        "prospective_baseline_brier", "prospective_model_brier",
                                        "validated_for_shadow", "score_validated_for_shadow",
                                        "authoritative_for_routing")}
    summary["local_prompt_analysis"] = analyze_local_prompt_experiments()
    if LOCAL_PROMPT_MODEL_PATH.exists():
        local_model = _read_json(LOCAL_PROMPT_MODEL_PATH)
        summary["local_prompt_model"] = {key: local_model.get(key) for key in
                                         ("trained_comparisons", "independent_products",
                                          "holdout_products", "baseline_brier", "model_brier",
                                          "prospective_holdout_products", "prospective_baseline_brier",
                                          "prospective_model_brier",
                                          "execution_model_verified_for_development",
                                          "current_training_execution_verified",
                                          "validated_for_shadow", "authoritative_for_routing",
                                          "browser_grades_included", "training_randomized",
                                          "causal_prompt_comparison")}
    summary["iterations"] = {"linked_rounds": traces[0], "task_groups": traces[1],
                             "revisions_with_parent_feedback": traces[2] or 0,
                             "adopted_browser_suggestions": traces[3] or 0,
                             "adopted_explicit_inline_revisions": traces[4] or 0,
                             "context_complete_revision_rounds": traces[5] or 0,
                             "context_complete_revision_task_groups": traces[6] or 0}
    summary["standalone_codex"] = {"completed_prompt_result_pairs": sessions[0],
                                   "thread_groups": sessions[1],
                                   "reported_usage_pairs": sessions[2] or 0,
                                   "complete_explicit_grade_exact_usage": sessions[3] or 0,
                                   "single_pre_inference_prompt_pairs": sessions[4] or 0,
                                   "observed_single_arm_turns": sessions[5] or 0,
                                   "training_role": "retrospective_observational_only"}
    summary["browser_codex_links"] = {"exact_prompt_result_review_links": cross_source[0] or 0,
                                      "managed": cross_source[1] or 0,
                                      "completed_session": cross_source[2] or 0,
                                      "ambiguous": cross_source[3] or 0,
                                      "unlinked_review_traces": unlinked_reviews[0],
                                      "unlinked_prompt_seen_managed": unlinked_reviews[1] or 0,
                                      "unlinked_prompt_seen_completed_session": unlinked_reviews[2] or 0,
                                      "unlinked_prompt_absent_both": unlinked_reviews[3] or 0,
                                      "label_role": "browser_review_only"}
    summary["browser_results"] = {"nonce_bound_feature_records": browser_result_count,
                                   "storage": "private_keyed_features_only"}
    summary["recovery_observations"] = {"ungraded_failed_response_records": recovered[0],
                                        "attachment_ui_confirmed_records": recovered[1] or 0,
                                        "training_role": "separate_observational_only"}
    if MODEL_PATH.exists():
        model = _read_json(MODEL_PATH)
        summary["review_model"] = {key: model.get(key) for key in
                                   ("trained_episodes", "task_groups", "task_group_policy",
                                    "holdout_episodes",
                                    "baseline_brier", "model_brier",
                                    "retrospective_holdout_is_diagnostic_only",
                                    "prospective_checkpoint_created_at",
                                    "prospective_holdout_episodes", "prospective_holdout_task_groups",
                                    "prospective_baseline_brier", "prospective_model_brier",
                                    "validated_for_shadow")}
    if SCORE_MODEL_PATH.exists():
        model = _read_json(SCORE_MODEL_PATH)
        summary["review_score_model"] = {key: model.get(key) for key in
                                          ("trained_episodes", "task_groups", "holdout_episodes",
                                           "baseline_mae_points", "model_mae_points",
                                           "retrospective_holdout_is_diagnostic_only",
                                           "prospective_checkpoint_created_at",
                                           "prospective_holdout_episodes", "prospective_holdout_task_groups",
                                           "prospective_baseline_mae_points", "prospective_model_mae_points",
                                           "validated_for_shadow", "causal_prompt_comparison")}
    if CODEX_MODEL_PATH.exists():
        model = _read_json(CODEX_MODEL_PATH)
        summary["codex_model"] = {key: model.get(key) for key in
                                  ("trained_turns", "task_groups", "holdout_turns",
                                   "baseline_brier", "model_brier",
                                   "retrospective_holdout_is_diagnostic_only",
                                   "prospective_checkpoint_created_at",
                                   "prospective_holdout_turns", "prospective_holdout_task_groups",
                                   "prospective_comparable_arms",
                                   "prospective_baseline_brier", "prospective_model_brier",
                                   "validated_for_shadow",
                                   "causal_model_comparison")}
    if ITERATION_MODEL_PATH.exists():
        model = _read_json(ITERATION_MODEL_PATH)
        summary["iteration_model"] = {key: model.get(key) for key in
                                      ("trained_rounds", "task_groups", "holdout_rounds",
                                       "context_complete_revision_rounds",
                                      "context_complete_revision_task_groups",
                                       "linked_parent_codex_results",
                                       "linked_parent_codex_arms",
                                       "linked_parent_browser_results",
                                       "minimum_revision_task_groups",
                                       "training_context_complete_revision_task_groups",
                                       "holdout_context_complete_revision_task_groups",
                                       "baseline_brier", "model_brier",
                                       "retrospective_holdout_is_diagnostic_only",
                                       "prospective_checkpoint_created_at",
                                       "prospective_holdout_rounds", "prospective_holdout_task_groups",
                                       "prospective_context_complete_revision_task_groups",
                                       "prospective_baseline_brier", "prospective_model_brier",
                                       "validated_for_shadow",
                                       "causal_prompt_comparison")}
    if IMPROVEMENT_MODEL_PATH.exists():
        model = _read_json(IMPROVEMENT_MODEL_PATH)
        summary["improvement_model"] = {key: model.get(key) for key in
                                        ("scored_revisions", "task_groups", "holdout_revisions",
                                         "linked_parent_codex_results",
                                         "linked_parent_codex_arms",
                                         "linked_parent_browser_results",
                                         "baseline_brier", "model_brier",
                                         "retrospective_holdout_is_diagnostic_only",
                                         "prospective_checkpoint_created_at",
                                         "prospective_holdout_revisions", "prospective_holdout_task_groups",
                                         "prospective_baseline_brier", "prospective_model_brier",
                                         "validated_for_shadow", "causal_prompt_comparison")}
    summary["forward_readiness"] = forward_readiness(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Local ModelLabs outcome learner")
    parser.add_argument("action", choices=("update", "import-auracall", "import-codex-sessions",
                                           "import-bound-loops", "import-recovery-observations",
                                           "export-codex-result", "export-codex-turn",
                                           "sync-codex", "sync-session-grades", "train", "train-score", "train-codex",
                                           "train-iteration", "train-improvement", "train-local-prompt",
                                           "train-bound-loops", "train-managed-routing",
                                           "train-managed-prompt",
                                           "status", "capabilities", "predict-review", "predict-codex",
                                           "predict-iteration", "predict-local-prompt"))
    parser.add_argument("--guard-root", type=Path)
    parser.add_argument("--loop-root", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--guard-id")
    parser.add_argument("--sessions-root", type=Path)
    parser.add_argument("--thread-id")
    parser.add_argument("--turn-id")
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--goal-file", type=Path)
    parser.add_argument("--revision-file", type=Path)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--baseline-prompt-file", type=Path)
    parser.add_argument("--candidate-prompt-file", type=Path)
    parser.add_argument("--task-class", choices=TASK_CLASSES)
    parser.add_argument("--origin-prompt-file", type=Path)
    parser.add_argument("--generation-prompt-file", type=Path)
    parser.add_argument("--parent-feedback-file", type=Path)
    parser.add_argument("--parent-result-file", type=Path)
    parser.add_argument("--parent-codex-result-file", type=Path)
    parser.add_argument("--parent-browser-result-file", type=Path)
    parser.add_argument("--parent-codex-model", choices=MODEL_IDS)
    parser.add_argument("--parent-codex-effort", choices=EFFORTS)
    parser.add_argument("--parent-quality-score", type=int)
    parser.add_argument("--parent-suggestion-file", type=Path)
    parser.add_argument("--prompt-author", choices=("user", "chatgpt", "codex", "mixed"))
    args = parser.parse_args()
    if args.action == "capabilities":
        print(json.dumps({"schema": "modellabs.learning_capabilities.v1",
                          "guard_review_formats": ["inline", "codex.pro_guard_file_handoff.v1"],
                          "guard_reference_files": "codex.pro_guard_reference_files.v1"}))
        return
    if args.action == "update":
        sessions = import_codex_sessions(args.sessions_root, args.thread_id)
        session_grades = sync_session_grades()
        # Exact completed Codex pairs must exist before AuraCall link matching.
        backfilled = backfill_managed_turns()
        synced = sync_codex_grades()
        imported = import_auracall(args.guard_root, args.runs_root, args.guard_id)
        bound_loops = import_bound_loops(args.loop_root, args.runs_root)
        recovery_observations = import_recovery_observations(args.runs_root)
        result = {"auracall": imported, "sessions": sessions,
                  "bound_loops": bound_loops,
                  "recovery_observations": recovery_observations,
                  "session_grades": session_grades, "managed_backfill": backfilled, "codex": synced,
                  "local_prompt_analysis": analyze_local_prompt_experiments(),
                  "browser_revision_product_outcomes": analyze_browser_revision_product_outcomes(),
                  "local_prompt_model": train_local_prompt_model(),
                  "managed_routing_policy": train_managed_routing_policy(),
                  "managed_prompt_policy": train_managed_prompt_policy(),
                  "review_model": train_review_model(), "review_score_model": train_review_score_model(),
                  "iteration_model": train_iteration_model(),
                  "improvement_model": train_improvement_model(),
                  "bound_loop_model": train_bound_loop_model(),
                  "codex_model": train_codex_model()}
    elif args.action == "import-auracall":
        result = import_auracall(args.guard_root, args.runs_root, args.guard_id)
    elif args.action == "import-bound-loops":
        result = import_bound_loops(args.loop_root, args.runs_root)
    elif args.action == "import-recovery-observations":
        result = import_recovery_observations(args.runs_root)
    elif args.action == "train-bound-loops":
        result = train_bound_loop_model()
    elif args.action == "train-managed-routing":
        result = train_managed_routing_policy()
    elif args.action == "train-managed-prompt":
        result = train_managed_prompt_policy()
    elif args.action == "import-codex-sessions":
        result = import_codex_sessions(args.sessions_root, args.thread_id)
    elif args.action == "export-codex-result":
        if not args.thread_id or not args.turn_id or not args.generation_prompt_file or not args.output_file:
            parser.error("export-codex-result requires thread, turn, generation prompt, and output file")
        result = export_codex_result(args.thread_id, args.turn_id, args.generation_prompt_file,
                                     args.output_file, args.sessions_root)
    elif args.action == "export-codex-turn":
        if not args.thread_id or not args.turn_id or not args.output_dir:
            parser.error("export-codex-turn requires thread, turn, and output directory")
        result = export_codex_turn(args.thread_id, args.turn_id, args.output_dir, args.sessions_root)
    elif args.action == "sync-session-grades":
        result = sync_session_grades()
    elif args.action == "sync-codex":
        result = sync_codex_grades()
    elif args.action == "train":
        result = train_review_model()
    elif args.action == "train-score":
        result = train_review_score_model()
    elif args.action == "train-codex":
        result = train_codex_model()
    elif args.action == "train-iteration":
        result = train_iteration_model()
    elif args.action == "train-improvement":
        result = train_improvement_model()
    elif args.action == "train-local-prompt":
        result = train_local_prompt_model()
    elif args.action == "predict-review":
        if not args.goal_file or not args.revision_file:
            parser.error("predict-review requires --goal-file and --revision-file")
        goal = args.goal_file.read_text(encoding="utf-8")
        revision = args.revision_file.read_text(encoding="utf-8")
        result = predict_review(goal, revision, args.round)
    elif args.action == "predict-codex":
        if not args.prompt_file or not args.task_class:
            parser.error("predict-codex requires --prompt-file and --task-class")
        result = predict_codex(args.prompt_file.read_text(encoding="utf-8"), args.task_class)
    elif args.action == "predict-local-prompt":
        if not args.baseline_prompt_file or not args.candidate_prompt_file:
            parser.error("predict-local-prompt requires baseline and candidate prompt files")
        result = predict_local_prompt(args.baseline_prompt_file.read_text(encoding="utf-8"),
                                      args.candidate_prompt_file.read_text(encoding="utf-8"))
    elif args.action == "predict-iteration":
        if not args.origin_prompt_file or not args.generation_prompt_file or not args.prompt_author:
            parser.error("predict-iteration requires origin prompt, generation prompt, and author")
        feedback = args.parent_feedback_file.read_text(encoding="utf-8") if args.parent_feedback_file else ""
        parent_result = args.parent_result_file.read_text(encoding="utf-8") if args.parent_result_file else ""
        parent_codex_result = (args.parent_codex_result_file.read_text(encoding="utf-8")
                               if args.parent_codex_result_file else "")
        parent_browser_result = (args.parent_browser_result_file.read_text(encoding="utf-8")
                                 if args.parent_browser_result_file else "")
        parent_suggestion = (args.parent_suggestion_file.read_text(encoding="utf-8")
                             if args.parent_suggestion_file else "")
        result = predict_iteration(args.origin_prompt_file.read_text(encoding="utf-8"),
                                   args.generation_prompt_file.read_text(encoding="utf-8"),
                                   args.prompt_author, args.round, feedback,
                                   parent_result, args.parent_quality_score, parent_suggestion,
                                   parent_codex_result, args.parent_codex_model,
                                   args.parent_codex_effort, parent_browser_result)
    else:
        result = status()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
