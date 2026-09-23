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
CODEX_MODEL_PATH = LEARNING_ROOT / "codex-model.json"
ITERATION_MODEL_PATH = LEARNING_ROOT / "iteration-model-v5.json"
IMPROVEMENT_MODEL_PATH = LEARNING_ROOT / "revision-improvement-model-v2.json"
REVIEW_EVAL_PATH = LEARNING_ROOT / "review-evaluation-checkpoint.json"
ITERATION_EVAL_PATH = LEARNING_ROOT / "iteration-evaluation-checkpoint-v5.json"
IMPROVEMENT_EVAL_PATH = LEARNING_ROOT / "revision-improvement-evaluation-checkpoint-v2.json"
CODEX_EVAL_PATH = LEARNING_ROOT / "codex-evaluation-checkpoint.json"
FEATURE_COUNT = 512
FEATURE_VERSION = 1
ITERATION_FEATURE_VERSION = 4
MIN_ITERATION_REVISION_GROUPS = 8
MODEL_IDS = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
             "gpt-6-luna", "gpt-6-sol", "gpt-6-astra")
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max", "ultra")
TASK_CLASSES = ("simple", "routine", "difficult", "consequential")
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_SESSION_BYTES = 64 * 1024 * 1024
MIN_REVIEW_HOLDOUT = 20
MIN_REVIEW_BRIER_GAIN = 0.005
MIN_REVIEW_RELATIVE_GAIN = 0.05
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
                             ("input_scope", "TEXT NOT NULL DEFAULT 'legacy_unverified'")):
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
    return {name: counts[name] for name in ("imported", "unchanged", "conflict", "turn_ineligible", "file_skipped")}


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
                    "route_accepted", "turn_completed", "turn_usage", "quality_grade"}:
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


def sync_codex_grades(metrics_path: Path | None = None) -> dict[str, int]:
    """Join exact terminal usage and explicit grades onto accepted turn features."""
    from telemetry import METRICS_PATH

    path = metrics_path or METRICS_PATH
    key = _key()
    counts = Counter()
    if not path.exists():
        return {"graded": 0, "usage_matched": 0}
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
    return {"graded": counts["graded"], "usage_matched": counts["usage_matched"]}


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
    instructions, revision = inputs.get("instructions"), inputs.get("requestInput")
    if not isinstance(instructions, str) or not isinstance(revision, str):
        return None
    start = instructions.find(GOAL_START)
    end = instructions.find(GOAL_END, start + len(GOAL_START))
    if start < 0 or end < 0 or end <= start + len(GOAL_START):
        return None
    goal = instructions[start + len(GOAL_START):end].strip()
    if not goal or not revision.strip():
        return None
    submitted_at = state.get("submitted_at")
    if not isinstance(submitted_at, str):
        return None
    datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
    context_digest = _digest(key, goal)
    return (_digest(key, response_id), context_digest, "auracall_pro_guard", submitted_at,
            round_number, "review_goal", context_digest, _digest(key, revision),
            json.dumps(_features(goal, revision, round_number, key), sort_keys=True),
            str(inputs.get("model") or ""), None, None,
            evaluation["score"], int(evaluation["passed"]), "nonce_bound_pro_guard",
            response_id, guard_id, FEATURE_VERSION)


def _bound_text(source: Any) -> str:
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
    value = content.decode("utf-8").strip()
    if not value:
        raise ValueError("bound prompt source is empty")
    return value


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
        parent_artifact = (((parent_record.get("bundle") or {}).get("run") or {})
                           .get("initialInputs") or {}).get("requestInput")
        if not isinstance(parent_artifact, str) or not parent_artifact.strip():
            raise ValueError("parent review artifact is missing")
        # The legacy column name says "result", but this is reviewer input,
        # not a verified Codex final answer. Exact answers are linked separately.
        parent_result_features = json.dumps(_features(parent_artifact, "", 1, key), sort_keys=True)
        parent_quality_score = parent_row[12]
        parent_suggestion = (parent.get("verdict") or {}).get("suggested_next_prompt")
        if isinstance(parent_suggestion, str) and parent_suggestion.strip() == generation:
            adopted_parent_suggestion = 1
    elif root_guard_id != state.get("guard_id") or state.get("round") != 1:
        raise ValueError("invalid first trace round")
    feedback = _verdict_feedback(state)
    suggestion = (state.get("verdict") or {}).get("suggested_next_prompt")
    if suggestion is not None and (not isinstance(suggestion, str) or len(suggestion) > 12000):
        raise ValueError("invalid browser suggested prompt")
    return (row[0], _digest(key, root_guard_id), parent_episode_id,
            _digest(key, origin), _digest(key, generation), trace["prompt_author"],
            json.dumps(_features(origin, generation, state["round"], key), sort_keys=True),
            parent_feedback, json.dumps(_features(feedback, "", 1, key), sort_keys=True),
            row[3], row[13], _digest(key, suggestion) if isinstance(suggestion, str) and suggestion else None,
            adopted_parent_suggestion, parent_result_features, parent_quality_score)


def _link_review_to_codex(connection: sqlite3.Connection, episode_id: str,
                          generation_digest: str, artifact_digest: str) -> str:
    """Link a browser grade only to one exact prompt-and-final-answer pair."""
    live = connection.execute("""SELECT thread_key,turn_key FROM codex_turns
                                 WHERE prompt_digest=? AND result_digest IS NOT NULL
                                 AND result_digest=?""",
                              (generation_digest, artifact_digest)).fetchall()
    completed_sessions = connection.execute("""SELECT thread_key,turn_key FROM session_turns
                                               WHERE prompt_digest=? AND result_digest=?
                                               AND input_scope='single_pre_inference_message'""",
                                            (generation_digest, artifact_digest)).fetchall()
    identities = set(live) | set(completed_sessions)
    existing = connection.execute("""SELECT thread_key,turn_key,link_status FROM review_codex_links
                                     WHERE episode_id=?""", (episode_id,)).fetchone()
    if not identities:
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
            existing = connection.execute("SELECT features,quality_score,passed FROM episodes WHERE episode_id=?", (row[0],)).fetchone()
            if existing:
                if existing != (row[8], row[12], row[13]):
                    counts["conflict"] += 1
                    continue
                else:
                    counts["unchanged"] += 1
            else:
                connection.execute("INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
                counts["imported"] += 1
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
                    "SELECT root_key,parent_episode_id,origin_prompt_digest,generation_prompt_digest,prompt_author,prompt_features,parent_feedback_features,result_feedback_features,submitted_at,passed,suggested_prompt_digest,adopted_parent_suggestion,parent_result_features,parent_quality_score FROM iteration_traces WHERE episode_id=?",
                    (trace[0],)).fetchone()
                if existing_trace:
                    if existing_trace[:12] != trace[1:13] or any(
                            old is not None and old != new
                            for old, new in zip(existing_trace[12:], trace[13:])):
                        counts["trace_conflict"] += 1
                        continue
                    if existing_trace[12:] != trace[13:]:
                        connection.execute("""UPDATE iteration_traces
                                           SET parent_result_features=?,parent_quality_score=?
                                           WHERE episode_id=?""", (*trace[13:], trace[0]))
                    counts["trace_unchanged"] += 1
                else:
                    connection.execute("INSERT INTO iteration_traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", trace)
                    counts["trace_imported"] += 1
                link_result = _link_review_to_codex(connection, row[0], trace[4], row[7])
                counts[f"codex_link_{link_result}"] += 1
                if link_result == "conflict":
                    counts["conflict"] += 1
    return {name: counts[name] for name in ("imported", "unchanged", "ineligible", "rejected", "conflict",
                                            "trace_imported", "trace_unchanged", "trace_missing",
                                            "trace_rejected", "trace_conflict",
                                            "codex_link_imported", "codex_link_unchanged",
                                            "codex_link_no_match", "codex_link_ambiguous", "codex_link_conflict")}


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
                                   effort,task_class,quality_score,verification,total_tokens
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
                                         model,effort,task_class,quality_score,verification,exact_total_tokens
                                         FROM session_turns WHERE grade_status='explicit'
                                         AND input_scope='single_pre_inference_message'
                                         AND quality_score IS NOT NULL AND verification IS NOT NULL
                                         AND exact_total_tokens IS NOT NULL""").fetchall()
    for thread_key, turn_key, completed_at, features, model, effort, task_class, score, verification, tokens in historical:
        if (thread_key, turn_key) in live_identities:
            continue
        try:
            observed_at_ms = int(datetime.fromisoformat(completed_at.replace("Z", "+00:00")).timestamp() * 1000)
        except (ValueError, OverflowError):
            continue
        raw.append((thread_key, observed_at_ms, features, model, effort, task_class,
                    score, verification, tokens))
    raw = [row for row in raw if row[3] in MODEL_IDS and row[4] in EFFORTS
           and row[5] in TASK_CLASSES]
    arms = Counter((row[5], row[3], row[4]) for row in raw)
    supported = {arm for arm, count in arms.items() if count >= 10}
    comparable_classes = {task_class for task_class in TASK_CLASSES
                          if sum(arm[0] == task_class for arm in supported) >= 2}
    selected = [row for row in raw if (row[5], row[3], row[4]) in supported
                and row[5] in comparable_classes]
    groups = sorted({row[0] for row in selected},
                    key=lambda group: min(row[1] for row in selected if row[0] == group))
    if len(selected) < 40 or len(groups) < 8:
        return {"status": "insufficient_comparable_outcomes", "graded_with_usage": len(raw),
                "supported_arms": len(supported), "comparable_task_classes": len(comparable_classes)}
    mapped = [(row[0], row[1], _codex_features(json.loads(row[2]), row[3], row[4], row[5]),
               int(row[7] == "passed" and row[6] >= 90)) for row in selected]
    holdout_count = max(2, math.ceil(len(groups) * 0.2))
    train_groups, test_groups = set(groups[:-holdout_count]), set(groups[-holdout_count:])
    training = [(vector, label) for group, _at, vector, label in mapped if group in train_groups]
    test = [(vector, label) for group, _at, vector, label in mapped if group in test_groups]
    if len({label for _vector, label in training}) < 2 or len(test) < 8:
        return {"status": "insufficient_holdout_diversity", "graded_with_usage": len(raw)}
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
    first_any = {group: min(row[1] for row in raw if row[0] == group) for group in future_groups}
    future_groups = [group for group in future_groups if first_any[group] > cutoff_ms]
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


def train_review_model() -> dict[str, Any]:
    with _connect() as connection:
        raw = connection.execute("""SELECT e.group_id,t.root_key,e.submitted_at,e.features,e.passed
                                    FROM episodes e LEFT JOIN iteration_traces t
                                    ON t.episode_id=e.episode_id
                                    WHERE e.source='auracall_pro_guard'
                                    AND e.verifier='nonce_bound_pro_guard'
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

    for goal_key, root_key, _at, _features_json, _passed in raw:
        goal_node = f"goal:{goal_key}"
        if root_key is not None:
            root_node = f"root:{root_key}"
            parents[find(root_node)] = find(goal_node)
    component_keys: dict[str, str] = {}
    for goal_key, _root_key, _at, _features_json, _passed in raw:
        component = find(f"goal:{goal_key}")
        component_keys[component] = min(component_keys.get(component, goal_key), goal_key)
    grouped: dict[str, list[tuple[str, dict[str, float], int]]] = {}
    for goal_key, _root_key, submitted_at, features, passed in raw:
        group_id = component_keys[find(f"goal:{goal_key}")]
        grouped.setdefault(group_id, []).append((submitted_at, json.loads(features), passed))
    groups = sorted(grouped, key=lambda item: min(row[0] for row in grouped[item]))
    if len(raw) < 20 or len(groups) < 6:
        return {"status": "insufficient_data", "episodes": len(raw), "task_groups": len(groups)}
    holdout_count = max(2, math.ceil(len(groups) * 0.2))
    train_groups, test_groups = groups[:-holdout_count], groups[-holdout_count:]
    training = [(vector, label) for group in train_groups for _at, vector, label in grouped[group]]
    test = [(vector, label) for group in test_groups for _at, vector, label in grouped[group]]
    if len({label for _vector, label in training}) < 2:
        return {"status": "insufficient_label_diversity", "episodes": len(raw), "task_groups": len(groups)}
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
                "trained_episodes": len(raw), "task_groups": len(groups),
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


def predict_review(goal: str, revision: str, round_number: int) -> dict[str, Any]:
    model = _read_json(MODEL_PATH)
    if model.get("schema") != "modellabs.review_outcome_model.v1" or model.get("feature_version") != FEATURE_VERSION:
        raise ValueError("review model version mismatch")
    vector = _features(goal, revision, round_number, _key())
    probability = _sigmoid(model["bias"] + sum(model["weights"][int(index)] * value
                                                   for index, value in vector.items()
                                                   if int(index) < len(model["weights"])))
    return {"predicted_review_pass_probability": round(probability, 4),
            "validated_for_shadow": model["validated_for_shadow"],
            "training_episodes": model["trained_episodes"]}


def _iteration_vector(prompt_features: dict[str, float],
                      feedback_features: dict[str, float] | None,
                      author: str,
                      parent_result_features: dict[str, float] | None = None,
                      parent_quality_score: int | None = None,
                      adopted_parent_suggestion: bool = False,
                      parent_codex_final_features: dict[str, float] | None = None) -> dict[str, float]:
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
    return vector


def _linked_codex_result_features(connection: sqlite3.Connection) -> dict[str, dict[str, float]]:
    """Return only final answers from valid exact browser-to-Codex links."""
    rows = connection.execute("""SELECT l.episode_id,
                                CASE WHEN l.source_kind='managed' THEN c.result_features
                                     WHEN l.source_kind='completed_session' THEN s.result_features END
                                FROM review_codex_links l
                                LEFT JOIN codex_turns c ON l.source_kind='managed'
                                    AND c.thread_key=l.thread_key AND c.turn_key=l.turn_key
                                LEFT JOIN session_turns s ON l.source_kind='completed_session'
                                    AND s.thread_key=l.thread_key AND s.turn_key=l.turn_key
                                WHERE l.link_status='valid'""").fetchall()
    return {episode_id: json.loads(features) for episode_id, features in rows if features}


def train_iteration_model() -> dict[str, Any]:
    """Learn prompt-revision outcomes only from explicitly linked review rounds."""
    with _connect() as connection:
        linked_results = _linked_codex_result_features(connection)
        raw = connection.execute("""SELECT root_key,submitted_at,prompt_features,
                                   parent_feedback_features,prompt_author,passed,
                                   parent_result_features,parent_quality_score,
                                   adopted_parent_suggestion,parent_episode_id
                                   FROM iteration_traces ORDER BY submitted_at,episode_id""").fetchall()
    groups: dict[str, list[tuple[str, dict[str, float], int]]] = {}
    revision_groups: set[str] = set()
    revision_rounds = 0
    linked_parent_codex_results = 0
    for root_key, submitted_at, prompt, feedback, author, passed, parent_result, parent_score, adopted, parent_id in raw:
        if parent_id is not None and parent_result is not None and parent_score is not None:
            revision_groups.add(root_key)
            revision_rounds += 1
        codex_result = linked_results.get(parent_id)
        if codex_result is not None:
            linked_parent_codex_results += 1
        vector = _iteration_vector(json.loads(prompt), json.loads(feedback) if feedback else None,
                                   author, json.loads(parent_result) if parent_result else None,
                                   parent_score, bool(adopted), codex_result)
        groups.setdefault(root_key, []).append((submitted_at, vector, passed))
    ordered = sorted(groups, key=lambda root: min(row[0] for row in groups[root]))
    if len(raw) < 30 or len(groups) < 10:
        return {"status": "insufficient_linked_rounds", "rounds": len(raw), "task_groups": len(groups),
                "context_complete_revision_rounds": revision_rounds,
                "context_complete_revision_task_groups": len(revision_groups),
                "linked_parent_codex_results": linked_parent_codex_results}
    if len(revision_groups) < MIN_ITERATION_REVISION_GROUPS:
        return {"status": "insufficient_context_complete_revisions", "rounds": len(raw),
                "task_groups": len(groups), "context_complete_revision_rounds": revision_rounds,
                "context_complete_revision_task_groups": len(revision_groups),
                "linked_parent_codex_results": linked_parent_codex_results,
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
                                        "modellabs.iteration_evaluation_checkpoint.v5", ordered, groups,
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
    artifact = {"schema": "modellabs.iteration_outcome_model.v5", "feature_version": ITERATION_FEATURE_VERSION,
                "trained_rounds": len(raw), "task_groups": len(groups), "holdout_rounds": len(test),
                "context_complete_revision_rounds": revision_rounds,
                "context_complete_revision_task_groups": len(revision_groups),
                "linked_parent_codex_results": linked_parent_codex_results,
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
        linked_results = _linked_codex_result_features(connection)
        raw = connection.execute("""SELECT t.root_key,
                                   (SELECT MIN(first.submitted_at) FROM iteration_traces first
                                    WHERE first.root_key=t.root_key),t.prompt_features,
                                   t.parent_feedback_features,t.prompt_author,
                                   t.parent_result_features,t.parent_quality_score,
                                   t.adopted_parent_suggestion,e.quality_score,t.parent_episode_id
                                   FROM iteration_traces t JOIN episodes e ON e.episode_id=t.episode_id
                                   WHERE t.parent_episode_id IS NOT NULL
                                   AND t.parent_result_features IS NOT NULL
                                   AND t.parent_quality_score IS NOT NULL
                                   AND e.quality_score IS NOT NULL
                                   ORDER BY t.submitted_at,t.episode_id""").fetchall()
    groups: dict[str, list[tuple[str, dict[str, float], int]]] = {}
    linked_parent_codex_results = 0
    for root, submitted_at, prompt, feedback, author, parent_result, parent_score, adopted, score, parent_id in raw:
        if not (0 <= parent_score <= 100 and 0 <= score <= 100):
            continue
        codex_result = linked_results.get(parent_id)
        if codex_result is not None:
            linked_parent_codex_results += 1
        vector = _iteration_vector(json.loads(prompt), json.loads(feedback) if feedback else None,
                                   author, json.loads(parent_result), parent_score, bool(adopted), codex_result)
        groups.setdefault(root, []).append((submitted_at, vector, int(score > parent_score)))
    ordered = sorted(groups, key=lambda root: min(row[0] for row in groups[root]))
    revisions = sum(len(rows) for rows in groups.values())
    if revisions < 20 or len(groups) < 10:
        return {"status": "insufficient_scored_revisions", "scored_revisions": revisions,
                "task_groups": len(groups),
                "linked_parent_codex_results": linked_parent_codex_results}
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
                                        "modellabs.revision_improvement_evaluation_checkpoint.v2",
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
    artifact = {"schema": "modellabs.revision_improvement_model.v2",
                "feature_version": ITERATION_FEATURE_VERSION,
                "scored_revisions": revisions, "task_groups": len(groups),
                "linked_parent_codex_results": linked_parent_codex_results,
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
                      parent_suggestion: str = "", parent_codex_result: str = "") -> dict[str, Any]:
    model = _read_json(ITERATION_MODEL_PATH)
    if (model.get("schema") != "modellabs.iteration_outcome_model.v5"
            or model.get("feature_version") != ITERATION_FEATURE_VERSION):
        raise ValueError("iteration model version mismatch")
    if (parent_result or parent_quality_score is not None or parent_suggestion
            or parent_codex_result) and round_number < 2:
        raise ValueError("first-round prediction cannot include parent evidence")
    adopted = bool(parent_suggestion.strip()) and parent_suggestion.strip() == generation_prompt.strip()
    key = _key()
    vector = _iteration_vector(_features(origin_prompt, generation_prompt, round_number, key),
                               _features(parent_feedback, "", 1, key) if parent_feedback else None,
                               author, _features(parent_result, "", 1, key) if parent_result else None,
                               parent_quality_score, adopted,
                               _features(parent_codex_result, "", 1, key) if parent_codex_result else None)
    probability = _sigmoid(model["bias"] + sum(model["weights"][int(index)] * value
                                                   for index, value in vector.items()
                                                   if int(index) < len(model["weights"])))
    improvement_probability = None
    improvement_validated = False
    if (IMPROVEMENT_MODEL_PATH.exists() and round_number >= 2
            and parent_result and parent_quality_score is not None):
        improvement = _read_json(IMPROVEMENT_MODEL_PATH)
        if (improvement.get("schema") != "modellabs.revision_improvement_model.v2"
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
            "parent_quality_score_supplied": parent_quality_score is not None,
            "adopted_parent_suggestion": adopted}


def status() -> dict[str, Any]:
    with _connect() as connection:
        rows = connection.execute("SELECT source,COUNT(*),COUNT(DISTINCT group_id),SUM(passed) FROM episodes GROUP BY source").fetchall()
        codex = connection.execute("""SELECT COUNT(*),COUNT(DISTINCT group_id),
                                     SUM(quality_score IS NOT NULL),SUM(total_tokens IS NOT NULL),
                                     SUM(result_digest IS NOT NULL),
                                     SUM(quality_score IS NOT NULL AND total_tokens IS NOT NULL
                                         AND result_digest IS NOT NULL)
                                     FROM codex_turns""").fetchone()
        traces = connection.execute("""SELECT COUNT(*),COUNT(DISTINCT root_key),
                                     SUM(parent_episode_id IS NOT NULL),
                                     SUM(adopted_parent_suggestion),
                                     SUM(parent_episode_id IS NOT NULL AND parent_result_features IS NOT NULL
                                         AND parent_quality_score IS NOT NULL),
                                     COUNT(DISTINCT CASE WHEN parent_episode_id IS NOT NULL
                                         AND parent_result_features IS NOT NULL
                                         AND parent_quality_score IS NOT NULL THEN root_key END)
                                     FROM iteration_traces""").fetchone()
        sessions = connection.execute("""SELECT COUNT(*),COUNT(DISTINCT thread_key),
                                       SUM(reported_total_tokens IS NOT NULL),
                                       SUM(grade_status='explicit' AND exact_total_tokens IS NOT NULL),
                                       SUM(input_scope='single_pre_inference_message')
                                       FROM session_turns""").fetchone()
        cross_source = connection.execute("""SELECT COUNT(*),SUM(source_kind='managed' AND link_status='valid'),
                                           SUM(source_kind='completed_session' AND link_status='valid'),
                                           SUM(link_status='ambiguous')
                                           FROM review_codex_links""").fetchone()
    summary: dict[str, Any] = {"sources": {source: {"episodes": count, "task_groups": groups,
                                                       "passing": passing or 0}
                                            for source, count, groups, passing in rows}}
    summary["codex"] = {"accepted_turns": codex[0], "thread_groups": codex[1],
                        "graded_turns": codex[2] or 0, "exact_usage_turns": codex[3] or 0,
                        "result_captured_turns": codex[4] or 0,
                        "complete_prompt_result_usage_grade_turns": codex[5] or 0}
    summary["iterations"] = {"linked_rounds": traces[0], "task_groups": traces[1],
                             "revisions_with_parent_feedback": traces[2] or 0,
                             "adopted_browser_suggestions": traces[3] or 0,
                             "context_complete_revision_rounds": traces[4] or 0,
                             "context_complete_revision_task_groups": traces[5] or 0}
    summary["standalone_codex"] = {"completed_prompt_result_pairs": sessions[0],
                                   "thread_groups": sessions[1],
                                   "reported_usage_pairs": sessions[2] or 0,
                                   "complete_explicit_grade_exact_usage": sessions[3] or 0,
                                   "single_pre_inference_prompt_pairs": sessions[4] or 0,
                                   "training_role": "retrospective_observational_only"}
    summary["browser_codex_links"] = {"exact_prompt_result_review_links": cross_source[0] - (cross_source[3] or 0),
                                      "managed": cross_source[1] or 0,
                                      "completed_session": cross_source[2] or 0,
                                      "ambiguous": cross_source[3] or 0,
                                      "label_role": "browser_review_only"}
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
                                         "baseline_brier", "model_brier",
                                         "retrospective_holdout_is_diagnostic_only",
                                         "prospective_checkpoint_created_at",
                                         "prospective_holdout_revisions", "prospective_holdout_task_groups",
                                         "prospective_baseline_brier", "prospective_model_brier",
                                         "validated_for_shadow", "causal_prompt_comparison")}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Local ModelLabs outcome learner")
    parser.add_argument("action", choices=("update", "import-auracall", "import-codex-sessions",
                                           "sync-codex", "sync-session-grades", "train", "train-codex",
                                           "train-iteration", "train-improvement", "status", "predict-review", "predict-codex",
                                           "predict-iteration"))
    parser.add_argument("--guard-root", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--guard-id")
    parser.add_argument("--sessions-root", type=Path)
    parser.add_argument("--thread-id")
    parser.add_argument("--goal-file", type=Path)
    parser.add_argument("--revision-file", type=Path)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--task-class", choices=TASK_CLASSES)
    parser.add_argument("--origin-prompt-file", type=Path)
    parser.add_argument("--generation-prompt-file", type=Path)
    parser.add_argument("--parent-feedback-file", type=Path)
    parser.add_argument("--parent-result-file", type=Path)
    parser.add_argument("--parent-codex-result-file", type=Path)
    parser.add_argument("--parent-quality-score", type=int)
    parser.add_argument("--parent-suggestion-file", type=Path)
    parser.add_argument("--prompt-author", choices=("user", "chatgpt", "codex", "mixed"))
    args = parser.parse_args()
    if args.action == "update":
        sessions = import_codex_sessions(args.sessions_root, args.thread_id)
        imported = import_auracall(args.guard_root, args.runs_root, args.guard_id)
        session_grades = sync_session_grades()
        synced = sync_codex_grades()
        result = {"auracall": imported, "sessions": sessions,
                  "session_grades": session_grades, "codex": synced,
                  "review_model": train_review_model(), "iteration_model": train_iteration_model(),
                  "improvement_model": train_improvement_model(),
                  "codex_model": train_codex_model()}
    elif args.action == "import-auracall":
        result = import_auracall(args.guard_root, args.runs_root, args.guard_id)
    elif args.action == "import-codex-sessions":
        result = import_codex_sessions(args.sessions_root, args.thread_id)
    elif args.action == "sync-session-grades":
        result = sync_session_grades()
    elif args.action == "sync-codex":
        result = sync_codex_grades()
    elif args.action == "train":
        result = train_review_model()
    elif args.action == "train-codex":
        result = train_codex_model()
    elif args.action == "train-iteration":
        result = train_iteration_model()
    elif args.action == "train-improvement":
        result = train_improvement_model()
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
    elif args.action == "predict-iteration":
        if not args.origin_prompt_file or not args.generation_prompt_file or not args.prompt_author:
            parser.error("predict-iteration requires origin prompt, generation prompt, and author")
        feedback = args.parent_feedback_file.read_text(encoding="utf-8") if args.parent_feedback_file else ""
        parent_result = args.parent_result_file.read_text(encoding="utf-8") if args.parent_result_file else ""
        parent_codex_result = (args.parent_codex_result_file.read_text(encoding="utf-8")
                               if args.parent_codex_result_file else "")
        parent_suggestion = (args.parent_suggestion_file.read_text(encoding="utf-8")
                             if args.parent_suggestion_file else "")
        result = predict_iteration(args.origin_prompt_file.read_text(encoding="utf-8"),
                                   args.generation_prompt_file.read_text(encoding="utf-8"),
                                   args.prompt_author, args.round, feedback,
                                   parent_result, args.parent_quality_score, parent_suggestion,
                                   parent_codex_result)
    else:
        result = status()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
