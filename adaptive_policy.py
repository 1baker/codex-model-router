"""Privacy-safe, evidence-gated adaptive routing."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable
from typing import Any

from telemetry import METRICS_PATH, record

RETRY = re.compile(r"\b(still|failed|failure|broken|incorrect|not working|try again|redo)\b", re.I)
VERIFIED = re.compile(r"\b(tests? passed|verified|works now|looks good|fixed)\b", re.I)
ORDER = ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra"]
WINDOW_MS = 30 * 24 * 60 * 60 * 1000
MIN_RETRY_SIGNALS = 4
MIN_VERIFIED_SIGNALS = 10


def feedback(prompt: str) -> str | None:
    """Classify possible follow-up feedback without retaining its text."""
    return "retry" if RETRY.search(prompt) else "verified" if VERIFIED.search(prompt) else None


def read_records() -> list[dict[str, Any]]:
    if not METRICS_PATH.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in METRICS_PATH.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            records.append(row)
    return records


def note_followup(thread_id: str | None, prompt: str) -> None:
    """Record weak, non-policy evidence against the prior accepted route."""
    signal = feedback(prompt)
    if not signal or not thread_id:
        return
    prior = next((row for row in reversed(read_records())
                  if row.get("event") == "route_accepted" and row.get("thread_id") == thread_id), None)
    if prior:
        record("outcome_observation", source="inferred", source_turn_id=prior.get("turn_id"),
               thread_id=thread_id, task_class=prior.get("task_class"), model=prior.get("model"),
               effort=prior.get("effort"), task_bucket=prior.get("task_bucket"), outcome=signal)


def record_outcome(records: list[dict[str, Any]], thread_id: str, turn_id: str, outcome: str,
                   route: dict[str, Any]) -> None:
    if outcome not in {"verified", "retry"}:
        raise ValueError("outcome must be 'verified' or 'retry'")
    records.append({"event": "outcome_signal", "source": "explicit", "thread_id": thread_id,
                    "source_turn_id": turn_id, "outcome": outcome, "task_class": route["class"],
                    "model": route["model"], "effort": route["effort"], "task_bucket": route["task_bucket"]})


def record_explicit_outcome(thread_id: str, turn_id: str, outcome: str) -> None:
    """Persist an operator-confirmed result for a previously accepted turn."""
    rows = read_records()
    route = next((row for row in reversed(rows) if row.get("event") == "route_accepted"
                  and row.get("thread_id") == thread_id and row.get("turn_id") == turn_id), None)
    if route is None:
        raise ValueError("No accepted ModelLabs route matches that thread and turn.")
    if any(row.get("event") == "outcome_signal" and row.get("source") == "explicit"
           and row.get("source_turn_id") == turn_id for row in rows):
        raise ValueError("An explicit outcome is already recorded for that turn.")
    staged: list[dict[str, Any]] = []
    record_outcome(staged, thread_id, turn_id, outcome, {"class": route["task_class"],
                   "model": route["model"], "effort": route["effort"],
                   "task_bucket": route.get("task_bucket", f"{route['task_class']}:legacy")})
    record("outcome_signal", **{key: value for key, value in staged[0].items() if key != "event"})


def adapt(choice: dict[str, Any], *, thread_id: str | None = None,
          records: Iterable[dict[str, Any]] | None = None, now_ms: int | None = None) -> dict[str, Any]:
    """Recommend or enforce only well-scoped, explicit, recent evidence."""
    result = dict(choice)
    if choice["class"] == "consequential":
        result["adaptive_reason"] = "consequential_baseline"
        return result
    if choice.get("explicit_model") or choice.get("explicit_effort"):
        result["adaptive_reason"] = "explicit_user_override"
        return result
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    relevant = [row for row in (list(records) if records is not None else read_records())
                if row.get("event") == "outcome_signal" and row.get("source") == "explicit"
                and row.get("task_bucket") == choice["task_bucket"]
                and (thread_id is None or row.get("thread_id") == thread_id)
                and int(row.get("recorded_at_ms", now_ms)) >= now_ms - WINDOW_MS]
    retries = sum(row.get("outcome") == "retry" for row in relevant)
    verified = sum(row.get("outcome") == "verified" for row in relevant)
    recommendation: dict[str, str] = {}
    if retries >= MIN_RETRY_SIGNALS and retries * 2 >= len(relevant):
        recommendation = {"model": ORDER[min(ORDER.index(choice["model"]) + 1, len(ORDER) - 1)], "effort": "high"}
        reason = "retry_escalation"
    elif verified >= MIN_VERIFIED_SIGNALS and not retries and choice["effort"] in {"high", "medium"}:
        recommendation = {"effort": "medium" if choice["effort"] == "high" else "low"}
        reason = "verified_effort_reduction"
    else:
        result["adaptive_reason"] = "baseline_insufficient_evidence"
        return result
    result["adaptive_recommendation"] = recommendation
    if os.environ.get("MODELLABS_ADAPTIVE_MODE", "shadow").lower() == "enforce":
        result.update(recommendation)
        result["intelligence_slider"] = result["effort"]
        result["adaptive_reason"] = reason
    else:
        result["adaptive_reason"] = f"shadow_{reason}"
    return result
