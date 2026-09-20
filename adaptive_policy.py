"""Privacy-safe, evidence-gated adaptive routing."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable
from collections import defaultdict
from statistics import median
from typing import Any

from paths import ROOT
from telemetry import METRICS_PATH, record

RETRY = re.compile(r"\b(still|failed|failure|broken|incorrect|not working|try again|redo)\b", re.I)
VERIFIED = re.compile(r"\b(tests? passed|verified|works now|looks good|fixed)\b", re.I)
ORDER = ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra"]
WINDOW_MS = 30 * 24 * 60 * 60 * 1000
MIN_RETRY_SIGNALS = 4
MIN_VERIFIED_SIGNALS = 10
MIN_BENCHMARK_SAMPLES = 3
MIN_BENCHMARK_SCENARIOS = 2
ADAPTIVE_MODE_PATH = ROOT / "adaptive-mode"


def adaptive_mode() -> str:
    value = os.environ.get("MODELLABS_ADAPTIVE_MODE")
    if value is None:
        try:
            value = ADAPTIVE_MODE_PATH.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            value = "shadow"
    return value.lower() if value.lower() in {"shadow", "enforce"} else "shadow"


def set_adaptive_mode(mode: str) -> None:
    if mode not in {"shadow", "enforce"}:
        raise ValueError("adaptive mode must be 'shadow' or 'enforce'")
    ADAPTIVE_MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ADAPTIVE_MODE_PATH.write_text(mode + "\n", encoding="utf-8")
    os.chmod(ADAPTIVE_MODE_PATH, 0o600)


def benchmark_recommendation(choice: dict[str, Any], records: Iterable[dict[str, Any]],
                             now_ms: int) -> tuple[dict[str, str], dict[str, Any]] | None:
    """Choose a verified class-level benchmark winner without using prompt text."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if (row.get("event") != "benchmark_result" or row.get("task_class") != choice["class"]
                or int(row.get("recorded_at_ms", now_ms)) < now_ms - WINDOW_MS):
            continue
        model, effort = row.get("model"), row.get("effort")
        if isinstance(model, str) and isinstance(effort, str):
            grouped[(model, effort)].append(row)
    baseline_key = (choice["model"], choice["effort"])
    if baseline_key not in grouped:
        return None
    baseline_scenarios = {row.get("scenario_id") for row in grouped[baseline_key]}
    eligible = []
    for (model, effort), all_rows in grouped.items():
        shared = baseline_scenarios & {row.get("scenario_id") for row in all_rows}
        rows = [row for row in all_rows if row.get("scenario_id") in shared]
        if not rows:
            continue
        passed = [row for row in rows if row.get("product_pass") is True]
        if len(passed) != len(rows):
            continue
        tokens = [int(row["total_tokens"]) for row in passed
                  if isinstance(row.get("total_tokens"), (int, float)) and row["total_tokens"] > 0]
        if len(tokens) != len(passed):
            continue
        satisfaction = [int(row.get("satisfaction_score", 0)) for row in passed]
        scenario_count = len({row.get("scenario_id") for row in rows})
        eligible.append({"model": model, "effort": effort, "samples": len(rows),
                         "scenarios": scenario_count,
                         "median_tokens": int(median(tokens)),
                         "median_satisfaction": int(median(satisfaction))})
    if not eligible:
        return None
    eligible.sort(key=lambda item: (-item["median_satisfaction"], item["median_tokens"],
                                    ORDER.index(item["model"]) if item["model"] in ORDER else len(ORDER)))
    winner = eligible[0]
    if (winner["model"], winner["effort"]) == (choice["model"], choice["effort"]):
        return None
    evidence = {"source": "benchmark", "samples": winner["samples"],
                "scenarios": winner["scenarios"],
                "median_tokens": winner["median_tokens"],
                "median_satisfaction": winner["median_satisfaction"],
                "provisional": (winner["samples"] < MIN_BENCHMARK_SAMPLES
                                or winner["scenarios"] < MIN_BENCHMARK_SCENARIOS)}
    return {"model": winner["model"], "effort": winner["effort"]}, evidence


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
    if choice.get("explicit_model") or choice.get("explicit_effort"):
        result["adaptive_reason"] = "explicit_user_override"
        return result
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    all_records = list(records) if records is not None else read_records()
    relevant = [row for row in all_records
                if row.get("event") == "outcome_signal" and row.get("source") == "explicit"
                and row.get("task_bucket") == choice["task_bucket"]
                # Operator outcomes are intentionally chat-local. A new chat
                # has no thread identity yet, so only benchmark evidence may
                # influence its initial route.
                and thread_id is not None and row.get("thread_id") == thread_id
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
        benchmark = benchmark_recommendation(choice, all_records, now_ms)
        if benchmark:
            recommendation, evidence = benchmark
            result["adaptive_recommendation"] = recommendation
            result["adaptive_evidence"] = evidence
            # Preliminary benchmark evidence and all consequential routes are
            # shadow-only. This lets real artifact checks inform the router
            # without allowing a tiny canary set to steer production traffic.
            enforceable = (not evidence["provisional"] and choice["class"] != "consequential")
            if adaptive_mode() == "enforce" and enforceable:
                result.update(recommendation)
                result["intelligence_slider"] = result["effort"]
                result["adaptive_reason"] = "benchmark_verified_efficiency"
            else:
                result["adaptive_reason"] = "shadow_benchmark_preliminary" if evidence["provisional"] else "shadow_benchmark_verified"
            return result
        result["adaptive_reason"] = ("consequential_baseline" if choice["class"] == "consequential"
                                     else "baseline_insufficient_evidence")
        return result
    result["adaptive_recommendation"] = recommendation
    if adaptive_mode() == "enforce":
        result.update(recommendation)
        result["intelligence_slider"] = result["effort"]
        result["adaptive_reason"] = reason
    else:
        result["adaptive_reason"] = f"shadow_{reason}"
    return result
