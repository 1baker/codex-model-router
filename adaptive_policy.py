"""Privacy-safe, evidence-gated adaptive routing."""

from __future__ import annotations

import json
import hashlib
import os
import re
import stat
import time
from collections.abc import Iterable
from collections import defaultdict
from statistics import median
from typing import Any

from paths import ROOT
from telemetry import METRICS_PATH, record

RETRY = re.compile(
    r"^(?:no[, ]+)?(?:that|it|this)\s+(?:is\s+)?(?:still\s+)?"
    r"(?:not working|broken|failing|incorrect)\b|"
    r"^still\s+(?:not working|broken|failing)\b|"
    r"^(?:that|it|this)\s+(?:didn't|did not)\s+(?:work|fix\s+it)\b|"
    r"^tests?\s+(?:are\s+)?(?:failing|failed)\b", re.I)
VERIFIED = re.compile(
    r"^(?:okay[, ]+|yes[, ]+)?(?:that|it|this)\s+(?:is\s+)?(?:now\s+)?working\b|"
    r"^(?:it\s+)?works\s+now\b|"
    r"^tests?\s+(?:have\s+)?passed\b", re.I)
ORDER = ["gpt-5.6-luna", "gpt-6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-sol", "gpt-6-astra"]
ESCALATION_NEXT = {
    "gpt-5.6-luna": "gpt-5.6-terra",
    "gpt-5.6-terra": "gpt-5.6-sol",
    "gpt-5.6-sol": "gpt-6-sol",
    "gpt-6-luna": "gpt-6-sol",
    "gpt-6-sol": "gpt-6-astra",
    "gpt-6-astra": "gpt-6-astra",
}
WINDOW_MS = 30 * 24 * 60 * 60 * 1000
MIN_RETRY_SIGNALS = 4
MIN_VERIFIED_SIGNALS = 10
MIN_BENCHMARK_SAMPLES = 3
MIN_BENCHMARK_SCENARIOS = 2
ADAPTIVE_MODE_PATH = ROOT / "adaptive-mode"
MANAGED_ROUTING_POLICY_PATH = ROOT / "learning/managed-routing-policy-v2.json"
PILOT_SAMPLE_MODULUS = 10


def adaptive_mode() -> str:
    value = os.environ.get("MODELLABS_ADAPTIVE_MODE")
    if value is None:
        try:
            value = ADAPTIVE_MODE_PATH.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            value = "shadow"
    return value.lower() if value.lower() in {"shadow", "pilot", "enforce"} else "shadow"


def set_adaptive_mode(mode: str) -> None:
    if mode not in {"shadow", "pilot", "enforce"}:
        raise ValueError("adaptive mode must be 'shadow', 'pilot', or 'enforce'")
    ADAPTIVE_MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ADAPTIVE_MODE_PATH.write_text(mode + "\n", encoding="utf-8")
    os.chmod(ADAPTIVE_MODE_PATH, 0o600)


def managed_policy_recommendation(choice: dict[str, Any],
                                  path: os.PathLike[str] | str = MANAGED_ROUTING_POLICY_PATH
                                  ) -> tuple[dict[str, str], dict[str, Any]] | None:
    """Read a currently validated receipt-bound class policy, failing closed."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "r", encoding="utf-8") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                return None
            artifact = json.load(source)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(artifact, dict):
        return None
    validations = artifact.get("validations")
    if artifact.get("schema") != "modellabs.managed_routing_policy.v2" or not isinstance(validations, dict):
        return None
    baseline = [choice.get("model"), choice.get("effort")]
    matches = []
    for value in validations.values():
        if not isinstance(value, dict):
            return None
        evidence = {name: value.get(name) for name in (
            "comparison_id", "task_class", "baseline_arm", "recommended_arm",
            "prospective_products", "prospective_wins", "prospective_losses",
            "checkpoint_sha256")}
        digest = hashlib.sha256(json.dumps(
            evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (value.get("evidence_sha256") != digest
                or not isinstance(value.get("baseline_arm"), list)
                or len(value["baseline_arm"]) != 2
                or not isinstance(value.get("recommended_arm"), list)
                or len(value["recommended_arm"]) != 2
                or type(value.get("prospective_products")) is not int
                or type(value.get("prospective_wins")) is not int
                or type(value.get("prospective_losses")) is not int
                or not isinstance(value.get("comparison_id"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", value["comparison_id"])
                or not isinstance(value.get("checkpoint_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", value["checkpoint_sha256"])):
            return None
        if (value.get("task_class") == choice.get("class")
                and value.get("baseline_arm") == baseline
                and value.get("recommended_arm") != baseline
                and value.get("prospective_products", 0) >= 8
                and value.get("prospective_wins", 0) >= 7
                and value.get("prospective_losses") == 0):
            matches.append(value)
    if len(matches) != 1:
        return None
    selected = matches[0]
    model, effort = selected["recommended_arm"]
    if model not in ORDER or effort not in {"none", "low", "medium", "high", "xhigh", "max", "ultra"}:
        return None
    return {"model": model, "effort": effort}, {
        "source": "managed_randomized_policy",
        "comparison_id": selected["comparison_id"],
        "prospective_products": selected["prospective_products"],
        "prospective_wins": selected["prospective_wins"],
        "prospective_losses": selected["prospective_losses"],
        "model_execution_verified": True, "provisional": False}


def _pilot_selected(choice: dict[str, Any]) -> bool:
    digest = choice.get("prompt_sha256")
    return (isinstance(digest, str) and bool(re.fullmatch(r"[0-9a-f]{64}", digest))
            and int(digest[:8], 16) % PILOT_SAMPLE_MODULUS == 0)


def benchmark_recommendation(choice: dict[str, Any], records: Iterable[dict[str, Any]],
                             now_ms: int) -> tuple[dict[str, str], dict[str, Any]] | None:
    """Rank legacy local benchmarks for shadow diagnosis only.

    Version 2 rows are caller-written telemetry.  Their model provenance fields
    are not a proxy-owned execution receipt, so they must never become an
    enforceable route even when a caller labels them observed_per_request.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if (row.get("event") != "benchmark_result" or row.get("task_class") != choice["class"]
                or row.get("schema") != "modellabs.benchmark-result.v2"
                or not isinstance(row.get("scenario_id"), str)
                or not isinstance(row.get("scenario_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["scenario_sha256"])
                or int(row.get("recorded_at_ms", now_ms)) < now_ms - WINDOW_MS):
            continue
        model, effort = row.get("model"), row.get("effort")
        if isinstance(model, str) and isinstance(effort, str):
            grouped[(model, effort)].append(row)
    baseline_key = (choice["model"], choice["effort"])
    if baseline_key not in grouped:
        return None
    baseline_scenarios = {(row["scenario_id"], row["scenario_sha256"])
                          for row in grouped[baseline_key]}
    eligible = []
    for (model, effort), all_rows in grouped.items():
        shared = baseline_scenarios & {(row["scenario_id"], row["scenario_sha256"])
                                       for row in all_rows}
        rows = [row for row in all_rows
                if (row["scenario_id"], row["scenario_sha256"]) in shared]
        if not rows:
            continue
        passed = [row for row in rows if row.get("product_pass") is True
                  and row.get("fixture_integrity") is True
                  and row.get("codex_exit_code") == 0
                  and row.get("verifier_exit_code") == 0]
        if len(passed) != len(rows):
            continue
        tokens = [int(row["total_tokens"]) for row in passed
                  if isinstance(row.get("total_tokens"), (int, float)) and row["total_tokens"] > 0]
        if len(tokens) != len(passed):
            continue
        satisfaction = [int(row.get("satisfaction_score", 0)) for row in passed]
        scenario_count = len({row["scenario_id"] for row in rows})
        eligible.append({"model": model, "effort": effort, "samples": len(rows),
                         "scenarios": scenario_count,
                         "model_execution_verified": False,
                         "median_tokens": int(median(tokens)),
                         "median_satisfaction": int(median(satisfaction))})
    if not eligible:
        return None
    eligible.sort(key=lambda item: (-item["median_satisfaction"], item["median_tokens"],
                                    ORDER.index(item["model"]) if item["model"] in ORDER else len(ORDER)))
    winner = eligible[0]
    if (winner["model"], winner["effort"]) == (choice["model"], choice["effort"]):
        return None
    baseline = next((item for item in eligible
                     if (item["model"], item["effort"]) == baseline_key), None)
    evidence = {"source": "benchmark", "samples": winner["samples"],
                "scenarios": winner["scenarios"],
                "median_tokens": winner["median_tokens"],
                "median_satisfaction": winner["median_satisfaction"],
                "model_execution_verified": bool(
                    winner["model_execution_verified"] and baseline
                    and baseline["model_execution_verified"]),
                "provisional": (winner["samples"] < MIN_BENCHMARK_SAMPLES
                                or winner["scenarios"] < MIN_BENCHMARK_SCENARIOS
                                or not winner["model_execution_verified"]
                                or not baseline or not baseline["model_execution_verified"])}
    return {"model": winner["model"], "effort": winner["effort"]}, evidence


def feedback(prompt: str) -> str | None:
    """Classify only short direct follow-ups as weak, non-grade evidence."""
    value = prompt.strip()
    if not value or len(value) > 240 or "\n" in value:
        return None
    return "retry" if RETRY.search(value) else "verified" if VERIFIED.search(value) else None


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


def record_explicit_grade(thread_id: str, turn_id: str, quality_score: int,
                          verification: str) -> dict[str, Any]:
    """Attach a prompt-free, operator-verified product grade to one routed turn.

    A high score is not enough by itself: only an independently verified product
    can be positive learning evidence.  A failed verification or a score below
    the release threshold becomes retry evidence so routing can escalate within
    the same managed chat and task bucket.
    """
    if not 0 <= quality_score <= 100:
        raise ValueError("quality_score must be an integer from 0 through 100")
    if verification not in {"passed", "failed"}:
        raise ValueError("verification must be 'passed' or 'failed'")
    rows = read_records()
    route = next((row for row in reversed(rows) if row.get("event") == "route_accepted"
                  and row.get("thread_id") == thread_id and row.get("turn_id") == turn_id), None)
    if route is None:
        raise ValueError("No accepted ModelLabs route matches that thread and turn.")
    if any(row.get("event") == "quality_grade" and row.get("thread_id") == thread_id
           and row.get("source_turn_id") == turn_id for row in rows):
        raise ValueError("An explicit quality grade is already recorded for that turn.")
    outcome = "verified" if verification == "passed" and quality_score >= 90 else "retry"
    payload = {"thread_id": thread_id, "source_turn_id": turn_id,
               "quality_score": quality_score, "verification": verification,
               "outcome": outcome, "task_class": route["task_class"],
               "model": route["model"], "effort": route["effort"],
               "task_bucket": route.get("task_bucket", f"{route['task_class']}:legacy")}
    record("quality_grade", source="explicit", **payload)
    record("outcome_signal", source="explicit", **payload)
    return payload


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
        recommendation = {"model": ESCALATION_NEXT.get(choice["model"], choice["model"]), "effort": "high"}
        reason = "retry_escalation"
    elif verified >= MIN_VERIFIED_SIGNALS and not retries and choice["effort"] in {"high", "medium"}:
        recommendation = {"effort": "medium" if choice["effort"] == "high" else "low"}
        reason = "verified_effort_reduction"
    else:
        managed = managed_policy_recommendation(choice)
        if managed:
            recommendation, evidence = managed
            result["adaptive_recommendation"] = recommendation
            result["adaptive_evidence"] = evidence
            mode = adaptive_mode()
            pilot_selected = mode == "pilot" and _pilot_selected(choice)
            if mode == "enforce" or pilot_selected:
                result.update(recommendation)
                result["intelligence_slider"] = result["effort"]
                result["adaptive_reason"] = ("managed_policy_pilot"
                                             if pilot_selected else "managed_policy_enforced")
            else:
                result["adaptive_reason"] = ("shadow_managed_policy_pilot_holdout"
                                             if mode == "pilot"
                                             else "shadow_managed_policy_validated")
            return result
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
