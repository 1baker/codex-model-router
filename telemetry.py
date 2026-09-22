"""Private, prompt-free routing and completion telemetry."""

from __future__ import annotations

import json
import fcntl
import math
import os
import time
from pathlib import Path
from typing import Any
from paths import ROOT


DEFAULT_PATH = ROOT / "metrics.jsonl"
METRICS_PATH = Path(os.environ.get("MODELLABS_METRICS_PATH", DEFAULT_PATH))
USAGE_FIELDS = ("inputTokens", "cachedInputTokens", "cacheWriteInputTokens",
                "outputTokens", "reasoningOutputTokens", "totalTokens")


def record(event: str, **fields: Any) -> None:
    """Append metadata only; callers must never supply prompt text or tokens."""
    payload = {"event": event, "recorded_at_ms": int(time.time() * 1000), **fields}
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    receipt_id = fields.get("receipt_id")
    lock_descriptor = None
    duplicate = False
    if receipt_id is not None:
        lock_descriptor = os.open(METRICS_PATH.with_suffix(METRICS_PATH.suffix + ".receipt.lock"),
                                  os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        try:
            with METRICS_PATH.open(encoding="utf-8") as source:
                duplicate = any(_has_receipt(line, receipt_id) for line in source)
        except FileNotFoundError:
            pass
    try:
        if duplicate:
            return
        fd = os.open(METRICS_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.chmod(METRICS_PATH, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as out:
            out.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            out.flush()
            os.fsync(out.fileno())
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)


def _has_receipt(line: str, receipt_id: str) -> bool:
    try:
        return json.loads(line).get("receipt_id") == receipt_id
    except (ValueError, TypeError):
        return False


def usage_from(params: dict[str, Any]) -> Any:
    usage = params.get("usage")
    return usage if isinstance(usage, dict) else None


def thread_usage_from(params: dict[str, Any]) -> Any:
    """Read one live per-response usage sample from a thread usage event."""
    token_usage = params.get("tokenUsage")
    if not isinstance(token_usage, dict):
        return None
    usage = token_usage.get("last")
    return usage if isinstance(usage, dict) else None


def usage_delta(baseline: dict[str, Any], total: dict[str, Any]) -> dict[str, int] | None:
    if not baseline or not total:
        return None
    if "totalTokens" not in baseline or "totalTokens" not in total:
        return None
    result: dict[str, int] = {}
    for field in USAGE_FIELDS:
        before, after = baseline.get(field, 0), total.get(field, 0)
        if (not isinstance(before, (int, float)) or isinstance(before, bool)
                or not isinstance(after, (int, float)) or isinstance(after, bool)
                or before < 0 or after < before):
            return None
        result[field] = int(after - before)
    return result


class UsageTracker:
    """Validate cumulative accounting and permanently fail closed after resets."""

    def __init__(self) -> None:
        self.baseline: dict[str, int] | None = None
        self.previous: dict[str, int] | None = None
        self.observed = False
        self.invalid_reason: str | None = None

    @staticmethod
    def _counters(value: Any) -> dict[str, int] | None:
        if not isinstance(value, dict):
            return None
        if any(field not in value for field in USAGE_FIELDS):
            return None
        result: dict[str, int] = {}
        for field in USAGE_FIELDS:
            item = value[field]
            if (not isinstance(item, (int, float)) or isinstance(item, bool)
                    or not math.isfinite(item) or item < 0 or int(item) != item):
                return None
            result[field] = int(item)
        if result["totalTokens"] != result["inputTokens"] + result["outputTokens"]:
            return None
        return result

    def observe(self, params: dict[str, Any]) -> None:
        self.observed = True
        if self.invalid_reason:
            return
        token_usage = params.get("tokenUsage")
        total = self._counters(token_usage.get("total") if isinstance(token_usage, dict) else None)
        last = self._counters(token_usage.get("last") if isinstance(token_usage, dict) else None)
        if total is None or last is None:
            self.invalid_reason = "malformed_usage_snapshot"
            return
        if any(last[field] > total[field] for field in USAGE_FIELDS):
            self.invalid_reason = "last_exceeds_total"
            return
        if self.previous is not None and any(total[field] < self.previous[field] for field in USAGE_FIELDS):
            self.invalid_reason = "cumulative_counter_decreased"
            return
        if self.baseline is None:
            self.baseline = {field: total[field] - last[field] for field in USAGE_FIELDS}
        self.previous = total

    def outcome(self) -> tuple[dict[str, int] | None, str | None]:
        if not self.observed:
            return None, "no_usage_evidence"
        if self.invalid_reason:
            return None, self.invalid_reason
        if self.baseline is None or self.previous is None:
            return None, "incomplete_usage_evidence"
        delta = usage_delta(self.baseline, self.previous)
        return (delta, None) if delta is not None else (None, "invalid_usage_delta")


def aggregate_usage(samples: list[dict[str, Any]]) -> dict[str, int] | None:
    """Return one exact per-turn sum from raw upstream completion samples."""
    totals: dict[str, int] = {}
    for sample in samples:
        for field in USAGE_FIELDS:
            value = sample.get(field)
            if isinstance(value, (int, float)) and value >= 0:
                totals[field] = totals.get(field, 0) + int(value)
    return totals or None
