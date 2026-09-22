"""Private, prompt-free routing and completion telemetry."""

from __future__ import annotations

import json
import fcntl
import math
import os
import time
import hashlib
import secrets
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
    lock_descriptor = os.open(METRICS_PATH.with_suffix(METRICS_PATH.suffix + ".lock"),
                              os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
    try:
        if receipt_id is not None:
            payload = _persist_receipt(receipt_id, payload)
        fd = os.open(METRICS_PATH, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            content = os.read(fd, os.fstat(fd).st_size)
            if content and not content.endswith(b"\n"):
                boundary = content.rfind(b"\n") + 1
                os.ftruncate(fd, boundary)
                content = content[:boundary]
            duplicate = receipt_id is not None and any(
                _has_receipt(line.decode("utf-8", "replace"), receipt_id)
                for line in content.splitlines())
            if not duplicate:
                os.lseek(fd, 0, os.SEEK_END)
                _write_all(fd, (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_directory(METRICS_PATH.parent)
    finally:
        os.close(lock_descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short telemetry write")
        view = view[written:]


def _persist_receipt(receipt_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    directory = METRICS_PATH.parent / "receipts"
    directory_created = not directory.exists()
    directory.mkdir(parents=True, exist_ok=True)
    if directory_created:
        _fsync_directory(directory.parent)
    path = directory / f"{hashlib.sha256(receipt_id.encode()).hexdigest()}.json"
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (existing.get("receipt_id") != receipt_id
                or existing.get("event") != payload.get("event")
                or existing.get("thread_id") != payload.get("thread_id")
                or existing.get("turn_id") != payload.get("turn_id")):
            raise RuntimeError("Receipt identity collision.")
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(directory)
        _fsync_directory(directory.parent)
        return existing
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            _write_all(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        _fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _has_receipt(line: str, receipt_id: str) -> bool:
    try:
        return json.loads(line).get("receipt_id") == receipt_id
    except (ValueError, TypeError):
        return False


def canonical_receipt(receipt_id: str) -> dict[str, Any] | None:
    path = METRICS_PATH.parent / "receipts" / f"{hashlib.sha256(receipt_id.encode()).hexdigest()}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict) or payload.get("receipt_id") != receipt_id:
        raise RuntimeError("Invalid canonical receipt.")
    return payload


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
