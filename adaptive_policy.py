"""Conservative, prompt-free adaptive routing evidence."""
from __future__ import annotations
import re
from collections import Counter
from typing import Any
from telemetry import METRICS_PATH, record

RETRY = re.compile(r"\b(still|failed|failure|broken|incorrect|not working|try again|redo)\b", re.I)
VERIFIED = re.compile(r"\b(tests? passed|verified|works now|looks good|fixed)\b", re.I)
ORDER = ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra"]

def feedback(prompt: str) -> str | None:
    return "retry" if RETRY.search(prompt) else "verified" if VERIFIED.search(prompt) else None

def note_followup(thread_id: str | None, prompt: str) -> None:
    signal = feedback(prompt)
    if not signal or not thread_id or not METRICS_PATH.exists(): return
    rows = []
    for line in METRICS_PATH.read_text(encoding="utf-8").splitlines():
        try:
            import json; rows.append(json.loads(line))
        except Exception: pass
    prior = next((r for r in reversed(rows) if r.get("event") == "route_accepted" and r.get("thread_id") == thread_id), None)
    if not prior or any(r.get("event") == "outcome_signal" and r.get("source_turn_id") == prior.get("turn_id") for r in rows): return
    record("outcome_signal", source_turn_id=prior.get("turn_id"), thread_id=thread_id, task_class=prior.get("task_class"),
           model=prior.get("model"), effort=prior.get("effort"), outcome=signal)

def adapt(choice: dict[str, Any]) -> dict[str, Any]:
    """Escalate on evidence; only lower effort after eight verified outcomes."""
    if not METRICS_PATH.exists() or choice["class"] == "consequential": return choice
    counts = Counter()
    for line in METRICS_PATH.read_text(encoding="utf-8").splitlines():
        try:
            import json; row = json.loads(line)
        except Exception: continue
        if row.get("event") == "outcome_signal" and row.get("task_class") == choice["class"]:
            counts[row.get("outcome")] += 1
    result = dict(choice)
    if counts["retry"] >= 2:
        pos = min(ORDER.index(choice["model"]) + 1, len(ORDER) - 1)
        result.update(model=ORDER[pos], effort="high", adaptive_reason="retry_escalation")
    elif counts["verified"] >= 8 and not counts["retry"] and choice["effort"] in {"high", "medium"}:
        result.update(effort="medium" if choice["effort"] == "high" else "low", adaptive_reason="verified_effort_reduction")
    else:
        result["adaptive_reason"] = "baseline_insufficient_evidence"
    return result
