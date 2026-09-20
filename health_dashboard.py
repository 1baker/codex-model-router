"""Prompt-free operational health summary for ModelLabs managed chats."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from telemetry import METRICS_PATH


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            records.append(row)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate accepted turns from telemetry event volume and token usage."""
    latest: dict[str, dict[str, Any]] = {}
    accepted: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    token_totals: dict[str, int] = defaultdict(int)
    scorecards: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    latencies: dict[tuple[str, str], list[int]] = defaultdict(list)
    accepted_turns: dict[tuple[str, str], tuple[str, str]] = {}
    for row in records:
        event = row.get("event")
        event_counts[str(event)] += 1
        model = row.get("model")
        thread = row.get("thread_id")
        if event == "route_accepted" and isinstance(model, str):
            accepted[model] += 1
            card = scorecards[(model, str(row.get("effort", "unknown")))]
            card["accepted_turns"] += 1
            if isinstance(thread, str) and isinstance(row.get("turn_id"), str):
                accepted_turns[(thread, row["turn_id"])] = (model, str(row.get("effort", "unknown")))
            if isinstance(thread, str):
                latest[thread] = {key: row.get(key) for key in ("model", "effort", "task_class", "adaptive_reason", "recorded_at_ms")}
    unpaired_usage_events = 0
    unpaired_completion_events = 0
    for row in records:
        model = row.get("model")
        thread = row.get("thread_id")
        turn = row.get("turn_id")
        key = accepted_turns.get((thread, turn)) if isinstance(thread, str) and isinstance(turn, str) else None
        event = row.get("event")
        usage = row.get("usage")
        if isinstance(usage, dict) and isinstance(model, str):
            total = int(usage.get("totalTokens", usage.get("total_tokens", 0)) or 0)
            token_totals[model] += total
            if key:
                card = scorecards[key]
                card["usage_events"] += 1
                card["total_tokens"] += total
            else:
                unpaired_usage_events += 1
        if event == "turn_completed" and isinstance(model, str):
            if key:
                card = scorecards[key]
                card["completed_turns"] += 1
                elapsed = row.get("elapsed_ms")
                if isinstance(elapsed, (int, float)) and elapsed >= 0:
                    latencies[key].append(int(elapsed))
            else:
                unpaired_completion_events += 1
        if event == "outcome_signal" and row.get("source") == "explicit" and isinstance(model, str):
            scorecards[(model, str(row.get("effort", "unknown")))][f"{row.get('outcome')}_outcomes"] += 1
    cards = []
    for (model, effort), values in sorted(scorecards.items()):
        card = {"model": model, "effort": effort, **dict(sorted(values.items()))}
        samples = sorted(latencies[(model, effort)])
        if samples:
            card["latency_p50_ms"] = samples[(len(samples) - 1) // 2]
            card["latency_p95_ms"] = samples[min(len(samples) - 1, (len(samples) * 95 + 99) // 100 - 1)]
        cards.append(card)
    return {"accepted_turn_counts": dict(sorted(accepted.items())),
            "event_counts": dict(sorted(event_counts.items())),
            "token_totals": dict(sorted(token_totals.items())),
            "unpaired_usage_events": unpaired_usage_events,
            "unpaired_completion_events": unpaired_completion_events,
            "scorecards": cards, "latest_by_thread": latest}


def tmux_tabs(session: str) -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(["tmux", "list-windows", "-t", session, "-F",
            "#{window_index}\t#{window_name}\t#{@byobu-codex-mode}\t#{@byobu-codex-thread-id}"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [{"index": parts[0], "name": parts[1], "mode": parts[2], "thread_id": parts[3]}
            for line in output.splitlines() for parts in [(line.split("\t") + ["", "", "", ""])[:4]]]


def process_count(pattern: str) -> int:
    result = subprocess.run(["pgrep", "-fc", pattern], capture_output=True, text=True)
    return int(result.stdout.strip() or 0) if result.returncode == 0 else 0


def report(metrics: Path, session: str) -> dict[str, Any]:
    summary = summarize(read_records(metrics))
    tabs = tmux_tabs(session)
    for tab in tabs:
        tab["latest_route"] = summary["latest_by_thread"].get(tab["thread_id"])
    return {"schema": "modellabs.health.v2", "metrics_path": str(metrics), "managed_tabs": tabs,
            "routing": {key: value for key, value in summary.items() if key != "latest_by_thread"},
            "mcp_processes": {"modelControl": process_count("model_host_mcp.py"),
                              "agentBrowser": process_count("agent-browser mcp serve"),
                              "litScout": process_count("litscout.mcp_server"),
                              "previews": process_count("previews_server.mcp"),
                              "codexResearch": process_count("codex-research/mcp-server/server.py")}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Show prompt-free ModelLabs and Byobu health")
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--session", default="recovered-tabs")
    args = parser.parse_args()
    print(json.dumps(report(args.metrics, args.session), sort_keys=True))


if __name__ == "__main__":
    main()
