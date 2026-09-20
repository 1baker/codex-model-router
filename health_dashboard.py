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
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    routes: dict[str, dict[str, Any]] = {}
    totals: dict[str, int] = defaultdict(int)
    route_counts: Counter[str] = Counter()
    for record in records:
        model = record.get("model")
        if isinstance(model, str):
            route_counts[model] += 1
        thread = record.get("thread_id")
        if isinstance(thread, str) and model:
            routes[thread] = {key: record.get(key) for key in ("model", "effort", "event", "recorded_at_ms")}
        usage = record.get("usage")
        if isinstance(usage, dict) and isinstance(model, str):
            totals[model] += int(usage.get("totalTokens", usage.get("total_tokens", 0)) or 0)
    return {"route_counts": dict(sorted(route_counts.items())),
            "token_totals": dict(sorted(totals.items())), "latest_by_thread": routes}


def tmux_tabs(session: str) -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            ["tmux", "list-windows", "-t", session, "-F",
             "#{window_index}\t#{window_name}\t#{@byobu-codex-mode}\t#{@byobu-codex-thread-id}"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    tabs = []
    for line in output.splitlines():
        index, name, mode, thread = (line.split("\t") + ["", "", "", ""])[:4]
        tabs.append({"index": index, "name": name, "mode": mode, "thread_id": thread})
    return tabs


def process_count(pattern: str) -> int:
    result = subprocess.run(["pgrep", "-fc", pattern], capture_output=True, text=True)
    return int(result.stdout.strip() or 0) if result.returncode == 0 else 0


def report(metrics: Path, session: str) -> dict[str, Any]:
    summary = summarize(read_records(metrics))
    tabs = tmux_tabs(session)
    for tab in tabs:
        tab["latest_route"] = summary["latest_by_thread"].get(tab["thread_id"])
    return {"schema": "modellabs.health.v1", "metrics_path": str(metrics),
            "managed_tabs": tabs, "routing": {key: value for key, value in summary.items() if key != "latest_by_thread"},
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
