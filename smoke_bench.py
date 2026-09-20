"""Run reproducible product smoke prompts and record prompt-free evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from paths import ROOT
from telemetry import record

DEFAULT_MANIFEST = ROOT / "benchmarks/smoke.json"


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "modellabs.smoke.v1" or not isinstance(data.get("scenarios"), list):
        raise ValueError("Unsupported smoke manifest.")
    return data


def safe_child(root: Path, name: str) -> Path:
    candidate = (root / name).resolve()
    if candidate == root.resolve() or root.resolve() not in candidate.parents:
        raise ValueError(f"Unsafe fixture path: {name}")
    return candidate


def materialize(root: Path, scenario: dict[str, Any]) -> None:
    for name, text in scenario.get("files", {}).items():
        path = safe_child(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(text), encoding="utf-8")


def parse_codex_jsonl(output: str) -> tuple[dict[str, int], str]:
    usage: dict[str, int] = {}
    final = ""
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "item.completed" and (event.get("item") or {}).get("type") == "agent_message":
            final = str(event["item"].get("text", ""))
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = {key: int(value or 0) for key, value in event["usage"].items()
                     if isinstance(value, (int, float))}
    return usage, final


def grade(workspace: Path, scenario: dict[str, Any], final: str,
          usage: dict[str, int]) -> dict[str, Any]:
    checked = subprocess.run(scenario["verify"], cwd=workspace, capture_output=True,
                             text=True, timeout=int(scenario.get("verify_timeout", 30)))
    product_pass = checked.returncode == 0
    exact = final.strip() == scenario["expected_final"]
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    total_tokens = input_tokens + output_tokens
    # Artifact usability is the hard gate. Response-format compliance matters,
    # but cannot rescue a broken product.
    satisfaction = (90 + (10 if exact else 0)) if product_pass else 0
    return {"product_pass": product_pass, "exact_final_response": exact,
            "satisfaction_score": satisfaction, "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
            "output_tokens": output_tokens,
            "reasoning_output_tokens": int(usage.get("reasoning_output_tokens", 0)),
            "verifier_exit_code": checked.returncode}


def run_one(scenario: dict[str, Any], model: str, effort: str, suite_id: str,
            codex: str = "codex", record_metrics: bool = False) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"modellabs-smoke-{scenario['id']}-") as temporary:
        workspace = Path(temporary)
        materialize(workspace, scenario)
        command = [codex, "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check",
                   "-C", str(workspace), "--sandbox", "workspace-write", "--model", model,
                   "--config", f'model_reasoning_effort="{effort}"', "--json", "-"]
        started = time.monotonic()
        completed = subprocess.run(command, input=scenario["prompt"], capture_output=True,
                                   text=True, timeout=int(scenario.get("agent_timeout", 300)))
        elapsed_ms = round((time.monotonic() - started) * 1000)
        usage, final = parse_codex_jsonl(completed.stdout)
        result = {"schema": "modellabs.benchmark-result.v1", "suite_id": suite_id,
                  "scenario_id": scenario["id"], "scenario_sha256": hashlib.sha256(
                      scenario["prompt"].encode()).hexdigest(),
                  "task_class": scenario["task_class"], "model": model, "effort": effort,
                  "codex_exit_code": completed.returncode, "elapsed_ms": elapsed_ms,
                  **grade(workspace, scenario, final, usage)}
    if record_metrics:
        record("benchmark_result", **{key: value for key, value in result.items()
                                      if key not in {"schema"}})
    return result


def ranked(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked_rows = []
    scenario_ids = sorted({str(row.get("scenario_id", "default")) for row in results})
    for scenario_id in scenario_ids:
        cohort = [row for row in results if str(row.get("scenario_id", "default")) == scenario_id]
        best = min((row["total_tokens"] for row in cohort
                    if row["product_pass"] and row["total_tokens"] > 0), default=0)
        for row in cohort:
            efficiency = round(20 * best / row["total_tokens"], 2) if row["product_pass"] and best else 0.0
            ranked_rows.append({**row, "efficiency_score": efficiency,
                                "overall_score": round(row["satisfaction_score"] * 0.8 + efficiency, 2)})
    return sorted(ranked_rows, key=lambda row: (str(row.get("scenario_id", "default")),
                                                -row["overall_score"], row["total_tokens"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run verified ModelLabs smoke prompts")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--pair", action="append", help="MODEL:EFFORT; repeat to override the scenario matrix")
    parser.add_argument("--record-metrics", action="store_true")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    manifest = load_manifest(args.manifest)
    selected = set(args.scenario or [])
    scenarios = [item for item in manifest["scenarios"] if not selected or item["id"] in selected]
    if selected - {item["id"] for item in scenarios}:
        raise SystemExit("Unknown scenario requested.")
    override = [tuple(item.rsplit(":", 1)) for item in (args.pair or [])]
    suite_id = str(uuid.uuid4())
    results = []
    for scenario in scenarios:
        pairs = override or [tuple(item) for item in scenario["matrix"]]
        for _ in range(args.repeat):
            for model, effort in pairs:
                results.append(run_one(scenario, model, effort, suite_id, args.codex, args.record_metrics))
    print(json.dumps({"schema": "modellabs.smoke-report.v1", "suite_id": suite_id,
                      "results": ranked(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
