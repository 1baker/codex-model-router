"""Run reproducible product smoke prompts and record prompt-free evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import websockets

from paths import ROOT
from telemetry import record

DEFAULT_MANIFEST = ROOT / "benchmarks/smoke.json"
DEFAULT_PROMPT_REGISTRY = ROOT / "prompt-experiments.json"
PRODUCT_FIELDS = ("files", "protected_files", "verify", "expected_final", "verify_timeout")
GUARD_ROOT = Path.home() / ".auracall/pro-guard/runs"
RUNS_ROOT = Path.home() / ".auracall/runtime/runs"


def product_definition(scenario: dict[str, Any]) -> dict[str, Any]:
    return {name: scenario.get(name) for name in PRODUCT_FIELDS}


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "modellabs.smoke.v1" or not isinstance(data.get("scenarios"), list):
        raise ValueError("Unsupported smoke manifest.")
    for scenario in data["scenarios"]:
        files = scenario.get("files")
        protected = scenario.get("protected_files")
        if (not isinstance(files, dict) or not isinstance(protected, list) or not protected
                or any(not isinstance(name, str) or name not in files for name in protected)
                or len(protected) != len(set(protected))):
            raise ValueError("Every smoke scenario must name unchanged protected fixture files.")
    by_id = {scenario.get("id"): scenario for scenario in data["scenarios"]}
    if (len(by_id) != len(data["scenarios"])
            or any(not isinstance(name, str) or not name for name in by_id)):
        raise ValueError("Smoke scenario IDs must be unique.")
    for scenario in data["scenarios"]:
        parent_id = scenario.get("variant_of")
        if parent_id is not None:
            parent = by_id.get(parent_id)
            if (parent is None or scenario.get("comparison_id") != parent.get("comparison_id")
                    or not isinstance(scenario.get("comparison_id"), str)
                    or product_definition(scenario) != product_definition(parent)
                    or scenario.get("prompt") == parent.get("prompt")
                    or scenario.get("prompt_author") != "codex"):
                raise ValueError("Prompt variant must share one exact product definition and comparison ID.")
        cohort = scenario.get("managed_cohort")
        if cohort in {"routine_luna_vs_sol_development", "routine_luna_vs_sol_prospective"}:
            from modellabs import route
            if (scenario.get("task_class") != "routine"
                    or route(str(scenario.get("prompt", ""))).get("class") != "routine"):
                raise ValueError("Managed routine cohort requires matching annotated and routed classes.")
    experiments = data.get("managed_experiments")
    if experiments is not None:
        if not isinstance(experiments, list):
            raise ValueError("Managed experiments must be a list.")
        from modellabs import route
        experiment_ids: set[str] = set()
        comparisons: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        valid_efforts = {"none", "low", "medium", "high", "xhigh", "max", "ultra"}
        for experiment in experiments:
            if (not isinstance(experiment, dict) or set(experiment) != {
                    "id", "task_class", "arms", "development_scenario_ids",
                    "prospective_scenario_ids", "enabled"}):
                raise ValueError("Managed experiment entry is malformed.")
            identifier = experiment.get("id")
            arms = experiment.get("arms")
            development_ids = experiment.get("development_scenario_ids")
            prospective_ids = experiment.get("prospective_scenario_ids")
            if (not isinstance(identifier, str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", identifier)
                    or identifier in experiment_ids
                    or not isinstance(experiment.get("task_class"), str)
                    or type(experiment.get("enabled")) is not bool
                    or not isinstance(arms, list) or len(arms) != 2
                    or any(not isinstance(arm, list) or len(arm) != 2
                           or not all(isinstance(value, str) and value for value in arm)
                           or arm[1] not in valid_efforts for arm in arms)
                    or len({tuple(arm) for arm in arms}) != 2
                    or not isinstance(development_ids, list) or len(development_ids) < 8
                    or not isinstance(prospective_ids, list) or len(prospective_ids) < 8
                    or any(not isinstance(value, str) or value not in by_id
                           for value in development_ids + prospective_ids)
                    or len(set(development_ids)) != len(development_ids)
                    or len(set(prospective_ids)) != len(prospective_ids)
                    or set(development_ids) & set(prospective_ids)):
                raise ValueError("Managed experiment definition is invalid.")
            comparison = (experiment["task_class"], tuple(sorted(tuple(arm) for arm in arms)))
            if comparison in comparisons:
                raise ValueError("Managed experiment comparisons must be unique.")
            development = [by_id[value] for value in development_ids]
            prospective = [by_id[value] for value in prospective_ids]
            if (len({stable_digest(product_definition(value)) for value in development})
                    != len(development)
                    or len({stable_digest(product_definition(value)) for value in prospective})
                    != len(prospective)
                    or ({stable_digest(product_definition(value)) for value in development}
                        & {stable_digest(product_definition(value)) for value in prospective})):
                raise ValueError("Managed experiment products must be unique and phase-disjoint.")
            selected = development + prospective
            routed = [route(str(value.get("prompt", ""))) for value in selected]
            if (any(value.get("variant_of") is not None
                    or value.get("task_class") != experiment["task_class"]
                    or chosen.get("class") != experiment["task_class"]
                    for value, chosen in zip(selected, routed))
                    or len({(chosen.get("model"), chosen.get("effort")) for chosen in routed}) != 1
                    or (routed[0].get("model"), routed[0].get("effort"))
                    not in {tuple(arm) for arm in arms}):
                raise ValueError("Managed experiment must bind one routed baseline and task class.")
            experiment_ids.add(identifier)
            comparisons.add(comparison)
    return data


def load_prompt_registry(path: Path) -> dict[str, Any]:
    """Read a private, operator-reviewed browser prompt experiment registry."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Prompt experiment registry is missing or linked.")
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size > 1024 * 1024:
        raise ValueError("Prompt experiment registry must be owner-only and bounded.")
    data = json.loads(path.read_text(encoding="utf-8"))
    experiments = data.get("experiments") if isinstance(data, dict) else None
    if (not isinstance(data, dict) or set(data) != {"schema", "experiments"}
            or data.get("schema") != "modellabs.prompt_experiment_registry.v1"
            or not isinstance(experiments, list)):
        raise ValueError("Unsupported prompt experiment registry.")
    seen: set[str] = set()
    allowed_root = (ROOT / "prompt-experiments").resolve()
    for experiment in experiments:
        if not isinstance(experiment, dict) or set(experiment) != {
                "id", "scenario_id", "guard_id", "prompt_file", "enabled"}:
            raise ValueError("Prompt experiment registry entry is malformed.")
        identifier = experiment.get("id")
        prompt_file = Path(str(experiment.get("prompt_file", "")))
        try:
            resolved = prompt_file.resolve(strict=True)
            resolved.relative_to(allowed_root)
        except (OSError, ValueError):
            raise ValueError("Prompt experiment file must be inside the private prompt root.") from None
        if (not isinstance(identifier, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", identifier)
                or identifier in seen
                or not isinstance(experiment.get("scenario_id"), str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}",
                                    experiment["scenario_id"])
                or not isinstance(experiment.get("guard_id"), str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}",
                                    experiment["guard_id"])
                or type(experiment.get("enabled")) is not bool
                or prompt_file.is_symlink() or not prompt_file.is_file()
                or prompt_file.stat().st_uid != os.getuid()
                or prompt_file.stat().st_mode & 0o077
                or prompt_file.stat().st_size > 256_000):
            raise ValueError("Prompt experiment registry entry is unsafe.")
        experiment["prompt_file"] = str(resolved)
        seen.add(identifier)
    return data


def explicit_inline_revision(state: dict[str, Any]) -> str | None:
    """Extract only a complete, explicitly quoted legacy inline revision."""
    verdict, evaluation = state.get("verdict") or {}, state.get("evaluation") or {}
    if (state.get("review_format") != "inline" or state.get("status") != "completed"
            or not isinstance(verdict, dict) or not isinstance(evaluation, dict)
            or verdict.get("suggested_next_prompt") is not None
            or verdict.get("pass") is not False or verdict.get("nonce") != state.get("nonce")
            or evaluation.get("valid") is not True or evaluation.get("nonce_matched") is not True):
        return None
    summary = verdict.get("summary")
    if not isinstance(summary, str):
        return None
    matches = re.findall(r"A complete revised prompt would be:\s*\n\s*“([^”]+)”", summary)
    if len(matches) != 1 or not matches[0].strip() or len(matches[0]) > 12_000:
        return None
    return matches[0]


def verified_browser_revision(guard_id: str, prompt: str, origin_prompt: str,
                              guard_root: Path = GUARD_ROOT,
                              runs_root: Path = RUNS_ROOT) -> dict[str, str]:
    """Bind an exact browser-proposed prompt to a completed, nonce-checked review."""
    if not isinstance(guard_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", guard_id):
        raise ValueError("Invalid browser review ID")
    path = guard_root / f"{guard_id}.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Browser review is missing or linked")
    state = json.loads(path.read_text(encoding="utf-8"))
    verdict, evaluation = state.get("verdict") or {}, state.get("evaluation") or {}
    source, trace = state.get("source_files") or {}, state.get("learning_trace") or {}
    if (state.get("schema") != "codex.pro_guard_run.v1" or state.get("guard_id") != guard_id
            or state.get("status") != "completed" or state.get("review_format") not in {"inline", "file"}
            or not isinstance(state.get("response_id"), str) or not state["response_id"].startswith("resp_")
            or not isinstance(state.get("conversation_url"), str)
            or not state["conversation_url"].startswith("https://chatgpt.com/")
            or evaluation.get("valid") is not True or evaluation.get("nonce_matched") is not True
            or verdict.get("nonce") != state.get("nonce") or verdict.get("pass") is not False
            or trace.get("root_guard_id") != guard_id or trace.get("parent_guard_id") is not None
            or trace.get("origin_prompt_sha256") != (source.get("origin_prompt") or {}).get("sha256")
            or trace.get("generation_prompt_sha256") != (source.get("generation_prompt") or {}).get("sha256")):
        raise ValueError("Browser review lacks completed, bound evidence")
    for name in ("origin_prompt", "generation_prompt", "artifact"):
        entry = source.get(name) or {}
        source_path = Path(entry.get("path", ""))
        if (source_path.is_symlink() or not source_path.is_file()
                or hashlib.sha256(source_path.read_bytes()).hexdigest() != entry.get("sha256")):
            raise ValueError("Browser review source changed")
    origin_sha256 = source["origin_prompt"]["sha256"]
    if (source["generation_prompt"]["sha256"] != origin_sha256
            or (state["review_format"] == "inline" and source["artifact"]["sha256"] != origin_sha256)
            or Path(source["origin_prompt"]["path"]).read_text(encoding="utf-8").strip() != origin_prompt):
        raise ValueError("Browser review is for another prompt")
    if state["review_format"] == "file":
        # A file review is only browser-authored evidence when the complete
        # provider handoff and the actual sent-turn attachment UI are present.
        import outcome_model
        try:
            if outcome_model._guard_row(path, runs_root, b"benchmark-review-validation") is None:
                raise ValueError("Browser review lacks a verified provider handoff")
            record = json.loads((runs_root / state["response_id"] / "record.json").read_text(encoding="utf-8"))
            bundle = record.get("bundle") or {}
            requested = ((bundle.get("run") or {}).get("initialInputs") or {}).get("attachments") or []
            expected_paths = []
            for item in requested:
                uri = urllib.parse.urlparse(item["uri"])
                if uri.scheme != "file" or uri.netloc not in {"", "localhost"}:
                    raise ValueError("Browser review attachment is not a local file")
                expected_paths.append(urllib.request.url2pathname(uri.path))
            browser_runs = [((step.get("output") or {}).get("structuredData") or {}).get("browserRun")
                            for step in bundle.get("steps") or [] if isinstance(step, dict)
                            and step.get("status") == "succeeded"]
            browser_runs = [run for run in browser_runs if isinstance(run, dict)]
            if len(browser_runs) != 1:
                raise ValueError("Browser review lacks one succeeded browser run")
            run = browser_runs[0]
            transport = (run.get("promptTransport") or {}).get("attachments") or []
            paths = [item.get("path") for item in transport if isinstance(item, dict)]
            receipt = run.get("attachmentUiReceipt") or {}
            extra_request_attachment = (len(paths) == len(expected_paths) + 1
                                        and (run.get("promptTransport") or {}).get("metadata", {}).get("mode")
                                        == "request_attachment"
                                        and transport[-1].get("displayPath") == "auracall-request.txt")
            if (run.get("service") != "chatgpt" or run.get("tabUrl") != state["conversation_url"]
                    or run.get("runtimeProfileId") != "agent-browser-chatgpt"
                    or len(paths) != len(transport) or paths[:len(expected_paths)] != expected_paths
                    or not (len(paths) == len(expected_paths) or extra_request_attachment)
                    or receipt.get("schema") != "auracall.browser_attachment_ui_receipt.v1"
                    or receipt.get("attachmentPaths") != paths
                    or receipt.get("uploadCompletion") != "confirmed"
                    or receipt.get("sentUserTurnAttachments") != "confirmed"
                    or not isinstance(receipt.get("submittedUserId"), str)
                    or not receipt["submittedUserId"].strip()):
                raise ValueError("Browser review lacks sent attachment UI proof")
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Browser review provider evidence is incomplete") from error
    suggestion = verdict.get("suggested_next_prompt")
    if suggestion is None:
        if state["review_format"] != "inline":
            raise ValueError("File review has no explicit prompt revision")
        suggestion = explicit_inline_revision(state)
        if suggestion is None:
            raise ValueError("Browser review has no unambiguous prompt revision")
    if not isinstance(suggestion, str) or prompt != suggestion:
        raise ValueError("Prompt does not match browser revision")
    return {"guard_id": guard_id, "response_id": state["response_id"],
            "origin_sha256": origin_sha256}


def browser_prompt_variant(base: dict[str, Any], guard_id: str, prompt_file: Path,
                           guard_root: Path = GUARD_ROOT,
                           runs_root: Path = RUNS_ROOT) -> dict[str, Any]:
    if prompt_file.is_symlink() or not prompt_file.is_file():
        raise ValueError("Browser revision file is missing or linked")
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    evidence = verified_browser_revision(guard_id, prompt, base["prompt"], guard_root, runs_root)
    return {**base, "id": f"{base['id']}_browser_{guard_id}", "prompt": prompt,
            "variant_of": base["id"], "prompt_author": "chatgpt",
            "browser_review": evidence, "browser_origin_prompt": base["prompt"]}


def materialize_browser_revision(guard_id: str, output_file: Path,
                                 guard_root: Path = GUARD_ROOT,
                                 runs_root: Path = RUNS_ROOT) -> dict[str, Any]:
    """Write one exact verified structured or legacy-inline revision exclusively."""
    if not isinstance(guard_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", guard_id):
        raise ValueError("Invalid browser review ID")
    guard_path = guard_root / f"{guard_id}.json"
    if guard_path.is_symlink() or not guard_path.is_file() or guard_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Browser review is missing, linked, or oversized")
    state = json.loads(guard_path.read_text(encoding="utf-8"))
    verdict = state.get("verdict") or {}
    revision = verdict.get("suggested_next_prompt")
    if revision is None:
        revision = explicit_inline_revision(state)
    source = ((state.get("source_files") or {}).get("origin_prompt") or {})
    origin_path = Path(source.get("path", ""))
    if (not isinstance(revision, str) or not revision.strip()
            or origin_path.is_symlink() or not origin_path.is_file()):
        raise ValueError("Browser review has no verified complete revision")
    origin = origin_path.read_text(encoding="utf-8").strip()
    evidence = verified_browser_revision(guard_id, revision, origin, guard_root, runs_root)
    parent = output_file.expanduser().parent.resolve(strict=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("Browser revision output directory is unavailable")
    target = parent / output_file.name
    if target != output_file.expanduser().resolve(strict=False) or target.name in {"", ".", ".."}:
        raise ValueError("Browser revision output path is unsafe")
    payload = (revision.strip() + "\n").encode()
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    return {"schema": "modellabs.browser-prompt-revision.v1",
            "guard_id": evidence["guard_id"], "response_id": evidence["response_id"],
            "output_file": str(target), "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload)}


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
          usage: dict[str, int], codex_exit_code: int = 0) -> dict[str, Any]:
    protected = scenario.get("protected_files") or []
    def unchanged(name: str) -> bool:
        path = workspace / name
        safe_child(workspace, name)
        return (not path.is_symlink() and path.is_file()
                and path.read_text(encoding="utf-8") == str(scenario["files"][name]))
    fixture_integrity = bool(protected) and all(unchanged(name) for name in protected)
    verifier_exit_code: int | None = None
    if fixture_integrity and codex_exit_code == 0:
        sandbox = shutil.which("bwrap")
        if sandbox is None:
            raise RuntimeError("bubblewrap is required for isolated benchmark verification")
        command = [sandbox, "--die-with-parent", "--unshare-net", "--unshare-pid",
                   "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
                   "--ro-bind", "/lib64", "/lib64", "--proc", "/proc",
                   "--dev-bind", "/dev", "/dev", "--tmpfs", "/tmp",
                   "--ro-bind", str(workspace), str(workspace), "--chdir", str(workspace),
                   *scenario["verify"]]
        checked = subprocess.run(command, cwd=workspace, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, timeout=int(scenario.get("verify_timeout", 30)),
                                 env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
        verifier_exit_code = checked.returncode
    product_pass = verifier_exit_code == 0 and fixture_integrity and codex_exit_code == 0
    exact = final.strip() == scenario["expected_final"]
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    total_tokens = input_tokens + output_tokens
    # Artifact usability is the hard gate. Response-format compliance matters,
    # but cannot rescue a broken product.
    satisfaction = (90 + (10 if exact else 0)) if product_pass else 0
    return {"product_pass": product_pass, "fixture_integrity": fixture_integrity,
            "exact_final_response": exact,
            "satisfaction_score": satisfaction, "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
            "output_tokens": output_tokens,
            "reasoning_output_tokens": int(usage.get("reasoning_output_tokens", 0)),
            "verifier_exit_code": verifier_exit_code}


def run_one(scenario: dict[str, Any], model: str, effort: str, suite_id: str,
            codex: str = "codex", record_metrics: bool = False,
            sequence_index: int | None = None,
            repetition_index: int | None = None) -> dict[str, Any]:
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
        result = {"schema": "modellabs.benchmark-result.v2", "suite_id": suite_id,
                  "run_id": str(uuid.uuid4()), "sequence_index": sequence_index,
                  "repetition_index": repetition_index,
                  "scenario_id": scenario["id"], "scenario_sha256": stable_digest(scenario),
                  "product_sha256": stable_digest(product_definition(scenario)),
                  "prompt_sha256": hashlib.sha256(scenario["prompt"].encode()).hexdigest(),
                  "comparison_id": scenario.get("comparison_id"),
                  "variant_of": scenario.get("variant_of"),
                  "prompt_author": scenario.get("prompt_author", "benchmark"),
                  "browser_review": scenario.get("browser_review"),
                  "task_class": scenario["task_class"], "model": model, "effort": effort,
                  "model_provenance": "cli_requested_only", "observed_model": None,
                  "codex_exit_code": completed.returncode, "elapsed_ms": elapsed_ms,
                  **grade(workspace, scenario, final, usage, completed.returncode)}
    if record_metrics:
        record("benchmark_result", receipt_id=f"benchmark:{result['run_id']}", **result)
        try:
            from outcome_model import capture_benchmark_prompt_run
            captured = capture_benchmark_prompt_run(result, scenario, final)
            result["learning_capture"] = "captured" if captured else "ineligible"
        except Exception as exc:
            result["learning_capture"] = f"failed:{type(exc).__name__}"
    return result


async def _managed_turn(workspace: Path, prompt: str, model: str, effort: str,
                        timeout: int) -> tuple[str, str, str, str]:
    """Run one ephemeral product turn through the authenticated proxy."""
    from host_control import _read_token
    from model_host_launcher import PROXY_URL

    async def receive_until(websocket: Any, request_id: int, *, terminal: bool = False) -> tuple[Any, str | None, str | None]:
        answer = None
        turn_id = None
        while True:
            message = json.loads(await asyncio.wait_for(websocket.recv(), timeout=timeout))
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(f"managed benchmark RPC {request_id} failed")
                if not terminal:
                    return message.get("result") or {}, answer, turn_id
                turn_id = ((message.get("result") or {}).get("turn") or {}).get("id") or turn_id
            method = message.get("method")
            params = message.get("params") or {}
            if method == "item/completed":
                item = params.get("item") or {}
                if item.get("type") == "agentMessage" and isinstance(item.get("text"), str):
                    answer = item["text"]
            elif method == "turn/started":
                turn_id = (params.get("turn") or {}).get("id") or turn_id
            elif method == "turn/completed":
                return params, answer, turn_id
            if message.get("id") is not None and method is not None:
                await websocket.send(json.dumps({"id": message["id"], "error": {
                    "code": -32001, "message": "managed benchmark refuses interactive requests"}}))

    headers = {"Authorization": f"Bearer {_read_token()}.explicit-both"}
    async with websockets.connect(PROXY_URL, additional_headers=headers,
                                  max_size=64 * 1024 * 1024) as websocket:
        await websocket.send(json.dumps({"id": 1, "method": "initialize", "params": {
            "clientInfo": {"name": "modellabs-managed-benchmark", "version": "1"},
            "capabilities": {"experimentalApi": True}}}))
        await receive_until(websocket, 1)
        await websocket.send(json.dumps({"method": "initialized"}))
        await websocket.send(json.dumps({"id": 2, "method": "thread/start", "params": {
            "cwd": str(workspace), "ephemeral": True, "sandbox": "workspace-write",
            "approvalPolicy": "never"}}))
        started, _answer, _turn = await receive_until(websocket, 2)
        thread_id = (started.get("thread") or {}).get("id")
        if not isinstance(thread_id, str):
            raise RuntimeError("managed benchmark host returned no thread identity")
        await websocket.send(json.dumps({"id": 3, "method": "turn/start", "params": {
            "threadId": thread_id, "model": model, "effort": effort,
            "input": [{"type": "text", "text": prompt}]}}))
        completed, answer, turn_id = await receive_until(websocket, 3, terminal=True)
    status = (completed.get("turn") or {}).get("status")
    if not isinstance(turn_id, str) or not isinstance(answer, str):
        raise RuntimeError("managed benchmark turn has no exact result identity")
    return thread_id, turn_id, status, answer


def run_managed_one(scenario: dict[str, Any], model: str, effort: str,
                    suite_id: str, block: dict[str, Any], arm_index: int,
                    sequence_index: int, repetition_index: int,
                    record_metrics: bool) -> dict[str, Any]:
    """Execute, independently verify, grade, and bind one managed benchmark arm."""
    if not record_metrics:
        raise ValueError("managed benchmarks require --record-metrics")
    from adaptive_policy import record_explicit_grade
    from outcome_model import (capture_managed_benchmark_prompt_run,
                               managed_turn_evidence, sync_codex_grades)

    run_id = str(uuid.uuid4())
    started_at = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"modellabs-managed-{scenario['id']}-") as temporary:
        workspace = Path(temporary)
        materialize(workspace, scenario)
        thread_id, turn_id, status, final = asyncio.run(_managed_turn(
            workspace, scenario["prompt"], model, effort,
            int(scenario.get("agent_timeout", 300))))
        if status != "completed":
            failure = {"schema": "modellabs.managed-benchmark-attempt.v1",
                       "suite_id": suite_id, "block_id": block["block_id"],
                       "prompt_set_id": block.get("prompt_set_id"),
                       "prompt_arm_index": block.get("prompt_arm_index"),
                       "arm_index": arm_index, "run_id": run_id,
                       "scenario_id": scenario["id"], "model": model, "effort": effort,
                       "thread_id": thread_id, "turn_id": turn_id, "status": status,
                       "sequence_index": sequence_index,
                       "repetition_index": repetition_index, "captured": False,
                       "product_pass": False, "satisfaction_score": 0,
                       "total_tokens": 0}
            record("managed_benchmark_attempt", receipt_id=f"managed-attempt:{run_id}", **failure)
            return failure
        evidence = managed_turn_evidence(thread_id, turn_id, require_grade=False)
        usage = evidence["usage"]
        assessed = grade(workspace, scenario, final, {
            "input_tokens": usage["inputTokens"],
            "cached_input_tokens": usage.get("cachedInputTokens", 0),
            "output_tokens": usage["outputTokens"],
            "reasoning_output_tokens": usage.get("reasoningOutputTokens", 0)}, 0)
        result = {"schema": "modellabs.managed-benchmark-result.v1",
                  "suite_id": suite_id, "block_id": block["block_id"],
                  "prompt_set_id": block.get("prompt_set_id"),
                  "prompt_arm_index": block.get("prompt_arm_index"),
                  "arm_index": arm_index, "run_id": run_id,
                  "sequence_index": sequence_index, "repetition_index": repetition_index,
                  "scenario_id": scenario["id"], "scenario_sha256": stable_digest(scenario),
                  "product_sha256": stable_digest(product_definition(scenario)),
                  "prompt_sha256": hashlib.sha256(scenario["prompt"].encode()).hexdigest(),
                  "comparison_id": scenario.get("comparison_id"),
                  "variant_of": scenario.get("variant_of"),
                  "prompt_author": scenario.get("prompt_author", "benchmark"),
                  "browser_review": scenario.get("browser_review"),
                  "scenario_task_class": scenario["task_class"],
                  "task_class": block["routing_task_class"],
                  "model": model, "effort": effort,
                  "thread_id": thread_id, "turn_id": turn_id, "status": status,
                  "codex_exit_code": 0,
                  "elapsed_ms": round((time.monotonic() - started_at) * 1000), **assessed}
        verification = "passed" if assessed["product_pass"] else "failed"
        record_explicit_grade(thread_id, turn_id, assessed["satisfaction_score"], verification)
        sync_codex_grades()
        captured = capture_managed_benchmark_prompt_run(result, scenario, final, workspace)
        if not captured:
            raise RuntimeError("managed benchmark evidence binding failed closed")
        record("managed_benchmark_result", receipt_id=f"managed-benchmark:{run_id}",
               **{key: value for key, value in result.items() if key != "browser_review"},
               assignment_commitment=block["commitment"],
               assignment_probability=block["assignment_probability"],
               order_position_probability=block["order_position_probability"],
               randomization_estimand=block["randomization_estimand"],
               prompt_assignment_commitment=block.get("prompt_set_commitment"),
               captured=True)
        result["learning_capture"] = "captured"
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


def daily_managed_selection(
        scenarios: list[dict[str, Any]],
        experiments: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, list[tuple[str, str]], str]:
    """Choose the least-sampled product without crossing the prospective boundary early."""
    from modellabs import route
    from outcome_model import (MAX_MANAGED_FAILURES_PER_PRODUCT, managed_scenario_attempts,
                               managed_scenario_progress, train_managed_routing_policy)

    if experiments:
        by_id = {scenario["id"]: scenario for scenario in scenarios}
        policy = train_managed_routing_policy(create_checkpoint=False)
        fallback_arms: list[tuple[str, str]] = []
        for experiment in experiments:
            if not experiment["enabled"]:
                continue
            arms = [tuple(arm) for arm in experiment["arms"]]
            if not fallback_arms:
                fallback_arms = arms
            comparison = next((row for row in policy["comparisons"]
                               if {tuple(row["left_arm"]), tuple(row["right_arm"])} == set(arms)
                               and row["task_class"] == experiment["task_class"]), None)
            phase = ("prospective" if comparison
                     and comparison["development_checkpoint_created"] else "development")
            scenario_ids = experiment[f"{phase}_scenario_ids"]
            candidates = [by_id[value] for value in scenario_ids
                          if value in by_id
                          and by_id[value].get("task_class") == experiment["task_class"]
                          and route(str(by_id[value].get("prompt", ""))).get("class")
                          == experiment["task_class"]]
            if not candidates:
                continue
            progress = managed_scenario_progress(candidates, arms)
            attempts = managed_scenario_attempts(candidates, arms)
            pending = [scenario for scenario in candidates
                       if progress.get(scenario["id"], 0) < 3
                       and attempts.get(scenario["id"], 0) - progress.get(scenario["id"], 0)
                       < MAX_MANAGED_FAILURES_PER_PRODUCT]
            if not pending:
                continue
            selected = min(pending, key=lambda scenario: (
                progress.get(scenario["id"], 0),
                attempts.get(scenario["id"], 0) - progress.get(scenario["id"], 0),
                scenario["id"]))
            return selected, arms, f"{phase}:{experiment['id']}"
        return (None, fallback_arms or [("gpt-6-luna", "low"), ("gpt-6-sol", "medium")],
                "registered_managed_experiments_complete_or_quarantined")

    tagged = [scenario for scenario in scenarios if scenario.get("managed_cohort") in {
        "routine_luna_vs_sol_development", "routine_luna_vs_sol_prospective"}]
    eligible = [scenario for scenario in tagged
                if scenario.get("task_class") == "routine"
                and route(str(scenario.get("prompt", ""))).get("class") == "routine"]
    arms = [("gpt-6-luna", "low"), ("gpt-6-sol", "medium")]
    policy = train_managed_routing_policy(create_checkpoint=False)
    comparison = next((row for row in policy["comparisons"]
                       if {tuple(row["left_arm"]), tuple(row["right_arm"])} == set(arms)
                       and row["task_class"] == "routine"), None)
    phase = ("prospective" if comparison
             and comparison["development_checkpoint_created"] else "development")
    cohort = f"routine_luna_vs_sol_{phase}"
    candidates = [scenario for scenario in eligible if scenario["managed_cohort"] == cohort]
    if not candidates:
        reason = (f"no_valid_{phase}_cohort" if any(
            scenario.get("managed_cohort") == cohort for scenario in tagged)
                  else f"no_{phase}_cohort")
        return None, arms, reason
    progress = managed_scenario_progress(candidates, arms)
    attempts = managed_scenario_attempts(candidates, arms)
    pending = [scenario for scenario in candidates
               if progress.get(scenario["id"], 0) < 3
               and attempts.get(scenario["id"], 0) - progress.get(scenario["id"], 0)
               < MAX_MANAGED_FAILURES_PER_PRODUCT]
    if not pending:
        incomplete = any(progress.get(scenario["id"], 0) < 3 for scenario in candidates)
        reason = (f"{phase}_failure_budget_exhausted" if incomplete
                  else f"{phase}_cohort_complete")
        return None, arms, reason
    selected = min(pending, key=lambda scenario: (progress.get(scenario["id"], 0),
                                                   attempts.get(scenario["id"], 0)
                                                   - progress.get(scenario["id"], 0),
                                                   scenario["id"]))
    return selected, arms, phase


def daily_prompt_selection(
        scenarios: list[dict[str, Any]], registry: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, list[tuple[str, str]], str]:
    """Choose one reviewed browser revision without crossing its checkpoint boundary."""
    from modellabs import route
    from outcome_model import (MAX_MANAGED_FAILURES_PER_PRODUCT,
                               managed_prompt_experiment_counts,
                               train_managed_prompt_policy)

    arms = [("gpt-6-luna", "low"), ("gpt-6-sol", "medium")]
    policy = train_managed_prompt_policy(create_checkpoint=False)
    comparisons = [row for row in policy["comparisons"]
                   if row.get("task_class") == "routine"
                   and (row.get("model"), row.get("effort")) in set(arms)]
    phase = ("prospective" if len(comparisons) == len(arms)
             and all(row.get("development_checkpoint_created") for row in comparisons)
             else "development")
    cohort = f"routine_luna_vs_sol_{phase}"
    by_id = {scenario["id"]: scenario for scenario in scenarios}
    candidates = []
    for experiment in registry["experiments"]:
        if not experiment["enabled"]:
            continue
        base = by_id.get(experiment["scenario_id"])
        if (base is None or base.get("managed_cohort") != cohort
                or base.get("task_class") != "routine"
                or route(str(base.get("prompt", ""))).get("class") != "routine"):
            continue
        candidate = browser_prompt_variant(base, experiment["guard_id"],
                                           Path(experiment["prompt_file"]))
        counts = managed_prompt_experiment_counts(base, candidate, arms)
        if (counts["complete"] < 3
                and counts["failed_or_incomplete"] < MAX_MANAGED_FAILURES_PER_PRODUCT):
            candidates.append((counts["complete"], counts["failed_or_incomplete"],
                               experiment["id"], base, candidate))
    if not candidates:
        matching = [entry for entry in registry["experiments"] if entry["enabled"]
                    and (by_id.get(entry["scenario_id"]) or {}).get("managed_cohort") == cohort]
        return None, arms, (f"no_registered_{phase}_prompt_experiments" if not matching
                            else f"registered_{phase}_prompt_experiments_complete_or_quarantined")
    _complete, _failures, identifier, base, candidate = min(candidates)
    return [base, candidate], arms, f"prompt_{phase}:{identifier}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run verified ModelLabs smoke prompts")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--pair", action="append", help="MODEL:EFFORT; repeat to override the scenario matrix")
    parser.add_argument("--record-metrics", action="store_true")
    parser.add_argument("--managed", action="store_true",
                        help="run randomized arms through the proxy with causal receipts")
    parser.add_argument("--daily-managed", action="store_true",
                        help="run one bounded phase-aware managed block")
    parser.add_argument("--daily-browser-prompts", action="store_true",
                        help="run one bounded reviewed browser-prompt block")
    parser.add_argument("--prompt-registry", type=Path, default=DEFAULT_PROMPT_REGISTRY)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--browser-variant-guard-id")
    parser.add_argument("--browser-variant-prompt-file", type=Path)
    parser.add_argument("--materialize-browser-revision-guard-id")
    parser.add_argument("--materialize-browser-revision-output", type=Path)
    args = parser.parse_args()
    if (args.materialize_browser_revision_guard_id
            or args.materialize_browser_revision_output):
        if (not args.materialize_browser_revision_guard_id
                or not args.materialize_browser_revision_output
                or args.daily_managed or args.daily_browser_prompts
                or args.managed or args.record_metrics
                or args.scenario or args.pair or args.repeat != 1
                or args.browser_variant_guard_id or args.browser_variant_prompt_file):
            parser.error("browser revision materialization requires only its guard ID and output")
        print(json.dumps(materialize_browser_revision(
            args.materialize_browser_revision_guard_id,
            args.materialize_browser_revision_output), sort_keys=True))
        return
    if args.daily_managed:
        if (args.daily_browser_prompts or args.managed or args.record_metrics or args.scenario or args.pair
                or args.repeat != 1 or args.browser_variant_guard_id
                or args.browser_variant_prompt_file):
            parser.error("--daily-managed cannot be combined with manual matrix options")
        args.managed = True
        args.record_metrics = True
    if args.daily_browser_prompts:
        if (args.daily_managed or args.managed or args.record_metrics or args.scenario or args.pair
                or args.repeat != 1 or args.browser_variant_guard_id
                or args.browser_variant_prompt_file):
            parser.error("--daily-browser-prompts cannot be combined with manual matrix options")
        args.managed = True
        args.record_metrics = True
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.managed and not args.record_metrics:
        parser.error("--managed requires --record-metrics")
    manifest = load_manifest(args.manifest)
    selected = set(args.scenario or [])
    scenarios = [item for item in manifest["scenarios"] if not selected or item["id"] in selected]
    if selected - {item["id"] for item in scenarios}:
        raise SystemExit("Unknown scenario requested.")
    if bool(args.browser_variant_guard_id) != bool(args.browser_variant_prompt_file):
        parser.error("browser variant requires both guard ID and prompt file")
    if args.browser_variant_guard_id:
        if len(scenarios) != 1 or scenarios[0].get("variant_of") is not None:
            parser.error("browser variant requires one base scenario")
        scenarios = [scenarios[0], browser_prompt_variant(
            scenarios[0], args.browser_variant_guard_id, args.browser_variant_prompt_file)]
    override = [tuple(item.rsplit(":", 1)) for item in (args.pair or [])]
    daily_phase = None
    if args.daily_browser_prompts:
        chosen_prompts, daily_arms, daily_phase = daily_prompt_selection(
            scenarios, load_prompt_registry(args.prompt_registry))
        if chosen_prompts is None:
            print(json.dumps({"schema": "modellabs.managed-prompt-daily-skip.v1",
                              "reason": daily_phase}, sort_keys=True))
            return
        scenarios = chosen_prompts
        override = daily_arms
    if args.daily_managed:
        chosen, daily_arms, daily_phase = daily_managed_selection(
            scenarios, manifest.get("managed_experiments"))
        if chosen is None:
            print(json.dumps({"schema": "modellabs.managed-daily-skip.v1",
                              "reason": daily_phase}, sort_keys=True))
            return
        scenarios = [chosen]
        override = daily_arms
    suite_id = str(uuid.uuid4())
    results = []
    if args.managed:
        from model_host_launcher import ensure_proxy, ensure_proxy_supervisor
        from outcome_model import create_managed_benchmark_block, create_managed_prompt_set

        ensure_proxy()
        ensure_proxy_supervisor()
        for repetition in range(args.repeat):
            prompt_experiment = (len(scenarios) >= 2
                                 and any(item.get("prompt_author") == "chatgpt"
                                         for item in scenarios))
            prompt_set = (create_managed_prompt_set(
                suite_id, str(uuid.uuid4()), scenarios, repetition)
                          if prompt_experiment else None)
            if prompt_set:
                by_digest = {stable_digest(scenario): scenario for scenario in scenarios}
                ordered_scenarios = [by_digest[item["scenario_sha256"]]
                                     for item in prompt_set["prompts"]]
            else:
                ordered_scenarios = scenarios
            execution_plan = []
            for prompt_arm_index, scenario in enumerate(ordered_scenarios):
                pairs = override or [tuple(item) for item in scenario["matrix"]]
                if len(set(pairs)) < 2:
                    parser.error("every managed scenario requires at least two distinct --pair arms")
                block = create_managed_benchmark_block(
                    suite_id, str(uuid.uuid4()), scenario, list(dict.fromkeys(pairs)), repetition,
                    prompt_set_id=prompt_set["prompt_set_id"] if prompt_set else None,
                    prompt_arm_index=prompt_arm_index if prompt_set else None)
                execution_plan.append((scenario, block))
            for scenario, block in execution_plan:
                for arm_index, (model, effort) in enumerate(block["arms"]):
                    results.append(run_managed_one(
                        scenario, model, effort, suite_id, block, arm_index,
                        len(results), repetition, args.record_metrics))
        print(json.dumps({"schema": "modellabs.managed-smoke-report.v1",
                          "suite_id": suite_id, "daily_phase": daily_phase,
                          "results": ranked(results)}, sort_keys=True))
        return
    for repetition in range(args.repeat):
        # Alternate order so prompt or model comparisons are not always
        # confounded with the first run's host/cache state.
        ordered_scenarios = scenarios if repetition % 2 == 0 else list(reversed(scenarios))
        for scenario in ordered_scenarios:
            pairs = override or [tuple(item) for item in scenario["matrix"]]
            if repetition % 2:
                pairs = list(reversed(pairs))
            for model, effort in pairs:
                results.append(run_one(scenario, model, effort, suite_id, args.codex,
                                       args.record_metrics, len(results), repetition))
    print(json.dumps({"schema": "modellabs.smoke-report.v1", "suite_id": suite_id,
                      "results": ranked(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
