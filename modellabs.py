"""Prompt-first intake for managed Codex chats.

The original prompt is sent unchanged to the app-server. Routing metadata never
contains prompt text, and server selection can only disable configured servers.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import websockets

from host_control import HOST_URL, _read_token, _rpc
from model_host_launcher import PROXY_URL, ensure_host, ensure_proxy
from paths import ROOT, real_codex_binary
from thread_owner import acquire_thread_ownership
from adaptive_policy import adapt, set_adaptive_mode
from telemetry import aggregate_usage, record as record_metric, usage_from


ROUTES = ROOT / "routes.jsonl"
SERVERS = frozenset({
    "agentBrowser", "cloudflare", "cloudflare-docs", "cloudflare-bindings",
    "cloudflare-builds", "cloudflare-observability", "openaiDeveloperDocs",
    "codegraph", "previews", "codexResearch", "litScout", "modelControl",
})
MODEL_BY_CLASS = {
    "simple": "gpt-5.6-luna",
    "routine": "gpt-5.6-terra",
    "difficult": "gpt-5.6-sol",
    "consequential": "gpt-6-astra",
}
EFFORT_LEVELS = ("none", "low", "medium", "high", "xhigh", "max", "ultra")


def choose_effort(prompt: str, task_class: str, model: str) -> str:
    """Choose the least reasoning effort expected to reach a checked result."""
    p = prompt.lower()
    requested = re.search(r"\b(?:reasoning|intelligence)(?:\s+effort|\s+slider)?\s*(?:at|to|=|:)?\s*"
                          r"(none|low|medium|high|xhigh|max|ultra)\b", p)
    if requested:
        effort = requested.group(1)
    elif re.search(r"\b(ultra|maximum rigor|exhaustive independent|leave no stone unturned)\b", p):
        effort = "ultra"
    elif re.search(r"\b(max(?:imum)? effort|formal verification|multi.?system (?:release|migration)|"
                   r"adversarial (?:review|audit))\b", p):
        effort = "max"
    elif task_class == "consequential" and re.search(r"\b(architecture|security|privacy|major refactor|"
                                                        r"cross.system|production|release|migration)\b", p):
        effort = "xhigh"
    elif task_class in {"consequential", "difficult"}:
        effort = "high"
    elif task_class == "routine":
        effort = "medium"
    else:
        effort = "low"
    # Preserve explicit choices: the live catalog gate must reject unsupported
    # combinations instead of silently lowering the requested intelligence.
    return effort


def explicit_effort_requested(prompt: str) -> bool:
    """Whether the user, rather than the router, selected a reasoning effort."""
    return bool(re.search(r"\b(?:reasoning|intelligence)(?:\s+effort|\s+slider)?\s*"
                          r"(?:at|to|=|:)?\s*(none|low|medium|high|xhigh|max|ultra)\b", prompt.lower()))


def route(prompt: str, model_override: str | None = None,
          effort_override: str | None = None) -> dict:
    if not prompt.strip():
        raise ValueError("A first prompt is required for prompt-first routing.")
    p = prompt.lower()
    words = len(prompt.split())
    high = bool(re.search(r"\b(architecture|security|privacy|production|release|deploy|migrat\w*|major refactor|end.to.end|cross.system|legal|medical|financial|high.stakes)\b", p))
    hard = bool(re.search(r"\b(debug|investigat\w*|root cause|race condition|intermittent|complex|optimi[sz]\w*|performance)\b", p))
    simple = words <= 55 and bool(re.search(r"\b(translate|format|extract|classify|summari[sz]e|rewrite|convert)\b", p)) and not high and not hard
    task_class = "consequential" if high else "difficult" if hard else "simple" if simple else "routine"
    model = MODEL_BY_CLASS[task_class]
    requested = re.search(r"\b(?:use|run|route to|switch to)\s+(?:the\s+)?(gpt-6[- ](?:astra|sol|luna)|gpt-5\.6[- ](?:luna|terra|sol)|astra|luna|terra|sol)\b", p)
    explicit_model = bool(requested or model_override)
    if requested:
        alias = requested.group(1).replace(" ", "-")
        model = alias if alias.startswith("gpt-") else {"astra": "gpt-6-astra", "luna": "gpt-5.6-luna",
                                                     "terra": "gpt-5.6-terra", "sol": "gpt-5.6-sol"}[alias]
    if model_override:
        model = model_override
    explicit_effort = explicit_effort_requested(prompt) or bool(effort_override)
    effort = choose_effort(prompt, task_class, model)
    if effort_override:
        effort = effort_override

    selected = {"modelControl"}
    if re.search(r"\b(web|browse|website|browser|chatgpt|pro session|online|internet)\b", p):
        selected.add("agentBrowser")
    if re.search(r"\b(research|literature|paper|citation|patent|grant|litscout)\b", p):
        selected.update({"codexResearch", "litScout"})
    if re.search(r"\b(codex research|graphiti|prior context|memory)\b", p):
        selected.add("codexResearch")
    if re.search(r"\b(codegraph|call graph|repository graph)\b", p):
        selected.add("codegraph")
    if re.search(r"\b(openai api|codex docs|codex documentation|openai docs)\b", p):
        selected.add("openaiDeveloperDocs")
    if re.search(r"\b(cloudflare|wrangler|durable object|turnstile)\b", p):
        selected.update(x for x in SERVERS if x.startswith("cloudflare"))
    if re.search(r"\b(preview|pdf|docx|pptx|xlsx|figure|image|report)\b", p):
        selected.add("previews")
    # A narrow bundle is a starting context choice, not an authorization gate.
    # Ambiguous requests retain evidence/browser access so they can be resolved.
    if p.strip().rstrip(".!?") in {"ok go", "go ahead", "continue", "yes", "do it"}:
        selected.update({"agentBrowser", "codexResearch", "litScout", "previews"})
    task_bucket = f"{task_class}:" + ",".join(sorted(selected))
    return {"class": task_class, "model": model, "effort": effort,
            "intelligence_slider": effort,
            "explicit_model": explicit_model, "explicit_effort": explicit_effort,
            "task_bucket": task_bucket,
            "servers": sorted(selected), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def config_for(selected: list[str]) -> dict:
    unknown = set(selected) - SERVERS
    if unknown:
        raise ValueError(f"Unknown server(s): {', '.join(sorted(unknown))}")
    return {"mcp_servers": {name: {"enabled": name in selected} for name in SERVERS}}


def record_route(choice: dict) -> None:
    ROUTES.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(ROUTES, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.chmod(ROUTES, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as out:
        out.write(json.dumps(choice, sort_keys=True) + "\n")


def launch_usage_observer(thread_id: str, turn_id: str, choice: dict) -> None:
    """Observe initial-turn usage without retaining user prompt text."""
    with (ROOT / "usage-observer.log").open("ab") as log:
        subprocess.Popen(
            [sys.executable, str(ROOT / "usage_observer.py"), "--thread-id", thread_id,
             "--turn-id", turn_id, "--model", choice["model"], "--effort", choice["effort"],
             "--task-class", choice["class"]],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )


async def start(prompt: str, cwd: str, override_model: str | None,
                override_effort: str | None) -> tuple[str, dict, int]:
    raise RuntimeError(
        "Headless first-turn submission was removed because it cannot relay interactive approvals. "
        "Use `modellabs run` or `codex-model-host start` so the real TUI owns the turn."
    )


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt:
        return " ".join(args.prompt)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("ModelLabs first prompt (finish with a line containing only .):", file=sys.stderr)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == ".":
            break
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Route a new Codex chat before its first model call")
    parser.add_argument("action", choices=["route", "start", "run", "outcome", "grade", "adaptive-mode"])
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("--prompt-file")
    parser.add_argument("--model")
    parser.add_argument("--effort")
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--thread-id")
    parser.add_argument("--turn-id")
    parser.add_argument("--outcome", choices=["verified", "retry"])
    parser.add_argument("--quality-score", type=int)
    parser.add_argument("--verification", choices=["passed", "failed"])
    parser.add_argument("--mode", choices=["shadow", "enforce"])
    args = parser.parse_intermixed_args()
    if args.action == "adaptive-mode":
        if not args.mode:
            parser.error("adaptive-mode requires --mode")
        set_adaptive_mode(args.mode)
        print(json.dumps({"adaptive_mode": args.mode}, sort_keys=True))
        return
    if args.action == "outcome":
        if not args.thread_id or not args.turn_id or not args.outcome:
            parser.error("outcome requires --thread-id, --turn-id, and --outcome")
        from adaptive_policy import record_explicit_outcome
        record_explicit_outcome(args.thread_id, args.turn_id, args.outcome)
        print(json.dumps({"recorded": "outcome_signal", "thread_id": args.thread_id,
                          "turn_id": args.turn_id, "outcome": args.outcome}, sort_keys=True))
        return
    if args.action == "grade":
        if not args.thread_id or not args.turn_id or args.quality_score is None or not args.verification:
            parser.error("grade requires --thread-id, --turn-id, --quality-score, and --verification")
        from adaptive_policy import record_explicit_grade
        grade = record_explicit_grade(args.thread_id, args.turn_id, args.quality_score, args.verification)
        print(json.dumps({"recorded": "quality_grade", **grade}, sort_keys=True))
        return
    prompt = read_prompt(args)
    if args.action == "route":
        print(json.dumps(adapt(route(prompt, args.model, args.effort)), sort_keys=True))
        return
    if args.action == "start":
        asyncio.run(start(prompt, str(Path(args.cwd).resolve()), args.model, args.effort))
        return
    command = [sys.executable, str(ROOT / "model_host_launcher.py"),
               "-C", str(Path(args.cwd).resolve())]
    if args.model:
        command.extend(["--model", args.model])
    if args.effort:
        command.extend(["--config", f'model_reasoning_effort="{args.effort}"'])
    if prompt:
        command.append(prompt)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"modellabs: {exc}", file=sys.stderr)
        raise SystemExit(1)
