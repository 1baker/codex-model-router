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


ROOT = Path(__file__).parent
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
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max", "ultra")


def choose_effort(prompt: str, task_class: str, model: str) -> str:
    """Choose the least reasoning effort expected to reach a checked result."""
    p = prompt.lower()
    requested = re.search(r"\b(?:reasoning|intelligence)(?:\s+effort|\s+slider)?\s*(?:at|to|=|:)?\s*"
                          r"(low|medium|high|xhigh|max|ultra)\b", p)
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
    # Luna's live catalog currently ends at max. Do not submit an unsupported
    # ultra selection when a user requested it on a simple task.
    return "max" if effort == "ultra" and model == "gpt-5.6-luna" else effort


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
    requested = re.search(r"\b(?:use|run|route to|switch to)\s+(?:the\s+)?(gpt-6-astra|gpt-5\.6-(?:luna|terra|sol)|astra|luna|terra|sol)\b", p)
    if requested:
        alias = requested.group(1)
        model = alias if alias.startswith("gpt-") else {"astra": "gpt-6-astra", "luna": "gpt-5.6-luna",
                                                     "terra": "gpt-5.6-terra", "sol": "gpt-5.6-sol"}[alias]
    if model_override:
        model = model_override
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
    return {"class": task_class, "model": model, "effort": effort,
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
    root = Path(__file__).parent
    with (root / "usage-observer.log").open("ab") as log:
        subprocess.Popen(
            [sys.executable, str(root / "usage_observer.py"), "--thread-id", thread_id,
             "--turn-id", turn_id, "--model", choice["model"], "--effort", choice["effort"],
             "--task-class", choice["class"]],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )


async def start(prompt: str, cwd: str, override_model: str | None,
                override_effort: str | None) -> tuple[str, dict]:
    choice = route(prompt, override_model, override_effort)
    ensure_host()
    async with websockets.connect(HOST_URL,
                                  additional_headers={"Authorization": f"Bearer {_read_token()}"},
                                  max_size=8 * 1024 * 1024) as ws:
        await _rpc(ws, "initialize", {"clientInfo": {"name": "modellabs", "title": "ModelLabs", "version": "0.1"},
                                      "capabilities": {"experimentalApi": True}}, 1)
        await ws.send(json.dumps({"method": "initialized", "params": {}}))
        catalog = await _rpc(ws, "model/list", {}, 2)
        entry = next((m for m in catalog.get("data", []) if m.get("id") == choice["model"] and not m.get("hidden")), None)
        if entry is None:
            raise ValueError(f"Model {choice['model']} is unavailable on this host.")
        efforts = {e.get("reasoningEffort") for e in entry.get("supportedReasoningEfforts", [])}
        if choice["effort"] not in efforts:
            raise ValueError(f"Model {choice['model']} does not support {choice['effort']} effort.")
        begun = await _rpc(ws, "thread/start", {"cwd": cwd, "model": choice["model"], "experimentalRawEvents": True,
                                                 "config": config_for(choice["servers"])}, 3)
        thread_id = begun["thread"]["id"]
        choice["thread_id"] = thread_id
        # The first-turn hook must see the launcher choice before turn/start.
        record_route({**choice, "status": "initial_route"})
        # The first inference happens only after both the model and tool config are set.
        started = await _rpc(ws, "turn/start", {"threadId": thread_id,
                                                 "input": [{"type": "text", "text": prompt}],
                                                 "model": choice["model"], "effort": choice["effort"]}, 4)
        turn_id = (started.get("turn") or {}).get("id")
        if isinstance(turn_id, str):
            launch_usage_observer(thread_id, turn_id, choice)
    return thread_id, choice


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
    parser.add_argument("action", choices=["route", "start", "run"])
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("--prompt-file")
    parser.add_argument("--model")
    parser.add_argument("--effort")
    parser.add_argument("--cwd", default=os.getcwd())
    args = parser.parse_intermixed_args()
    prompt = read_prompt(args)
    if args.action == "route":
        print(json.dumps(route(prompt, args.model, args.effort), sort_keys=True))
        return
    thread_id, choice = asyncio.run(start(prompt, str(Path(args.cwd).resolve()), args.model, args.effort))
    print(json.dumps(choice, sort_keys=True), flush=True)
    if args.action == "run":
        ensure_proxy()
        env = os.environ.copy()
        env["MODEL_SELECTOR_HOST_TOKEN"] = _read_token()
        os.execvpe("codex", ["codex", "--remote", PROXY_URL,
                             "--remote-auth-token-env", "MODEL_SELECTOR_HOST_TOKEN",
                             "resume", thread_id], env)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"modellabs: {exc}", file=sys.stderr)
        raise SystemExit(1)
