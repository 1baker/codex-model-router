"""Codex UserPromptSubmit hook providing advisory routing context only.

The authenticated proxy is the sole routing authority. A post-admission hook
must never overwrite a model or effort selected before host admission.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from modellabs import record_route, route


CONTINUATIONS = {"ok go", "go ahead", "continue", "yes", "do it"}


def prior_user_prompt(transcript_path: str | None, current: str) -> str | None:
    if not transcript_path:
        return None
    try:
        path = Path(transcript_path).resolve()
        if not path.is_relative_to(Path.home() / ".codex" / "sessions") or not path.is_file():
            return None
        with path.open("rb") as source:
            source.seek(max(0, path.stat().st_size - 1024 * 1024))
            if source.tell():
                source.readline()
            lines = source.readlines()
        for raw in reversed(lines):
            item = json.loads(raw)
            payload = item.get("payload", {})
            if item.get("type") != "response_item" or payload.get("role") != "user":
                continue
            message = "\n".join(c.get("text", "") for c in payload.get("content", []) if c.get("type") == "input_text")
            if message.strip() and message.strip() != current.strip() and message.lower().strip().rstrip(".!?") not in CONTINUATIONS:
                return message
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return None


def main() -> None:
    payload = json.load(sys.stdin)
    if payload.get("hook_event_name") != "UserPromptSubmit":
        return
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return
    context = prior_user_prompt(payload.get("transcript_path"), prompt) if prompt.lower().strip().rstrip(".!?") in CONTINUATIONS else None
    choice = route(context or prompt)
    choice["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    session_id = payload.get("session_id", "")
    turn_id = payload.get("turn_id", "")
    record = {**choice, "session_id": session_id, "turn_id": turn_id,
              "status": "advisory_only"}
    record_route(record)
    message = (f"ModelLabs route: {choice['model']}; intelligence slider: "
               f"{choice['intelligence_slider']} ({choice['effort']} reasoning effort); "
               f"suggested MCPs: {', '.join(choice['servers'])}. ")
    message += "Advisory only: the authenticated proxy is the sole routing authority. Do not claim a model or tool switch from this hook."
    if "agentBrowser" in choice["servers"]:
        message += (
            " For ChatGPT browser-backed intelligence, use AuraCall as the provider bridge with "
            "the semantic selector chatgpt:premium while agent-browser retains browser lifecycle "
            "ownership. Fail closed unless the completed receipt binds an observed Pro/premium "
            "selection, response and assistant-message identities, conversation, runtime profile, "
            "and browser account/profile; a requested selector alone is not execution proof."
        )
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                              "additionalContext": message}}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ModelLabs hook skipped: {type(exc).__name__}", file=sys.stderr)
