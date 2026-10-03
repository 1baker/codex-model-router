# ModelLabs orientation for Claude Code

ModelLabs is a local router for interactive Codex CLI chats. Its managed launcher
connects the Codex TUI to an authenticated loopback proxy and app-server. Before
each managed `turn/start`, the proxy chooses a listed model and reasoning effort
from the user's text, honors explicit choices, and provides a tool shortlist.
The original user text is forwarded unchanged. New threads can scope their MCP
servers; later tool shortlists are advice because the attached MCP set cannot
change per turn.

## Start with these files

- `README.md`: behavior, boundaries, adaptive evidence, operation, and checks.
- `modellabs.py`: task classification, contextual follow-ups, model/effort and
  tool selection, and the route preview command.
- `turn_proxy.py`: authenticated turn admission, context lookup, ownership,
  receipt reconciliation, and relay behavior.
- `host_control.py`: exact active-thread model control and compatibility checks.
- `model_host_launcher.py`, `proxy_supervisor.py`, `install.py`: process and
  installation lifecycle, immutable proxy generations, and guarded updates.
- `telemetry.py`, `receipt_journal.py`, `authority.py`, `thread_owner.py`:
  prompt-free evidence and ownership safeguards.
- `adaptive_policy.py`, `outcome_model.py`, `smoke_bench.py`: shadow routing
  evidence, outcome learning, and benchmark gates.
- `tests/`: routing, proxy, installer, receipt, and release behavior.

## Current design contracts

- Defaults: short checked work uses GPT-6 Luna/low; routine work uses GPT-6
  Sol/medium; difficult work uses Sol/high; consequential work uses GPT-6
  Astra, with effort based on the task. The live Codex catalog admits choices.
- A short continuation such as `ok go`, `proceed`, or `yes please` inherits its
  preceding task class, effort, and tool context.
  Referentials inherit difficult or consequential risk, while a simple
  question can remain simple. Explicit model and effort selections take priority.
- Active-turn switching applies only to the exact live managed thread and a
  later inference step. A known Node REPL review-contract mismatch is rejected
  before the host call. An accepted settings update alone is not proof that a
  later model actually ran. A durable switch-attempt marker excludes that turn
  from single-model execution evidence.
- The TUI owns approvals, input, cancellation, and interrupts. The proxy uses
  exact thread ownership and admission receipts. Do not attach a second owner
  to a live standalone or managed conversation.
- Telemetry stores route and usage metadata, not prompt or response text.
  Completion, exact usage, and actual execution have separate evidence. An
  unavailable usage receipt is preferable to invented precision.
- Adaptive routing is in shadow mode by default. Outcome or benchmark evidence
  must pass its existing gates before it can change a standing route.
- Proxy generations are immutable, digest-bound bundles on separate loopback
  ports. Candidate bundles must pass an import preflight in the installed
  environment. Existing chats may still use an older generation. Do not restart or
  retire a live owner or proxy just to test a candidate.
- The working tree may contain uncommitted changes. Inspect `git status` and
  `git diff` before proposing edits, and preserve unrelated work.

## Useful checks

Use the installed virtual environment for the test suite:

    /home/bak3r/.local/share/model-selector/venv/bin/python -m unittest discover -s tests

Preview a route with `modellabs route 'prompt'`. A route preview is a prediction;
only an accepted managed turn plus completion, usage, and execution receipts
proves the live path. Keep reviews read-only unless the user asks for a change.
