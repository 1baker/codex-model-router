# ModelLabs

ModelLabs is a local Codex prompt router. It selects a model, reasoning effort,
and starting MCP shortlist before the host begins inference. A managed Codex
chat can be adjusted again on later turns through the protected local host.

ModelLabs writes a mode-0600 local `metrics.jsonl` file containing route choice,
host admission, completion status, elapsed time, and server-reported token
usage when Codex delivers it to the proxy. It records no prompt or response
text in telemetry.

For the initial turn, ModelLabs starts a short-lived local observer after turn
admission. It resumes the managed thread and records the exact upstream token
usage from `rawResponse/completed`; later turns use the managed proxy's event
stream and durable completion poller.

The launcher directs new managed chats to the current loopback proxy and starts
a local watchdog for that proxy. Older proxies are left in place for already
connected chats; no new managed launcher uses their port. Retire an older proxy
only after its connected chats have exited.

## Network boundary

The ModelLabs host and proxy bind only to `127.0.0.1`. They are not registered
with Traefik, Cloudflare Tunnel, or Authelia because they are bearer-token
control-plane WebSockets rather than a browser application. Authelia protects
browser-facing services at the existing Traefik edge; it cannot replace the
ModelLabs capability token. Keep any future browser dashboard on a separate,
Authelia-protected HTTPS route, and keep the control-plane ports local-only.

## Reasoning effort

The router selects the least effort expected to reach a verified result:

| Task signal | Default effort |
| --- | --- |
| Short, fully specified work | `low` |
| Routine implementation or research | `medium` |
| Difficult investigation or debugging | `high` |
| Consequential architecture, security, release, or migration work | `xhigh` |
| Formal verification, adversarial audit, or multi-system release work | `max` |
| An explicit request for exhaustive independent rigor | `ultra` |

An explicit model or effort request in the prompt takes precedence. The live
Codex model catalog remains the authority: ModelLabs rejects unavailable models
and unsupported effort values before a new managed turn starts. Route output and
managed-turn context expose the same value as `intelligence_slider`, while the
host receives it as Codex's `effort` setting before inference.

## Adaptive routing evidence

Adaptive routing defaults to **shadow mode**. It records a recommendation but
does not change a selected route until `MODELLABS_ADAPTIVE_MODE=enforce` is set
deliberately. Only an explicit, prompt-free outcome record can influence a
future route; keyword matches in a later user message are observation-only.
Evidence is limited to the same managed chat, task bucket, and a recent
30-day window. Explicit model or reasoning-effort choices always win.

Verified smoke benchmarks provide a second, prompt-free evidence stream. Each
run uses a disposable workspace, independently executes the produced artifact,
and records only the scenario digest, model, effort, pass/fail result, latency,
and token counts. Correctness is a hard gate; token efficiency ranks only
products that pass. Fewer than three successful samples or fewer than two
distinct scenarios for a model/effort pair can produce a shadow recommendation
but can never change a route, even when adaptive enforcement is enabled.
Consequential routes remain benchmark shadow-only.

Run the representative matrix and add its verified metadata to the dashboard:

```bash
modellabs-smoke --record-metrics
```

In plain English: This gives several models identical small product tasks,
checks each generated product by actually running it, and records private
scorecard metadata for future routing decisions. It consumes model tokens and
creates only temporary workspaces, which are deleted when each run finishes.
Use `--repeat 3` when promoting a fresh comparison beyond preliminary shadow
evidence; repeated runs still remain subject to the configured adaptive mode.
After reviewing verified multi-scenario evidence, use
`modellabs adaptive-mode --mode enforce` to permit qualified non-consequential
recommendations to affect routing. Explicit user selections always win, and
consequential tasks remain benchmark shadow-only.

To add an operator-confirmed outcome for a managed turn, use:

`modellabs outcome --thread-id THREAD_ID --turn-id TURN_ID --outcome verified`

The command writes metadata only, with no prompt or response content. The
health summary is available via
`python3 ~/.local/share/model-selector/health_dashboard.py`; its model counts
represent accepted turns, not duplicated telemetry events.

For a completed product with an independently checked final result, record a
quality grade rather than relying on a conversational success signal:

`modellabs grade --thread-id THREAD_ID --turn-id TURN_ID --quality-score 94 --verification passed`

The grade stores only the numeric score, pass/fail verification state, routed
model, effort, task bucket, final per-turn upstream token sum, and latency. The dashboard
reports a quality letter grade, a token-efficiency score relative to the lowest
token verified result in the same task bucket, and an overall grade weighted
80% to quality and 20% to token efficiency. Missing verification, missing
usage, or a non-comparable task remains `ungraded`; it never becomes positive
adaptive evidence. A verified score below 90 and any failed verification record
retry evidence, while a verified score of at least 90 records positive evidence
for the same managed chat and task bucket.

Raw upstream completions are buffered until the turn finishes and then written
as one `turn_usage` record. This prevents partial tool-step usage from being
mistaken for the final product's cost and makes an unmatched legacy raw-usage
record visibly non-gradeable.

## Run a managed chat

Install or upgrade for the current user:

```bash
python3 install.py
```

In plain English: This copies ModelLabs into your local data directory, creates
its Python environment and private host token, installs the global Codex skill,
preserves existing hooks, and writes a user service that restarts the local
proxy after reboot. It updates your Codex configuration and local command
wrappers, but does not publish a network port.

The installed `modellabs-proxy.service` is a lingered user service. It keeps the
updated loopback proxy healthy after logout and restart; it does not replace an
older proxy that still serves an existing chat.

```bash
codex-model-host start
```

In plain English: This starts a new Codex chat through ModelLabs. It reads your
first prompt before any model responds, then chooses a suitable model and
reasoning effort. It only starts a local process and does not modify your files.

To move an already closed standalone chat to ModelLabs, use its exact thread
UUID:

```bash
codex-model-host resume THREAD_ID
```

In plain English: This reconnects the same saved conversation through the
managed host. It refuses if the original standalone Codex process is still
open, preventing two processes from writing to the same chat.

## Verification

Install the one runtime dependency in a virtual environment first:

```bash
python3 -m pip install -r requirements.txt
```

In plain English: This installs the WebSocket library ModelLabs uses to talk to
Codex's local host. It changes the selected Python environment by adding that
library.

Run the local routing tests with:

```bash
python3 -m unittest discover -s tests -v
```

In plain English: This checks that representative prompts select the intended
model and effort levels. It does not contact an AI model or change settings.
