# ModelLabs

ModelLabs is a local Codex prompt router. It selects a model, reasoning effort,
and starting MCP shortlist before the host begins inference. A managed Codex
chat can be adjusted again on later turns through the protected local host.

ModelLabs writes a mode-0600 local `metrics.jsonl` file containing route choice,
host admission, completion status, elapsed time, and server-reported token
usage when Codex delivers it to the proxy. It records no prompt or response
text in telemetry.

For initial and later turns, the owning proxy retains matching events received
during admission and computes a turn delta from complete, monotonic,
turn-ID-scoped `thread/tokenUsage/updated` totals; repeated snapshots do not
add tokens. A durable-only completion still produces one terminal receipt and
an explicit unavailable-usage receipt when exact evidence is absent.

The launcher directs new managed chats to the current loopback proxy and starts
a local watchdog for that proxy. Older proxies are left in place for already
connected chats; no new managed launcher uses their port. Retire an older proxy
only after its connected chats have exited and its accounting work has settled.

## Network boundary

The ModelLabs host and proxy bind only to `127.0.0.1`. They are not registered
with Traefik, Cloudflare Tunnel, or Authelia because they are bearer-token
control-plane WebSockets rather than a browser application. Authelia protects
browser-facing services at the existing Traefik edge; it cannot replace the
ModelLabs capability token. Keep any future browser dashboard on a separate,
Authelia-protected HTTPS route, and keep the control-plane ports local-only.

## Model choices

The current defaults are GPT-6 Luna for short, checked work; GPT-6 Sol for
routine and difficult work; and GPT-6 Astra for consequential work. These
choices are admitted only when the live Codex catalog lists them. Existing short names
`Sol`, `Luna`, and `Terra` retain their GPT-5.6 meanings.

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
and unsupported effort values before a new managed turn starts. It never
silently lowers an explicit `ultra` request to `max`. The `none` choice can be
requested, but is admitted only if that model's live Codex catalog entry lists
it; API support alone is not sufficient. Route output and
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

## Local outcome model

ModelLabs now has a separate private learning store under
`~/.local/share/model-selector/learning/`. It stores keyed text features and
source identities, never raw prompt or response text. The ordinary
`metrics.jsonl` remains metadata-only. Accepted managed Codex turns contribute
features from the user prompt; explicit quality grades and exact token usage
are joined later by the same thread and turn identities. The last completed
Codex agent message contributes keyed result features after the turn ends;
result text is not retained in the learning database or used to predict a
model before the turn. Weak keyword feedback is excluded from model labels.
Managed Codex outcome training requires the same turn's captured final answer,
explicit grade, completed status, and exact usage from the owning proxy. A
grade or token total alone cannot promote an incomplete turn into training.
`modellabs grade` attempts a local learner sync after recording the grade and
reports that sync separately; a learning error cannot erase an accepted grade.
Standalone local Codex rollout logs can also be imported with
`modellabs-learn import-codex-sessions`. This captures completed user-prompt
and final-answer feature pairs, observed model/effort, and reported turn usage
without storing the text. These examples remain explicitly **ungraded** and
are excluded from model training unless `modellabs-learn sync-session-grades`
finds an exact same-thread/turn accepted route, completed turn, proxy usage,
and explicit grade in local telemetry. That retrospective join also requires
the session's model, effort, and reported usage to agree with telemetry and
the completed turn to have exactly one genuine user message before any recorded
assistant inference. Older ambiguous, multi-message, or late-message rows are
excluded from training. It is
reported separately from live pre-inference managed capture and is not by
itself evidence of prospective adaptation. Explicitly graded retrospective
examples can enter the observational Codex outcome trainer, where independent
task-group holdout and multiple supported model/effort arms are still required
before a model is trained. Re-imports and joins are
idempotent, with changed session records reported as conflicts rather than
overwritten.

`modellabs-learn import-auracall` reads completed AuraCall Pro guard rounds from
their saved guard state and exact durable response record. It requires the
response ID, guard ID, nonce, round, submission fingerprint, succeeded run, and
parsed grade to agree before adding an example. The model sees the review goal
and the submitted revised artifact as pre-grade inputs. A Pro review grade is
review evidence, not independent proof that the delivered product works.
Use `--guard-id ID` to import only one completed guard round. Run
`modellabs-learn update` after new guard results or explicit Codex grades to
import browser rounds and local Codex sessions, join available explicit grades
and exact usage, and retrain the local models in one step. This update does not
submit prompts to a provider or alter live routing.
The AuraCall guard also calls this local update after a completed poll when
the command is installed. A failed learning update does not change the review
verdict and is recorded as `learning_sync.status: failed`.

For future iterative reviews, the guard can bind the original user prompt,
the generation prompt that produced each candidate, its author, and the exact
parent guard. ModelLabs imports those linked rounds, the prior review artifact,
its numeric review score, and the prior reviewer feedback as private keyed
features. The artifact may be a packet containing an answer, checks, and other
evidence; it is not assumed to be the Codex final answer. Only the **previous**
round's artifact and score enter a candidate prompt's feature vector; the
candidate's own answer and grade are never
available to its pre-generation prediction. `modellabs-learn train-iteration` fits a
separate predictor only after enough linked task groups exist; its held-out
score and `validated_for_shadow` flag are reported in `status`. The prompt
model remains observational: it cannot prove that a prompt edit caused an
improvement, and it does not rewrite prompts automatically. Once trained,
`predict-iteration` scores a candidate generation prompt against the original
prompt and optional prior feedback, reviewed artifact, score, and exact browser-suggested
prompt without submitting it to ChatGPT. Supply `--parent-feedback-file`,
`--parent-result-file`, `--parent-quality-score`, and
`--parent-suggestion-file` for a complete revised-round context. Exact adoption
of that suggestion is a separate model feature, not merely a claimed prompt
author. When the prior review artifact and bound generation prompt exactly
match a Codex turn's completed final answer and prompt, the model adds that
separately as verified Codex-result evidence. A candidate prediction can also
accept `--parent-codex-result-file`; the caller-supplied text is reported as
such, not claimed to be a verified training link. The upgraded iteration
predictor uses a separate v5 model and frozen checkpoint, so an old
evaluation cannot silently validate the new feature set.
Training also requires at least eight distinct training task groups and two
diagnostic holdout groups with revised rounds that include the previous
reviewed result and score. Prospective shadow
validation requires eight such new groups as well: first-round reviews alone
cannot validate an adaptive iteration predictor.
`modellabs-learn train-improvement` fits a separate revision-only outcome:
whether the new browser review score exceeds its exact parent's score. Unlike
the pass target, this captures progress that still falls short of the release
threshold. It uses the bound original and revised prompts, prior feedback,
prior review artifact and score, prompt author, exact suggestion adoption,
and a separate Codex final-answer feature only when exact linkage proves it.
`predict-iteration` reports `predicted_score_improvement_probability` only
when this model exists and the parent result and score are supplied. Its frozen
prospective checkpoint and `improvement_validated_for_shadow` flag are separate
from the pass predictor; neither probability is a causal estimate or an
automatic prompt rewrite. The prospective holdout admits only task roots that
began after its checkpoint; a late revision of an older root cannot enter it.
When a later prompt exactly adopts the browser reviewer's suggested next
prompt, `status` counts that link separately rather than trusting an author
label alone. The iteration model also freezes a private evaluation checkpoint
once enough linked task groups exist; later independent groups, not a moving
rolling holdout, determine whether its shadow-validation threshold is met.
If the guard's bound generation prompt and submitted artifact exactly match a
single Codex turn's prompt and completed final answer, the learner records a
cross-source link. Ambiguous matches are rejected. The linked browser score
remains a browser-review label, distinct from an operator-verified Codex grade;
it does not silently train the managed Codex outcome model.
`modellabs-learn update` reports `trace_missing` for otherwise eligible guard
reviews that lack bound prompt lineage. These reviews can still inform the
ordinary review-pass model, but not the adaptive iteration model; their missing
generation prompts cannot be safely inferred from round numbers.

`modellabs-learn train` fits a regularized review-pass predictor and evaluates
it on later, whole task groups withheld from training. `modellabs-learn status`
reports counts and the holdout result. `modellabs-learn predict-review
--goal-file GOAL --revision-file CANDIDATE --round N` scores a proposed review
round locally. Its probability is advisory; a model with
`validated_for_shadow: false` must not influence routing or release gates.
The review model now requires at least 20 held-out episodes and both a 0.005
absolute and 5% relative Brier-score improvement over the constant baseline
before it can even qualify for shadow validation. An immutable private
checkpoint freezes the evaluation predictor, baseline, and development task
groups before future grades arrive. Only new task groups submitted after that
checkpoint count toward this prospective test, and at least ten such groups
are required. Episodes with the same review goal remain conservatively grouped;
verified AuraCall iteration roots also connect rounds even if their review-goal
wording changes. A later round of an older task therefore cannot enter the
prospective holdout by changing its goal text. The older rolling holdout scores remain diagnostic only: their
groups can move into later training, so they cannot validate an adaptive
model. Live shadow predictions may continue fitting new graded data, but
their validation flag remains false until the prospective checkpoint passes.
`modellabs-learn sync-codex` joins accepted turns with explicit grades and exact
usage; `modellabs-learn train-codex` trains only when at least two model/effort
arms have enough comparable graded turns in a task class. Predictions remain
observational and shadow-only: routine traffic is heavily concentrated in one
older arm, so the model cannot infer what an untried arm would have done. The
Codex outcome model freezes its development groups and eligible model/effort
arms before prospective evaluation; later new arms cannot count as evidence
for an arm the checkpoint never learned. Future validation also requires
comparable arms in a common task class. The current task-based router remains
authoritative until paired, independently verified comparisons support a
policy change.

## Run a managed chat

Install or upgrade for the current user:

```bash
python3 install.py
```

In plain English: This copies ModelLabs into your local data directory, creates
its Python environment and private host token, installs the global Codex skill,
preserves unrelated hooks, removes the old ModelLabs prompt-injection hook, and writes a user service that restarts the local
proxy after reboot. It also installs a managed `codex` wrapper first in the
shell path, while preserving the underlying executable as `codex-direct`. It
does not publish a network port. New Codex launches are affected; already
running chats are not replaced.

The installed `modellabs-proxy.service` is a lingered user service. Each proxy
implementation receives a revision-bound loopback port. Upgrades restart the
lightweight supervisor onto the new revision while older proxy processes keep
their existing chat connections until those sessions drain. The unit uses
supervisor-only termination so an installer restart does not kill established
proxy or host children in the service cgroup.

```bash
codex
```

In plain English: This is now the normal way to start Codex. The installed
wrapper opens the TUI through the local authenticated proxy. Enter the first
prompt in the TUI; the proxy reads its `turn/start` before any model responds
and chooses a suitable model and reasoning effort. A prompt supplied as a
command-line argument can also scope the initial MCP servers before the TUI
opens. The TUI remains the owner of approvals, user input, and
interrupts. Every thread mutation must come from the connection holding that
thread's ownership lock, and server-to-client requests use a separate RPC
namespace so approval or input request IDs cannot consume admission state.
`codex-model-host start` remains
as a compatibility alias, while `codex-direct` bypasses routing for maintenance
or recovery. Starting a chat only starts local processes and does not modify
your project files.

To move an already closed standalone chat to ModelLabs, use its exact thread
UUID:

```bash
codex resume THREAD_ID
```

In plain English: This reconnects the same saved conversation through the
managed host. It refuses if the original standalone Codex process is still
open, preventing two processes from writing to the same chat.

Noninteractive inference through `codex exec` currently fails closed. Use the
interactive TUI until ModelLabs can bind that branch to admission, terminal,
cancellation, and exact-or-unavailable usage receipts. Help and version queries
remain available only as genuine exec options; flag-shaped literal prompts
remain refused. `codex-direct` remains the explicit maintenance bypass.

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
~/.local/share/model-selector/venv/bin/python -m unittest discover -s tests -v
```

In plain English: This checks that representative prompts select the intended
model and effort levels. It does not contact an AI model or change settings.
