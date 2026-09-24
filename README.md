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

Smoke benchmarks provide a second, prompt-free routing evidence stream. Each
run uses a disposable workspace, independently executes the produced artifact,
and records the scenario digest, requested model and effort, pass/fail result, latency,
and token counts. Correctness is a hard gate; token efficiency ranks only
products that pass. Fewer than three successful samples or fewer than two
distinct scenarios for a model/effort pair can produce a shadow recommendation
but can never change a route, even when adaptive enforcement is enabled.
The current CLI runner records requested-model evidence only, not per-request
proof of the model that executed inference. Such rows remain shadow-only even
after the sample and scenario thresholds are met; route enforcement requires
observed per-request model evidence for both the baseline and challenger.
Consequential routes remain benchmark shadow-only.
Version 2 benchmark evidence binds the complete scenario definition, checks
that protected input and verifier files are unchanged, and runs verification
inside a read-only, network-isolated bubblewrap workspace with a minimal
environment. A failed agent or verifier cannot pass. Routing ignores older
version 1 rows because their verifier integrity was not captured; matching
scenario names without matching full digests are not comparable. Keep adaptive
mode in `shadow` until enough new, intact, multi-scenario comparisons are
reviewed.
Controlled prompt variants can share a `comparison_id` only when their product
fixtures, verifier, and expected response match exactly. Each result records
the prompt author, parent scenario, prompt digest, and shared product digest.
These local experiments are separate from user prompts and AuraCall/ChatGPT
iterations; they are not graded browser suggestions or causal proof from a
single pair. Repeats alternate order to reduce first-run/cache confounding.
New benchmark runs use unique receipts and may stage keyed features from the
exact final Codex answer in the private learning database. The operational
metrics log still contains only metadata. `modellabs-learn status` reports
these as `local_prompt_experiments`, separate from browser-reviewed iterations;
they cannot validate or enable the browser prompt-adjustment model.
`modellabs-learn train-local-prompt` fits a separate pairwise preference model
only from repeated managed comparisons of an original benchmark prompt and a
browser-authored revision on the exact same product definition and observed
model/effort.
Correctness wins first; equal passing scores need at least a 10% median-token
difference before a pair receives a preference label. Inconclusive pairs are
excluded. Training needs eight independent products, splits by product, and
freezes a checkpoint before prospective evaluation. The current CLI runner
without `--managed` does not prove the executed model and remains observational;
its rows can never train this model. Verified browser-authored revisions may be
compared against the same local product baseline, but browser review grades are
never treated as product outcomes. The separately lineage-bound iteration
learner handles the browser review loop itself.

For a controlled browser-authored prompt variant, the source runner accepts
`--scenario BASE_ID --browser-variant-guard-id GUARD_ID
--browser-variant-prompt-file PATH`. It requires a completed Pro-guard review
with a matching nonce, unchanged hash-bound original files, and an exact match
to the review's proposed prompt. File-backed reviews additionally require the
bound downloaded review record, matching provider artifact, and confirmed
sent-turn attachment UI receipt; a review request alone is insufficient. The
base and variant are then run on the same product and model/effort matrix. With
`--managed --record-metrics`, the runner precommits the complete randomized
prompt order and every randomized model block before inference. It accepts a
pair only when proxy-owned execution receipts, exact result identity, the frozen
verifier grade, tokens, and browser guard/response lineage agree. Without
`--managed`, the same run remains calibration evidence only. The browser grade
is never substituted for the local product grade.
New smoke receipts include an explicit repetition index. Prompt preferences
are labeled only from baseline/candidate runs matched within the same suite,
repetition, product, and model/effort arm; older unblocked runs remain stored
as observational evidence but cannot become token-efficiency wins by comparing
unpaired medians. The learner requires consistent within-block direction for
a token-based winner.
The benchmark capture API rejects a caller-declared `observed_per_request`
label: only an independently verified execution receipt can establish that
provenance, and the current local CLI runner does not supply one.
`modellabs-learn status` and `update` also report
`browser_revision_product_outcomes`. It joins only an exact adopted browser
suggestion to its nonce-bound review and the matching baseline/revision
benchmark blocks. The review score and local product score remain different
labels; the report explicitly counts cases where the review score improves
while the locally passing revision uses more tokens in every matched block.
Ambiguous duplicate blocks are excluded. This is post-result diagnosis, not
a feature leaked into an earlier review prediction or a live routing signal.
The live-pilot diagnostic gate additionally requires both the prospectively
validated local prompt preference model and the separate managed prompt policy.
Each managed policy needs eight development products followed by eight new
prospective products, three randomized repeats per product, at least seven
prospective wins, no losses, and verified passes throughout. The observed model
and effort must match the precommitted arm on every paired run; an
`observed_per_request` label alone is insufficient. Version 2
`benchmark_result` telemetry is caller-written and is always shadow-only,
including when a caller supplies that label. Future enforcement must consume a
separate randomized-block receipt derived from proxy-owned execution evidence.
Reviewer grades alone cannot satisfy this gate. The current CLI benchmark
marks its model choice as requested-only, so these runs remain useful for
development but do not count as execution-verified wins. A separate live
canary and rollback check are still required even if the diagnostic gate passes.
The managed proxy also records a turn-bound `model/rerouted` server event as
metadata-only evidence. The learner marks that turn as rerouted and excludes
it from clean single-model comparisons. No reroute event is not proof that
the assigned model executed every request; the app-server's raw completion
event supplies usage and response identity, but not an execution-model ID.
Accepted text-only `turn/steer` user messages are stored separately as keyed,
prompt-free context records for their exact thread and turn. A rejected steer,
an image-bearing steer, or an unconfirmed response is not learned. Steers do
not become independent graded turns and are not assigned the turn's final
answer, browser grade, or token total. This preserves real mid-turn user input
for later context-aware learning without inventing causal outcome labels.
The daily session importer also backfills durable text-only user messages when
their own message ID and recorded turn ID agree. It distinguishes messages
after prior model output, imports idempotently, and keeps raw text out of the
learning database. Those context records do not relax the exact single-prompt
requirement for linking a completed Codex result to a browser review.
Rollout reads stream up to a 256 MiB file bound and skip individual lines above
4 MiB; oversized input cannot silently become an exact result pair.

Run the representative matrix and add its verified metadata to the dashboard:

```bash
modellabs-smoke --record-metrics
```

In plain English: This gives several models identical small product tasks,
checks each generated product by actually running it, and records private
scorecard metadata for future routing decisions. It consumes model tokens and
creates only temporary workspaces, which are deleted when each run finishes.
Use `--repeat 3` to improve a fresh comparison's precision; repetition alone
does not turn requested-model evidence into observed-model evidence. After the
forward gate, a separate canary, and rollback validation pass, use
`modellabs adaptive-mode --mode pilot` for a deterministic 10% pilot. Full
`enforce` remains a later promotion. Explicit user selections always win, and
consequential tasks remain benchmark shadow-only.

For forward model/effort experiments, use the managed runner explicitly:

```bash
modellabs-smoke --managed --record-metrics --scenario routine_cli \
  --pair gpt-6-luna:low --pair gpt-6-sol:medium
```

In plain English: This precommits a private randomized two-arm block, sends
each arm through the authenticated proxy in a fresh workspace, runs the frozen
network-isolated verifier, and imports the result only after canonical model,
completion, usage, and grade evidence agree. It consumes a full model turn for
every arm. Missing or failed arms remain visible as an incomplete block and do
not silently become causal evidence. Version 2 CLI benchmark rows remain
observational and can never enforce a route.

The managed policy report keeps a benchmark's product-difficulty label
separate from the live router's independently derived task class. It requires
at least three complete randomized blocks per product, eight independent
development products, and then eight new prospective products after an
immutable checkpoint. A candidate needs at least seven prospective product
wins, no losses, and verified passes on every prospective run. Even then the
report is shadow-only; it cannot change a route until the remaining forward
readiness checks, bounded live canary, and rollback test pass.
The stored probability is the probability of an arm occupying an order
position; every arm runs, so it is not a treatment-assignment probability.
Daily collection rejects a cohort whose annotated class differs from the live
router class and quarantines a product after two incomplete or failed blocks.
This bounds token spend and prevents one broken fixture from monopolizing the
daily canary. The frozen `managed_experiments` registry in `benchmarks/smoke.json`
can schedule more than one baseline-bound model/effort comparison. It completes
or quarantines each comparison in declared order, advances past an inconclusive
development set without weakening the predeclared effect threshold, and exposes
prospective products only after that exact comparison has an immutable
development checkpoint. Reusing the same frozen product set for a different
model/effort contrast does not pool those contrasts or count a product twice
inside either comparison.
The current registry follows the routine contrasts with a separate difficult
task experiment comparing the routed `gpt-6-sol`/`high` baseline against
`gpt-6-astra`/`medium`. Its debugging and optimization products are frozen into
disjoint development and prospective sets; routine evidence cannot satisfy its
checkpoint.

Reviewed browser-prompt collection uses a separate private registry at
`~/.local/share/model-selector/prompt-experiments.json`. Each enabled entry
binds one frozen scenario to one exact Pro-guard ID and one owner-only prompt
file under `prompt-experiments/`. `modellabs-smoke --daily-browser-prompts`
runs at most one randomized base/revision block, stops after three complete
sets, and applies the same two-failure quarantine. It does not discover or
adopt suggestions automatically: a revision must first be reviewed and
materialized through the nonce-, origin-, response-, and attachment-bound
guard path. Development products remain separated from prospective products.
Different prompt variants of one frozen product are robustness strata and
count as one independent product for model-routing validation; conflicting
strata produce a tie. A user timer may run this prompt canary before the model
canary and daily learner, but a successful timer run never changes adaptive
mode or promotes a policy by itself.

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
`modellabs-learn import-recovery-observations` separately reads AuraCall's
write-once `recovery-observation.json` sidecars for failed direct review runs.
It checks the original run digest, saved guard nonce and prompt, answer hash,
and any attached file's submitted-user receipt before storing only keyed
prompt/result features. These rows remain ungraded and do not enter the
review-pass trainer or satisfy the forward-validation gate. The daily
`modellabs-learn update` includes this import, but source changes need
installation before the scheduled job uses them.
After validating the learner, `python3 install.py --learning-only
--expect-installed-sha256 CURRENT_SHA256` updates only its installed Python
module and ownership-manifest entry. It first checks the exact installed hash
and saves a private backup; it does not rewrite Codex wrappers, restart the
proxy, or enable the daily timer. The timer's actual running state must be
verified separately.
The AuraCall guard also calls this local update after a completed poll when
the command is installed. A failed learning update does not change the review
verdict and is recorded as `learning_sync.status: failed`.

`modellabs-learn capabilities` reports supported guard formats without opening
the learning database. File-backed Pro Guard reviews use
`codex.pro_guard_file_handoff.v1`: a normal chat request with Markdown source and
review-record attachments, not a visible scoring template. The importer verifies
the submitted file manifest, exact goal/candidate bytes (including whitespace),
reviewer prompt, downloaded verdict, assistant reply and recorded output artifact.
Parent-round features come from the reviewed candidate file, never the short
handoff message. Only keyed features are retained in the learning database.
The installed learner must advertise this format before a traced file review is
submitted; source tests alone do not establish installed compatibility.

Optional reviewed Markdown, DOCX and PDF references require the additional
`guard_reference_files: codex.pro_guard_reference_files.v1` capability. The
importer verifies exact original and retained document bytes, the submitted
manifest, and the actual attachment paths and MIME types. Reference contents do
not enter the learning database. Candidate identity includes the document
manifest, so changing a PDF or Word document cannot be mistaken for reviewing
the same unchanged Markdown overview. An overview alone is not an exact Codex
final answer; the existing explicit final-answer binding remains separate.

For future iterative reviews, the guard can bind the original user prompt,
the generation prompt that produced each candidate, its author, and the exact
parent guard. ModelLabs imports those linked rounds, the prior review artifact,
its numeric review score, and the prior reviewer feedback as private keyed
features. The guard requires this lineage on every new review by default,
including round one. `--allow-untraced-review` explicitly excludes a review
from iteration training; old untraced reviews are never assigned invented
prompts. The artifact may be a packet containing an answer, checks, and other
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
author. Legacy inline reviews sometimes placed one complete revised prompt
inside a clearly labeled quote rather than the structured field. For
completed, nonce-checked inline reviews only, ModelLabs records an exact
child-prompt match separately as `adopted_explicit_inline_revisions`. Ordinary
critique and file-backed reviews cannot gain this label. It is observational
provenance, not causal improvement or live-route authority.
When the prior review artifact and bound generation prompt exactly
match a Codex turn's completed final answer and prompt, the model adds that
separately as verified Codex-result evidence. A candidate prediction can also
accept `--parent-codex-result-file`; the caller-supplied text is reported as
such, not claimed to be a verified training link. When that exact link also
contains a supported Codex model and reasoning effort, the learner includes
the prior turn's model/effort as separate features. For managed turns it requires
an exact usage receipt matching the accepted arm; a selected arm alone does
not establish execution. A prediction may supply
`--parent-codex-model` and `--parent-codex-effort` together with the prior
Codex result; caller-supplied values are not represented as verified links.
For a completed local turn, `modellabs-learn export-codex-result` stages its
exact final answer for AuraCall's `--codex-result-file`. Supply exact
`--thread-id`, `--turn-id`, `--generation-prompt-file`, and `--output-file`;
the command requires one completed turn whose single original user message
matches that generation prompt. The output must be a new file in an existing
private directory. It returns only the file path and SHA-256 digest, not the
answer text. Review the staged file before using it in a browser submission;
the guard still binds its digest and the learner still requires all exact
prompt/result/thread/turn identities before linking outcomes.
`modellabs-learn export-codex-turn --thread-id THREAD_ID --turn-id TURN_ID
--output-dir PRIVATE_DIR` stages the single pre-inference user message (with
outer whitespace normalized as in session import) and its exact completed
final answer as new mode-0600 files in an existing
mode-0700 directory. It returns only paths and SHA-256 digests, not the text.
Use the exported `generation-prompt.txt` and `codex-final.txt` with the guard's
`--generation-prompt-file` and `--codex-result-file`, together with the exact
thread and turn IDs. The original user request still needs its own bound
`--origin-prompt-file` on the first review round. No files are overwritten;
inspect these private exports before browser submission. This makes lineage
staging easier, but does not create a grade or submit anything to ChatGPT.
Requested AuraCall model selectors are not treated as observed execution.
For managed Codex turns, `observed_per_request` is recorded only when the host
emits an exact `thread/settings/updated` notice while that turn is still pending,
the turn completes with exact proxy usage, and no `model/rerouted` event exists.
The confirmation is prompt-free and bound to the exact thread and turn receipt.
The completed browser assistant output is also reduced to private keyed
features when the saved AuraCall record contains a nonce-bound assistant
message, or an exact assistant reply bound to a verified file verdict. It enters
only later rounds as the prior browser result, separate
from parsed reviewer feedback and any Codex answer. No raw browser output is
stored in the learning database or operational telemetry. A caller may supply
`--parent-browser-result-file` for prediction; as with other caller-supplied
context, that does not assert a verified training link. The upgraded iteration
predictor uses a separate v7 model and frozen checkpoint, so an old
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
The same exact link supplies a prior Codex model/effort only when separate
per-request execution evidence proves a single observed arm. A proxy-selected
model, session metadata, or exact token count alone never fills that feature.
It also uses the previous nonce-bound browser assistant result when present.
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
For review packets that include checks or context around a Codex answer, the
guard can instead bind an exact completed Codex thread ID, turn ID, and private
final-answer file. ModelLabs verifies the bound result bytes and requires the
recorded turn's prompt and final-answer digests to match before linking; the
packet itself is not mislabeled as the Codex answer. This optional binding is
part of the submission fingerprint and cannot be added after the browser run.
`modellabs-learn update` reports `trace_missing` for otherwise eligible guard
reviews that lack bound prompt lineage. These reviews can still inform the
ordinary review-pass model, but not the adaptive iteration model; their missing
generation prompts cannot be safely inferred from round numbers.

For the older Codex Research unattended loop, new rounds can carry a
submission-bound `learning-trace.json`. `modellabs-learn import-bound-loops`
checks the loop, trace, adapter state, and AuraCall's completed response record
before storing keyed prompt/result features and the browser grade. Daily
`modellabs-learn update` performs this import too. These rounds remain in a
separate observational table: local file hashes do not prove that the browser
loaded identical attachment bytes, and the imported grades do not yet train or
control the live router. Earlier unbound rounds are not retroactively promoted.
The loop now requests a concise browser `suggested_next_prompt` on failed
reviews and passes it to the next Codex round as bounded, quoted, untrusted
review data. Codex still decides whether its substance fits the original goal;
the durable next-round prompt records what was actually supplied. This is an
observational prompt-adjustment path, not automatic adoption or causal proof.
New child traces bind a digest of the parent suggestion and an exact quoted-
prompt inclusion flag. ModelLabs checks that claim against the parent verdict
and child prompt, counts `browser_suggestion_supplied_rounds`, and gives the
post-Codex predictor a separate pre-browser feature. Older traces default to
unproven, and the predictor uses a new feature version and frozen checkpoint;
neither a quoted suggestion nor a better later grade proves causation.
The importer separately checks the AuraCall request attachment list against
the bound manifest and records whether the succeeded browser run dispatched
those same direct paths. A matched dispatch path is stronger than request
intent, but is still not proof of provider-consumed bytes; bundled uploads and
unconfirmed paths remain outside the post-result model's eligible set.
New AuraCall records may also carry `browserRun.attachmentUiReceipt`. ModelLabs
requires its exact attempted paths to match the transport, an observed upload
completion, attachment UI on the sent user turn, and a nonempty submitted user
ID before admitting a bound round to the post-result predictor. Older records
remain observational with `browser_ui_confirmed_rounds=0`; no grade is
retroactively promoted. This browser UI evidence still cannot prove the
provider read identical attachment bytes or make the model authoritative.
`modellabs-learn train-bound-loops` (included in the daily update) fits a
separate post-Codex, pre-browser pass and numeric-grade predictor once enough
independent, linked loop rounds exist. Its features include the original goal,
revised prompt, current Codex result, and prior browser feedback/grade; the
current browser answer is excluded from its own prediction. Separate frozen
pass/fail and numeric-score checkpoints evaluate later task groups prospectively.
The score gate requires at least 20 later rounds from 10 new task groups,
including 8 revision groups, and at least a three-point absolute and 5%
relative improvement in mean absolute error over the frozen median baseline.
Retrospective holdout scores alone
cannot activate it, and its browser-grade labels remain observational until
attachment delivery and downstream use are independently verified. It does not
change live routing or prompts merely because training completes.

`modellabs-learn train` fits a regularized review-pass predictor and evaluates
it on later, whole task groups withheld from training. `modellabs-learn status`
reports counts and the holdout result. `modellabs-learn predict-review
--goal-file GOAL --revision-file CANDIDATE --round N` scores a proposed review
round locally. Its probability is advisory; a model with
`validated_for_shadow: false` must not influence routing or release gates.
`modellabs-learn train-score` separately learns the browser reviewer's numeric
0–100 grade from the same nonce-bound goal/artifact examples, preserving grade
differences that a pass/fail label discards. `predict-review` adds
`predicted_review_score` when this model exists. Its independent frozen
prospective checkpoint requires at least 20 later episodes from 10 new task
groups and a three-point absolute plus 5% relative improvement in mean
absolute error over the development-score median. A retrospective holdout is
diagnostic only; the score model is observational, shadow-only, and cannot
authorize automatic prompt edits or routing changes. It does not replace the
lineage-dependent iteration and score-improvement models.
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
arms have enough comparable graded turns with observed per-request model and
effort evidence in a task class. Existing selected-model and session-metadata
labels are retained for diagnostics but excluded from arm training. Predictions remain
observational and shadow-only: routine traffic is heavily concentrated in one
older arm, so the model cannot infer what an untried arm would have done. The
Codex outcome model freezes its development groups and eligible model/effort
arms before prospective evaluation; later new arms cannot count as evidence
for an arm the checkpoint never learned. Future validation also requires
comparable arms in a common task class. The current task-based router remains
authoritative until paired, independently verified comparisons support a
policy change.
`modellabs-learn status` also reports `forward_readiness`: a diagnostic list of
prospective, lineage-bound, attachment-confirmed, and causal evidence still
missing before a live adaptive-routing pilot can be considered. Its
`candidate_for_live_pilot` flag does not change routes or authorize a release;
a separate bounded live canary and tested rollback are still required. A
browser-visible answer from a failed AuraCall run, requested-only Codex
model/effort labels, or retrospective holdout scores cannot satisfy these
checks.
`managed_routing_policy` is the separate causal model/effort report. It uses
only precommitted randomized blocks with exact proxy execution receipts and
never promotes ordinary chat traffic or caller-written benchmark metadata.
`managed_prompt_policy` applies the same development/checkpoint/prospective
discipline to nonce-bound ChatGPT revisions. A validated model/effort policy is
published as a digest-bound artifact. `pilot` mode applies it to a deterministic
10% of eligible non-explicit prompts; `shadow` applies nothing, and explicit
user model or effort choices always win.
`browser_codex_links` separately counts unlinked review traces whose exact
generation prompt appears in managed turns, completed single-message sessions,
or neither. These are acquisition diagnostics only: a prompt match is not a
review-to-result link, and an absent match must never be guessed from a similar
conversation or packet.
The bound-loop importer reports `external_blocker_excluded` separately from
other ineligible historical rounds; these attempts remain ungraded and do not
enter training even if a browser answer was later seen in the retained tab.

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
The source supervisor now leaves old proxy generations running by default:
a disconnected TUI can still need its original proxy URL. Retiring a generation
based only on zero established sockets requires the explicit
`MODELLABS_RETIRE_DRAINED_PROXY_GENERATIONS=1` operator opt-in. That override
is not a substitute for inspecting exact thread owners and unresolved receipts.
Installing this source change does not retroactively alter an already-running
supervisor process.
For a no-restart safety publication, `install.py --supervisor-source-only
--expect-installed-sha256 <inspected-sha256>` checks the owned manifest, keeps
a private byte-exact backup, and updates only the on-disk supervisor source.
It does not activate a new proxy generation or change the running supervisor;
that process keeps its previously loaded code until it is naturally replaced.
`install.py --proxy-source-only --expect-installed-sha256
<inspected-turn-proxy-sha256>` is the guarded proxy equivalent. Before changing
the mutable entrypoint, it snapshots both the installed and candidate dependency
closures under private, revision-named directories and binds each revision to
its loopback port with a digest manifest. A restarted old supervisor therefore
executes the exact old bundle for its remembered port, while a later chat can
start the candidate bundle on its new port. The operation itself starts no proxy
and restarts no service; a missing, redirected, changed, or port-colliding bundle
fails closed.

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
