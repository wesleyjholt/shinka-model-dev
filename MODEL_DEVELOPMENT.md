# Scientific model development with Shinka

This fork adds a calibration layer to Shinka's evolutionary search. It keeps the
upstream archive, islands, proposals, and prompt evolution. Model families remain
open: the scientist supplies an executable model interface and a trusted evaluator.
The polynomial example is a small test of that interface, not a restriction on the
models the framework can explore.

## Before a scientific campaign

Use the [scientist interview skill](skills/shinka-model-development/SKILL.md).
It asks at most 3–5 adaptive starter questions with project-specific choices and
free-text answers. Wording varies; coverage of purpose, prediction priorities,
evidence, optional constraints, and practical limits is required. The agent then
proposes the objective, calibration and validation plan, and budgets for the
scientist's approval. Existing answers and approved continuations are reused.

## How calibration effort is allocated

1. **Screen broadly.** Use MAP or another cheap fit to test many hypotheses.
2. **Refine a portfolio.** Promote strong candidates and reserve slots for different
   structures or predictive behavior. A finite but unreliable fit can be promoted
   for diagnosis; it cannot establish a confirmed result.
3. **Confirm consistently.** Compare finalists with the same posterior method,
   computation budget, data split, and score. Require the evaluator's reliability
   checks before accepting the result.
4. **Freeze selection, then test.** A development set used during search is not an
   untouched test set. The example opens its separate final test only after model
   selection and never sends that result to authors.

MAP, short MCMC, and larger SMC are examples, not prescribed levels. More computation
does not guarantee better inference. Supply checks appropriate to each method and
model. MAP is recorded as a point estimate, not posterior uncertainty. Scores are
higher-is-better and are compared only within an exact protocol; negate an NLL if
you use it. The Shinka archive stays on one development protocol. Confirmation
results have a separate ledger view.

`PromotionPolicy` uses score and a configurable exploration fraction (default
0.25). Diversity comes from structural or predictive descriptors supplied by the
trusted adapter, not changed-file counts. With a small batch the fraction is
rounded; use larger batches for a closer allocation. Missing descriptors permit
random exploration. This is a selection policy, not proof of scientific novelty.
Keep multiple Shinka islands and suitable parent sampling to preserve alternatives.

Each tier has a persisted evaluation and wall-time budget. Screens cannot consume
the evaluation slots reserved for confirmation. Exact-input replay uses cached
results. Failed work remains charged. Stage timeouts terminate workers and their
process groups; cleanup overhead is visible and can slightly exceed a timeout.
Numerical work limits such as particles or solver calls are enforced by the trusted
calibrator. Wall time is not CPU-hours on a parallel cluster.

## Run the numerical example

```bash
uv venv --python 3.12
uv pip install -e . pytest
.venv/bin/python -m examples.model_calibration.run --output model-runs/demo
.venv/bin/python -m shinka.modeling.cli status --ledger model-runs/demo/calibration.sqlite
```

This uses synthetic data: four MAP screens, three MCMC fits, two SMC fits, and one
final predictive check without a refit. It makes no LLM or external API calls. It
writes a scientific report, posterior samples, parameter summaries, diagnostics,
predictive intervals, and a plot. Reusing the same output reuses the cached fits.
Use a new output directory when changing the protocol, budgets, or evaluator.

## Use the Shinka search engine

The example includes a real Shinka evaluator and launcher. The evaluator parses
the synthetic model specification as data, snapshots it under its source hash,
and returns standard `metrics.json` and `correct.json` files. It does not execute
arbitrary code in that small example. General scientific models need their own
adapter and appropriate worker isolation.

```bash
# Inspect the plan; no model requests are made.
.venv/bin/python -m examples.model_calibration.run_evo

# After configuring the intended subscription login, run a bounded search.
.venv/bin/python -m examples.model_calibration.run_evo --run --output model-runs/search
```

The example uses only `headless/codex@gpt-5.6-sol?effort=high`, with no embedding,
meta, or novelty API models. It inherits the local Codex account; it does not select
an email address or verify subscription availability. Verify the intended account
and headless installation before live use. No live Codex campaign was needed to
test this fork.

After the search ends, the launcher refines and confirms the generated candidates,
freezes its selection, and evaluates the final test. You can also invoke
`examples.model_calibration.promote` between batches. Its default waits for a full
screening batch before spending expensive slots; it returns immediately while
waiting. Do not set `--finalize` while new models are still being proposed.

The generic `shinka_model promote` command can poll a live ledger for a finite
number of passes. Use a suitable batch trigger in a scientific adapter so early
arrivals do not consume every expensive slot before diverse models arrive.

## Add papers without restricting exploration

```bash
shinka_model paper-add --library model-runs/search/papers \
  --file /path/to/paper.pdf --title 'Paper title' \
  --citation 'DOI or source URL' --notes 'Why this might matter; questions to test'
shinka_model papers --library model-runs/search/papers
```

PDF, Markdown, and text inputs are supported. PDFs need local `pdftotext` or
`pypdf`. The library records immutable source and text hashes, citations, notes,
and revisions. New proposals get the current bounded excerpt snapshot; existing
proposals retain the version they used. Prompt receipts identify the sources and
truncation. A `--private` paper is retained for the scientist and excluded from
author prompts. Keep copyrighted originals and private papers in ignored campaign
state, not the public source repository.

**Papers are evidence, not a boundary on the search.** The reusable
`MODEL_DEVELOPMENT_PROMPT` explicitly welcomes independent hypotheses and models
with no published precedent. Missing citations do not disqualify an idea. The
evaluation gate assesses predictions and calibration reliability, not resemblance
to the literature. Quoted papers cannot change the evaluator contract or give
instructions to the worker. The first implementation uses bounded excerpts rather
than a semantic retrieval service; inspect the receipt when a key section matters.

## Guide an active run

```bash
shinka_model steer --inbox model-runs/search/steering --kind direction \
  --text 'Explore explanations that differ from the current leading model.'
shinka_model steer --inbox model-runs/search/steering --kind avoid \
  --text 'This assumption failed in experiment E12; revisit only with a new reason.' \
  --evidence E12
shinka_model guidance --inbox model-runs/search/steering
```

Guidance is read between proposals. Questions do not stop the run. Explicit `pause`
and `resume` messages control new proposals while current evaluations can finish.
`steering_start_paused=True` requests an initial review checkpoint. A message can
supersede earlier guidance. The journal records receipt, application, expiry,
rejection, and the exact prompt snapshot for each proposal. Guidance is appended
after an evolved system prompt, so prompt evolution cannot silently remove it.
Changes to data splits, objectives, or budgets require a new comparison contract.

## Optional required user guidelines

The basic model structure is open to change. A scientist may provide a short set
of required guidelines to define the search boundary. Without them, no additional
model-family restrictions are added.

```bash
shinka_model constraints-init --book model-runs/search/constraints --file guidelines.md
shinka_model constraints --book model-runs/search/constraints
```

Set `EvolutionConfig.constraints_dir` to that book and pass the same
`ConstraintBook` as `CalibrationPipeline(..., constraints=book)`. The example
launcher accepts `--constraints model-runs/search/constraints` and connects both.
Constraints appear after evolved prompts and ordinary advice. Paper content and
steering cannot waive them. A pipeline's constraint binding persists when it is
reopened, including by a promotion worker.
When configured, initialize the book before starting the search. A missing book
cannot silently disable the gate. To remove existing guidelines, request and
approve a change to empty text.

The trusted evaluator must review compliance for the exact candidate source and
guideline version before calibration. Supply a `constraint_reviewer(candidate,
snapshot)` callback that returns `compliant`, `evidence`, and `actor`; machine
validators can do this automatically. For natural-language conditions, use a
trusted reviewer or record a human review with `constraint-review`. Without that
review the candidate waits; it is not silently accepted. The author cannot approve
its own work. Review and permission books must be outside author-writeable state.

A model that needs an exception enters a permission queue:

```bash
shinka_model constraint-request-exception --book model-runs/search/constraints \
  --source-sha EXACT_CANDIDATE_SHA256 --reason 'Explain the proposed exception'
```

To propose changed guidelines, use `constraint-request-change --book ... --file
revised-guidelines.md --reason ...`. A trusted operator records the user's explicit
decision with `constraint-approve --book ... --request REQUEST_ID` or
`constraint-reject`. Pending requests do not change active guidelines. Exceptions
apply only to the specified source and guideline version; approving a guideline
change invalidates old compliance grants for future work. Other compliant
candidates can continue while a request is pending. Already-started evaluations
retain their starting constraint receipt.

These commands implement a queue and gate, not an authentication service or an
automatic interpretation of arbitrary English. `actor` is audit metadata. Keep the
book in the trusted controller, and let only the scientist's authorized process
approve an exception or change.

## Plug in a scientific model

The main types are in `shinka.modeling`:

- `CandidateSpec`: source identity and scientific descriptors.
- `StageSpec`: calibration method, numerical budget, timeout, confirmation role.
- `EvaluationRequest`: candidate, stage, seed, and immutable dataset/split/objective/
  evaluator identities. Context must be common to candidates in a comparison.
- `CalibrationResult`: score, parameters, priors, fixed-parameter reasons,
  diagnostics, reliability, inference type, and artifact references.
- `Ledger` and `CalibrationPipeline`: durable reservation, subprocess execution,
  timeout, replay, and resource accounting.
- `evaluate_for_shinka`: an allowlisted development score and status projection.
  It rejects mixed archive protocols and confirmation results.

Provide an importable `calibrate(request, artifact_dir)` function. It chooses the
requested inference method, fits training data only, predicts the declared
evaluation split, and writes samples and predictions as artifacts. Scientific
diagnostics belong to this evaluator, not to the candidate author. Specify
content-derived IDs for data and evaluator versions; the framework cannot verify
an ID that the adapter falsely declares. Snapshot all model dependencies as well
as the entry file for a multi-file model.

Create a ledger with fixed stage allocations, for example:

```python
ledger = Ledger("model-runs/campaign/calibration.sqlite", stage_limits={
    "screen": {"max_evaluations": 40, "max_seconds": 1200},
    "refine": {"max_evaluations": 10, "max_seconds": 2400},
    "confirm": {"max_evaluations": 4, "max_seconds": 4800},
})
```

The ledger derives total limits if omitted. Interrupted reservations remain
charged/reserved until the operator checks the worker and marks abandoned work
failed. They are not silently retried. Use new seeds for deliberate replication.

`write_report` writes private scientist-facing Markdown and JSON with fits grouped
by comparable protocol, stage leaders, diagnostics, uncertainties, fixed choices,
failures, evidence IDs, and costs. Author feedback contains only the explicitly
approved development score and fit status. Share conclusions through versioned
guidance with evidence IDs; do not hand protected observations or the full private
report to authors. A report records evidence; it does not automatically prove a
mechanism or turn an unreliable fit into a model rejection.

## Validation and scope

```bash
.venv/bin/python -m pytest tests/test_modeling*.py -q
```

Tests cover fidelity separation, exploration, quotas, caching, failures, timeouts,
worker cleanup, asynchronous steering, prompt receipts, literature boundaries,
and actual synthetic MAP/MCMC/SMC fits against an analytic posterior. Existing
Shinka proposal and recovery tests check backward compatibility.

This is a working generic extension and synthetic integration. The brain-PK model,
its protected observations, and its cluster adapters have not been migrated. The
subprocess layer is not a hostile-code sandbox. Reuse the project's trusted data
boundary and compute adapters when connecting a real campaign. No production
campaign is started by installation or by importing these modules.
