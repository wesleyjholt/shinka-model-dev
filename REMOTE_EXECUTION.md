# Remote scientific work

Keep model authors and the research controller separate from numerical workers.
The controller can run on a Mac. A trusted worker can run a calibration or a
bounded experiment on Gautschi through the local `cluster-slurm` tool.

Local tests of this adapter do not establish that Gautschi is commissioned.
No live SSH connection, scheduler submission, or protected-data transfer was
performed as part of this implementation review.

## Choose the unit of remote work

Usually submit one complete calibration stage per allocation. Keep the solver,
likelihood, optimizer or sampler, random state, and observed data in that worker.
For example, one job can perform a complete MAP fit or one bounded MCMC or SMC
run. Use the declared method budget inside the allocation.

Do not submit a separate Slurm job for each likelihood evaluation. Queue delays,
transfers, and process startup can exceed the numerical work and make sampler
state difficult to manage. Batch independent cheap fits when appropriate. A
persistent worker pool is a separate scheduling feature, not an implicit part of
this adapter.

An author can also request a bounded experiment during model development. The
trusted controller checks its candidate source hash, evaluator identity, user
constraints, data policy, and available budget before dispatch. An unscored
diagnostic experiment must remain unscored; it cannot become a model-ranking
result. These experiments consume the same tracked resource budget as other
work. A pending experiment does not require every author to wait.

## Two different Slurm paths

Upstream Shinka's `slurm_conda` and `slurm_docker` launch modes invoke `sbatch`,
`squeue`, and `scancel` on the controller's own host. They assume that job results
are visible through the same filesystem. Use them when the controller is already
on a suitable cluster host.

The model-development remote path uses the configured `cluster-slurm` command
line tool to reach a remote host. It must retain remote job identities and
transfer explicit input and result files. Setting upstream `job_type` alone does
not turn a Mac into a remote Gautschi controller.

Do not call a remote submission command inside a local
`CalibrationPipeline.evaluate()` callback. Killing that callback's local process
group does not cancel the remote job. Use the durable remote lifecycle instead.

## Controller API

`shinka.modeling.remote` provides `RemoteSpec`, `RemoteController`, and immutable
`RemoteJob` records. `RemoteSpec` contains the explicit profile, CPU/GPU counts,
memory, allocation runtime, and a separate queue timeout.

Initialize a controller with finite `max_jobs`, `max_inflight`,
`max_cpu_seconds`, and `max_gpu_seconds` limits. On restart, omitted limits reuse
the saved values; conflicting limits are rejected. Submission returns a work
record without waiting for the scheduler queue:

```python
job = controller.submit(
    request,
    spec,
    files={"candidate.json": candidate_source, "trusted_model.py": evaluator_source},
    command=(
        "python", "-m", "shinka.modeling.remote_worker",
        "--packet", "packet.json", "--output", "results",
        "--calibrator", "trusted_model:calibrate",
    ),
)
```

The trusted deployment must provide the fork and required numerical packages in
the profile-managed remote environment. The evaluator module and its identity
belong to the campaign owner; an author must not choose or replace them.

Call `poll(job.key)` to advance a job, `cancel(job.key)` to request cancellation,
and `reconcile(job.key)` after a submission became uncertain. `get()`,
`records()`, and `usage()` expose durable status. Each call can perform bounded
transport work, but no call waits for the entire scheduler queue or fit.

For intermediate work, use a non-calibration `task_kind` and
`parent_evaluation_key` to preserve its research lineage. Its result manifest has
no rankable calibration `result_path`. The controller verifies this distinction.

The state directory contains `remote-state.json`, immutable input packets under
`packets/<work-key>/`, and verified retrieved outputs under
`results/<work-key>/`. Keep this state in the trusted controller area. A worker
packet is not a replacement for the local calibration/promotion ledger.

## Connect the author search

`RemoteCalibrationPipeline` connects remote work to the scientific fit ledger.
Its `submit()` returns a ledger record; it does not wait through the queue.
`poll(cache_key)` advances that record, and `job_for(cache_key)` exposes the
corresponding remote job. Use `submit_probe()` for bounded intermediate work.

`RemoteCalibrationScheduler` plugs this pipeline into Shinka's existing author
search. It handles generation zero, asynchronous submissions, status checks,
result export, and cancellation through the same remote lifecycle:

```python
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.launch import LocalJobConfig
from shinka.modeling.remote_scheduler import RemoteCalibrationScheduler

scheduler = RemoteCalibrationScheduler(
    pipeline=remote_pipeline,
    build_request=trusted_request_factory,
    build_inputs=trusted_input_factory,
    spec=spec,
    command=worker_argv,
    state_dir=trusted_state / "shinka-scheduler",
)
runner = ShinkaEvolveRunner(
    evo_config=evolution_config,
    job_config=LocalJobConfig(time=None),
    db_config=database_config,
    scheduler=scheduler,
)
await runner.run_async()
```

The factories are ordinary trusted functions. The request factory receives the
candidate's local source path and returns an `EvaluationRequest` with its exact
source hash. The input factory receives that request and returns an explicit
relative-path-to-local-file mapping. Fix the development stage, data split,
objective, evaluator, and score convention in these factories. The scheduler
binds one scientific protocol to the archive and rejects confirmation stages.

Do not set a generic local job timeout on this injected scheduler. The remote
controller already separates queue and runtime limits. Unknown scheduler state,
pending retrieval, and unconfirmed cancellation remain pending; they are not
exported as failed fits. Only terminal, verified records produce author metrics.
Result export reads cached local evidence and does not perform remote transport.

Pending constraint reviews remain durable and are retried by polling. Other
available evaluation slots can continue. A fully occupied set of pending review
slots still requires the operator to resolve those requests or cancel them.
Generation zero must finish before the search has a reference model to evolve.

The scheduler stores request/source/result-directory bindings in
`scheduler-state.json`. Reopening it with the same campaign reuses existing work.
Shutdown records cancellation intent and makes one cancellation pass. Jobs whose
termination is unconfirmed remain in state for a later observer; shutdown does
not claim that they stopped.

## Keep execution state durable

The controller must retain these distinct facts:

- The immutable scientific request: candidate source, method and budget, seed,
  dataset/split/objective/evaluator identities, and accepted constraint version.
- The staged packet identity, selected cluster profile, environment and job
  script identity, remote run ID, and Slurm job ID.
- Submission intent, observed scheduler states, cancellation requests, and
  verified terminal state.
- Result and checkpoint manifests, file sizes and hashes, and the corresponding
  ledger record.

Keep run IDs, host paths, and transport details out of
`EvaluationRequest.context`. That field contributes to the scientific protocol
hash. Adding an execution-specific value there would make otherwise comparable
scores appear to use different protocols.

Persist submission intent before invoking the scheduler. A transport error can
occur after Slurm accepted a job but before the controller received its ID.
That state is uncertain, not proof that no job exists. Reconcile the existing
run; do not submit another copy automatically.

In particular, `cluster-slurm init-run` is not a resume command: its current
implementation can replace local run metadata for an existing run ID. Reattach
to a saved, verified run instead of reinitializing it.

## Runtime, queues, and cancellation

Reserve the work allowance before submission. Report queue wait, numerical
runtime, and controller/transfer overhead separately. A wall-time budget is not
CPU-hour or GPU-hour billing; retain the requested resource counts as well.

Use separate limits for queue wait, job runtime, and transport operations. A
Slurm allocation also needs time for its controlled startup and result writing.
The worker must enforce method-specific limits such as draws, particles,
iterations, or solver calls. A scheduler time limit does not enforce these
scientific work limits.

`UNKNOWN` scheduler status does not mean success or cancellation. Keep the work
reservation until the job's status is resolved. Likewise, a successful `scancel`
command only acknowledges the cancellation request. Confirm a terminal state
before releasing the reservation or reporting that the job has stopped.

Never accept empty `squeue` output as a successful fit. Consult scheduler
accounting, then verify the expected result manifest and every accepted file.
The absence of a job from the queue and the presence of an old result file are
not sufficient evidence.

## Checkpoints and restart

Controller restart means reattaching to a known remote job and ledger entry.
Sampler restart means reconstructing an interrupted numerical calculation from
a checkpoint. These are different capabilities.

A scientific adapter must define a checkpoint format before sampler resume can
be supported. Include the request identity, source and evaluator hashes,
algorithm version, seed/random state, completed work count, and all state needed
by that algorithm. Write checkpoints atomically. Validate them before resuming,
and record which checkpoint was used. A partial file is not a valid checkpoint.

Do not claim that a restarted MCMC or SMC run continues the same calculation
unless its adapter restores the required state. A fresh run needs an explicit
new attempt and corresponding budget accounting.

## Protected workers and author feedback

Keep subscription Codex author processes on the author side of the boundary.
Numerical jobs must not launch an author agent or receive Codex login files, API
keys, or unrelated credentials. Use an explicit environment and file allowlist.

Keep observed data, posterior samples, checkpoint state, and full evaluator
reports in the trusted worker/evaluator area. Upload and download only the
approved files. Do not recursively copy a campaign directory because it may
contain private data, credentials, or unrelated results.

Return the development feedback allowed by the campaign. The existing
`author_feedback()` bridge exposes a development score and a small status
summary; it excludes posterior samples, raw observations, raw errors, and final
confirmation results. The full scientist report is a private evaluator artifact.

The Python process boundary and the cluster adapter are not hostile-code
sandboxes. Review or isolate executable candidate code before it can access
protected observations. Use spawned numerical workers and do not inherit an
author process's memory or credentials through a fork.

## Before a first real Gautschi run

Use the installed `~/.codex/skills/cluster-slurm/SKILL.md` and its
`references/agent-integration.md` contract.
Supply an explicit configured profile and configuration path. If setup is
missing, obtain the SSH target and environment setup mode from the operator;
do not infer them from unrelated logs or old runs.

Run the profile health check and retain its real receipt. Preview the exact
workload without side effects. A live `test-workload` check can validate a script
with `sbatch --test-only`; it does not submit a job. Neither a dry-run nor a
successful test-only receipt is evidence that a research loop is running.

`shinka.modeling.cluster_slurm.ClusterSlurmBackend` requires an explicit profile,
configuration file, saved successful doctor receipt, and approved local download
root. It records the skill CLI and configuration identities. A saved health
receipt is evidence of the earlier check, not a fresh probe or authenticated
proof. Do not use a fake-test receipt to commission a real campaign.

The backend transfers individual declared input files, including on a resumed
partial staging operation. Retrieval is limited both by the result manifest and
the configured `approved_artifact_patterns` and `max_download_bytes` policy.
The default patterns permit only `calibration-result.json`,
`intermediate-result.json`, and `worker-resources.json`. Configure any intended
array, checkpoint, or plot file types in the trusted deployment. A worker's
manifest does not itself authorize new kinds of protected output.

Then run a small synthetic job through submission, status polling, manifest
retrieval, and cancellation/restart handling. Check the profile, environment,
resource request, timeout, and file hashes. Only after these checks should a
campaign use its authorized scientific data and expensive calibration budgets.

The low-level adapter uses explicit submission operations. Automatic package
repair, environment changes, GPU fallback, and blind submission retries would
change the execution contract; they must not occur unnoticed inside a scientific
evaluation. Record any approved environment or resource change and revalidate
the affected protocol before comparing its results.
