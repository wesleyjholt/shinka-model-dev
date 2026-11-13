# Requested changes to cluster-slurm for durable scientific evaluation

These requests concern the installed [cluster-slurm skill](/Users/holtw/.codex/skills/cluster-slurm/SKILL.md) inspected on
2026-11-13. They were identified by reading its source and by local fake-transport
tests. No cluster jobs, file transfers, or live health checks were performed.

Source SHA256: `9444286fb4a2067c2e311c6e73c455645ab678f30dfad6921b788daeda290b2e`.

## 1. Make a charged submission a single transport attempt

The shared transport loop stops when `attempt > retries + 1`. With
`max_retries=0`, the first failed attempt can therefore be repeated once if the
transient-error predicate returns true. With `max_retries=2`, it can make four
attempts. This is an off-by-one error. See
[the retry condition](/Users/holtw/.codex/skills/cluster-slurm/scripts/cluster_slurm.py:905).

More seriously, `submit-job` invokes `ssh_exec` for real `sbatch` using the same
retry policy as reads. If SLURM accepts a job but the response is lost, a
transport retry can create another charged job. See
[real submission](/Users/holtw/.codex/skills/cluster-slurm/scripts/cluster_slurm.py:4701)
and [transport defaults](/Users/holtw/.codex/skills/cluster-slurm/scripts/cluster_slurm.py:1010).

Requested changes:

- Fix the attempt bound to make `max_retries=0` mean exactly one attempt.
- Make the real `sbatch` operation single-attempt regardless of generic transport
  settings. Reads, uploads, and scheduler validation can have separate policies.
- Return a structured `SUBMISSION_UNKNOWN` result when acceptance is uncertain.
  Do not interpret transport failure as proof that no job was created.
- Write a submission-intent receipt before the charged operation. Include a
  unique operation token, profile identity, script hash, and requested resources.
- Add reconciliation that searches SLURM accounting/queue records by the unique
  token and returns `found`, `confirmed_absent`, or `unknown`. Reusing a local
  run ID is not scheduler deduplication.

Acceptance tests must simulate acceptance followed by lost stdout, controller
termination before the local receipt is saved, and a restart. Each case must
produce at most one charged submission and retain an unresolved reservation
until scheduler evidence settles it.

## 2. Do not erase a run when initialization is repeated

`create_run` creates the directory with `exist_ok=True`, then writes a fresh
record with `job_id=None` and empty history. Repeating `init-run` on an existing
run ID can erase the record needed to reconcile a previous submission. See
[create_run](/Users/holtw/.codex/skills/cluster-slurm/scripts/cluster_slurm.py:2847).

Requested behavior: fail if the run exists, or add a separate idempotent
`open-run` command that validates identity and preserves all state. A restart
must never clear a job ID or submission history. Test existing submitted,
completed, and ambiguous runs.

## 3. Expose scheduler allocation and terminal accounting explicitly

Current `status` uses `squeue` fields for state/elapsed/nodes/reason, then `sacct`
fields for job ID/state/elapsed/exit code/nodes. It does not expose allocated
CPUs or GPUs. See
[status_for_job](/Users/holtw/.codex/skills/cluster-slurm/scripts/cluster_slurm.py:1752).

Add stable fields for scheduler job identity, state, exit code, elapsed seconds,
allocated CPUs, allocated GPUs/TRES, accounting source, and whether accounting
is final. Keep unknown values distinct from zero. Report queue time separately
from allocated runtime. Requested resources are not always the allocation
actually charged by the scheduler.

Tests should cover running jobs, delayed accounting, cancellation that is still
pending, failed jobs with nonzero elapsed time, GPU TRES, and scheduler resource
scaling. A successful `scancel` call must not be reported as completed
cancellation before terminal status is confirmed.

## 4. Keep validated scripts and resource reservations immutable

`submit-job` may repair a rendered header after `sbatch --test-only`, including
CPU scaling, before actual submission. That can change an already reserved
budget and the script identity. See
[bounded header repair](/Users/holtw/.codex/skills/cluster-slurm/scripts/cluster_slurm.py:4453)
and [submit-job repair callback](/Users/holtw/.codex/skills/cluster-slurm/scripts/cluster_slurm.py:4662).

Provide a strict submission mode that accepts expected profile/script hashes
and CPU/GPU/runtime limits. If scheduler validation requires a change, return a
proposed replacement receipt for a new reservation; do not apply it and submit
silently. Profile `submit_args` must not override the validated resource limits.
Expose final script and effective resource identities in the submission receipt.

## 5. Make integrity and retrieval scope part of the transport contract

Current upload/download receipts contain paths, but no content hashes. Download
without an explicit remote path retrieves the whole run directory. See
[upload](/Users/holtw/.codex/skills/cluster-slurm/scripts/cluster_slurm.py:4392)
and [download](/Users/holtw/.codex/skills/cluster-slurm/scripts/cluster_slurm.py:4999).

Add manifests with path, byte count, and SHA256 for staged inputs and retrieved
outputs. Verify staged inputs on the worker before execution and verify returned
outputs locally. Require explicit retrieval scope for automated callers. A
worker-generated manifest proves which bytes were returned; it does not grant
permission to disclose a new protected result type.

## Temporary fork adapter

`shinka/modeling/cluster_slurm.py` uses the existing profile-managed CLI through
a thin local loader. It disables transport replay and automatic header repair,
restricts resource overrides, pins CLI/configuration identities, preserves its
own submission intent, and refuses blind resubmission. It copies only declared
input files and retrieves only allowed result paths with hash checks.

The loader is implementation-specific. An official stable API for the contracts
above would let the fork remove it. The adapter's tests use simulated command
responses. They do not establish that a particular cluster profile is ready.
