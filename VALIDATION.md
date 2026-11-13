# Validation record

Validated locally on Python 3.12.9 on 2026-11-13. No live LLM requests or private
scientific data were used.

```bash
python -m pytest tests/test_modeling*.py \
  tests/test_proposal_llm_kwargs.py tests/test_billed_proposal_cancellation.py \
  tests/test_async_runner_recovery.py tests/test_prompt_evolution.py \
  tests/test_billed_runner_failures.py tests/test_billed_failure_persistence.py -q
```

Result after remote integration: **214 passed**. The final scheduler shutdown
hardening also passed its focused suite of ten tests. Compilation and
`git diff --check` passed. Remote tests use a fake scheduler and fake transport;
they do not claim live Gautschi commissioning.

The numerical demonstration completed four MAP screens, three MCMC fits, and two
SMC confirmations. It selected the quadratic synthetic model, then opened a
separate final test after selection. Replay reused the same nine fit records.
Real numerical tests compare posterior moments with the analytic Gaussian result;
this reference is available only because the example is deliberately simple.

The Shinka evaluator CLI produced valid standard result files from a source
snapshot. The installed `shinka_model` command and launcher were checked. A
subsequent real subscription-Codex demonstration completed three proposals and
ten local synthetic calibrations, then stopped. See [DEMO_REPORT.md](DEMO_REPORT.md).
Live cluster submission and the NHP scientific adapter still require target-campaign
commissioning.

The new tests exercise stage quotas, cache identity, failure charges, process
timeouts and cleanup, diversity promotion, calibration-method separation,
constraint approval and restart enforcement, paper revisions and omissions,
asynchronous guidance, and exact per-proposal context receipts. The upstream
tests exercise proposal settings, cancellation costs, recovery, and prompt
evolution with the optional hooks unset.
