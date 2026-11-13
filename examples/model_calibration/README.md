# Staged calibration with a known reference

This local example makes no language-model or paid API calls. It fits four
polynomial structures to synthetic training observations, using three stages:

| Stage | Calibration | Role |
|---|---|---|
| Screen | Closed-form MAP | Fast point-prediction test for all four structures |
| Refine | Four short MCMC chains | Estimate uncertainty for three promoted structures |
| Confirm | Tempered SMC, 1,024 particles | Compare two structures with one common protocol |

Run from the repository root:

```bash
python -m examples.model_calibration.run --output ./results/calibration-demo
```

A rerun in the same directory uses the ledger cache. Use a new directory after
changing the comparison contract or budgets.

The promotion policy reserves capacity for exploration. The polynomial choices
are only a small demonstration; the framework does not restrict model classes.
Scores compare candidates **within the same stage**. MAP is a point estimate;
it does not produce posterior uncertainty. Its score is a screening proxy and
is not compared numerically with an MCMC or SMC score.

All fitting uses the training split. Search uses a separate development split.
That split becomes part of model selection; it is not an untouched test set.
The runner freezes the selected structure before evaluating one final test from
the saved posterior. The test score does not feed selection or promotion.

Outputs include `report.md`, `report.json`, the SQLite ledger, sample arrays,
parameter summaries, diagnostic values, predictive intervals, a frozen
selection, and the single final-test result. `selected_predictions.png` shows
50% and 90% posterior predictive bands including observation noise. Training and
development points are marked separately.

## Limits of this example

The known noise standard deviation is fixed because it is part of the synthetic
data generator. Each coefficient has an independent Normal(0, 2) prior. The
Gaussian model has an analytical posterior. That known answer checks the toy
MCMC and SMC implementations. The MAP Hessian also gives proposal scales.

These numerical routines illustrate the framework; they are not a general
inference library. The MCMC check uses basic split R-hat and autocorrelation
ESS. SMC reports weight ESS, ancestors, and acceptance. None of those values
alone proves convergence. Real scientific evaluators must supply suitable
samplers and model-specific reliability checks. They cannot rely on an
analytical reference that does not exist.

A method with a larger budget is not automatically more accurate. A failed
inference check is evidence about the calibration run, not necessarily about
the model. Keep a path for extra calibration work and for diverse structures.

## Use with Shinka-generated candidates

`initial_program.py` exposes a literal `MODEL_SPEC` inside an evolution block.
`evaluate.py` reads it without running candidate code. It snapshots the candidate
by its SHA256 and writes standard Shinka `metrics.json` and `correct.json` files.
Only fixed-protocol MAP development feedback enters the evolutionary archive.

```bash
python examples/model_calibration/evaluate.py \
  --program_path examples/model_calibration/initial_program.py \
  --results_dir results/live-calibration/initial-metrics \
  --ledger_root results/live-calibration
```

Set `MODEL_CALIBRATION_RUN` to that shared ledger root when launching Shinka.
The trusted ledger reserves separate budgets for 10 MAP screens, 4 MCMC fits,
and 2 SMC confirmations, so screening cannot consume confirmation capacity.

A separate numerical worker can poll this command while Shinka works:

```bash
python -m examples.model_calibration.promote \
  --ledger-root results/live-calibration
```

It returns immediately without extra work until the bounded screening batch is
complete. It then chooses candidates for MCMC and SMC, with an exploration quota,
and writes the scientific report. It does not expose the final test. Use
`--minimum-screens N` only when intentionally changing the batch length.

After model search has stopped, add `--finalize` to freeze selection and report
the one final test. Finalization prevents the toy evaluator from accepting more
model search in that campaign. Start a new campaign with a new untouched test
set for further development; changing the output folder alone does not make an
already inspected test set new.
