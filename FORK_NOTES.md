# Fork provenance

- Upstream: https://github.com/SakanaAI/ShinkaEvolve
- Starting commit: `805a2265053418150800974d4e1e5d9db03b4b38`
- Fork: https://github.com/wesleyjholt/shinka-model-dev
- Development branch: `model-development`
- License: Apache-2.0, retained with upstream history and attribution.

The fork adds `shinka/modeling`, an optional `shinka_model` CLI, and the synthetic
`examples/model_calibration` integration. Changes to the upstream runner are
optional guidance and literature hooks. Vanilla configurations retain their prior
behavior. Scientific data, local identities, credentials, run databases, and
posterior artifacts are not part of this source distribution.

See [MODEL_DEVELOPMENT.md](MODEL_DEVELOPMENT.md) for the comparison contract,
calibration stages, exploration policy, reports, and integration limits.

The fork also includes a durable remote work controller, profile-managed
cluster-slurm adapter, bounded numerical worker, and an injected Shinka scheduler.
See [REMOTE_EXECUTION.md](REMOTE_EXECUTION.md) and the completed
[synthetic autoresearch demonstration](DEMO_REPORT.md).
