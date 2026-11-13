"""Scientist-facing reports. These files are private evaluator artifacts by default."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .contracts import EvaluationRecord
from .policy import confirmed_records


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_report(
    records: Iterable[EvaluationRecord], output_dir: str | Path, usage: dict | None = None
) -> Path:
    """Write full calibration evidence; never pass this report to authors implicitly.

    Scores are ranked only within an identical evaluation protocol. A high score
    alone is not evidence of a reliable calibration or a correct mechanism.
    """
    records = list(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        groups[record.request.protocol_key].append(record)
    lines = [
        "# Model development report", "",
        "Scores are higher-is-better and comparable only within each protocol below.",
        "MAP is a point estimate. It does not establish posterior uncertainty.",
        "Repeated development-set use does not constitute an untouched final test.", "",
    ]
    if usage is not None:
        lines += ["## Resource use", "", "```json", json.dumps(usage, indent=2), "```", ""]
    findings = []
    for protocol_key, group in sorted(groups.items()):
        stage = group[0].request.stage
        lines += [f"## {stage.name}: {stage.method}", "", f"Protocol: `{protocol_key}`", "",
                  f"Confirmation stage: {stage.confirmation}. Budget: `{json.dumps(dict(stage.budget), sort_keys=True)}`.", "",
                  "| Model | Status | Score | Reliable | Inference result | Seconds | Evidence |",
                  "| --- | --- | ---: | --- | --- | ---: | --- |"]
        # Pending and failed work remains visible; it cannot win a comparison.
        ordered = sorted(group, key=lambda r: (
            r.result is None or r.status != "completed",
            -(r.result.score if r.result is not None and r.result.score is not None else float("-inf")), r.cache_key))
        for record in ordered:
            result = record.result
            lines.append("| " + " | ".join([
                _cell(record.request.candidate.candidate_id), record.status,
                f"{result.score:.6g}" if result is not None and result.score is not None else "—",
                str(result.reliable) if result is not None else "—",
                _cell(result.posterior_kind) if result is not None else "—",
                f"{record.elapsed_seconds:.3f}", f"`{record.cache_key[:12]}`",
            ]) + " |")
        usable = [r for r in ordered if r.status == "completed" and r.result and r.result.reliable and r.result.score is not None]
        if stage.confirmation:
            usable = confirmed_records(usable)
        if usable:
            by_source = defaultdict(list)
            for record in usable:
                by_source[(record.request.candidate.candidate_id, record.request.candidate.source_sha)].append(record)
            winning_group = max(by_source.values(), key=lambda rs: sum(r.result.score for r in rs) / len(rs))
            leader = winning_group[0]
            mean_score = sum(r.result.score for r in winning_group) / len(winning_group)
            label = "Reliable stage leader"
            if not stage.confirmation:
                label += " (provisional)"
            findings.append({"protocol_key": protocol_key, "candidate_id": leader.request.candidate.candidate_id,
                             "evidence_keys": [r.cache_key for r in winning_group],
                             "mean_score": mean_score, "replicates": len(winning_group),
                             "confirmation": stage.confirmation})
            lines += ["", f"{label}: **{_cell(leader.request.candidate.candidate_id)}**, mean score {mean_score:.6g} over {len(winning_group)} reliable fit(s)."]
        else:
            lines += ["", "No reliable completed fit in this protocol. Diagnose calibration before drawing model conclusions."]
        lines.append("")
        for record in ordered:
            lines += [f"### {_cell(record.request.candidate.candidate_id)} — `{record.cache_key[:12]}`", ""]
            if record.result is None:
                lines += [f"Status: {record.status}. Error: {_cell(record.error or 'none recorded')}.", ""]
                continue
            result = record.result
            payload = {key: result.to_dict()[key] for key in (
                "parameters", "priors", "fixed_parameters", "diagnostics", "reliability_reasons",
                "artifacts", "notes")}
            lines += ["```json", json.dumps(payload, indent=2, sort_keys=True), "```", ""]
    failures = [r for r in records if r.status == "failed" or (r.result and not r.result.reliable)]
    lines += ["## Interpretation and next work", "",
              f"{len(failures)} failed or unreliable fits need diagnosis. A calibration failure does not rule out a model structure.",
              "Keep alternative structures in promotion slots. Recheck promising models with the same confirmation method and budget.",
              "Treat improved prediction as evidence about prediction; use separate checks or experiments to assess a mechanism.",
              "Record each proposed explanation with the evidence keys above. Human notes can enter the steering inbox without stopping the run.", ""]
    report = {"schema_version": 1, "visibility": "evaluator_private", "usage": usage,
              "stage_leaders": findings, "records": [r.to_dict() for r in records]}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    path = output_dir / "report.md"
    path.write_text("\n".join(lines))
    return path
