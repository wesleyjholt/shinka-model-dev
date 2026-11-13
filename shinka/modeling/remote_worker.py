"""Trusted numerical entry point for a staged remote work packet.

This process never launches a model author. Its environment and calibrator are
provided by the trusted cluster profile and campaign, not by candidate prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from .contracts import CalibrationResult, EvaluationRequest
from .pipeline import CalibrationPipeline
from .store import Ledger
from .remote import RemoteSpec


def file_identity(path: Path, relative: str) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": relative, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def safe_path(root: Path, relative: str) -> Path:
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or not name.parts or "\\" in relative:
        raise ValueError("Packet paths must be relative and stay within the declared root")
    result = root.joinpath(*name.parts)
    if not result.resolve().is_relative_to(root.resolve()) or result.is_symlink():
        raise ValueError("Packet symlinks or escaping paths are not permitted")
    return result


def verify_inputs(packet: dict, packet_root: Path) -> None:
    for item in packet["files"]:
        path = safe_path(packet_root / "files", item["path"])
        if file_identity(path, item["path"]) != item:
            raise ValueError(f"Input identity mismatch: {item['path']}")


def _invoke(request, artifact_dir, *, calibrator, output, task_kind):
    """Import and run the trusted callback inside the bounded worker group."""
    module, _, name = calibrator.partition(":")
    callback = getattr(importlib.import_module(module), name)
    result = callback(request, output)
    if task_kind == "calibration":
        return result
    if result is not None:
        payload = result.to_dict() if isinstance(result, CalibrationResult) else result
        encoded = json.dumps(payload, indent=2, allow_nan=False)
        if len(encoded.encode()) > 1024 * 1024:
            raise ValueError("Intermediate metadata exceeds 1 MiB")
        (Path(output) / "intermediate-result.json").write_text(encoded + "\n")
    # Intermediate work gets the same process lifetime controls without
    # representing its return value as a scientific model-selection score.
    return CalibrationResult(score=None, posterior_kind="none")


def run(packet_path: str | Path, output: str | Path, calibrator: str) -> dict:
    packet_path = Path(packet_path).resolve()
    if packet_path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Remote packet metadata exceeds 8 MiB")
    packet = json.loads(packet_path.read_text())
    if packet["schema_version"] != 1:
        raise ValueError("Unsupported remote packet version")
    request = EvaluationRequest.from_dict(packet["request"])
    resources = RemoteSpec(**packet["spec"])
    if request.stage.max_seconds > resources.runtime_seconds:
        raise ValueError("Calibration stage timeout exceeds the remote compute allocation")
    if (request.cache_key != packet["request_key"] or request.protocol_key != packet["protocol_key"]
            or request.candidate.source_sha != packet["source_sha"]):
        raise ValueError("Packet evaluation identities disagree")
    verify_inputs(packet, packet_path.parent)
    source = packet.get("candidate_source")
    if source is None and packet["task_kind"] == "calibration":
        raise ValueError("Remote calibration requires a declared candidate source snapshot")
    if source is not None:
        source_path = safe_path(packet_path.parent / "files", source)
        if file_identity(source_path, source)["sha256"] != request.candidate.source_sha:
            raise ValueError("Remote candidate source does not match the request")
        request = replace(request, candidate=replace(request.candidate, source_path=str(source_path)))
    else:
        request = replace(request, candidate=replace(request.candidate, source_path=""))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty result directory for each remote attempt")
    module, separator, name = calibrator.partition(":")
    if not separator:
        raise ValueError("calibrator must be a trusted module:function")
    # The allowlisted packet may contain trusted evaluator modules. This is not
    # permission for a candidate author to supply or replace the evaluator.
    sys.path.insert(0, str(packet_path.parent / "files"))
    try:
        with TemporaryDirectory(prefix="shinka-remote-worker-") as private:
            ledger = Ledger(Path(private) / "worker.sqlite", max_evaluations=1,
                            max_seconds=request.stage.max_seconds)
            pipeline = CalibrationPipeline(ledger, Path(private) / "artifacts")
            record = pipeline.evaluate(request, partial(
                _invoke, calibrator=calibrator, output=str(output), task_kind=packet["task_kind"]
            ))
            if record.status != "completed":
                raise RuntimeError(f"Remote work failed: {record.error}")
            result, elapsed = record.result, record.elapsed_seconds
    finally:
        sys.path.pop(0)
    verify_inputs(packet, packet_path.parent)
    result_path = None
    if packet["task_kind"] == "calibration":
        if not isinstance(result, CalibrationResult):
            raise TypeError("Calibration callbacks must return CalibrationResult")
        if request.stage.method.lower() == "map" and result.posterior_kind not in {"none", "point_estimate"}:
            raise ValueError("MAP cannot be labelled posterior uncertainty")
        artifacts = {}
        for label, raw_path in result.artifacts.items():
            path = Path(raw_path)
            path = path if path.is_absolute() else output / path
            if not path.resolve().is_relative_to(output) or path.is_symlink() or not path.is_file():
                raise ValueError("Calibration artifacts must be files within this result directory")
            artifacts[label] = str(path.resolve().relative_to(output))
        result = replace(result, artifacts=artifacts)
        encoded = json.dumps(result.to_dict(), indent=2, allow_nan=False)
        if len(encoded.encode()) > 1024 * 1024:
            raise ValueError("Calibration metadata exceeds 1 MiB; write large arrays as artifacts")
        result_path = "calibration-result.json"
        (output / result_path).write_text(encoded + "\n")
    (output / "worker-resources.json").write_text(json.dumps({
        "evaluation_elapsed_seconds": elapsed,
        "slurm": {key: os.environ[key] for key in (
            "SLURM_JOB_ID", "SLURM_JOB_NUM_NODES", "SLURM_CPUS_ON_NODE",
            "SLURM_NTASKS", "SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE"
        ) if key in os.environ},
        "interpretation": "Worker elapsed time excludes queue time and setup; scheduler allocation accounting remains authoritative.",
    }, indent=2) + "\n")
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError("Output symlinks are not allowed")
        if path.is_file():
            artifacts.append(file_identity(path, str(path.relative_to(output))))
    manifest = {
        "schema_version": 1, "work_key": packet["work_key"],
        "request_key": packet["request_key"], "protocol_key": packet["protocol_key"],
        "source_sha": packet["source_sha"], "stage": request.stage.to_dict(),
        "dataset_id": request.dataset_id, "split_id": request.split_id,
        "objective_id": request.objective_id, "evaluator_id": request.evaluator_id,
        "task_kind": packet["task_kind"], "parent_evaluation_key": packet.get("parent_evaluation_key"),
        "artifacts": artifacts, "result_path": result_path,
    }
    temporary = output / ".result-manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "result-manifest.json")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibrator", required=True)
    args = parser.parse_args(argv)
    run(args.packet, args.output, args.calibrator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
