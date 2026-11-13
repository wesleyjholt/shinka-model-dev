"""Run trusted scientific callbacks under durable, wall-time bounded accounting."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import signal
import time
import traceback
from dataclasses import replace
from pathlib import Path

from .contracts import CalibrationResult


class EvaluationInProgress(RuntimeError):
    pass


def _worker(connection, calibrate, request, artifact_dir):
    # On POSIX this lets timeout cleanup stop descendants as well as the worker.
    if hasattr(os, "setsid"):
        os.setsid()
    notice_sent = False
    try:
        result = calibrate(request, artifact_dir)
        if not isinstance(result, CalibrationResult):
            raise TypeError("calibrator must return CalibrationResult")
        if request.stage.method.lower() == "map" and result.posterior_kind not in {"none", "point_estimate"}:
            raise ValueError("MAP is a point estimate, not posterior uncertainty; use a separate approximation stage")
        serialized = json.dumps(result.to_dict(), allow_nan=False).encode()
        if len(serialized) > 1024 * 1024:
            raise ValueError("result metadata exceeds 1 MiB; store large samples or predictions as artifacts")
        destination = Path(artifact_dir) / ".calibration-result.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(serialized)
        temporary.replace(destination)
        # Large pipe messages can block recv() after poll() signals only the
        # first bytes. Send a tiny notice after the complete file is available.
        connection.send((True, None))
        notice_sent = True
    except BaseException:
        connection.send((False, traceback.format_exc(limit=12).encode()[-3000:].decode(errors="replace")))
        notice_sent = True
    finally:
        # Keep the group leader owned and alive until the controller cleans up
        # the whole group. Reaping the leader first can make killpg fail with
        # EPERM under macOS sandbox rules, even for a group we just created.
        if notice_sent:
            try:
                connection.recv()
            except (EOFError, OSError):
                pass
        connection.close()


def _stop(process):
    if process.pid is None:
        return
    # A completion notice leaves the leader waiting for controller cleanup.
    # Kill the complete group in one operation: TERM followed by join/reaping
    # and a later KILL would recreate the dead-leader permissions race.
    group_stopped = False
    if hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGKILL)
            group_stopped = True
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            if not process.is_alive():
                # A crashed or previously reaped worker no longer grants a
                # sandbox permission to address its former process group.
                return
            # Stop the owned leader if possible, but do not claim descendants
            # were cleaned up when permission to signal their group failed.
            process.kill()
            process.join(0.5)
            raise RuntimeError("permission denied cleaning up a live worker process group; descendants may remain") from exc
    if process.is_alive():
        if not group_stopped:
            process.kill()
        process.join(0.5)
    if process.is_alive():
        raise RuntimeError("calibration worker survived forced cleanup")


def _verify_source(candidate):
    if candidate.source_path:
        source = Path(candidate.source_path)
        if not source.is_file():
            raise ValueError("candidate source_path must refer to an immutable source file")
        if hashlib.sha256(source.read_bytes()).hexdigest() != candidate.source_sha:
            raise ValueError("candidate source SHA does not match its file")


class CalibrationPipeline:
    """Callbacks are trusted, importable top-level Python functions.

    The subprocess supplies timeout/process isolation, not a hostile-code
    sandbox. Scientific adapters must keep protected observations outside
    candidate-author processes. A callback is responsible for enforcing its
    declared method work budget (particles, draws, solver calls, and so on).
    """

    def __init__(self, ledger, artifact_root, *, constraints=None, constraint_reviewer=None):
        self.ledger = ledger
        self.artifact_root = Path(artifact_root).resolve()
        # Bind once in trusted campaign state. A resumed promotion worker cannot
        # bypass active constraints merely by omitting the constructor argument.
        from .constraints import ConstraintBook
        import fcntl
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        binding = self.artifact_root / ".constraint-book.json"
        with (self.artifact_root / ".constraint-binding.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if binding.exists():
                root = Path(json.loads(binding.read_text())["root"]).resolve()
                if constraints is not None and constraints.root.resolve() != root:
                    raise ValueError("Campaign is already bound to a different user constraint book")
                constraints = ConstraintBook(root)
            elif constraints is not None:
                temporary = binding.with_suffix(".tmp")
                temporary.write_text(json.dumps({"root": str(constraints.root.resolve())}))
                temporary.replace(binding)
        self.constraints = constraints
        self.constraint_reviewer = constraint_reviewer

    def evaluate(self, request, calibrate):
        _verify_source(request.candidate)
        constraint_snapshot = self.constraints.snapshot() if self.constraints is not None else None
        if constraint_snapshot is not None and constraint_snapshot.version == 0:
            raise ValueError("Configured user constraint book is missing or uninitialized; it cannot silently disable the gate")
        if constraint_snapshot is not None and constraint_snapshot.active:
            if self.constraint_reviewer is not None:
                # This reviewer belongs to the trusted evaluator. Authors cannot
                # approve their own compliance or exceptions.
                review = self.constraint_reviewer(request.candidate, constraint_snapshot)
                if review is not None:
                    self.constraints.review_compliance(request.candidate.source_sha,
                        expected_constraint_sha=constraint_snapshot.sha256, **review)
                    if not review["compliant"]:
                        self.constraints.request_exception(request.candidate.source_sha, reason=review["evidence"])
            permission = self.constraints.assert_allowed(request.candidate.source_sha)
            # Capture the exact version accepted by the gate, even if a user
            # approved an update while the trusted review was underway.
            from .constraints import ConstraintSnapshot
            constraint_snapshot = ConstraintSnapshot(permission["version"], permission["sha256"], permission["text"], permission["active"])
        artifact_dir = self.artifact_root / request.cache_key
        record, claimed = self.ledger.reserve(request, str(artifact_dir))
        if not claimed:
            if record.status in {"reserved", "running"}:
                raise EvaluationInProgress(
                    "evaluation is already reserved/running; inspect the worker before fail_abandoned()"
                )
            return record
        started = time.monotonic()
        context = multiprocessing.get_context("spawn")
        reader, writer = context.Pipe(duplex=True)
        process = context.Process(target=_worker, args=(writer, calibrate, request, str(artifact_dir)))
        try:
            artifact_dir.mkdir(parents=True, exist_ok=False)
            self.ledger.start(request.cache_key)
            process.start()
            writer.close()
            remaining = max(0.0, request.stage.max_seconds - (time.monotonic() - started))
            if not reader.poll(remaining):
                raise TimeoutError(f"calibration exceeded {request.stage.max_seconds:g} seconds")
            succeeded, payload = reader.recv()
            if not succeeded:
                raise RuntimeError(payload)
            _verify_source(request.candidate)
            result = CalibrationResult.from_dict(json.loads(
                (artifact_dir / ".calibration-result.json").read_text()
            ))
            if constraint_snapshot is not None and constraint_snapshot.active:
                result = replace(result, notes=result.notes + (
                    f"User constraints at evaluation start: version={constraint_snapshot.version}, sha256={constraint_snapshot.sha256}",
                ))
            # A callback may return before children exit; no process survives a stage.
            _stop(process)
            return self.ledger.complete(request.cache_key, result, time.monotonic() - started)
        except Exception as exc:
            _stop(process)
            return self.ledger.fail(request.cache_key, f"{type(exc).__name__}: {exc}", time.monotonic() - started)
        except BaseException:
            _stop(process)
            self.ledger.fail(request.cache_key, "controller interrupted", time.monotonic() - started)
            raise
        finally:
            reader.close()
            writer.close()
