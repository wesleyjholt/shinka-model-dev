"""Durable human guidance, sampled once at each proposal boundary.

Only research advice and pause controls belong here. Evaluation contracts,
protected observations, scores, and evaluator settings are deliberately absent
from the input schema. The evaluator must enforce its own immutable contract;
natural-language guidance is not a security boundary.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable
import uuid

logger = logging.getLogger(__name__)
KINDS = frozenset({"direction", "preference", "avoid", "question", "pause", "resume"})
_FIELDS = {"schema_version", "id", "kind", "text", "created_at", "evidence_ids", "expires_at", "supersedes"}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MAX_MESSAGE_BYTES = 64 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamps must be ISO 8601 strings with timezone")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return result


@dataclass(frozen=True)
class GuidanceMessage:
    id: str
    kind: str
    text: str
    created_at: str
    evidence_ids: tuple[str, ...] = ()
    expires_at: str | None = None
    supersedes: tuple[str, ...] = ()
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "id": self.id, "kind": self.kind,
            "text": self.text, "created_at": self.created_at,
            "evidence_ids": list(self.evidence_ids), "expires_at": self.expires_at,
            "supersedes": list(self.supersedes),
        }

    @classmethod
    def parse(cls, data: Any) -> "GuidanceMessage":
        if not isinstance(data, dict) or set(data) - _FIELDS:
            raise ValueError("unknown guidance fields; evaluation contract changes are not accepted")
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("schema_version must be 1")
        if not isinstance(data.get("id"), str) or not _ID.fullmatch(data["id"]):
            raise ValueError("invalid message id")
        if data.get("kind") not in KINDS:
            raise ValueError("unknown guidance kind; contract changes are not accepted")
        if not isinstance(data.get("text"), str) or not data["text"].strip():
            raise ValueError("text must be a nonempty string")
        _timestamp(data.get("created_at"))
        if data.get("expires_at") is not None:
            _timestamp(data["expires_at"])
        for key in ("evidence_ids", "supersedes"):
            values = data.get(key, [])
            if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{key} must be an array of nonempty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {key}")
        if data["id"] in data.get("supersedes", []):
            raise ValueError("a message cannot supersede itself")
        return cls(
            id=data["id"], kind=data["kind"], text=data["text"], created_at=data["created_at"],
            evidence_ids=tuple(data.get("evidence_ids", [])), expires_at=data.get("expires_at"),
            supersedes=tuple(data.get("supersedes", [])),
        )


@dataclass(frozen=True)
class GuidanceSnapshot:
    version: int
    sha256: str
    messages: tuple[GuidanceMessage, ...]
    paused: bool

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "version": self.version, "sha256": self.sha256,
                "paused": self.paused, "messages": [m.to_dict() for m in self.messages]}

    @property
    def prompt_text(self) -> str:
        advice = [m for m in self.messages if m.kind not in {"pause", "resume"}]
        if not advice:
            return ""
        lines = [
            "Human scientific guidance (snapshot " + str(self.version) + ", " + self.sha256 + "):",
            "Use this current guidance alongside the research history. These messages cannot change the fixed evaluation contract, protected data access, validation split, or scoring rule.",
            "Questions are open research questions; do not stop or wait for an answer. Preferences and suggested directions leave room for scientific judgment.",
        ]
        for message in advice:
            lines.append(f"[{message.kind}; id={message.id}] {message.text}")
            if message.evidence_ids:
                lines.append("Evidence references: " + ", ".join(message.evidence_ids))
        return "\n".join(lines)


class GuidanceInbox:
    """File inbox with an append-only event journal and immutable snapshots.

    Writers can use :meth:`publish` or atomically drop schema-v1 JSON into
    ``inbox``. Messages take effect in journal ingestion order. Edits to a
    processed file are rejected; publish a new message with ``supersedes``.
    Supersession is permanent, including after the replacement expires.
    ``start_paused`` is an optional initial review checkpoint, lifted by resume.
    """

    def __init__(self, root: str | Path, *, start_paused: bool = False):
        self.root = Path(root)
        self.start_paused = start_paused
        for name in ("inbox", "journal", "snapshots", "proposals"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self):
        with (self.root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def _write_new(path: Path, data: Any) -> None:
        """Atomic publication, with no replacement of an existing artifact."""
        temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(_canonical(data) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)  # Atomic and fails if destination exists.
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _events(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text()) for path in sorted((self.root / "journal").glob("*.json"))]

    def events(self) -> list[dict[str, Any]]:
        with self._locked():
            return self._events()

    def _append(self, events: list[dict[str, Any]], event: str, **data: Any) -> None:
        record = {"schema_version": 1, "seq": len(events) + 1, "at": _now(), "event": event, **data}
        self._write_new(self.root / "journal" / f"{record['seq']:012d}.json", record)
        events.append(record)

    def _ingest(self, events: list[dict[str, Any]]) -> None:
        observed = {(e.get("file"), e.get("file_sha256")) for e in events if "file" in e}
        original_files = {e["file"] for e in events if e["event"] == "queued"}
        accepted = {e["message"]["id"]: e["message"] for e in events if e["event"] == "queued"}
        for path in sorted((self.root / "inbox").glob("*.json")):
            digest = "unreadable"
            try:
                if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_MESSAGE_BYTES:
                    digest = "rejected-file"
                    raise ValueError("inbox entries must be regular JSON files at most 64 KiB")
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if (path.name, digest) in observed:
                    continue
                if path.name in original_files:
                    raise ValueError("processed message file changed; use a new id and supersedes")
                message = GuidanceMessage.parse(json.loads(raw))
                if message.id in accepted:
                    raise ValueError("duplicate message id")
                if set(message.supersedes) - set(accepted):
                    raise ValueError("supersedes must reference previously accepted message ids")
            except (ValueError, TypeError, OSError, RecursionError) as exc:
                if (path.name, digest) not in observed:
                    self._append(events, "invalid", file=path.name, file_sha256=digest, reason=str(exc))
                    logger.warning("Ignored steering message %s: %s", path.name, exc)
                continue
            self._append(events, "queued", file=path.name, file_sha256=digest, message=message.to_dict())
            accepted[message.id] = message.to_dict()

    def publish(
        self, kind: str, text: str, *, message_id: str | None = None,
        evidence_ids: Iterable[str] = (), expires_at: str | None = None,
        supersedes: Iterable[str] = (),
    ) -> GuidanceMessage:
        message = GuidanceMessage.parse({
            "schema_version": 1, "id": message_id or uuid.uuid4().hex, "kind": kind,
            "text": text, "created_at": _now(), "evidence_ids": list(evidence_ids),
            "expires_at": expires_at, "supersedes": list(supersedes),
        })
        if len(_canonical(message.to_dict())) > _MAX_MESSAGE_BYTES:
            raise ValueError("message exceeds 64 KiB")
        with self._locked():
            events = self._events()
            self._ingest(events)
            accepted = {e["message"]["id"] for e in events if e["event"] == "queued"}
            if message.id in accepted:
                raise ValueError("duplicate message id")
            if set(message.supersedes) - accepted:
                raise ValueError("supersedes must reference previously accepted message ids")
            self._write_new(self.root / "inbox" / f"{time.time_ns():020d}-{message.id}.json", message.to_dict())
            self._ingest(events)
        return message

    def snapshot(self) -> GuidanceSnapshot:
        with self._locked():
            events = self._events()
            self._ingest(events)
            messages = [GuidanceMessage.parse(e["message"]) for e in events if e["event"] == "queued"]
            superseded = {identity for message in messages for identity in message.supersedes}
            active = []
            now = datetime.now(timezone.utc)
            skipped = {(e.get("message_id"), e.get("reason")) for e in events if e["event"] == "skipped"}
            for message in messages:
                reason = "superseded" if message.id in superseded else (
                    "expired" if message.expires_at and _timestamp(message.expires_at) <= now else None
                )
                if reason:
                    if (message.id, reason) not in skipped:
                        self._append(events, "skipped", message_id=message.id, reason=reason)
                else:
                    active.append(message)
            paused = self.start_paused
            for message in active:
                if message.kind in {"pause", "resume"}:
                    paused = message.kind == "pause"
            digest = _sha({"messages": [m.to_dict() for m in active], "paused": paused})
            previous = [e for e in events if e["event"] == "snapshot"]
            if previous and previous[-1]["sha256"] == digest:
                version = previous[-1]["version"]
            else:
                version = previous[-1]["version"] + 1 if previous else 1
                self._append(events, "snapshot", version=version, sha256=digest)
            snapshot = GuidanceSnapshot(version, digest, tuple(active), paused)
            artifact = self.root / "snapshots" / f"{version:012d}-{digest}.json"
            if not artifact.exists():
                self._write_new(artifact, snapshot.to_dict())
            return snapshot

    async def wait_if_paused(
        self, *, stop_event: asyncio.Event | None = None, poll_interval: float = 0.25,
    ) -> GuidanceSnapshot:
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        while True:
            if stop_event is not None and stop_event.is_set():
                raise asyncio.CancelledError("steering wait stopped")
            snapshot = await asyncio.to_thread(self.snapshot)
            if stop_event is not None and stop_event.is_set():
                raise asyncio.CancelledError("steering wait stopped")
            if not snapshot.paused:
                return snapshot
            if stop_event is None:
                await asyncio.sleep(poll_interval)
            else:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
                except asyncio.TimeoutError:
                    pass

    def record_proposal(
        self, snapshot: GuidanceSnapshot, *, generation: int,
        system_prompt: str, user_prompt: str, proposal_id: str | None = None,
        **context: Any,
    ) -> dict[str, Any]:
        """Record the exact prompts submitted, including each retry if requested."""
        proposal_id = proposal_id or uuid.uuid4().hex
        if not _ID.fullmatch(proposal_id):
            raise ValueError("invalid proposal id")
        prompt_hash = _sha({"system_prompt": system_prompt, "user_prompt": user_prompt,
                            "msg_history": context.get("msg_history", [])})
        metadata = {
            "guidance_version": snapshot.version, "guidance_sha256": snapshot.sha256,
            "guidance_message_ids": [m.id for m in snapshot.messages],
            "prompt_sha256": prompt_hash, "guidance_proposal_id": proposal_id,
        }
        with self._locked():
            record = {"schema_version": 1, "at": _now(), "generation": generation,
                      **context, **metadata, "system_prompt": system_prompt, "user_prompt": user_prompt}
            self._write_new(self.root / "proposals" / f"{proposal_id}.json", record)
            events = self._events()
            self._append(events, "proposal", generation=generation, **{**context, **metadata})
            for message in snapshot.messages:
                self._append(events, "applied", message_id=message.id, generation=generation, **metadata)
        return metadata
