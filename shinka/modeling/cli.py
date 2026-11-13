"""Local scientific ledger, promotion, reports, and asynchronous guidance tools."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "report"):
        command = commands.add_parser(name)
        command.add_argument("--ledger", required=True)
        if name == "report":
            command.add_argument("--output", required=True)
    steer = commands.add_parser("steer")
    steer.add_argument("--inbox", required=True)
    steer.add_argument("--kind", choices=["direction", "preference", "avoid", "question", "pause", "resume"], required=True)
    steer.add_argument("--text", required=True)
    steer.add_argument("--evidence", action="append", default=[])
    steer.add_argument("--supersedes", action="append", default=[])
    steer.add_argument("--expires-at")
    guidance = commands.add_parser("guidance")
    guidance.add_argument("--inbox", required=True)
    papers = commands.add_parser("paper-add")
    papers.add_argument("--library", required=True)
    papers.add_argument("--file", required=True)
    papers.add_argument("--title", required=True)
    papers.add_argument("--citation", default="")
    papers.add_argument("--notes", default="")
    papers.add_argument("--tag", action="append", default=[])
    papers.add_argument("--private", action="store_true", help="Retain for the scientist; exclude from author prompts")
    paper_list = commands.add_parser("papers")
    paper_list.add_argument("--library", required=True)
    for name in ("constraints-init", "constraints", "constraint-request-change",
                 "constraint-request-exception", "constraint-approve", "constraint-reject", "constraint-review"):
        command = commands.add_parser(name)
        command.add_argument("--book", required=True)
        if name in {"constraints-init", "constraint-request-change"}:
            command.add_argument("--file", required=True, help="Short user-written guidelines in plain text")
        if name in {"constraint-request-change", "constraint-request-exception", "constraint-reject"}:
            command.add_argument("--reason", required=True)
        if name in {"constraint-request-exception", "constraint-review"}:
            command.add_argument("--source-sha", required=True)
        if name in {"constraint-approve", "constraint-reject"}:
            command.add_argument("--request", required=True)
        if name in {"constraint-approve", "constraint-reject", "constraint-review"}:
            command.add_argument("--actor", default="human", help="Trusted operator identity for the audit record")
        if name == "constraint-review":
            command.add_argument("--compliant", choices=["yes", "no"], required=True)
            command.add_argument("--evidence", required=True)
    promotion = commands.add_parser("promote")
    promotion.add_argument("--ledger", required=True)
    promotion.add_argument("--artifacts", required=True)
    promotion.add_argument("--stages", required=True, help="JSON array of StageSpec objects")
    promotion.add_argument("--calibrator", required=True, help="Trusted module:function, shared by all stages")
    promotion.add_argument("--limit", type=int, default=2, help="Promotions per stage per pass")
    promotion.add_argument("--exploration-fraction", type=float, default=0.25)
    promotion.add_argument("--passes", type=int, default=1, help="Finite background polling passes")
    promotion.add_argument("--interval", type=float, default=15.0)
    promotion.add_argument("--seed", type=int, default=0)
    promotion.add_argument("--report-output")
    args = parser.parse_args(argv)
    if args.command.startswith("constraint"):
        from .constraints import ConstraintBook
        book = ConstraintBook(args.book)
        if args.command == "constraints-init":
            result = book.initialize(Path(args.file).read_text()).to_dict()
        elif args.command == "constraints":
            result = {"active": book.snapshot().to_dict(), "events": book.events()}
        elif args.command == "constraint-request-change":
            result = book.request_change(Path(args.file).read_text(), args.reason)
        elif args.command == "constraint-request-exception":
            result = book.request_exception(args.source_sha, args.reason)
        elif args.command == "constraint-approve":
            result = book.approve(args.request, actor=args.actor)
        elif args.command == "constraint-reject":
            result = book.reject(args.request, actor=args.actor, reason=args.reason)
        else:
            result = book.review_compliance(args.source_sha, args.compliant == "yes", args.evidence, args.actor)
        print(json.dumps(result, indent=2))
        return 0
    if args.command in {"paper-add", "papers"}:
        from .literature import PaperLibrary
        library = PaperLibrary(args.library)
        if args.command == "paper-add":
            record = library.add(args.file, title=args.title, citation=args.citation,
                                 notes=args.notes, tags=args.tag, author_visible=not args.private)
            print(json.dumps(record.to_dict() if hasattr(record, "to_dict") else record, indent=2, default=str))
        else:
            print(json.dumps(library.snapshot().to_dict(), indent=2))
        return 0
    if args.command in {"steer", "guidance"}:
        from .steering import GuidanceInbox
        inbox = GuidanceInbox(args.inbox)
        if args.command == "steer":
            message = inbox.publish(args.kind, args.text, evidence_ids=args.evidence,
                                    supersedes=args.supersedes, expires_at=args.expires_at)
            print(json.dumps(message.to_dict(), indent=2))
        else:
            print(json.dumps(inbox.snapshot().to_dict(), indent=2))
        return 0
    from .store import Ledger
    # This CLI only opens a ledger created by the trusted campaign setup, with
    # persisted finite limits. A typo must not create an unlimited new campaign.
    if not Path(args.ledger).is_file():
        parser.error("Ledger does not exist. Create a bounded campaign through the Python API first.")
    ledger = Ledger(args.ledger)
    if args.command == "status":
        print(json.dumps(ledger.usage(), indent=2))
    elif args.command == "report":
        from .reports import write_report
        print(write_report(ledger.records(), args.output, usage=ledger.usage()))
    elif args.command == "promote":
        from .campaign import promote
        from .contracts import StageSpec
        from .pipeline import CalibrationPipeline
        if args.passes < 1 or args.limit < 1 or not math.isfinite(args.interval) or args.interval < 0:
            parser.error("passes and limit must be positive; interval must be nonnegative")
        if not (0 <= args.exploration_fraction <= 1):
            parser.error("exploration-fraction must be between 0 and 1")
        usage = ledger.usage()
        if any(usage.get(key) is None for key in ("max_evaluations", "max_seconds")):
            parser.error("Promotion requires persisted finite evaluation and time budgets")
        stages = [StageSpec(**entry) for entry in json.loads(Path(args.stages).read_text())]
        module, separator, name = args.calibrator.partition(":")
        if not separator:
            parser.error("calibrator must be module:function")
        calibrate = getattr(importlib.import_module(module), name)
        pipeline = CalibrationPipeline(ledger, args.artifacts)
        try:
            for index in range(args.passes):
                if (Path(args.artifacts) / "STOP").exists():
                    break
                records = promote(pipeline, calibrate, stages,
                                  lock_path=Path(args.artifacts) / "promotion.lock",
                                  limit_per_stage=args.limit, exploration_fraction=args.exploration_fraction,
                                  seed=args.seed + index)
                print(json.dumps({"pass": index + 1, "evaluations": [r.cache_key for r in records],
                                  "usage": ledger.usage()}), flush=True)
                if args.report_output:
                    from .reports import write_report
                    write_report(ledger.records(), args.report_output, usage=ledger.usage())
                if index + 1 < args.passes:
                    # Respond to STOP even while polling an otherwise idle search.
                    deadline = time.monotonic() + args.interval
                    while time.monotonic() < deadline:
                        if (Path(args.artifacts) / "STOP").exists():
                            return 0
                        time.sleep(min(0.25, max(0, deadline - time.monotonic())))
        except KeyboardInterrupt:
            return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
