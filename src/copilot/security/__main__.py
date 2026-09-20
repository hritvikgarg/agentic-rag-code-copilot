"""Secret-scan command: ``python -m copilot.security scan PATH``.

Runs the existing safe repository ingestion, scans the accepted files' *content* and prints a safe
summary (rule, relative path, line, column; never the value). Exit status: 0 = no findings,
1 = potential secrets found, 2 = error or incomplete scan (invalid path, truncated ingestion).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from copilot.config import get_settings, setup_logging
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.ingestion.errors import IngestionError
from copilot.security.secret_scanner import scan_repository


def _cmd_scan(args: argparse.Namespace) -> int:
    policy = IngestionPolicy.from_settings(get_settings()).with_extra_ignored_directories(
        *args.ignore_dir
    )
    ingestion = ingest_repository(args.path, policy)
    if ingestion.stats.truncated:
        print(
            f"error: ingestion stopped early ({ingestion.stats.truncation_reason}); the scan is "
            "incomplete, so nothing can be declared safe",
            file=sys.stderr,
        )
        return 2
    report = scan_repository(ingestion.files)

    if args.json:
        payload = {
            "repository_name": ingestion.repository_name,
            "files_scanned": report.files_scanned,
            "findings_count": report.findings_count,
            "counts_by_rule": report.counts_by_rule,
            "counts_by_severity": report.counts_by_severity,
            "findings": [
                f.model_dump(mode="json", exclude=None if args.show_preview else {"preview"})
                for f in report.findings
            ],
        }
        print(json.dumps(payload, indent=2))
        return 1 if report.blocking else 0

    print(f"repository:       {ingestion.repository_name}")
    print(f"files scanned:    {report.files_scanned}")
    print(f"findings:         {report.findings_count}")
    if report.counts_by_rule:
        print(f"by rule:          {report.counts_by_rule}")
        print(f"by severity:      {report.counts_by_severity}")
    for finding in report.findings:
        extra = f"  {finding.preview}" if args.show_preview else ""
        print(f"  {finding.location}  [{finding.severity.value}] {finding.rule_id}{extra}")
    if report.blocking:
        print(
            "RESULT: potential secrets found. This repository's text is REFUSED for external LLM "
            "use until they are removed (rotate any real credential) or the files are excluded.",
            file=sys.stderr,
        )
        return 1
    print("RESULT: no potential secrets found (this reduces risk; it is not a guarantee).")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m copilot.security", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="scan the content of a repository for secrets")
    scan.add_argument("path", help="repository directory")
    scan.add_argument(
        "--ignore-dir",
        action="append",
        default=[],
        help="extra directory to skip (as in the index build); repeatable",
    )
    scan.add_argument("--json", action="store_true", help="machine-readable output")
    scan.add_argument(
        "--show-preview",
        action="store_true",
        help="also show the masked preview (at most 6 characters of long values)",
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):  # a Windows console may not encode every character
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    setup_logging()
    try:
        return _cmd_scan(args)
    except IngestionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
