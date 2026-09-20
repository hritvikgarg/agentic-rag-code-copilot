"""Manual inspection tool: ``python -m copilot.ingestion PATH``.

Prints ingestion statistics and relative paths only - never file contents.
"""

from __future__ import annotations

import argparse
import json
import sys

from copilot.config import setup_logging
from copilot.ingestion.errors import IngestionError
from copilot.ingestion.loader import ingest_repository


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m copilot.ingestion",
        description="Run safe repository ingestion and print statistics (no file contents).",
    )
    parser.add_argument("path", help="repository directory")
    parser.add_argument("--skipped", action="store_true", help="also list skipped files")
    parser.add_argument("--json", action="store_true", help="print machine-readable statistics")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        result = ingest_repository(args.path)
    except IngestionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    stats = result.stats
    if args.json:
        payload = {
            "repository_name": result.repository_name,
            "stats": stats.model_dump(),
            "accepted": result.relative_paths,
            "skipped": [s.model_dump(mode="json") for s in result.skipped],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"repository:          {result.repository_name}")
    print(f"files discovered:    {stats.files_discovered}")
    print(f"files accepted:      {stats.files_accepted}")
    print(f"files skipped:       {stats.files_skipped}")
    print(f"skip reasons:        {stats.skip_reasons or '{}'}")
    print(f"directories pruned:  {stats.directories_pruned} {stats.pruned_directory_names or ''}")
    print(f"languages:           {stats.languages}")
    print(f"bytes accepted:      {stats.total_bytes_accepted}")
    print(f"truncated:           {stats.truncated} {stats.truncation_reason or ''}")
    print("accepted files:")
    for path in result.relative_paths:
        print(f"  {path}")
    if args.skipped:
        print("skipped files:")
        for item in result.skipped:
            print(f"  {item.relative_path}  [{item.reason.value}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
