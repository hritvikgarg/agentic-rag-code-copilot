"""Manual inspection tool: ``python -m copilot.chunking PATH``.

Ingests a repository, chunks it with the configured strategy and prints statistics. With
``--samples N`` it also lists metadata of the first N chunks. It never prints chunk contents.
"""

from __future__ import annotations

import argparse
import json
import sys

from copilot.chunking.base import chunk_repository
from copilot.chunking.errors import ChunkingError
from copilot.chunking.registry import create_chunker
from copilot.config import get_settings, setup_logging
from copilot.ingestion.errors import IngestionError
from copilot.ingestion.loader import ingest_repository
from copilot.ingestion.policy import IngestionPolicy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m copilot.chunking",
        description="Ingest and chunk a repository, then print statistics (no file contents).",
    )
    parser.add_argument("path", help="repository directory")
    parser.add_argument("--strategy", help="chunking strategy (default: from settings)")
    parser.add_argument("--samples", type=int, default=0, help="list metadata of first N chunks")
    parser.add_argument(
        "--ignore-dir", action="append", default=[], help="extra directory name to prune"
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable statistics")
    args = parser.parse_args(argv)

    setup_logging()
    settings = get_settings()
    try:
        policy = IngestionPolicy.from_settings(settings).with_extra_ignored_directories(
            *args.ignore_dir
        )
        ingestion = ingest_repository(args.path, policy)
        chunker = create_chunker(args.strategy, settings=settings)
    except (IngestionError, ChunkingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = chunk_repository(ingestion, chunker)
    stats = result.stats
    if args.json:
        print(json.dumps({"params": chunker.params(), "stats": stats.model_dump()}, indent=2))
        return 0

    print(f"repository:          {result.repository_name}")
    print(f"strategy:            {chunker.name} {chunker.params()}")
    print(f"files ingested:      {ingestion.stats.files_accepted}")
    print(
        f"files chunked:       {stats.files_chunked} (without chunks: {stats.files_without_chunks})"
    )
    print(f"chunks:              {stats.chunk_count}")
    print(f"line fragments:      {stats.line_fragment_count}")
    print(
        f"tokens (estimate):   mean={stats.tokens_mean:.1f} median={stats.tokens_median:.1f} "
        f"p95={stats.tokens_p95:.1f} max={stats.tokens_max}"
    )
    print(f"by chunk type:       {stats.by_chunk_type}")
    print(f"by language:         {stats.by_language}")
    print(f"elapsed:             {stats.elapsed_seconds:.3f}s")
    for chunk in result.chunks[: max(args.samples, 0)]:
        heading = f"  heading={chunk.heading!r}" if chunk.heading else ""
        print(
            f"  {chunk.chunk_id}  {chunk.file_path}:{chunk.start_line}-{chunk.end_line}  "
            f"{chunk.chunk_type.value}  tokens={chunk.token_estimate}{heading}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
