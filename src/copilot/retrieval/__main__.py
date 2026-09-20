"""Terminal search: ``python -m copilot.retrieval search INDEX_DIR --repo PATH "question"``.

Prints, for each result, the rank and cosine score, ``file:start-end``, the chunk id and a short
source preview. Output uses repository-relative paths only (never absolute host paths) and never
dumps a whole file. The index is built with ``python -m copilot.vectorstore build``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from copilot.chunking.errors import ChunkingError
from copilot.config import get_settings, setup_logging
from copilot.embeddings.errors import EmbeddingError
from copilot.embeddings.factory import create_embedder
from copilot.ingestion.errors import IngestionError
from copilot.retrieval.errors import RetrievalError
from copilot.retrieval.results import RetrievalResult
from copilot.retrieval.retriever import MAX_TOP_K, Retriever, validate_query, validate_top_k
from copilot.vectorstore.errors import VectorStoreError

PREVIEW_LINES = 8
PREVIEW_WIDTH = 100


def _clean(line: str) -> str:
    """Printable text: control characters (e.g. terminal escapes) become spaces; tabs stay."""
    return "".join(ch if ch == "\t" or ch.isprintable() else " " for ch in line)


def preview(text: str, max_lines: int = PREVIEW_LINES, width: int = PREVIEW_WIDTH) -> list[str]:
    """The first ``max_lines`` lines of ``text``, indentation preserved, long lines truncated."""
    lines = text.split("\n")
    shown = [_clean(line.rstrip()) for line in lines[:max_lines]]
    shown = [line if len(line) <= width else line[: width - 3] + "..." for line in shown]
    if len(lines) > max_lines:
        shown.append(f"... (+{len(lines) - max_lines} more lines)")
    return shown


def format_result(result: RetrievalResult, max_lines: int = PREVIEW_LINES) -> str:
    """One result block: score, location, chunk id and a short indented source preview."""
    symbol = f"  {result.qualified_name}" if result.qualified_name else ""
    head = [
        f"#{result.rank:<3} score {result.score:.4f}",
        f"{result.location}{symbol}",
        f"chunk {result.chunk_id}  {result.chunk_type.value}  {result.line_count} lines",
    ]
    return "\n".join([*head, *(f"    | {line}" for line in preview(result.text, max_lines))])


def _cmd_search(args: argparse.Namespace) -> int:
    validate_query(args.query)  # a bad query fails before the model is loaded
    top_k = validate_top_k(args.top_k)
    settings = get_settings()
    embedder = create_embedder(settings)
    print(
        f"note: the query is embedded locally with {settings.embedding_model}; loading the model "
        "takes a few seconds.",
        file=sys.stderr,
    )
    retriever = Retriever.open(
        args.index_dir,
        args.repo,
        embedder=embedder,
        settings=settings,
        ignore_directories=args.ignore_dir,
    )
    results = retriever.retrieve(args.query, top_k)
    spec = retriever.index.spec
    print(
        f"index {retriever.index.index_id}  repository {spec.repository_name}  "
        f"{len(results)} of {retriever.index.count} chunks shown (top_k={top_k})"
    )
    for result in results:
        print()
        print(format_result(result, args.preview_lines))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m copilot.retrieval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search", help="semantic search over a built index (no LLM)")
    search.add_argument("index_dir", help="index directory (data/indexes/<index id>)")
    search.add_argument("--repo", required=True, help="the repository that was indexed")
    search.add_argument("query", help="natural-language question (quote it)")
    search.add_argument(
        "--top-k", type=int, default=get_settings().retrieval_top_k, help=f"results (1-{MAX_TOP_K})"
    )
    search.add_argument(
        "--ignore-dir",
        action="append",
        default=[],
        help="extra directory pruned at build time (repeat the build's --ignore-dir flags)",
    )
    search.add_argument(
        "--preview-lines", type=int, default=PREVIEW_LINES, help="source lines shown per result"
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):  # a Windows console may not encode every character
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    setup_logging()
    try:
        return _cmd_search(args)
    except (
        RetrievalError,
        VectorStoreError,
        EmbeddingError,
        IngestionError,
        ChunkingError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
