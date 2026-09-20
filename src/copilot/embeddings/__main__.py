"""Inspection commands: ``python -m copilot.embeddings {info,smoke,tokens,sizes,representations}``.

Prints metadata and statistics only: never source contents and never full vectors.
The first command that loads the model downloads it (~0.64 GB for the default) into the model
cache directory shown by ``info``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence

import numpy as np

from copilot.chunking import chunk_repository, create_chunker
from copilot.config import Settings, get_settings, setup_logging
from copilot.embeddings.errors import EmbeddingError
from copilot.embeddings.factory import create_embedder
from copilot.embeddings.representation import REPRESENTATION_VERSION, embedding_text
from copilot.embeddings.token_validation import CapAssessment, assess_caps, compare_token_counts
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.models.chunk import Chunk

DEFAULT_CAPS = (256, 384, 512, 768, 1024, 2048)
SAMPLE_TEXTS = (
    "def add(a, b):\n    return a + b",
    "class UserRepository:\n    def get(self, user_id):\n        ...",
    "# Installation\nRun `uv sync` to install dependencies.",
)  # generic snippets used when no repository is given


def _peak_rss_mb() -> str:
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return f"{kb / 1024:.0f} MB (peak RSS, Linux units)"
    except ImportError:  # Windows has no `resource`
        return "not available on this platform"


def _chunks(path: str, ignore: Sequence[str], settings: Settings) -> tuple[list[Chunk], object]:
    policy = IngestionPolicy.from_settings(settings).with_extra_ignored_directories(*ignore)
    ingestion = ingest_repository(path, policy)
    result = chunk_repository(ingestion, create_chunker(settings=settings))
    return list(result.chunks), ingestion


def _cmd_info(args: argparse.Namespace, settings: Settings) -> int:
    from fastembed import TextEmbedding

    entry = next(
        (
            m
            for m in TextEmbedding.list_supported_models()
            if m["model"] == settings.embedding_model
        ),
        None,
    )
    print(f"configured model:    {settings.embedding_model}")
    print(f"model cache dir:     {settings.model_cache_dir}")
    style = f"{settings.embedding_text_style} (representation v{REPRESENTATION_VERSION})"
    print(f"text style:          {style}")
    print(f"batch size:          {settings.embedding_batch_size}")
    if entry is None:
        print("fastembed support:   NOT SUPPORTED by the installed fastembed")
        return 2
    print("fastembed support:   supported")
    print(f"  dimension:         {entry['dim']}")
    print(f"  onnx file:         {entry['model_file']}")
    print(f"  source:            {entry['sources']['hf']} (Hugging Face)")
    print(f"  approx. size:      {entry['size_in_GB']} GB")
    print(f"  license:           {entry['license']}")
    print(f"  description:       {entry['description']}")
    return 0


def _cmd_smoke(args: argparse.Namespace, settings: Settings) -> int:
    embedder = create_embedder(settings)
    if args.path:
        chunks, _ = _chunks(args.path, args.ignore_dir, settings)
        texts = [embedding_text(c, settings.embedding_text_style) for c in chunks[: args.n]]
    else:
        texts = list(SAMPLE_TEXTS)
    if not texts:
        print("error: nothing to embed", file=sys.stderr)
        return 2

    started = time.perf_counter()
    info = embedder.info  # loads (and possibly downloads) the model
    load_s = time.perf_counter() - started

    started = time.perf_counter()
    first = embedder.embed_documents(texts)
    embed_s = time.perf_counter() - started
    second = embedder.embed_documents(texts)
    query = embedder.embed_query("where is the database connection configured?")
    norms = np.linalg.norm(first, axis=1)

    print(f"model:               {info.model_id}")
    print(f"runtime:             {info.runtime} {info.runtime_version or ''}")
    print(f"dimension:           {info.dimension}")
    print(f"max input tokens:    {info.max_input_tokens}")
    print(f"vectors embedded:    {len(first)} (batch size {info.batch_size})")
    print(f"query vector shape:  {query.shape}")
    print(f"norm min/max:        {norms.min():.6f} / {norms.max():.6f}")
    print(f"all finite:          {bool(np.isfinite(first).all())}")
    print(f"repeat max |diff|:   {float(np.abs(first - second).max()):.3e}")
    print(f"load time:           {load_s:.2f}s")
    print(f"embed time:          {embed_s:.2f}s ({len(first) / max(embed_s, 1e-9):.1f} texts/s)")
    print(f"memory:              {_peak_rss_mb()}")
    return 0


def _cmd_tokens(args: argparse.Namespace, settings: Settings) -> int:
    embedder = create_embedder(settings)
    chunks, ingestion = _chunks(args.path, args.ignore_dir, settings)
    if not chunks:
        print("error: no chunks", file=sys.stderr)
        return 2
    info = embedder.info
    limits = embedder.runtime_limits()
    model_limit = (
        limits["tokenizer_truncation_max_length"] or limits["config_max_position_embeddings"]
    )
    if model_limit is None:
        print("error: could not determine the model input limit", file=sys.stderr)
        return 2

    actual_raw = embedder.count_tokens([c.content for c in chunks])
    actual_embedded = embedder.count_tokens(
        [embedding_text(c, settings.embedding_text_style) for c in chunks]
    )
    comparison = compare_token_counts(chunks, actual_raw, top=args.examples)
    caps = assess_caps(
        ingestion,  # type: ignore[arg-type]
        embedder.count_tokens,
        caps=args.caps,
        size_lines=settings.chunk_size_lines,
        overlap_lines=settings.chunk_overlap_lines,
        model_limit=model_limit,
        style=settings.embedding_text_style,
    )
    overhead = [e - r for e, r in zip(actual_embedded, actual_raw, strict=True)]

    if args.json:
        print(
            json.dumps(
                {
                    "model": info.model_dump(),
                    "limits": limits,
                    "comparison": comparison.model_dump(),
                    "prefix_overhead_tokens": {
                        "median": statistics.median(overhead),
                        "max": max(overhead),
                    },
                    "caps": [c.model_dump() for c in caps],
                },
                indent=2,
            )
        )
        return 0

    print(f"model: {info.model_id}  dimension: {info.dimension}")
    print(f"limits reported by the model files: {limits}")
    print(
        f"configured cap: {settings.chunk_max_tokens} estimated tokens, "
        f"{settings.chunk_size_lines} lines, overlap {settings.chunk_overlap_lines}\n"
    )
    c = comparison
    print(f"== estimated vs actual tokens ({c.sample_count} chunks, raw content) ==")
    print(f"total estimated / actual:   {c.total_estimated} / {c.total_actual}")
    print(
        f"abs error mean/median/p95/max: {c.mean_abs_error:.1f} / {c.median_abs_error:.1f} / "
        f"{c.p95_abs_error:.1f} / {c.max_abs_error}"
    )
    print(f"mean signed error (est-actual): {c.mean_signed_error:+.1f}")
    print(
        f"underestimated / overestimated / exact: {c.pct_underestimated:.1f}% / "
        f"{c.pct_overestimated:.1f}% / {c.pct_exact:.1f}%"
    )
    print(
        f"actual/estimated ratio median/p95/max: {c.actual_over_estimated_median:.2f} / "
        f"{c.actual_over_estimated_p95:.2f} / {c.actual_over_estimated_max:.2f}"
    )
    for title, rows in (
        ("largest underestimates", c.largest_underestimates),
        ("largest overestimates", c.largest_overestimates),
    ):
        print(f"{title}:")
        for r in rows:
            where = f"{r.file_path}:{r.start_line}-{r.end_line}"
            print(f"  {where}  estimated={r.estimated} actual={r.actual}")
    print(
        f"\nprefix overhead (embedded - raw actual tokens): median {statistics.median(overhead)} "
        f"max {max(overhead)}"
    )
    _print_caps(caps, model_limit, with_tokens=True)
    return 0


def _print_caps(caps: Sequence[CapAssessment], model_limit: int, *, with_tokens: bool) -> None:
    print(f"\n== candidate caps (model limit {model_limit}; same line size/overlap) ==")
    head = "cap    chunks frags cap-cut%  est med/p95/max   bytes med/p95/max    bytes-bound>limit"
    if with_tokens:
        head += "  | actual med/p95/max  embedded-max  >limit  >512  >=90%"
    print(head)
    for a in caps:
        row = (
            f"{a.cap:<6} {a.chunk_count:<6} {a.fragment_chunks:<5} {a.cap_shortened_pct:>7.1f}  "
            f"{a.estimated_median:>4.0f}/{a.estimated_p95:>4.0f}/{a.estimated_max:<5} "
            f"{a.embedded_bytes_median:>6.0f}/{a.embedded_bytes_p95:>5.0f}/{a.embedded_bytes_max:<6}"
            f" {a.bytes_bound_exceeds_limit:>10}"
        )
        if with_tokens:
            row += (
                f"  | {a.actual_median:>6.0f}/{a.actual_p95:>4.0f}/{a.actual_max:<5} "
                f"{a.embedded_max:>12}  {a.over_model_limit:>6}  {a.over_512_actual:>4} "
                f"{a.at_least_90pct_of_limit:>6}"
            )
        print(row)


def _cmd_sizes(args: argparse.Namespace, settings: Settings) -> int:
    """Tokenizer-free chunk-size report; needs no model and no network."""
    policy = IngestionPolicy.from_settings(settings).with_extra_ignored_directories(
        *args.ignore_dir
    )
    ingestion = ingest_repository(args.path, policy)
    caps = assess_caps(
        ingestion,
        None,
        caps=args.caps,
        size_lines=settings.chunk_size_lines,
        overlap_lines=settings.chunk_overlap_lines,
        model_limit=args.model_limit,
        style=settings.embedding_text_style,
    )
    print(
        f"line size {settings.chunk_size_lines}, overlap {settings.chunk_overlap_lines}, "
        f"text style {settings.embedding_text_style}; NO tokenizer used (estimates and bytes only)"
    )
    _print_caps(caps, args.model_limit, with_tokens=False)
    return 0


def _cmd_representations(args: argparse.Namespace, settings: Settings) -> int:
    embedder = create_embedder(settings)
    chunks, _ = _chunks(args.path, args.ignore_dir, settings)
    step = max(1, len(chunks) // args.n)
    sample = chunks[::step][: args.n]
    if not sample:
        print("error: no chunks", file=sys.stderr)
        return 2
    raw = embedder.embed_documents([embedding_text(c, "raw") for c in sample])
    prefixed = embedder.embed_documents([embedding_text(c, "prefixed") for c in sample])
    cos = (raw * prefixed).sum(axis=1)  # unit vectors: dot == cosine
    raw_tok = embedder.count_tokens([embedding_text(c, "raw") for c in sample])
    pre_tok = embedder.count_tokens([embedding_text(c, "prefixed") for c in sample])
    print(f"sample: {len(sample)} chunks (evenly spaced), model {embedder.info.model_id}")
    print(
        f"cosine(raw, prefixed) per chunk: mean {cos.mean():.3f} min {cos.min():.3f} "
        f"max {cos.max():.3f}"
    )
    overhead = [p - r for p, r in zip(pre_tok, raw_tok, strict=True)]
    print(f"prefix token overhead: median {statistics.median(overhead)} max {max(overhead)}")
    print("NOTE: this shows how much the prefix moves vectors, not whether retrieval improves.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m copilot.embeddings", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show the configured model and fastembed's facts (no download)")

    def add_repo(p: argparse.ArgumentParser, required: bool) -> None:
        p.add_argument("path", nargs=None if required else "?", help="repository directory")
        p.add_argument("--ignore-dir", action="append", default=[], help="extra directory to prune")

    smoke = sub.add_parser(
        "smoke", help="load the model and embed a few texts (downloads on first run)"
    )
    add_repo(smoke, required=False)
    smoke.add_argument("-n", type=int, default=8, help="number of chunks to embed")

    tokens = sub.add_parser("tokens", help="estimated vs actual tokens and cap assessment")
    add_repo(tokens, required=True)
    tokens.add_argument(
        "--caps", type=lambda s: [int(x) for x in s.split(",")], default=list(DEFAULT_CAPS)
    )
    tokens.add_argument("--examples", type=int, default=5)
    tokens.add_argument("--json", action="store_true")

    sizes = sub.add_parser("sizes", help="tokenizer-free chunk sizes per cap (no model needed)")
    add_repo(sizes, required=True)
    sizes.add_argument(
        "--caps", type=lambda s: [int(x) for x in s.split(",")], default=list(DEFAULT_CAPS)
    )
    sizes.add_argument("--model-limit", type=int, default=8192, help="model input limit in tokens")

    reps = sub.add_parser("representations", help="compare raw vs metadata-prefixed embedding text")
    add_repo(reps, required=True)
    reps.add_argument("-n", type=int, default=20)

    args = parser.parse_args(argv)
    setup_logging()
    settings = get_settings()
    commands = {
        "info": _cmd_info,
        "smoke": _cmd_smoke,
        "tokens": _cmd_tokens,
        "sizes": _cmd_sizes,
        "representations": _cmd_representations,
    }
    try:
        return commands[args.command](args, settings)
    except EmbeddingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
