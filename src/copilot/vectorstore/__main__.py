"""Vector-index commands: ``python -m copilot.vectorstore {build,info,validate}``.

* ``build PATH``: ingest, chunk, embed (loads the embedding model) and save a FAISS index.
* ``info INDEX_DIR``: print the manifest summary. Reads ``manifest.json`` only.
* ``validate INDEX_DIR [--repo PATH]``: full integrity check (checksums, mapping, FAISS type,
  dimension, counts, stored vectors). With ``--repo`` it also checks that the index still matches
  that repository and the current configuration (fingerprint, chunking, model id, representation).

``info`` and ``validate`` never load the embedding model. Output contains counts, ids and sizes,
never vectors or source text.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from copilot.config import Settings, get_settings, setup_logging
from copilot.embeddings.errors import EmbeddingError
from copilot.embeddings.factory import create_embedder
from copilot.ingestion.errors import IngestionError
from copilot.vectorstore.builder import build_repository_index, expectation_for_repository
from copilot.vectorstore.errors import (
    IndexCompatibilityError,
    IndexCorruptError,
    VectorStoreError,
)
from copilot.vectorstore.index import VectorIndex, read_manifest
from copilot.vectorstore.manifest import MANIFEST_FILE, IndexExpectation


def _cmd_build(args: argparse.Namespace, settings: Settings) -> int:
    embedder = create_embedder(settings)
    print(
        f"note: the first run downloads the embedding model ({settings.embedding_model}, ~0.64 GB) "
        f"into {settings.model_cache_dir}; later runs reuse that cache. Embedding takes a while.",
        file=sys.stderr,
    )
    report = build_repository_index(
        args.path,
        embedder=embedder,
        settings=settings,
        indexes_dir=args.index_dir,
        repository_name=args.name,
        ignore_directories=args.ignore_dir,
        text_style=args.text_style,
        overwrite=args.force,
        allow_truncated=args.allow_truncated,
    )
    print(f"repository:          {report.repository_name}")
    print(f"files indexed:       {report.files}")
    print(f"chunks:              {report.chunks}")
    print(f"vectors:             {report.vectors}")
    print(f"dimension:           {report.dimension}")
    print(f"embedding model:     {report.embedding_model_id}")
    print(f"representation:      {report.text_style} (v{report.representation_version})")
    print(f"index type:          {report.index_type}")
    print(f"index id:            {report.index_id}")
    print(f"location:            {report.index_path}")
    if report.ingestion_truncated:
        print("WARNING:             ingestion was truncated; the index is incomplete")
    print(
        f"time (s):            total {report.seconds_total:.1f} = "
        f"ingest {report.seconds_ingest:.2f}"
        f" + chunk {report.seconds_chunk:.2f} + embed {report.seconds_embed:.1f}"
        f" + index/save {report.seconds_index_and_save:.2f}"
    )
    for name, size in sorted(report.artifact_sizes.items()):
        print(f"  {name:<16} {size:>12,} bytes")
    return 0


def _cmd_info(args: argparse.Namespace, settings: Settings) -> int:
    manifest = read_manifest(args.index_dir)
    if args.json:
        print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    spec, prov = manifest.spec, manifest.provenance
    print(f"index id:            {manifest.index_id}")
    print(f"manifest schema:     {manifest.schema_version}")
    print(f"created (UTC):       {prov.created_at}")
    print(f"repository:          {spec.repository_name}")
    print(f"fingerprint:         {spec.repository_fingerprint}")
    print(f"source files:        {prov.source_file_count}  (truncated: {prov.ingestion_truncated})")
    print(f"chunks / vectors:    {manifest.chunk_count} / {manifest.vector_count}")
    print(f"dimension:           {spec.embedding_dimension}")
    print(f"embedding model:     {spec.embedding_model_id}")
    runtime = f"{prov.embedding_runtime} {prov.embedding_runtime_version or ''}".strip()
    print(f"embedding runtime:   {runtime}")
    print(f"representation:      {spec.text_style} (v{spec.representation_version})")
    print(
        f"chunking:            {spec.chunking_strategy} v{spec.chunking_strategy_version} "
        f"{spec.chunk_params}"
    )
    print(f"chunk id schema:     {spec.chunk_id_schema}")
    print(
        f"index type:          {spec.index_type} "
        f"(metric: {spec.metric}, FAISS {prov.faiss_version})"
    )
    for name, art in sorted(manifest.artifacts.items()):
        print(f"  {name:<16} {art.size_bytes:>12,} bytes  sha256 {art.sha256[:16]}...")
    manifest_path = Path(args.index_dir) / MANIFEST_FILE
    print(f"  {MANIFEST_FILE:<16} {manifest_path.stat().st_size:>12,} bytes")
    return 0


def _cmd_validate(args: argparse.Namespace, settings: Settings) -> int:
    expected = IndexExpectation()
    if args.repo:
        expected = expectation_for_repository(
            args.repo,
            settings=settings,
            repository_name=args.name,
            ignore_directories=args.ignore_dir,
        )
    try:
        index = VectorIndex.load(args.index_dir, expected=expected)
        check = index.verify_vectors()
    except IndexCompatibilityError as exc:
        print(f"INCOMPATIBLE: {exc}", file=sys.stderr)
        return 1
    except IndexCorruptError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"OK: index {index.index_id} is intact")
    print(
        f"  chunks / vectors:  {len(index.records)} / {index.count} (dimension {check.dimension})"
    )
    print(f"  max |norm - 1|:    {check.max_norm_deviation:.2e}")
    print(
        f"  self-match sample: {check.self_match_sampled} vectors, "
        f"min score {check.min_self_score:.6f}"
    )
    if args.repo:
        print(
            "  compatible with the repository and the current configuration (fingerprint, "
            "chunking, model id, representation)"
        )
    else:
        print("  compatibility with a repository/configuration was NOT checked (use --repo PATH)")
    print("  no embedding model was loaded")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m copilot.vectorstore", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_repo_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--name", help="repository name (default: the directory name)")
        p.add_argument("--ignore-dir", action="append", default=[], help="extra directory to prune")

    build = sub.add_parser("build", help="build and save an index (loads the embedding model)")
    build.add_argument("path", help="repository directory")
    add_repo_options(build)
    build.add_argument("--index-dir", help="where indexes are stored (default: data/indexes)")
    build.add_argument("--text-style", choices=["prefixed", "raw"], help="embedding text style")
    build.add_argument("--force", action="store_true", help="replace an existing identical index")
    build.add_argument(
        "--allow-truncated", action="store_true", help="index even if ingestion hit a limit"
    )

    info = sub.add_parser("info", help="show an index's manifest (no model, no FAISS load)")
    info.add_argument("index_dir", help="index directory (contains manifest.json)")
    info.add_argument("--json", action="store_true", help="print the manifest as JSON")

    validate = sub.add_parser(
        "validate", help="check integrity (and optionally repo compatibility)"
    )
    validate.add_argument("index_dir", help="index directory (contains manifest.json)")
    validate.add_argument("--repo", help="also require the index to match this repository")
    add_repo_options(validate)

    args = parser.parse_args(argv)
    setup_logging()
    settings = get_settings()
    commands = {"build": _cmd_build, "info": _cmd_info, "validate": _cmd_validate}
    try:
        return commands[args.command](args, settings)
    except (VectorStoreError, EmbeddingError, IngestionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
