"""Terminal front-end of the plain baseline and the RAG service.

    python -m copilot.rag ask   INDEX_DIR --repo PATH "question"   # repository-aware RAG
    python -m copilot.rag plain "question"                         # plain LLM, no repository

Both commands send text to the hosted model configured in ``.env`` (``COPILOT_LLM_MODEL`` and
``GEMINI_API_KEY``). Every request passes the content-secret gate first; a refusal exits with
status 1 and nothing is sent. There is no flag that disables the gate.

Exit status: 0 answered (or ``insufficient_evidence``), 1 refused by the secret gate, 2 any
other error.
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
from copilot.llm import LLMError, create_llm_client
from copilot.rag.models import AnswerStatus, RagAnswer
from copilot.rag.service import answer_plain, answer_with_rag
from copilot.retrieval.errors import RetrievalError
from copilot.retrieval.retriever import MAX_TOP_K, validate_query
from copilot.security import RepositorySecretRiskError, SecurityError
from copilot.vectorstore.errors import VectorStoreError


def format_rag_answer(answer: RagAnswer, *, show_context: bool = False) -> str:
    """ANSWER, then SOURCES (``file:start-end``). Metadata only; never source text."""
    lines: list[str] = []
    if answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
        lines += [
            "ANSWER",
            f"insufficient repository evidence ({answer.insufficient_reason}); "
            "the model was not called.",
        ]
    else:
        lines += ["ANSWER", answer.answer or ""]
    lines += ["", "SOURCES"]
    if answer.sources:
        for source in answer.sources:
            lines.append(
                f"[{source.label}] {source.location}  "
                f"(rank {source.rank}, score {source.score:.4f})"
            )
    else:
        lines.append("(none)")
    if answer.unknown_source_numbers:
        numbers = ", ".join(str(n) for n in answer.unknown_source_numbers)
        lines += [
            "",
            f"warning: the answer mentions source numbers that were not supplied: {numbers}",
        ]
    if show_context:
        lines += ["", "CONTEXT (metadata only)"]
        lines.append(f"retrieved chunks: {answer.retrieved_count}")
        lines.append(f"chunks in context: {len(answer.sources)}")
        lines.append(f"context estimated tokens: {answer.context_estimated_tokens}")
        for dropped in answer.dropped_evidence:
            lines.append(
                f"dropped ({dropped.reason}): rank {dropped.rank} "
                f"{dropped.file_path}:{dropped.start_line}-{dropped.end_line}"
            )
        lines.append(f"prompt version: {answer.prompt_version}")
        if answer.generation:
            g = answer.generation
            lines.append(f"model: {g.provider}/{g.model}  finish: {g.finish_reason}")
            lines.append(
                f"tokens: prompt {g.prompt_tokens} completion {g.completion_tokens} "
                f"total {g.total_tokens}"
            )
        lines.append(
            f"retrieval {answer.retrieval_seconds}s  generation {answer.generation_seconds}s"
        )
    return "\n".join(lines)


def _cmd_ask(args: argparse.Namespace) -> int:
    validate_query(args.question)  # bad input fails before the embedding model is loaded
    settings = get_settings()
    llm = create_llm_client(settings)  # fails early when the model / key is not configured
    embedder = create_embedder(settings)
    print(
        f"note: the question and the retrieved code chunks are sent to {llm.provider}/{llm.model} "
        "after the secret scan.",
        file=sys.stderr,
    )
    answer = answer_with_rag(
        args.question,
        args.repo,
        args.index_dir,
        args.top_k,
        llm=llm,
        embedder=embedder,
        settings=settings,
        ignore_directories=args.ignore_dir,
    )
    print(format_rag_answer(answer, show_context=args.show_context))
    return 0


def _cmd_plain(args: argparse.Namespace) -> int:
    validate_query(args.question)
    settings = get_settings()
    llm = create_llm_client(settings)
    print(
        f"note: the question is sent to {llm.provider}/{llm.model}; there is NO repository "
        "context, so the answer is not grounded.",
        file=sys.stderr,
    )
    answer = answer_plain(args.question, llm, settings=settings)
    print("ANSWER (plain LLM, no repository evidence)")
    print(answer.answer)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m copilot.rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="repository-aware RAG answer with source citations")
    ask.add_argument("index_dir", help="index directory (data/indexes/<index id>)")
    ask.add_argument("--repo", required=True, help="the repository that was indexed")
    ask.add_argument("question", help="natural-language question (quote it)")
    ask.add_argument(
        "--top-k", type=int, default=get_settings().retrieval_top_k, help=f"chunks (1-{MAX_TOP_K})"
    )
    ask.add_argument(
        "--ignore-dir",
        action="append",
        default=[],
        help="extra directory pruned at build time (repeat the build's --ignore-dir flags)",
    )
    ask.add_argument(
        "--show-context",
        action="store_true",
        help="also print context statistics (metadata only, never source text)",
    )

    plain = sub.add_parser("plain", help="plain LLM baseline: no repository context")
    plain.add_argument("question", help="natural-language question (quote it)")

    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # a Windows console may not encode every character
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    setup_logging()
    try:
        return _cmd_ask(args) if args.command == "ask" else _cmd_plain(args)
    except RepositorySecretRiskError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        print("nothing was sent to the model.", file=sys.stderr)
        return 1
    except (
        SecurityError,
        LLMError,
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
