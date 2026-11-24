"""Terminal front-end of the agentic workflow (separate from ``python -m copilot.rag``).

    python -m copilot.agents ask INDEX_DIR --repo PATH "question" [--max-rounds N] [--trace]
    python -m copilot.agents ask INDEX_DIR --repo PATH "question" --fake-llm   # offline, no model
    python -m copilot.agents graph                                             # Mermaid diagram

Without ``--fake-llm`` the question and the retrieved chunks go to the hosted model configured in
``.env``, after the same secret scan as ``copilot.rag`` (the guarded client cannot be bypassed).
``--fake-llm`` replaces only the model with a deterministic stand-in that makes no network call;
retrieval still uses the real local index and embedding model.

Exit status: 0 answered or abstained, 1 refused by the secret gate, 2 any other error.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from copilot.agents.state import MAX_ROUNDS_LIMIT, AgentConfig
from copilot.agents.workflow import AgentResult, AgentWorkflow
from copilot.chunking.errors import ChunkingError
from copilot.config import get_settings, setup_logging
from copilot.embeddings.errors import EmbeddingError
from copilot.embeddings.factory import create_embedder
from copilot.ingestion.errors import IngestionError
from copilot.llm import FakeLLMClient, LLMError, LLMRequest, create_llm_client
from copilot.rag.models import AnswerStatus
from copilot.retrieval import Retriever
from copilot.retrieval.errors import RetrievalError
from copilot.security import SecurityError
from copilot.vectorstore.errors import VectorStoreError


def positive_rounds(value: str) -> int:
    """argparse type: an integer in 1..MAX_ROUNDS_LIMIT."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if not 1 <= number <= MAX_ROUNDS_LIMIT:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_ROUNDS_LIMIT}, got {number}")
    return number


def fake_reply(request: LLMRequest) -> str:
    """Deterministic offline stand-in for the model: reports what it was given, nothing more."""
    sources = request.user_prompt.count("BEGIN SOURCE ")
    return (
        f"[offline fake model: no real model was called] I received {sources} source(s). "
        "This text only demonstrates the workflow; it is not an answer about the repository."
    )


def format_result(result: AgentResult, *, show_trace: bool = False) -> str:
    """ANSWER, SOURCES, WORKFLOW (and TRACE): metadata only, never source text."""
    lines = ["ANSWER"]
    answer = result.answer
    if result.failure is not None:
        lines.append(f"no answer: {result.failure.stage} failed ({result.failure.error_type}).")
        lines.append(result.failure.message)
    elif answer is None or answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
        reason = result.abstention_reason
        lines.append(f"insufficient repository evidence ({reason}); the model was not called.")
    else:
        lines.append(answer.answer or "")
    lines += ["", "SOURCES"]
    if result.citations:
        by_source = {p.source_number: p for p in result.provenance}
        for c in result.citations:
            p = by_source.get(c.source_number)
            where = (
                f"round {p.retrieval_round} rank {p.retrieval_rank}, "
                f"that round's score {p.score:.4f}"
                if p is not None
                else f"rank {c.rank}, score {c.score:.4f}"
            )
            lines.append(f"[{c.label}] {c.location}  ({where})")
    else:
        lines.append("(none)")
    for warning in result.warnings:
        lines += ["", f"warning: {warning}"]

    lines += ["", "WORKFLOW"]
    lines.append(f"intent: {result.decision.intent.value} (rule {result.decision.rule_id})")
    lines.append(f"retrieval rounds: {result.rounds} of at most {result.max_rounds}")
    lines.append(f"evidence chunks: {result.evidence_count}")
    if result.assessment is not None:
        cov = result.assessment.term_coverage
        lines.append(
            f"assessment: {result.assessment.outcome.value} ({result.assessment.reason}"
            + (f", term coverage {cov:.2f}" if cov is not None else "")
            + ")"
        )
    lines.append(f"termination: {result.termination.value}")
    lines.append(f"evidence: {result.evidence_status}")
    if answer is not None and answer.sources:
        lines.append(
            f"context: {len(answer.sources)} chunk(s), ~{answer.context_estimated_tokens} "
            "estimated tokens"
        )
    if answer is not None and answer.generation is not None:
        g = answer.generation
        lines.append(f"model: {g.provider}/{g.model}  finish: {g.finish_reason}")
    if show_trace:
        lines += ["", "TRACE"]
        for event in result.trace:
            fields = event.model_dump(exclude_none=True, exclude={"node"})
            detail = " ".join(f"{k}={v}" for k, v in fields.items())
            lines.append(f"{event.node}: {detail}")
    return "\n".join(lines)


def _cmd_graph() -> int:
    class _NoRetriever:
        def retrieve(self, query: str, top_k: int | None = None) -> list:
            return []

    print(AgentWorkflow(_NoRetriever(), FakeLLMClient()).mermaid())
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from copilot.retrieval import validate_query

    validate_query(args.question)  # bad input fails before the embedding model is loaded
    settings = get_settings()
    config = AgentConfig.from_settings(settings, max_rounds=args.max_rounds)
    if args.fake_llm:
        llm = FakeLLMClient(fake_reply, model="offline-fake")
        print(
            "note: --fake-llm: no model is called; the answer text is a fixed stand-in.",
            file=sys.stderr,
        )
    else:
        llm = create_llm_client(settings)
        print(
            f"note: the question and the retrieved code chunks are sent to {llm.provider}/"
            f"{llm.model} after the secret scan.",
            file=sys.stderr,
        )
    retriever = Retriever.open(
        args.index_dir,
        args.repo,
        embedder=create_embedder(settings),
        settings=settings,
        ignore_directories=args.ignore_dir,
    )
    workflow = AgentWorkflow(retriever, llm, settings=settings, config=config)
    result = workflow.run(args.question)
    print(format_result(result, show_trace=args.trace))
    if result.failure is not None:
        if result.failure.blocked_by_secret_gate:
            print("refused: nothing was sent to the model.", file=sys.stderr)
            return 1
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m copilot.agents", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="agentic RAG answer with citations and a workflow summary")
    ask.add_argument("index_dir", help="index directory (data/indexes/<index id>)")
    ask.add_argument("--repo", required=True, help="the repository that was indexed")
    ask.add_argument("question", help="natural-language question (quote it)")
    ask.add_argument(
        "--max-rounds",
        type=positive_rounds,
        default=None,
        help=f"retrieval rounds including the first (1-{MAX_ROUNDS_LIMIT}; default from settings)",
    )
    ask.add_argument("--trace", action="store_true", help="print the structured execution trace")
    ask.add_argument(
        "--fake-llm", action="store_true", help="offline: replace only the model with a stand-in"
    )
    ask.add_argument(
        "--ignore-dir",
        action="append",
        default=[],
        help="extra directory pruned at build time (repeat the build's --ignore-dir flags)",
    )
    sub.add_parser("graph", help="print the workflow graph as a Mermaid diagram")

    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # a Windows console may not encode every character
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    setup_logging()
    try:
        return _cmd_graph() if args.command == "graph" else _cmd_ask(args)
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
