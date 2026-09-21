"""Plain LLM vs repository-aware RAG: a comparison artifact and a manual-review rubric.

This module *collects* paired answers and *summarises human ratings*. It never scores answer
quality itself (the LLM is not used as a judge) and never declares a winner. Workflow:

1. ``run_comparison`` asks every selected benchmark question through both systems and records
   the answers, the retrieved evidence, the citations, the status and the latency.
2. A person reads the JSON file and fills in ``plain_rating`` / ``rag_rating`` per question,
   following ``RUBRIC`` (checking claims against the repository at the pinned commit).
3. ``summarize_ratings`` aggregates **only rated entries** and says how many there are.

The retrieval hit (does an included source overlap a ground-truth region?) is objective and
computed automatically; everything about answer quality is a human judgement.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from copilot.config import Settings, get_settings
from copilot.evaluation.benchmark import Benchmark, BenchmarkQuestion
from copilot.evaluation.matching import first_hit_rank
from copilot.llm import LLMClient, LLMError
from copilot.rag.models import AnswerStatus
from copilot.rag.service import RagService, SupportsRetrieve, answer_plain
from copilot.security import RepositorySecretRiskError

logger = logging.getLogger(__name__)

COMPARISON_SCHEMA = "answer-comparison/1"

RUBRIC: dict[str, str] = {
    "grounded_correctness": (
        "0-2. Check the answer's repository-specific claims against the source at the pinned "
        "commit. 0 = mostly wrong or unsupported; 1 = partly correct (or correct but vague); "
        "2 = correct and specific."
    ),
    "citation_correctness": (
        "0-2, RAG only (leave null for the plain answer). Open each cited file:lines. 0 = cited "
        "sources do not support the claims; 1 = some do; 2 = all cited sources support what "
        "they are cited for."
    ),
    "hallucinated_claims": (
        "Integer >= 0. Count distinct invented repository facts: files, functions, classes, "
        "parameters or line numbers that do not exist or do not do what is claimed. General "
        "programming knowledge is not counted."
    ),
    "completeness": (
        "0-2. 0 = misses what the question asks; 1 = answers part; 2 = covers what the "
        "reference regions show."
    ),
    "appropriate_abstention": (
        "true/false/null. true = the system said it could not answer AND the evidence really "
        "was missing (or the question is unanswerable); false = it abstained though the "
        "evidence was available, or answered confidently without support. null = not applicable."
    ),
}


class ManualRating(BaseModel):
    """One reviewer's rubric scores for one answer. ``None`` = not rated / not applicable."""

    model_config = ConfigDict(extra="forbid")

    grounded_correctness: int | None = Field(default=None, ge=0, le=2)
    citation_correctness: int | None = Field(default=None, ge=0, le=2)
    hallucinated_claims: int | None = Field(default=None, ge=0)
    completeness: int | None = Field(default=None, ge=0, le=2)
    appropriate_abstention: bool | None = None
    notes: str = ""

    @property
    def is_rated(self) -> bool:
        return any(
            v is not None
            for v in (
                self.grounded_correctness,
                self.citation_correctness,
                self.hallucinated_claims,
                self.completeness,
                self.appropriate_abstention,
            )
        )


class SourceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_number: int
    file_path: str
    start_line: int
    end_line: int
    chunk_id: str
    score: float
    rank: int


class PlainRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str | None = None
    error: str | None = None  # a safe, fixed-message error class name + message
    latency_seconds: float | None = None


class RagRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["answered", "insufficient_evidence", "error"]
    answer: str | None = None
    insufficient_reason: str | None = None
    error: str | None = None
    sources: tuple[SourceRecord, ...] = ()
    cited_source_numbers: tuple[int, ...] = ()
    unknown_source_numbers: tuple[int, ...] = ()
    retrieved_count: int = 0
    first_relevant_source: int | None = None  # objective: first included source in a region
    retrieval_seconds: float | None = None
    latency_seconds: float | None = None  # retrieval + generation


class ComparisonEntry(BaseModel):
    """One benchmark question through both systems, plus room for the human ratings."""

    question_id: str
    question: str
    expected_regions: tuple[str, ...]  # "file:start-end", from the benchmark (reviewer aid)
    plain: PlainRecord
    rag: RagRecord
    plain_rating: ManualRating | None = None
    rag_rating: ManualRating | None = None


class ComparisonRun(BaseModel):
    """The whole artifact. Contains model answers and repository-relative paths only."""

    schema_version: str = COMPARISON_SCHEMA
    benchmark_name: str | None = None
    repository_name: str | None = None
    index_id: str | None = None
    llm_provider: str
    llm_model: str
    plain_prompt_version: str
    rag_prompt_version: str
    top_k: int
    temperature: float
    max_output_tokens: int
    rubric: dict[str, str] = Field(default_factory=lambda: dict(RUBRIC))
    entries: list[ComparisonEntry]


def select_questions(
    benchmark: Benchmark, ids: Sequence[str] = (), limit: int | None = None
) -> list[BenchmarkQuestion]:
    """Questions by explicit id (in the order given), else the first ``limit`` (deterministic)."""
    by_id = {q.id: q for q in benchmark.questions}
    if ids:
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise ValueError(f"unknown question id(s): {', '.join(unknown)}")
        return [by_id[i] for i in ids]
    questions = list(benchmark.questions)
    return questions[:limit] if limit is not None else questions


def _safe_error(exc: Exception) -> str:
    # LLM errors and gate errors carry fixed, secret-free messages by construction.
    return f"{type(exc).__name__}: {exc}"


def _ask_plain(question: str, llm: LLMClient, settings: Settings) -> PlainRecord:
    started = time.perf_counter()
    try:
        answer = answer_plain(question, llm, settings=settings)
    except (LLMError, RepositorySecretRiskError) as exc:
        return PlainRecord(error=_safe_error(exc), latency_seconds=_elapsed(started))
    return PlainRecord(answer=answer.answer, latency_seconds=_elapsed(started))


def _ask_rag(question: BenchmarkQuestion, service: RagService, top_k: int) -> RagRecord:
    started = time.perf_counter()
    try:
        answer = service.answer(question.question, top_k)
    except (LLMError, RepositorySecretRiskError) as exc:
        return RagRecord(status="error", error=_safe_error(exc), latency_seconds=_elapsed(started))
    sources = tuple(
        SourceRecord(
            source_number=s.source_number,
            file_path=s.file_path,
            start_line=s.start_line,
            end_line=s.end_line,
            chunk_id=s.chunk_id,
            score=s.score,
            rank=s.rank,
        )
        for s in answer.sources
    )
    return RagRecord(
        status=("answered" if answer.status is AnswerStatus.ANSWERED else "insufficient_evidence"),
        answer=answer.answer,
        insufficient_reason=answer.insufficient_reason,
        sources=sources,
        cited_source_numbers=answer.cited_source_numbers,
        unknown_source_numbers=answer.unknown_source_numbers,
        retrieved_count=answer.retrieved_count,
        first_relevant_source=first_hit_rank(sources, question.relevant),
        retrieval_seconds=answer.retrieval_seconds,
        latency_seconds=_elapsed(started),
    )


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 3)


def run_comparison(
    questions: Sequence[BenchmarkQuestion],
    retriever: SupportsRetrieve,
    llm: LLMClient,
    *,
    top_k: int | None = None,
    settings: Settings | None = None,
    benchmark_name: str | None = None,
    repository_name: str | None = None,
    index_id: str | None = None,
    delay_seconds: float = 0.0,
    progress: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ComparisonRun:
    """Ask each question through the plain baseline and through RAG; rate nothing.

    Both paths go through ``GuardedLLMClient`` (the services wrap ``llm``). A gate refusal or a
    provider error on one question is recorded and the run continues with the next question.
    """
    from copilot.rag.prompts import PLAIN_PROMPT_VERSION, RAG_PROMPT_VERSION

    settings = settings or get_settings()
    k = top_k if top_k is not None else settings.retrieval_top_k
    service = RagService(retriever, llm, settings=settings)
    entries: list[ComparisonEntry] = []
    for number, question in enumerate(questions, start=1):
        if progress:
            progress(f"[{number}/{len(questions)}] {question.id}")
        plain = _ask_plain(question.question, llm, settings)
        if delay_seconds:
            sleep(delay_seconds)
        rag = _ask_rag(question, service, k)
        if delay_seconds:
            sleep(delay_seconds)
        entries.append(
            ComparisonEntry(
                question_id=question.id,
                question=question.question,
                expected_regions=tuple(
                    f"{r.file_path}:{r.start_line}-{r.end_line}" for r in question.relevant
                ),
                plain=plain,
                rag=rag,
            )
        )
    return ComparisonRun(
        benchmark_name=benchmark_name,
        repository_name=repository_name,
        index_id=index_id,
        llm_provider=llm.provider,
        llm_model=llm.model,
        plain_prompt_version=PLAIN_PROMPT_VERSION,
        rag_prompt_version=RAG_PROMPT_VERSION,
        top_k=k,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        entries=entries,
    )


def dump_comparison(run: ComparisonRun) -> str:
    """Stable, human-editable JSON (UTF-8, sorted by question order, trailing newline)."""
    return json.dumps(run.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"


def write_comparison(run: ComparisonRun, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_comparison(run), encoding="utf-8", newline="\n")


def load_comparison(path: str | Path) -> ComparisonRun:
    """Read a comparison file (possibly with human ratings filled in)."""
    try:
        return ComparisonRun.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"cannot read comparison file: {type(exc).__name__}") from exc


class RatingSummary(BaseModel):
    """Aggregates over the rated entries of ONE system. Means are None when nothing was rated."""

    n_total: int
    n_rated: int
    mean_grounded_correctness: float | None = None
    mean_citation_correctness: float | None = None
    mean_completeness: float | None = None
    total_hallucinated_claims: int | None = None
    mean_hallucinated_claims: float | None = None
    abstention_appropriate: int = 0
    abstention_inappropriate: int = 0
    mean_latency_seconds: float | None = None


class ComparisonSummary(BaseModel):
    n_questions: int
    plain: RatingSummary
    rag: RatingSummary
    rag_status_counts: dict[str, int]
    rag_first_relevant_source_hits: int  # questions whose included sources overlap a region
    note: str = (
        "Ratings are human judgements on a small, self-authored benchmark; no significance "
        "is claimed and no system is declared better."
    )


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _summarize_system(
    ratings: list[ManualRating | None], latencies: list[float | None]
) -> RatingSummary:
    rated = [r for r in ratings if r is not None and r.is_rated]
    halluc = [r.hallucinated_claims for r in rated if r.hallucinated_claims is not None]
    seen = [x for x in latencies if x is not None]
    return RatingSummary(
        n_total=len(ratings),
        n_rated=len(rated),
        mean_grounded_correctness=_mean(
            [r.grounded_correctness for r in rated if r.grounded_correctness is not None]
        ),
        mean_citation_correctness=_mean(
            [r.citation_correctness for r in rated if r.citation_correctness is not None]
        ),
        mean_completeness=_mean([r.completeness for r in rated if r.completeness is not None]),
        total_hallucinated_claims=sum(halluc) if halluc else None,
        mean_hallucinated_claims=_mean([float(h) for h in halluc]),
        abstention_appropriate=sum(1 for r in rated if r.appropriate_abstention is True),
        abstention_inappropriate=sum(1 for r in rated if r.appropriate_abstention is False),
        mean_latency_seconds=_mean(seen),
    )


def summarize_ratings(run: ComparisonRun) -> ComparisonSummary:
    """Aggregate the human ratings already present in ``run`` (unrated entries are excluded)."""
    entries = run.entries
    statuses: dict[str, int] = {}
    for e in entries:
        statuses[e.rag.status] = statuses.get(e.rag.status, 0) + 1
    return ComparisonSummary(
        n_questions=len(entries),
        plain=_summarize_system(
            [e.plain_rating for e in entries], [e.plain.latency_seconds for e in entries]
        ),
        rag=_summarize_system(
            [e.rag_rating for e in entries], [e.rag.latency_seconds for e in entries]
        ),
        rag_status_counts=dict(sorted(statuses.items())),
        rag_first_relevant_source_hits=sum(
            1 for e in entries if e.rag.first_relevant_source is not None
        ),
    )


def format_summary(summary: ComparisonSummary) -> str:
    def row(label: str, s: RatingSummary) -> str:
        def f(value: float | int | None) -> str:
            return "-" if value is None else str(value)

        return (
            f"{label:<6} rated {s.n_rated}/{s.n_total}"
            f"  correctness {f(s.mean_grounded_correctness)}"
            f"  citations {f(s.mean_citation_correctness)}  completeness {f(s.mean_completeness)}"
            f"  hallucinated claims {f(s.total_hallucinated_claims)}"
            f"  abstention ok/bad {s.abstention_appropriate}/{s.abstention_inappropriate}"
            f"  mean latency {f(s.mean_latency_seconds)}s"
        )

    return "\n".join(
        [
            f"{summary.n_questions} questions",
            row("plain", summary.plain),
            row("rag", summary.rag),
            f"rag statuses: {summary.rag_status_counts}",
            f"rag: an included source overlaps the expected region in "
            f"{summary.rag_first_relevant_source_hits}/{summary.n_questions} questions "
            "(objective retrieval check, not answer quality)",
            summary.note,
        ]
    )
