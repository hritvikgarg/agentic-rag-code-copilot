"""Resumable, multi-repository execution of the plain-vs-RAG answer experiment.

This extends the Milestone 7 comparison machinery (side states, resume checks, atomic checkpoints,
bounded retry, the HTTP-429 circuit breaker) to several repositories and adds identity-keyed
response caching. Nothing here talks to a provider by itself: the caller passes an ``LLMClient``
(a real one in Milestone 11B, a ``FakeLLMClient`` in tests) and a retriever.

**One file per repository** (``generation-run/1``). Every requested question is always
present, so an interrupted file is a valid resume input. Each side of each question is in
exactly one state: ``successful`` (an answer or a RAG system abstention is recorded),
``failed`` (a call ended in an error) or ``not_attempted``. Resume reuses ``successful`` sides
byte-for-byte and retries the rest;
it refuses - before any call - if the protocol hash, benchmark digest, repository, index, model,
settings, prompt versions or question set differ (``RunMismatchError``).

**Generation cache.** ``MeteredLLM`` sits *under* the secret gate and keys each response by
``{provider, model, temperature, max_output_tokens, prompt_version, SHA-256 of the exact outbound
text}``. The outbound text contains the question, the system prompt and, for RAG, the retrieved
evidence, so a different model, setting, prompt or evidence can never be served another run's
answer, and two generation models can never be mixed. Only successful responses are cached. A cache
hit is not a provider call: it neither extends nor resets the 429 streak, and its recorded side is
marked ``origin="cache"`` so latency analysis can exclude it.

**Test artifacts.** A run made with the fake provider is stamped ``test_artifact=true``; analysis
labels it and never presents it as an experimental result.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from copilot.config import Settings
from copilot.evaluation.comparison import SourceRecord
from copilot.evaluation.generation.protocol import PrimaryProtocol, RepositorySpec, protocol_hash
from copilot.evaluation.generation.schema import GenerationQuestion
from copilot.evaluation.matching import first_hit_rank
from copilot.llm import LLMClient, LLMError, LLMRateLimitError
from copilot.llm.models import LLMRequest, LLMResponse
from copilot.llm.retry import DEFAULT_RETRY_POLICY, RetryPolicy, call_with_retry
from copilot.rag.models import AnswerStatus, GenerationInfo, PlainAnswer, RagAnswer
from copilot.rag.service import RagService, SupportsRetrieve, answer_plain
from copilot.security import RepositorySecretRiskError, assert_safe_for_external_llm
from copilot.utils.atomic_file import write_text_atomically

logger = logging.getLogger(__name__)

RUN_SCHEMA = "generation-run/1"
Arm = Literal["plain", "rag"]
State = Literal["successful", "failed", "not_attempted"]
Origin = Literal["generated", "cache", "historical"]
DEFAULT_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES = 3


class RunMismatchError(ValueError):
    """The existing file does not describe this experiment; refuse to mix two experiments."""


# --------------------------------------------------------------------------- records


class SideRecord(BaseModel):
    """One arm's response to one question."""

    model_config = ConfigDict(frozen=True)

    arm: Arm
    state: State
    prompt_version: str
    status: Literal["answered", "insufficient_evidence"] | None = None
    answer: str | None = None
    insufficient_reason: str | None = None
    error: str | None = None
    origin: Origin = "generated"
    origin_file: str | None = None
    latency_seconds: float | None = None  # wall clock of this attempt, generated sides only
    input_sha256: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    sources: tuple[SourceRecord, ...] = ()
    cited_source_numbers: tuple[int, ...] = ()
    unknown_source_numbers: tuple[int, ...] = ()
    retrieved_count: int = 0
    retrieval_seconds: float | None = None
    first_relevant_source: int | None = None

    @property
    def has_text(self) -> bool:
        return self.state == "successful" and bool(self.answer and self.answer.strip())

    @property
    def system_insufficient(self) -> bool:
        """RAG answered "insufficient evidence" without calling the model."""
        return self.state == "successful" and self.status == "insufficient_evidence"


def placeholder(arm: Arm, prompt_version: str) -> SideRecord:
    return SideRecord(arm=arm, state="not_attempted", prompt_version=prompt_version)


class GenerationEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    question_id: str
    question: str
    answerable: bool
    plain: SideRecord
    rag: SideRecord


class GenerationRun(BaseModel):
    """The whole per-repository artifact: answers, evidence and run provenance only."""

    schema_version: str = RUN_SCHEMA
    repository_id: str
    protocol_hash: str
    benchmark_digest: str
    index_id: str | None
    llm_provider: str
    llm_model: str
    temperature: float
    max_output_tokens: int
    top_k: int
    plain_prompt_version: str
    rag_prompt_version: str
    test_artifact: bool = False
    stop_reason: str | None = None
    entries: list[GenerationEntry]

    def side_counts(self) -> dict[str, dict[str, int]]:
        counts = {a: {"successful": 0, "failed": 0, "not_attempted": 0} for a in ("plain", "rag")}
        for entry in self.entries:
            counts["plain"][entry.plain.state] += 1
            counts["rag"][entry.rag.state] += 1
        return counts


def dump_run(run: GenerationRun) -> str:
    return run.model_dump_json(indent=2) + "\n"


def write_run(run: GenerationRun, path: str | Path) -> None:
    write_text_atomically(Path(path), dump_run(run))


def load_run(path: str | Path) -> GenerationRun:
    try:
        return GenerationRun.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValidationError, UnicodeDecodeError) as exc:
        raise RunMismatchError(f"cannot read run file: {type(exc).__name__}") from exc


# --------------------------------------------------------------------------- the generation cache


def response_cache_key(provider: str, model: str, request: LLMRequest) -> tuple[str, str]:
    """``(cache key, SHA-256 of the exact outbound text)`` for one generation request."""
    input_sha = hashlib.sha256(request.outbound_text().encode("utf-8")).hexdigest()
    payload = {
        "provider": provider,
        "model": model,
        "temperature": request.temperature,
        "max_output_tokens": request.max_output_tokens,
        "prompt_version": request.prompt_version,
        "input_sha256": input_sha,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), input_sha


class ResponseCache:
    """Content-addressed store of successful generation responses (one JSON file per key)."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return LLMResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, OSError, UnicodeDecodeError):
            logger.warning("ignoring unreadable generation cache entry %s", path.name)
            return None

    def put(self, key: str, response: LLMResponse) -> None:
        write_text_atomically(self._path(key), response.model_dump_json(indent=2) + "\n")


class MeteredLLM:
    """An ``LLMClient`` wrapper: counts provider calls, records the input hash and serves the cache.

    It sits *under* the secret gate (the services wrap whatever they are given) and additionally
    scans every request itself before any cache lookup or provider call, so the policy holds even
    if a caller forgets the guard: a cache can never bypass the gate.
    """

    def __init__(
        self, inner: LLMClient, cache: ResponseCache | None = None, *, allow_calls: bool = True
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._allow_calls = allow_calls
        self.provider = inner.provider
        self.model = inner.model
        self.provider_calls = 0
        self.cache_hits = 0
        self.last_input_sha: str | None = None
        self.last_from_cache = False

    def reset_last(self) -> None:
        self.last_input_sha = None
        self.last_from_cache = False

    def generate(self, request: LLMRequest) -> LLMResponse:
        assert_safe_for_external_llm(request.scan_targets())  # raises RepositorySecretRiskError
        key, self.last_input_sha = response_cache_key(self.provider, self.model, request)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                self.last_from_cache = True
                return cached
        self.last_from_cache = False
        if not self._allow_calls:
            raise LLMError("provider calls are disabled for this run")
        self.provider_calls += 1
        response = self._inner.generate(request)
        if self._cache is not None:
            self._cache.put(key, response)
        return response


# --------------------------------------------------------------------------- the circuit breaker


@dataclass
class CircuitBreaker:
    """Stops all work after ``limit`` consecutive hosted sides that ended in HTTP 429."""

    limit: int | None = DEFAULT_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES
    streak: int = 0
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be >= 1 (or None to disable)")

    @property
    def tripped(self) -> bool:
        return self.reason is not None

    def record(self, failure: Exception | None) -> None:
        self.streak = self.streak + 1 if isinstance(failure, LLMRateLimitError) else 0
        if self.limit is not None and self.streak >= self.limit:
            self.reason = (
                f"stopped after {self.streak} consecutive hosted request(s) ended in HTTP 429 "
                "(rate or quota limit) after exhausting their retries. The provider did not say "
                "which limit was hit. Completed answers are saved; resume later."
            )


# --------------------------------------------------------------------------- contexts and checks


@dataclass(frozen=True)
class RepositoryContext:
    spec: RepositorySpec
    questions: tuple[GenerationQuestion, ...]
    benchmark_digest: str
    retriever: SupportsRetrieve | None = None
    index_id: str | None = None


def settings_for(protocol: PrimaryProtocol, base: Settings) -> Settings:
    """``base`` with the protocol's generation settings (never read from the environment)."""
    return base.model_copy(
        update={
            "llm_temperature": protocol.temperature,
            "llm_max_output_tokens": protocol.max_output_tokens,
            "retrieval_top_k": protocol.top_k,
        }
    )


def is_test_provider(llm: LLMClient) -> bool:
    return llm.provider == "fake"


def check_resume_compatible(
    existing: GenerationRun,
    ctx: RepositoryContext,
    protocol: PrimaryProtocol,
    llm: LLMClient,
) -> None:
    """Fail closed (before any call) unless resuming means "the same experiment"."""
    mismatches: list[str] = []

    def check(label: str, old: object, new: object) -> None:
        if old != new:
            mismatches.append(f"{label} (existing={old!r}, requested={new!r})")

    check("schema_version", existing.schema_version, RUN_SCHEMA)
    check("protocol_hash", existing.protocol_hash, protocol_hash(protocol))
    check("repository_id", existing.repository_id, ctx.spec.repository_id)
    check("benchmark_digest", existing.benchmark_digest, ctx.benchmark_digest)
    check("index_id", existing.index_id, ctx.index_id)
    check("llm_provider", existing.llm_provider, llm.provider)
    check("llm_model", existing.llm_model, llm.model)
    check("temperature", existing.temperature, protocol.temperature)
    check("max_output_tokens", existing.max_output_tokens, protocol.max_output_tokens)
    check("top_k", existing.top_k, protocol.top_k)
    check("plain_prompt_version", existing.plain_prompt_version, protocol.plain_prompt_version)
    check("rag_prompt_version", existing.rag_prompt_version, protocol.rag_prompt_version)
    check("test_artifact", existing.test_artifact, is_test_provider(llm))
    old_ids = {e.question_id for e in existing.entries}
    new_ids = {q.id for q in ctx.questions}
    if old_ids != new_ids:
        mismatches.append(f"question_id set differs ({sorted(old_ids ^ new_ids)[:5]})")
    else:
        by_id = {e.question_id: e for e in existing.entries}
        changed = sorted(q.id for q in ctx.questions if by_id[q.id].question != q.question)
        if changed:
            mismatches.append(f"question text changed for {changed[:5]}")
    if mismatches:
        raise RunMismatchError(
            "cannot resume: the existing run file does not match this run - "
            + "; ".join(mismatches)
            + ". Use a different output directory for a new experiment."
        )


# --------------------------------------------------------------------------- one side


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def _usage(info: GenerationInfo | None) -> dict[str, object]:
    if info is None:
        return {}
    return {
        "finish_reason": info.finish_reason,
        "prompt_tokens": info.prompt_tokens,
        "completion_tokens": info.completion_tokens,
        "reasoning_tokens": info.reasoning_tokens,
        "total_tokens": info.total_tokens,
    }


def _origin(llm: MeteredLLM) -> Origin:
    return "cache" if llm.last_from_cache else "generated"


def _plain_record(
    answer: PlainAnswer, llm: MeteredLLM, protocol: PrimaryProtocol, seconds: float
) -> SideRecord:
    return SideRecord(
        arm="plain", state="successful", prompt_version=answer.prompt_version, status="answered",
        answer=answer.answer, origin=_origin(llm), latency_seconds=seconds,
        input_sha256=llm.last_input_sha, **_usage(answer.generation),
    )  # fmt: skip


def _rag_record(
    answer: RagAnswer, question: GenerationQuestion, llm: MeteredLLM, seconds: float
) -> SideRecord:
    sources = tuple(
        SourceRecord(
            source_number=s.source_number,
            file_path=s.file_path,
            start_line=s.start_line,
            end_line=s.end_line,
            chunk_id=s.chunk_id,
            score=s.score,
            rank=s.rank,
        )  # fmt: skip
        for s in answer.sources
    )
    answered = answer.status is AnswerStatus.ANSWERED
    return SideRecord(
        arm="rag", state="successful", prompt_version=answer.prompt_version,
        status="answered" if answered else "insufficient_evidence", answer=answer.answer,
        insufficient_reason=answer.insufficient_reason,
        origin=_origin(llm) if answered else "generated", latency_seconds=seconds,
        input_sha256=llm.last_input_sha if answered else None, sources=sources,
        cited_source_numbers=answer.cited_source_numbers,
        unknown_source_numbers=answer.unknown_source_numbers,
        retrieved_count=answer.retrieved_count, retrieval_seconds=answer.retrieval_seconds,
        first_relevant_source=first_hit_rank(sources, question.relevant_regions),
        **_usage(answer.generation),
    )  # fmt: skip


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 3)


def ask_plain(
    question: GenerationQuestion,
    llm: MeteredLLM,
    protocol: PrimaryProtocol,
    settings: Settings,
    *,
    retry_policy: RetryPolicy,
    retry_sleep: Callable[[float], None],
) -> tuple[SideRecord, Exception | None, bool]:
    """``(record, ending exception, was_a_provider_call)``."""
    from copilot.rag import prompts

    system = {
        "plain-v1": prompts.PLAIN_SYSTEM_PROMPT,
        "plain-v2": prompts.PLAIN_SYSTEM_PROMPT_V2,
    }[protocol.plain_prompt_version]
    llm.reset_last()
    calls_before = llm.provider_calls
    started = time.perf_counter()
    try:
        answer = call_with_retry(
            lambda: answer_plain(
                question.question, llm, settings=settings, system_prompt=system,
                prompt_version=protocol.plain_prompt_version,
            ),
            policy=retry_policy,
            sleep=retry_sleep,
        )  # fmt: skip
    except (LLMError, RepositorySecretRiskError) as exc:
        record = SideRecord(
            arm="plain", state="failed", prompt_version=protocol.plain_prompt_version,
            error=_safe_error(exc), latency_seconds=_elapsed(started),
            input_sha256=llm.last_input_sha,
        )  # fmt: skip
        return record, exc, llm.provider_calls > calls_before
    return (
        _plain_record(answer, llm, protocol, _elapsed(started)),
        None,
        llm.provider_calls > calls_before,
    )


def ask_rag(
    question: GenerationQuestion,
    service: RagService,
    llm: MeteredLLM,
    protocol: PrimaryProtocol,
    *,
    retry_policy: RetryPolicy,
    retry_sleep: Callable[[float], None],
) -> tuple[SideRecord, Exception | None, bool]:
    llm.reset_last()
    calls_before = llm.provider_calls
    started = time.perf_counter()
    try:
        answer = call_with_retry(
            lambda: service.answer(question.question, protocol.top_k),
            policy=retry_policy,
            sleep=retry_sleep,
        )
    except (LLMError, RepositorySecretRiskError) as exc:
        record = SideRecord(
            arm="rag", state="failed", prompt_version=protocol.rag_prompt_version,
            error=_safe_error(exc), latency_seconds=_elapsed(started),
            input_sha256=llm.last_input_sha,
        )  # fmt: skip
        return record, exc, llm.provider_calls > calls_before
    return (
        _rag_record(answer, question, llm, _elapsed(started)),
        None,
        llm.provider_calls > calls_before,
    )


# --------------------------------------------------------------------------- the runner


def _entries_for(
    ctx: RepositoryContext,
    protocol: PrimaryProtocol,
    prior: GenerationRun | None,
    historical: Mapping[tuple[str, str], SideRecord],
) -> list[GenerationEntry]:
    prior_by_id = {e.question_id: e for e in prior.entries} if prior else {}
    out: list[GenerationEntry] = []
    for q in ctx.questions:
        old = prior_by_id.get(q.id)
        sides: dict[str, SideRecord] = {}
        for arm, version in (("plain", protocol.plain_prompt_version),
                             ("rag", protocol.rag_prompt_version)):  # fmt: skip
            existing = getattr(old, arm) if old else None
            if existing is not None and existing.state != "not_attempted":
                sides[arm] = existing
            elif (q.id, arm) in historical:
                sides[arm] = historical[(q.id, arm)]
            else:
                sides[arm] = placeholder(arm, version)  # type: ignore[arg-type]
        out.append(
            GenerationEntry(
                question_id=q.id, question=q.question, answerable=q.is_answerable, **sides
            )
        )
    return out


def run_repository(
    ctx: RepositoryContext,
    protocol: PrimaryProtocol,
    llm: LLMClient,
    settings: Settings,
    *,
    checkpoint_path: str | Path,
    resume_from: GenerationRun | None = None,
    historical: Mapping[tuple[str, str], SideRecord] | None = None,
    cache: ResponseCache | None = None,
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    retry_sleep: Callable[[float], None] = time.sleep,
    breaker: CircuitBreaker | None = None,
    progress: Callable[[str], None] | None = None,
    delay_seconds: float = 0.0,
) -> GenerationRun:
    """Ask every question of one repository through both arms, resumably.

    Sides already ``successful`` in ``resume_from`` (or supplied by ``historical``, which the caller
    has verified with ``compat.check_historical_run``) are reused unchanged; ``failed`` and
    ``not_attempted`` sides are asked. The run is checkpointed atomically after every side.
    """
    if not protocol.model_confirmed:
        raise RunMismatchError("the generation model has not been confirmed (--confirm-model)")
    if (protocol.llm_provider, protocol.llm_model) != (llm.provider, llm.model):
        raise RunMismatchError("the client is not the model confirmed in the protocol")
    if ctx.retriever is None:
        raise RunMismatchError(f"no retriever for repository {ctx.spec.repository_id!r}")
    if resume_from is not None:
        check_resume_compatible(resume_from, ctx, protocol, llm)
    breaker = breaker or CircuitBreaker()
    metered = MeteredLLM(llm, cache)
    service = RagService(ctx.retriever, metered, settings=settings)
    entries = _entries_for(ctx, protocol, resume_from, historical or {})

    def make_run() -> GenerationRun:
        return GenerationRun(
            repository_id=ctx.spec.repository_id, protocol_hash=protocol_hash(protocol),
            benchmark_digest=ctx.benchmark_digest, index_id=ctx.index_id,
            llm_provider=llm.provider, llm_model=llm.model, temperature=protocol.temperature,
            max_output_tokens=protocol.max_output_tokens, top_k=protocol.top_k,
            plain_prompt_version=protocol.plain_prompt_version,
            rag_prompt_version=protocol.rag_prompt_version,
            test_artifact=is_test_provider(llm), stop_reason=breaker.reason, entries=entries,
        )  # fmt: skip

    def checkpoint() -> None:
        write_run(make_run(), checkpoint_path)

    checkpoint()  # every requested question is on disk before the first call
    questions = {q.id: q for q in ctx.questions}
    for index, entry in enumerate(list(entries)):
        question = questions[entry.question_id]
        if progress:
            progress(f"[{ctx.spec.repository_id} {index + 1}/{len(entries)}] {question.id}")
        for arm in ("plain", "rag"):
            current = entries[index]
            if getattr(current, arm).state == "successful":
                continue  # reuse: not a call, so the 429 streak is untouched
            if breaker.tripped:
                checkpoint()
                return make_run()
            if arm == "plain":
                record, failure, called = ask_plain(
                    question, metered, protocol, settings,
                    retry_policy=retry_policy, retry_sleep=retry_sleep,
                )  # fmt: skip
            else:
                record, failure, called = ask_rag(
                    question, service, metered, protocol,
                    retry_policy=retry_policy, retry_sleep=retry_sleep,
                )  # fmt: skip
            entries[index] = current.model_copy(update={arm: record})
            checkpoint()
            if called:  # cache hits and system abstentions are not hosted calls
                breaker.record(failure)
                if breaker.tripped:
                    checkpoint()
                    if progress:
                        progress(f"  circuit breaker: {breaker.reason}")
                    return make_run()
                if delay_seconds:
                    retry_sleep(delay_seconds)
    return make_run()


def run_repositories(
    contexts: Sequence[RepositoryContext],
    protocol: PrimaryProtocol,
    llm: LLMClient,
    settings: Settings,
    *,
    output_dir: str | Path,
    historical: Mapping[str, Mapping[tuple[str, str], SideRecord]] | None = None,
    cache: ResponseCache | None = None,
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    retry_sleep: Callable[[float], None] = time.sleep,
    max_consecutive_rate_limit_failures: int | None = DEFAULT_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES,
    progress: Callable[[str], None] | None = None,
    delay_seconds: float = 0.0,
) -> dict[str, GenerationRun]:
    """Run several repositories into ``output_dir`` (one file each); a shared 429 breaker."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    breaker = CircuitBreaker(max_consecutive_rate_limit_failures)
    runs: dict[str, GenerationRun] = {}
    for ctx in contexts:
        path = run_path(out, ctx.spec.repository_id)
        prior = load_run(path) if path.is_file() else None
        if breaker.tripped:
            # not attempted: keep the on-disk state (or write complete placeholders) for honesty
            entries = _entries_for(
                ctx, protocol, prior, (historical or {}).get(ctx.spec.repository_id, {})
            )
            run = GenerationRun(
                repository_id=ctx.spec.repository_id, protocol_hash=protocol_hash(protocol),
                benchmark_digest=ctx.benchmark_digest, index_id=ctx.index_id,
                llm_provider=llm.provider, llm_model=llm.model,
                temperature=protocol.temperature, max_output_tokens=protocol.max_output_tokens,
                top_k=protocol.top_k, plain_prompt_version=protocol.plain_prompt_version,
                rag_prompt_version=protocol.rag_prompt_version,
                test_artifact=is_test_provider(llm), stop_reason=breaker.reason, entries=entries,
            )  # fmt: skip
            write_run(run, path)
            runs[ctx.spec.repository_id] = run
            continue
        runs[ctx.spec.repository_id] = run_repository(
            ctx, protocol, llm, settings, checkpoint_path=path, resume_from=prior,
            historical=(historical or {}).get(ctx.spec.repository_id), cache=cache,
            retry_policy=retry_policy, retry_sleep=retry_sleep, breaker=breaker,
            progress=progress, delay_seconds=delay_seconds,
        )  # fmt: skip
    return runs


def run_path(output_dir: str | Path, repository_id: str) -> Path:
    return Path(output_dir) / f"generation-{repository_id}.json"


# --------------------------------------------------------------------------- historical bridge


def historical_sides(
    run_entries: Sequence[object], reusable: Mapping[str, Sequence[str]], source_name: str
) -> dict[tuple[str, str], SideRecord]:
    """Convert verified-reusable historical sides to ``SideRecord`` (``origin="historical"``).

    ``run_entries`` are ``comparison.ComparisonEntry`` objects; ``reusable`` maps an arm to the
    question ids ``compat.check_historical_run`` verified. Token usage was not recorded historically
    and stays ``None``.
    """
    from copilot.evaluation.comparison import ComparisonEntry

    out: dict[tuple[str, str], SideRecord] = {}
    for entry in run_entries:
        assert isinstance(entry, ComparisonEntry)
        if entry.question_id in reusable.get("plain", ()):
            out[(entry.question_id, "plain")] = SideRecord(
                arm="plain", state="successful", prompt_version="plain-v1", status="answered",
                answer=entry.plain.answer, origin="historical", origin_file=source_name,
            )  # fmt: skip
        if entry.question_id in reusable.get("rag", ()):
            rag = entry.rag
            out[(entry.question_id, "rag")] = SideRecord(
                arm="rag", state="successful", prompt_version="rag-v1", status=rag.status,
                answer=rag.answer, insufficient_reason=rag.insufficient_reason,
                origin="historical", origin_file=source_name, sources=rag.sources,
                cited_source_numbers=rag.cited_source_numbers,
                unknown_source_numbers=rag.unknown_source_numbers,
                retrieved_count=rag.retrieved_count, retrieval_seconds=rag.retrieval_seconds,
                first_relevant_source=rag.first_relevant_source,
            )  # fmt: skip
    return out


@dataclass(frozen=True)
class PlanLine:
    repository_id: str
    questions: int
    answerable: int
    unanswerable: int
    generation_calls: int
    judge_calls: int
    reused_sides: int
    artifact: str


@dataclass(frozen=True)
class Plan:
    lines: tuple[PlanLine, ...]
    artifacts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def generation_calls(self) -> int:
        return sum(line.generation_calls for line in self.lines)

    @property
    def judge_calls(self) -> int:
        return sum(line.judge_calls for line in self.lines)


def plan_run(
    contexts: Sequence[RepositoryContext],
    protocol: PrimaryProtocol,
    *,
    output_dir: str | Path,
    historical: Mapping[str, Mapping[tuple[str, str], SideRecord]] | None = None,
) -> Plan:
    """Upper bound of calls a run would make, without constructing or invoking any provider.

    Generation calls: two sides per question, minus sides that are already successful on disk or
    verified-reusable historical sides. Judge calls: one per answer with text (RAG system
    abstentions are scored by rule, not judged); an upper bound, since the judge cache is not
    consulted here. A cache can only lower these numbers.
    """
    lines: list[PlanLine] = []
    artifacts: list[str] = []
    for ctx in contexts:
        path = run_path(output_dir, ctx.spec.repository_id)
        artifacts.append(path.as_posix())
        prior = load_run(path) if path.is_file() else None
        entries = _entries_for(
            ctx, protocol, prior, (historical or {}).get(ctx.spec.repository_id, {})
        )
        gen = sum(
            1 for e in entries for arm in ("plain", "rag") if getattr(e, arm).state != "successful"
        )
        reused = len(entries) * 2 - gen
        # judge calls: every side that is (or will be) a text answer; unanswered RAG system
        # abstentions cannot be known in advance, so all sides are counted (upper bound)
        judge = len(entries) * 2
        answerable = sum(1 for q in ctx.questions if q.is_answerable)
        lines.append(
            PlanLine(
                ctx.spec.repository_id,
                len(entries),
                answerable,
                len(entries) - answerable,
                gen,
                judge,
                reused,
                path.as_posix(),
            )  # fmt: skip
        )
    return Plan(tuple(lines), tuple(artifacts))
