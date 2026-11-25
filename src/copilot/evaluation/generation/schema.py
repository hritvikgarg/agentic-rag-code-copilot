"""The generation-benchmark file format (``generation-benchmark/1``), loader and checks.

The retrieval benchmark (``retrieval-benchmark/1``) only says *where* the answer lives. To grade a
**generated answer** we also need *what a correct answer must say* and *whether the repository can
answer the question at all*. This format adds exactly that and reuses the retrieval format's
``Region`` (file + inclusive line range + sha256 seal), so the same ground-truth sealing and
``verify_regions`` machinery apply.

One JSON object per line::

    {"id": "c01-ingestion", "repository_id": "copilot", "question": "...", "category": "...",
     "answerability": "answerable",
     "key_facts": [{"id": "f1", "statement": "...", "regions": [{"file_path": "...", ...}]}],
     "acceptable_answer": "...", "annotation_rationale": "..."}

    {"id": "c35-...", "answerability": "unanswerable", "key_facts": [],
     "unanswerable_rationale": "...", "expected_abstention": "...",
     "general_knowledge_sufficient": false, ...}

Rules (all enforced by ``load_generation_benchmark``):

* an *answerable* question has at least one key fact, each backed by at least one source region;
* an *unanswerable* question has no key facts, a rationale, an expected abstention behaviour, and
  ``general_knowledge_sufficient == false`` (a question a model could answer from general
  programming knowledge alone is **not** a repository-unanswerable question; it would reward
  confident general answers and punish honest ones);
* ids are unique and every question of a file carries the same ``repository_id``;
* the question text is a single field, so both arms necessarily receive identical text.

Both arms are given the same question string; nothing in this file is ever sent to a model except
``question``. Key facts, regions and rationales are evaluator-only reference material.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from copilot.evaluation.benchmark import (
    BenchmarkFormatError,
    BenchmarkQuestion,
    Region,
    RegionProblem,
    seal_regions,
    verify_regions,
)
from copilot.models.ingestion import SourceFile

GENERATION_SCHEMA = "generation-benchmark/1"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class Answerability(StrEnum):
    """Whether the repository contains what the question asks for."""

    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"


class KeyFact(BaseModel):
    """One checkable statement a correct answer must convey, with the source that supports it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    statement: str
    regions: tuple[Region, ...] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _id_format(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError("fact id must be lower-case letters, digits, '-' or '_'")
        return value

    @field_validator("statement")
    @classmethod
    def _statement_not_blank(cls, value: str) -> str:
        if len(value.strip()) < 8:
            raise ValueError("a key fact needs a real statement")
        return value


class GenerationQuestion(BaseModel):
    """A question, its answerability and the reference needed to grade an answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    repository_id: str
    question: str
    category: str
    answerability: Answerability
    key_facts: tuple[KeyFact, ...] = ()
    acceptable_answer: str = ""  # what a good answer looks like (answerable)
    unanswerable_rationale: str = ""  # why the repository cannot answer it (unanswerable)
    expected_abstention: str = ""  # what appropriate abstention looks like (unanswerable)
    near_miss_regions: tuple[Region, ...] = ()  # related code that does NOT contain the answer
    general_knowledge_sufficient: bool = False
    annotation_rationale: str = ""

    @field_validator("id", "repository_id", "category")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError("must be lower-case letters, digits, '-' or '_' (max 64)")
        return value

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question is blank")
        if value != value.strip():
            raise ValueError("question must not have leading/trailing whitespace")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> GenerationQuestion:
        fact_ids = [f.id for f in self.key_facts]
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("duplicate key-fact ids")
        if self.answerability is Answerability.ANSWERABLE:
            if not self.key_facts:
                raise ValueError("an answerable question needs at least one key fact")
            if not self.acceptable_answer.strip():
                raise ValueError("an answerable question needs acceptable_answer")
            if self.unanswerable_rationale or self.expected_abstention:
                raise ValueError("answerable questions must not carry abstention fields")
        else:
            if self.key_facts:
                raise ValueError("an unanswerable question cannot have key facts")
            if not self.unanswerable_rationale.strip() or not self.expected_abstention.strip():
                raise ValueError(
                    "an unanswerable question needs a rationale and expected_abstention"
                )
            if self.general_knowledge_sufficient:
                raise ValueError(
                    "general_knowledge_sufficient must be false for repository-unanswerable "
                    "questions (a general-programming question is not an abstention test)"
                )
        return self

    @property
    def is_answerable(self) -> bool:
        return self.answerability is Answerability.ANSWERABLE

    @property
    def relevant_regions(self) -> tuple[Region, ...]:
        """The union of every key fact's regions, first-seen order, duplicates removed."""
        seen: dict[tuple[str, int, int], Region] = {}
        for fact in self.key_facts:
            for region in fact.regions:
                seen.setdefault((region.file_path, region.start_line, region.end_line), region)
        return tuple(seen.values())

    def all_regions(self) -> tuple[Region, ...]:
        return (*self.relevant_regions, *self.near_miss_regions)


class GenerationBenchmarkMeta(BaseModel):
    """Provenance and honest limits of one repository's generation benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = GENERATION_SCHEMA
    name: str
    repository_id: str
    repository_name: str  # the name the index is built under
    source_kind: str  # "existing-snapshot" | "synthetic-authored"
    repository_fingerprint: str | None = None  # deterministic ingestion fingerprint, if frozen
    commit: str | None = None
    ignore_directories: tuple[str, ...] = ()
    independent: bool = False  # True only if the project authors did not write the code
    description: str = ""
    contamination_notes: str = ""
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationBenchmark:
    questions: tuple[GenerationQuestion, ...]
    meta: GenerationBenchmarkMeta | None = None

    @property
    def repository_id(self) -> str:
        return self.questions[0].repository_id

    def by_id(self) -> dict[str, GenerationQuestion]:
        return {q.id: q for q in self.questions}

    def counts(self) -> dict[str, int]:
        answerable = sum(1 for q in self.questions if q.is_answerable)
        return {
            "total": len(self.questions),
            "answerable": answerable,
            "unanswerable": len(self.questions) - answerable,
        }

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for q in self.questions:
            counts[q.category] = counts.get(q.category, 0) + 1
        return dict(sorted(counts.items()))


def meta_path_for(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".meta.json")


def _short(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        return f"{where}: {first['msg']}" if where else str(first["msg"])
    return f"{type(exc).__name__}: {exc}"


def load_generation_benchmark(path: str | Path) -> GenerationBenchmark:
    """Read and validate a generation benchmark (and its ``.meta.json`` when present)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise BenchmarkFormatError(f"cannot read benchmark file: {type(exc).__name__}") from exc
    questions: list[GenerationQuestion] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(GenerationQuestion.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise BenchmarkFormatError(f"line {number}: {_short(exc)}") from exc
    if not questions:
        raise BenchmarkFormatError("the benchmark contains no questions")
    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise BenchmarkFormatError(f"duplicate question ids: {', '.join(duplicates)}")
    repositories = {q.repository_id for q in questions}
    if len(repositories) != 1:
        raise BenchmarkFormatError("a benchmark file must cover exactly one repository_id")
    texts = [q.question.casefold() for q in questions]
    repeated = sorted({t[:60] for t in texts if texts.count(t) > 1})
    if repeated:
        raise BenchmarkFormatError(f"duplicate question text: {repeated[0]!r}")
    meta = None
    meta_file = meta_path_for(path)
    if meta_file.is_file():
        try:
            meta = GenerationBenchmarkMeta.model_validate_json(
                meta_file.read_text(encoding="utf-8")
            )
        except (ValidationError, OSError, UnicodeDecodeError) as exc:
            raise BenchmarkFormatError(f"{meta_file.name}: {_short(exc)}") from exc
        if meta.repository_id != questions[0].repository_id:
            raise BenchmarkFormatError("meta repository_id does not match the questions")
    return GenerationBenchmark(questions=tuple(questions), meta=meta)


def dump_generation_benchmark(questions: Sequence[GenerationQuestion]) -> str:
    """Canonical JSONL (compact lines, stable key order, UTF-8, trailing newline)."""
    return "".join(
        json.dumps(q.model_dump(mode="json"), ensure_ascii=False) + "\n" for q in questions
    )


def benchmark_digest(benchmark: GenerationBenchmark) -> str:
    """SHA-256 of the canonical text; any change to a question, fact or region changes it."""
    text = dump_generation_benchmark(benchmark.questions)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_retrieval_questions(questions: Sequence[GenerationQuestion]) -> list[BenchmarkQuestion]:
    out: list[BenchmarkQuestion] = []
    for q in questions:
        regions = q.all_regions()
        if regions:
            out.append(BenchmarkQuestion(id=q.id, question=q.question, relevant=regions))
    return out


def verify_generation_regions(
    benchmark: GenerationBenchmark, files: Sequence[SourceFile]
) -> list[RegionProblem]:
    """Every region of every question (key-fact and near-miss) against the ingested files.

    Reuses the retrieval benchmark's checks: the file was ingested, the lines exist, the region is
    sealed and its text still hashes to the sealed value.
    """
    return verify_regions(_as_retrieval_questions(benchmark.questions), files)


def seal_generation_regions(
    questions: Sequence[GenerationQuestion], files: Sequence[SourceFile]
) -> list[GenerationQuestion]:
    """Copies of ``questions`` with every region's sha256 computed from ``files``."""

    def seal(regions: tuple[Region, ...], qid: str) -> tuple[Region, ...]:
        if not regions:
            return ()
        probe = BenchmarkQuestion(id=qid, question="x", relevant=regions)
        return seal_regions([probe], files)[0].relevant

    sealed: list[GenerationQuestion] = []
    for q in questions:
        facts = tuple(
            fact.model_copy(update={"regions": seal(fact.regions, q.id)}) for fact in q.key_facts
        )
        sealed.append(
            q.model_copy(
                update={"key_facts": facts, "near_miss_regions": seal(q.near_miss_regions, q.id)}
            )
        )
    return sealed
