"""The benchmark file format, its loader and its integrity checks.

**Format** (``retrieval-benchmark/1``): a UTF-8 JSONL file, one question per line::

    {"id": "q01-ingestion", "question": "Where is repository ingestion implemented?",
     "relevant": [{"file_path": "src/copilot/ingestion/loader.py", "start_line": 68,
                   "end_line": 140, "sha256": "<hex>"}], "notes": "why these lines"}

* ``relevant`` lists the *source regions* that answer the question (one or more). A retrieved chunk
  is a hit when it is in the same file and overlaps a region by at least one physical line.
* Chunk ids are never used: they change with every chunking parameter.
* ``sha256`` is the SHA-256 of the region's text (lines ``start_line..end_line`` of the
  newline-normalised file joined with ``"\\n"``). It **pins the ground truth to the exact source**:
  ``verify_regions`` refuses a repository whose region text differs, so a benchmark can never be
  scored against a different version of the code by accident (this works across Windows/Linux).
* An optional sibling ``<name>.meta.json`` records the repository name, the pinned commit, the
  ignore directories used when indexing, and an honest description of the benchmark's limits.

Line numbers are 1-based and inclusive, exactly like chunk line ranges.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from copilot.chunking.windows import split_lines
from copilot.models.chunk import sha256_text
from copilot.models.ingestion import SourceFile
from copilot.utils.paths import validate_relative_posix

BENCHMARK_SCHEMA = "retrieval-benchmark/1"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class BenchmarkFormatError(ValueError):
    """The benchmark file is malformed (bad JSON, missing/unknown field, duplicate id...)."""


class Region(BaseModel):
    """A span of source lines that answers a question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("file_path")
    @classmethod
    def _relative_path(cls, value: str) -> str:
        return validate_relative_posix(value)

    @model_validator(mode="after")
    def _order(self) -> Region:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class BenchmarkQuestion(BaseModel):
    """One natural-language question and the source regions that answer it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    relevant: tuple[Region, ...] = Field(min_length=1)
    notes: str = ""

    @field_validator("id")
    @classmethod
    def _id_format(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError("id must be lower-case letters, digits, '-' or '_' (max 64)")
        return value

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question is blank")
        return value


class BenchmarkMeta(BaseModel):
    """Provenance and honest limits of a benchmark (``<name>.meta.json``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = BENCHMARK_SCHEMA
    name: str
    repository_name: str  # the name the index is built under (chunk ids include it)
    commit: str  # the commit the regions were verified against
    ignore_directories: tuple[str, ...] = ()  # pass these as --ignore-dir when indexing
    independent: bool = False  # True only for a corpus the project authors did not write
    description: str = ""
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Benchmark:
    """Loaded questions plus optional metadata."""

    questions: tuple[BenchmarkQuestion, ...]
    meta: BenchmarkMeta | None = None


def meta_path_for(path: str | Path) -> Path:
    """``foo.jsonl`` -> ``foo.meta.json`` (next to it)."""
    path = Path(path)
    return path.with_name(path.stem + ".meta.json")


def load_benchmark(path: str | Path) -> Benchmark:
    """Read and validate a benchmark JSONL file (and its ``.meta.json`` if present)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise BenchmarkFormatError(f"cannot read benchmark file: {type(exc).__name__}") from exc
    questions: list[BenchmarkQuestion] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(BenchmarkQuestion.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise BenchmarkFormatError(f"line {number}: {_short(exc)}") from exc
    if not questions:
        raise BenchmarkFormatError("the benchmark contains no questions")
    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise BenchmarkFormatError(f"duplicate question ids: {', '.join(duplicates)}")
    meta = None
    meta_file = meta_path_for(path)
    if meta_file.is_file():
        try:
            meta = BenchmarkMeta.model_validate_json(meta_file.read_text(encoding="utf-8"))
        except (ValidationError, OSError, UnicodeDecodeError) as exc:
            raise BenchmarkFormatError(f"{meta_file.name}: {_short(exc)}") from exc
    return Benchmark(questions=tuple(questions), meta=meta)


def _short(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        return f"{where}: {first['msg']}" if where else str(first["msg"])
    return f"{type(exc).__name__}: {exc}"


def dump_benchmark(questions: Sequence[BenchmarkQuestion]) -> str:
    """Canonical JSONL text (one compact line per question, stable key order, UTF-8)."""
    lines = [
        json.dumps(q.model_dump(mode="json", exclude_defaults=False), ensure_ascii=False)
        for q in questions
    ]
    return "".join(line + "\n" for line in lines)


def region_text(file: SourceFile, region: Region) -> str:
    """The text of ``region`` in ``file`` (lines joined with ``"\\n"``)."""
    return "\n".join(split_lines(file.content)[region.start_line - 1 : region.end_line])


@dataclass(frozen=True)
class RegionProblem:
    """Why a benchmark region is not valid for a repository."""

    question_id: str
    file_path: str
    start_line: int
    end_line: int
    problem: str

    def __str__(self) -> str:
        return (
            f"{self.question_id}: {self.file_path}:{self.start_line}-{self.end_line}: "
            f"{self.problem}"
        )


def verify_regions(
    questions: Sequence[BenchmarkQuestion], files: Sequence[SourceFile]
) -> list[RegionProblem]:
    """Every problem of every region against the ingested ``files`` (empty list: all valid).

    A region is valid when its file was ingested, its lines exist, and (when it carries a
    ``sha256``) its text hashes to that value. A region without ``sha256`` is reported as
    unsealed, because an unsealed benchmark is not pinned to any source version.
    """
    by_path = {f.relative_path: f for f in files}
    problems: list[RegionProblem] = []
    for question in questions:
        for region in question.relevant:

            def bad(message: str, q=question, r=region) -> None:
                problems.append(RegionProblem(q.id, r.file_path, r.start_line, r.end_line, message))

            file = by_path.get(region.file_path)
            if file is None:
                bad("file is not in the ingested repository (missing, ignored or unsupported)")
                continue
            total = len(split_lines(file.content))
            if region.end_line > total:
                bad(f"file has only {total} lines")
                continue
            if region.sha256 is None:
                bad("region is not sealed (no sha256); run the 'seal' command")
            elif sha256_text(region_text(file, region)) != region.sha256:
                bad("the source text differs from the text the benchmark was written against")
    return problems


def seal_regions(
    questions: Sequence[BenchmarkQuestion], files: Sequence[SourceFile]
) -> list[BenchmarkQuestion]:
    """Copies of ``questions`` with every region's ``sha256`` computed from ``files``.

    Raises ``BenchmarkFormatError`` if a region's file or lines do not exist. Sealing records
    "the ground truth was verified against exactly this text"; it does not verify relevance,
    which a human must do by reading the region.
    """
    by_path = {f.relative_path: f for f in files}
    sealed: list[BenchmarkQuestion] = []
    for question in questions:
        regions = []
        for region in question.relevant:
            file = by_path.get(region.file_path)
            if file is None or region.end_line > len(split_lines(file.content)):
                raise BenchmarkFormatError(
                    f"{question.id}: {region.file_path}:{region.start_line}-{region.end_line} "
                    "does not exist in the repository"
                )
            regions.append(
                region.model_copy(update={"sha256": sha256_text(region_text(file, region))})
            )
        sealed.append(question.model_copy(update={"relevant": tuple(regions)}))
    return sealed
