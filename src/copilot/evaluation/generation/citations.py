"""A programmatic citation checker.

Two kinds of citation are checked, and the report keeps them apart:

* **Source labels** ``[Source n]`` (the RAG arm). Each label that the *answer text* uses is resolved
  to the evidence block it names. A label with no matching block is ``unknown_source`` (invented).
  Blocks that retrieval returned but the answer never mentions are *not* citations of the answer;
  they are listed as ``retrieved_uncited`` and never scored.
* **Explicit location mentions** such as ``src/a/b.py:10-20``, ``a.py:42`` or
  ``src/a/b.py (lines 10-20)``, which either arm may write.

For every citation the checker reports, deterministically: whether the syntax is valid, whether the
file exists in the repository, whether the line range is valid for that file, whether it overlaps
the evidence the system actually retrieved, and whether it overlaps an expected (ground-truth)
region. ``invented`` means the cited file does not exist or the range lies beyond the file.

Applicability: a **plain** answer has no source labels by construction, so it is never penalised for
lacking them (``applicable=False``); it is checked only if it chooses to write explicit locations.
A RAG answer that makes an answer but cites nothing is applicable and reported ``missing``.
Abstentions and answers without status ``answered`` carry no claims and are not applicable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from copilot.evaluation.benchmark import Region
from copilot.evaluation.generation.inventory import RepoInventory
from copilot.rag.citations import extract_source_numbers

Arm = Literal["plain", "rag"]
Status = Literal["valid", "malformed", "file_missing", "invalid_range", "unknown_source"]

_PATH = r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,5}"
_LOCATION_RE = re.compile(
    rf"(?P<path>{_PATH})(?::|\s*\(?\s*lines?\s+|\s+L)(?P<start>\d+)(?:\s*[-–]\s*L?(?P<end>\d+))?",
    re.I,
)
_BROKEN_RE = re.compile(rf"(?P<path>{_PATH}):(?P<junk>[A-Za-z?#!]\w*)")


class RetrievedRegion(BaseModel):
    """One evidence block that retrieval supplied (file + line range)."""

    model_config = ConfigDict(frozen=True)

    source_number: int
    file_path: str
    start_line: int
    end_line: int


class CitationCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw: str
    origin: Literal["source_label", "location"]
    status: Status
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    in_retrieved_evidence: bool | None = None  # None when no retrieval applies (plain arm)
    overlaps_expected: bool | None = None  # None when the question has no expected regions
    invented: bool = False


@dataclass(frozen=True)
class CitationReport:
    arm: Arm
    applicable: bool
    not_applicable_reason: str | None
    missing: bool  # an applicable answer that cites nothing at all
    citations: tuple[CitationCheck, ...]
    retrieved_uncited: tuple[int, ...]  # source numbers retrieved but not used by the answer

    def _count(self, predicate) -> int:
        return sum(1 for c in self.citations if predicate(c))

    @property
    def valid(self) -> int:
        return self._count(lambda c: c.status == "valid")

    @property
    def invented(self) -> int:
        return self._count(lambda c: c.invented)

    def _fraction(self, predicate) -> float | None:
        if not self.applicable or not self.citations:
            return None
        return self._count(predicate) / len(self.citations)

    @property
    def valid_fraction(self) -> float | None:
        return self._fraction(lambda c: c.status == "valid")

    @property
    def retrieved_support_fraction(self) -> float | None:
        if self.arm == "plain":
            return None
        return self._fraction(lambda c: c.in_retrieved_evidence is True)

    @property
    def expected_overlap_fraction(self) -> float | None:
        if all(c.overlaps_expected is None for c in self.citations):
            return None
        return self._fraction(lambda c: c.overlaps_expected is True)


def _overlap(path: str, start: int, end: int, region_path: str, rs: int, re_: int) -> bool:
    return path == region_path and start <= re_ and rs <= end


def _check_location(
    path: str,
    start: int,
    end: int,
    raw: str,
    origin: Literal["source_label", "location"],
    inventory: RepoInventory,
    retrieved: Sequence[RetrievedRegion] | None,
    expected: Sequence[Region] | None,
) -> CitationCheck:
    norm = path[2:] if path.startswith("./") else path
    if norm not in inventory.paths:
        candidates = inventory.basenames.get(norm.rsplit("/", 1)[-1], ())
        if len(candidates) == 1 and "/" not in norm:
            norm = candidates[0]  # a bare file name that names exactly one repository file
        else:
            return CitationCheck(
                raw=raw,
                origin=origin,
                status="file_missing",
                file_path=norm,
                start_line=start,
                end_line=end,
                invented=True,
                in_retrieved_evidence=False if retrieved is not None else None,
            )
    total = inventory.line_count(norm) or 0
    if start < 1 or end < start:
        return CitationCheck(
            raw=raw,
            origin=origin,
            status="malformed",
            file_path=norm,
            start_line=start,
            end_line=end,
        )
    if end > total:
        return CitationCheck(
            raw=raw,
            origin=origin,
            status="invalid_range",
            file_path=norm,
            start_line=start,
            end_line=end,
            invented=True,
            in_retrieved_evidence=False if retrieved is not None else None,
        )
    in_evidence = None
    if retrieved is not None:
        in_evidence = any(
            _overlap(norm, start, end, r.file_path, r.start_line, r.end_line) for r in retrieved
        )
    overlaps = None
    if expected:
        overlaps = any(
            _overlap(norm, start, end, r.file_path, r.start_line, r.end_line) for r in expected
        )
    return CitationCheck(
        raw=raw,
        origin=origin,
        status="valid",
        file_path=norm,
        start_line=start,
        end_line=end,
        in_retrieved_evidence=in_evidence,
        overlaps_expected=overlaps,
    )


def check_citations(
    answer: str | None,
    *,
    arm: Arm,
    inventory: RepoInventory,
    answered: bool = True,
    sources: Mapping[int, RetrievedRegion] | None = None,
    expected: Sequence[Region] | None = None,
) -> CitationReport:
    """Check every citation ``answer`` makes. ``sources`` maps the supplied ``[Source n]`` numbers
    to their evidence block (the RAG arm); ``expected`` are the question's ground-truth regions."""
    if not answered or not answer or not answer.strip():
        return CitationReport(arm, False, "no answer text to check", False, (), ())
    retrieved = list(sources.values()) if sources is not None else None
    checks: list[CitationCheck] = []
    used: set[int] = set()
    seen_locations: set[tuple[str, int, int]] = set()

    if arm == "rag" and sources is not None:
        for number in extract_source_numbers(answer):
            if number in used:
                continue
            used.add(number)
            label = f"[Source {number}]"
            block = sources.get(number)
            if block is None:
                checks.append(
                    CitationCheck(
                        raw=label,
                        origin="source_label",
                        status="unknown_source",
                        invented=True,
                        in_retrieved_evidence=False,
                    )
                )
                continue
            checks.append(
                _check_location(
                    block.file_path,
                    block.start_line,
                    block.end_line,
                    label,
                    "source_label",
                    inventory,
                    retrieved,
                    expected,
                )
            )
    for match in _LOCATION_RE.finditer(answer):
        start = int(match.group("start"))
        end = int(match.group("end")) if match.group("end") else start
        key = (match.group("path"), start, end)
        if key in seen_locations:
            continue
        seen_locations.add(key)
        checks.append(
            _check_location(
                match.group("path"),
                start,
                end,
                match.group(0).strip(),
                "location",
                inventory,
                retrieved,
                expected,
            )
        )
    for match in _BROKEN_RE.finditer(answer):
        raw = match.group(0)
        if not any(c.raw.startswith(match.group("path")) for c in checks):
            checks.append(
                CitationCheck(
                    raw=raw, origin="location", status="malformed", file_path=match.group("path")
                )
            )
    uncited = tuple(sorted(n for n in (sources or {}) if n not in used))
    if arm == "plain" and not checks:
        return CitationReport(arm, False, "plain answers carry no source labels", False, (), ())
    missing = arm == "rag" and not checks
    return CitationReport(arm, True, None, missing, tuple(checks), uncited)
