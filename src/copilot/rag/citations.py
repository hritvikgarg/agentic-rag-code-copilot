"""Citation objects and answer-label checking.

The authoritative citations are built from retrieval metadata (``Citation``), never parsed from
the model's prose. The prose may contain ``[Source n]`` labels; ``resolve_source_numbers`` only
reports which supplied sources the prose mentions and which numbers do not exist, so an invented
label can never turn into a fake citation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from copilot.utils.paths import validate_relative_posix

# "[Source 1]", "(Source 2)", "[Sources 1, 2 and 3]", "[Source 1, Source 2]"
_LABEL = re.compile(
    r"[\[(]\s*sources?\s+(\d+(?:\s*(?:,|and|&)\s*(?:sources?\s+)?\d+)*)\s*[\])]", re.IGNORECASE
)


class Citation(BaseModel):
    """One piece of evidence given to the model, described by retrieval metadata only."""

    model_config = ConfigDict(frozen=True)

    source_number: int = Field(ge=1)  # the label the model sees: "Source n"
    file_path: str  # repository-relative POSIX path
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    chunk_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    score: float  # cosine similarity from retrieval
    rank: int = Field(
        ge=1
    )  # retrieval rank (may differ from source_number if evidence was dropped)

    @field_validator("file_path")
    @classmethod
    def _relative(cls, value: str) -> str:
        return validate_relative_posix(value)

    @property
    def label(self) -> str:
        return f"Source {self.source_number}"

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


def extract_source_numbers(answer: str) -> list[int]:
    """Every source number mentioned in ``[Source n]``-style labels, in order of appearance."""
    numbers: list[int] = []
    for match in _LABEL.finditer(answer):
        numbers.extend(int(n) for n in re.findall(r"\d+", match.group(1)))
    return numbers


def resolve_source_numbers(
    answer: str, citations: Sequence[Citation]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split the labels in ``answer`` into ``(cited, unknown)``.

    ``cited`` are supplied source numbers the prose mentions; ``unknown`` are numbers with no
    supplied source (the model invented or mis-numbered them). Both are sorted and unique.
    """
    valid = {c.source_number for c in citations}
    mentioned = set(extract_source_numbers(answer))
    return tuple(sorted(mentioned & valid)), tuple(sorted(mentioned - valid))
