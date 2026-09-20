"""``RetrievalResult``: one ranked, verified search hit.

Carries everything a later citation needs (repository, repo-relative path, line range, chunk id,
symbol metadata) plus the verified source text. Paths are repository-relative POSIX paths, never
host absolute paths (enforced by validation).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from copilot.models.chunk import ChunkType
from copilot.utils.paths import validate_relative_posix


class RetrievalResult(BaseModel):
    """A retrieved chunk, ranked by cosine similarity to the query."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)  # 1 = most similar
    score: float  # cosine similarity in [-1, 1] (inner product of unit vectors)
    position: int = Field(ge=0)  # vector position in the index
    chunk_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    repository_name: str = Field(min_length=1)
    file_path: str  # repository-relative POSIX path
    language: str
    chunk_type: ChunkType
    chunk_index: int = Field(ge=0)
    start_line: int = Field(ge=1)  # 1-based, inclusive
    end_line: int = Field(ge=1)
    fragment_index: int | None = None
    fragment_count: int | None = None
    symbol_name: str | None = None
    qualified_name: str | None = None
    parent_class: str | None = None
    token_estimate: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(repr=False)  # source text, verified against ``content_sha256``

    @field_validator("file_path")
    @classmethod
    def _relative_path(cls, value: str) -> str:
        return validate_relative_posix(value)

    @model_validator(mode="after")
    def _line_order(self) -> RetrievalResult:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self

    @property
    def line_count(self) -> int:
        """Physical source lines covered (a fragment of one long line counts as 1)."""
        return self.end_line - self.start_line + 1

    @property
    def location(self) -> str:
        """``path:start-end``, the form used in the terminal output and later in citations."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"
