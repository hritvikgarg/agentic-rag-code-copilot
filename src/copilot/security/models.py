"""Typed models of the secret scanner.

Design rules:
* A finding never stores the matched value - only rule id, location, a static reason and an
  optional heavily masked preview. So ``repr``, ``model_dump``, logs and exceptions cannot leak it.
* Paths are repository-relative POSIX strings (validated); absolute host paths are rejected.
* ``ScanTarget.text`` is excluded from ``repr``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from copilot.utils.paths import validate_relative_posix

_MIN_PREVIEW_LENGTH = 16  # shorter values are masked completely
_PREVIEW_HEAD = 4
_PREVIEW_TAIL = 2
MASK = "****"


class Severity(StrEnum):
    """How sure the rule is. Informational: EVERY finding blocks external LLM use."""

    HIGH = "high"  # provider-specific token formats, private keys, random values on secret names
    MEDIUM = "medium"  # generic assignments, bearer values, credentials embedded in URLs


def redact(value: str) -> str:
    """A masked preview: ``ghp_...ab`` style for long values, ``****`` for anything under 16 chars.

    At most 6 characters of a value of 16 or more characters are shown, so a preview cannot be
    used to reconstruct the value.
    """
    if len(value) < _MIN_PREVIEW_LENGTH:
        return MASK
    return f"{value[:_PREVIEW_HEAD]}...{value[-_PREVIEW_TAIL:]}"


@dataclass(frozen=True)
class ScanTarget:
    """One piece of text to scan: a whole file, or a chunk of one (``first_line`` offsets lines)."""

    path: str  # repository-relative POSIX path; validated
    text: str = field(repr=False)
    first_line: int = 1

    def __post_init__(self) -> None:
        validate_relative_posix(self.path)
        if self.first_line < 1:
            raise ValueError("first_line must be >= 1")


class SecretFinding(BaseModel):
    """One potential secret. Safe metadata only."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    severity: Severity
    file_path: str
    line: int = Field(ge=1)  # 1-based line in the file
    column: int = Field(ge=1)  # 1-based column where the suspicious value starts
    reason: str  # static, human-readable category; never contains the value
    preview: str  # masked value, see ``redact``

    @field_validator("file_path")
    @classmethod
    def _relative(cls, value: str) -> str:
        return validate_relative_posix(value)

    @property
    def location(self) -> str:
        """``path:line:column``."""
        return f"{self.file_path}:{self.line}:{self.column}"


class SecretScanReport(BaseModel):
    """Result of scanning many targets. Safe metadata only; findings are in deterministic order."""

    model_config = ConfigDict(frozen=True)

    files_scanned: int = Field(ge=0)  # distinct paths scanned
    findings: tuple[SecretFinding, ...]  # sorted by (path, line, column, rule)
    counts_by_rule: dict[str, int]
    counts_by_severity: dict[str, int]

    @property
    def findings_count(self) -> int:
        return len(self.findings)

    @property
    def blocking(self) -> bool:
        """True if external LLM use must be refused. Every finding blocks (fail closed)."""
        return bool(self.findings)

    @property
    def files_with_findings(self) -> list[str]:
        return sorted({f.file_path for f in self.findings})
