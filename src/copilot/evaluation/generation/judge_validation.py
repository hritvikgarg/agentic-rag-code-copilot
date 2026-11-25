"""The predeclared validation plan for the local judge (Milestone 11B).

``generation-eval/judge-validation/judge-validation-v1.json`` was committed before any local or
hosted judge execution on the pilot or the synthetic controls. This module loads it, exposes its
SHA-256 (every control result and validation report records that hash, so a result can always be
tied to the exact thresholds it was judged against) and evaluates artifacts against it. The
thresholds are read from the file, never hard-coded here, and a test pins the file's hash: changing
a threshold after seeing results therefore shows up as a failing test and a new version.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

VALIDATION_SPEC_PATH = "generation-eval/judge-validation/judge-validation-v1.json"
VALIDATION_SPEC_SCHEMA = "judge-validation/1"
REQUIRED_SECTIONS = (
    "judge",
    "controls",
    "technical_acceptance",
    "semantic_pilot_acceptance",
    "primary_audit_acceptance",
)


class ValidationSpecError(ValueError):
    """The validation specification is missing, malformed or not the expected version."""


def spec_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_validation_spec(path: str | Path) -> dict[str, Any]:
    try:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidationSpecError(f"cannot read the validation specification: {path}") from exc
    if not isinstance(spec, dict) or spec.get("schema") != VALIDATION_SPEC_SCHEMA:
        raise ValidationSpecError("not a judge-validation/1 specification")
    missing = [name for name in REQUIRED_SECTIONS if name not in spec]
    if missing:
        raise ValidationSpecError(f"the validation specification lacks: {', '.join(missing)}")
    return spec
