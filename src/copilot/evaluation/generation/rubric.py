"""The correctness rubric and the deterministic scoring of per-key-fact verdicts.

**Why per-fact verdicts instead of one number:** a single 0-2 grade from a judge or a person is
hard to audit. Here the evaluator (an LLM judge or a human) answers narrow questions - is *this*
key fact conveyed correctly, contradicted or missing? - and the score is computed from those
answers by fixed rules, so the same verdicts always give the same score and a reader can see why.

Scores for an **answerable** question (``key facts`` known):

* ``2`` full: every key fact is ``supported``, none ``contradicted``, no material unsupported claim;
* ``0`` incorrect: no key fact is ``supported``, or more facts are ``contradicted`` than
  ``supported``;
* ``1`` partial: everything else - some facts missing, or all facts present but the answer also
  makes a material unsupported claim or contradicts one fact while supporting more.

A *material unsupported claim* is a repository-specific statement that the reference source text
does not support (an invented file, function, behaviour or number) and that a reader could act on.
General programming knowledge that is clearly labelled and not about this repository is not
counted. A single ``contradicted`` fact or a material unsupported claim caps the score at ``1``.

Scores for an **unanswerable** question (the repository cannot answer it):

* ``2`` the answer abstains (says the repository does not show it) and makes no material
  unsupported repository claim;
* ``1`` it abstains but also asserts unsupported repository specifics (a hedged guess);
* ``0`` it answers as though the repository supported it.

Keyword or string overlap with the key facts is **never** used as correctness: paraphrase is
normal, and overlap can be high for a wrong answer. Verdicts come from reading, not matching.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from copilot.evaluation.generation.schema import GenerationQuestion

RUBRIC_VERSION = "gen-rubric-1"


class EvaluationError(ValueError):
    """An assessment is malformed or does not match its question. Never replaced by a default."""


class FactVerdict(StrEnum):
    SUPPORTED = "supported"  # the answer conveys this fact correctly (paraphrase is fine)
    MISSING = "missing"  # the answer does not convey it
    CONTRADICTED = "contradicted"  # the answer states something that conflicts with it


ResponseType = Literal["answered", "insufficient_evidence", "generic_refusal", "mixed", "ambiguous"]

#: ``abstained`` is true exactly for these response types (``ambiguous`` may be either).
ABSTAINING_TYPES = frozenset({"insufficient_evidence", "mixed"})
NON_ABSTAINING_TYPES = frozenset({"answered", "generic_refusal"})


class Assessment(BaseModel):
    """What one evaluator concluded about one answer. Raw answer text is stored elsewhere."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluator_kind: Literal["judge", "human", "rule"]
    evaluator_id: str  # the judge model + prompt version, or the annotator's pseudonym
    fact_verdicts: dict[str, FactVerdict] = Field(default_factory=dict)
    material_unsupported_claims: tuple[str, ...] = ()
    groundedness: int = Field(ge=0, le=2)  # 0 mostly unsupported, 1 mixed, 2 all supported
    abstained: bool  # the answer says the repository evidence/knowledge is insufficient
    notes: str = ""
    # What kind of response this is (judge prompt gen-judge-2 and the system rule). ``None`` for
    # assessments made without the distinction (human worksheets, gen-judge-1 files).
    response_type: ResponseType | None = None

    @model_validator(mode="after")
    def _type_matches_abstained(self) -> Assessment:
        if self.response_type in ABSTAINING_TYPES and not self.abstained:
            raise ValueError(f"response_type {self.response_type!r} requires abstained=true")
        if self.response_type in NON_ABSTAINING_TYPES and self.abstained:
            raise ValueError(f"response_type {self.response_type!r} requires abstained=false")
        return self

    @model_validator(mode="after")
    def _claims_are_short(self) -> Assessment:
        if any(len(c) > 400 for c in self.material_unsupported_claims):
            raise ValueError("an unsupported-claim description must be 400 characters or fewer")
        return self


class CorrectnessScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: int = Field(ge=0, le=2)
    total_facts: int
    supported: int
    missing: int
    contradicted: int
    material_unsupported: int
    completeness: float | None  # supported / total_facts (answerable only)
    appropriate_abstention: bool | None  # unanswerable only: abstained and no unsupported claim
    rule: str


def validate_assessment(question: GenerationQuestion, assessment: Assessment) -> None:
    """Raise ``EvaluationError`` unless the assessment covers exactly this question's facts."""
    expected = {f.id for f in question.key_facts}
    given = set(assessment.fact_verdicts)
    if question.is_answerable:
        if given != expected:
            raise EvaluationError(
                f"{question.id}: verdicts cover {sorted(given)}, key facts are {sorted(expected)}"
            )
    elif given:
        raise EvaluationError(f"{question.id}: an unanswerable question has no key facts to judge")


def score_assessment(question: GenerationQuestion, assessment: Assessment) -> CorrectnessScore:
    """Apply the rubric. Raises ``EvaluationError`` for an assessment that does not fit."""
    validate_assessment(question, assessment)
    unsupported = len(assessment.material_unsupported_claims)
    if not question.is_answerable:
        if assessment.abstained and unsupported == 0:
            score, rule = 2, "abstained with no unsupported repository claim"
        elif assessment.abstained:
            score, rule = 1, "abstained but asserted unsupported repository specifics"
        else:
            score, rule = 0, "answered a question the repository cannot answer"
        return CorrectnessScore(
            score=score, total_facts=0, supported=0, missing=0, contradicted=0,
            material_unsupported=unsupported, completeness=None,
            appropriate_abstention=assessment.abstained and unsupported == 0, rule=rule,
        )  # fmt: skip
    verdicts = list(assessment.fact_verdicts.values())
    total = len(verdicts)
    supported = verdicts.count(FactVerdict.SUPPORTED)
    missing = verdicts.count(FactVerdict.MISSING)
    contradicted = verdicts.count(FactVerdict.CONTRADICTED)
    if supported == 0 or contradicted > supported:
        score, rule = 0, "no key fact supported, or more contradicted than supported"
    elif supported == total and contradicted == 0 and unsupported == 0:
        score, rule = 2, "all key facts supported, none contradicted, no unsupported claim"
    else:
        score, rule = (
            1,
            "some key facts supported, but not all or with a contradiction/unsupported claim",
        )
    return CorrectnessScore(
        score=score, total_facts=total, supported=supported, missing=missing,
        contradicted=contradicted, material_unsupported=unsupported,
        completeness=supported / total, appropriate_abstention=None, rule=rule,
    )  # fmt: skip
