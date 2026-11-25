"""Reference material shown to evaluators (the judge and human annotators).

Both arms' answers are graded against the *same* reference: the question's key facts and the exact
source text of the regions that support them, read from the frozen repository. Nothing about the
arm that produced an answer is part of the reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from copilot.evaluation.benchmark import Region, region_text
from copilot.evaluation.generation.schema import GenerationQuestion
from copilot.models.ingestion import SourceFile

MAX_REGION_LINES = 160  # a safety bound on what is shown per region


@dataclass(frozen=True)
class FactReference:
    fact_id: str
    statement: str
    evidence: tuple[tuple[str, str], ...]  # (location label, source text)


@dataclass(frozen=True)
class ReferenceMaterial:
    question_id: str
    answerable: bool
    facts: tuple[FactReference, ...]
    unanswerable_rationale: str
    expected_abstention: str
    acceptable_answer: str


def _label(region: Region) -> str:
    return f"{region.file_path}:{region.start_line}-{region.end_line}"


def build_reference(question: GenerationQuestion, files: Sequence[SourceFile]) -> ReferenceMaterial:
    """Read the supporting source text for ``question`` from the frozen ``files``."""
    by_path = {f.relative_path: f for f in files}
    facts = []
    for fact in question.key_facts:
        evidence = []
        for region in fact.regions:
            file = by_path.get(region.file_path)
            if file is None:
                raise ValueError(f"{question.id}: {region.file_path} is not in the repository")
            text = region_text(file, region)
            lines = text.split("\n")
            if len(lines) > MAX_REGION_LINES:
                text = "\n".join(lines[:MAX_REGION_LINES]) + "\n[region truncated for display]"
            evidence.append((_label(region), text))
        facts.append(FactReference(fact.id, fact.statement, tuple(evidence)))
    return ReferenceMaterial(
        question_id=question.id,
        answerable=question.is_answerable,
        facts=tuple(facts),
        unanswerable_rationale=question.unanswerable_rationale,
        expected_abstention=question.expected_abstention,
        acceptable_answer=question.acceptable_answer,
    )
