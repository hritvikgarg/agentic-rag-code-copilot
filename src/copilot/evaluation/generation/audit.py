"""Manual-audit tooling: reproducible sampling, blinded worksheets and agreement statistics.

**Sampling.** ``sample_audit`` draws about 20 % of the *questions* (both arms of each sampled
question are audited, so comparisons stay paired). It is stratified by repository x answerability
so every repository and both kinds of question are represented, reproducible from a fixed seed, and
independent of the order of the input.

**Blinding.** ``build_worksheet`` gives every (question, arm) answer an anonymous item id, shuffles
the order with the seed, redacts ``[Source n]`` labels (which would reveal the RAG arm) and writes a
separate key file that maps item ids back to question and arm. The worksheet carries the reference
key facts and their source text but **no ratings**: every human field starts empty and nothing is
pre-filled. Blinding is imperfect - answer style can still hint at the arm - and the key file must
stay away from annotators.

**Agreement.** Raw agreement and chance-corrected agreement (Cohen's kappa; quadratic-weighted
kappa for ordinal 0-2 scores) between the judge and the human annotator(s). Kappa is ``None`` when
undefined (no variation in either rater). Kappa is also low when one class dominates even with high
raw agreement, so both numbers are always reported together.

Nothing here states or assumes that an audit has taken place.
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from copilot.evaluation.generation.judge import redact_source_labels
from copilot.evaluation.generation.reference import ReferenceMaterial
from copilot.evaluation.generation.rubric import Assessment, EvaluationError, FactVerdict

DEFAULT_AUDIT_FRACTION = 0.20


@dataclass(frozen=True)
class AuditUnit:
    repository_id: str
    question_id: str
    answerable: bool


def sample_audit(
    units: Sequence[AuditUnit], *, fraction: float = DEFAULT_AUDIT_FRACTION, seed: int
) -> list[AuditUnit]:
    """Stratified (repository x answerability), seeded sample of questions; at least one per
    non-empty stratum; deterministic and independent of the input order."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    strata: dict[tuple[str, bool], list[AuditUnit]] = {}
    for unit in sorted(units, key=lambda u: (u.repository_id, u.question_id)):
        strata.setdefault((unit.repository_id, unit.answerable), []).append(unit)
    chosen: list[AuditUnit] = []
    for key in sorted(strata):
        members = strata[key]
        rng = random.Random(f"{seed}|{key[0]}|{key[1]}")
        count = min(len(members), max(1, round(fraction * len(members))))
        chosen.extend(rng.sample(members, count))
    return sorted(chosen, key=lambda u: (u.repository_id, u.question_id))


class WorksheetKeyEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    repository_id: str
    question_id: str
    arm: str


@dataclass(frozen=True)
class AuditAnswer:
    """One arm's answer to one sampled question (only successful answers can be audited)."""

    repository_id: str
    question_id: str
    arm: str
    answer: str


@dataclass(frozen=True)
class Worksheet:
    rows: list[dict[str, object]]  # what annotators see (no arm, no ratings)
    key: list[WorksheetKeyEntry]  # keep away from annotators

    def to_csv(self) -> str:
        buffer = io.StringIO()
        fields = list(self.rows[0]) if self.rows else []
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(self.rows)
        return buffer.getvalue()

    def to_jsonl(self) -> str:
        return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.rows)

    def key_json(self) -> str:
        return json.dumps([k.model_dump() for k in self.key], indent=2) + "\n"


HUMAN_FIELDS = (
    "human_fact_verdicts",  # e.g. "f1=supported;f2=missing"
    "human_correctness",  # 0, 1 or 2 (optional cross-check of the computed score)
    "human_groundedness",  # 0, 1 or 2
    "human_unsupported_claims",  # one per line or separated by " | "
    "human_abstained",  # yes / no
    "human_notes",
)


def build_worksheet(
    answers: Sequence[AuditAnswer],
    questions: Mapping[tuple[str, str], str],
    references: Mapping[tuple[str, str], ReferenceMaterial],
    *,
    seed: int,
) -> Worksheet:
    """A blinded, shuffled worksheet with empty human fields and a separate key."""
    order = sorted(answers, key=lambda a: (a.repository_id, a.question_id, a.arm))
    random.Random(f"worksheet|{seed}").shuffle(order)
    rows: list[dict[str, object]] = []
    key: list[WorksheetKeyEntry] = []
    for number, item in enumerate(order, start=1):
        ref = references[(item.repository_id, item.question_id)]
        item_id = f"A-{number:04d}"
        if ref.answerable:
            facts = "\n".join(f"{f.fact_id}: {f.statement}" for f in ref.facts)
            evidence = "\n\n".join(
                f"[{label}]\n{text}" for f in ref.facts for label, text in f.evidence
            )
        else:
            facts = f"(none - the repository cannot answer this) {ref.unanswerable_rationale}"
            evidence = f"Appropriate answer: {ref.expected_abstention}"
        row: dict[str, object] = {
            "item_id": item_id,
            "repository_id": item.repository_id,
            "question": questions[(item.repository_id, item.question_id)],
            "reference_key_facts": facts,
            "reference_source_evidence": evidence,
            "answer": redact_source_labels(item.answer),
        }
        row.update(dict.fromkeys(HUMAN_FIELDS, ""))
        rows.append(row)
        key.append(WorksheetKeyEntry(item_id=item_id, repository_id=item.repository_id,
                                     question_id=item.question_id, arm=item.arm))  # fmt: skip
    return Worksheet(rows, key)


def parse_human_row(
    row: Mapping[str, str], fact_ids: Sequence[str], rater: str
) -> Assessment | None:
    """A filled worksheet row as an ``Assessment``; ``None`` for an unrated (blank) row.

    A partly filled row is an error (``EvaluationError``), not a guess."""
    filled = {f: (row.get(f) or "").strip() for f in HUMAN_FIELDS}
    required = [filled["human_groundedness"], filled["human_abstained"]]
    if fact_ids:
        required.append(filled["human_fact_verdicts"])
    if not any(filled[f] for f in ("human_fact_verdicts", "human_groundedness", "human_abstained")):
        return None
    if not all(required):
        raise EvaluationError(
            f"{row.get('item_id')}: incomplete rating (fill every required field)"
        )
    verdicts: dict[str, FactVerdict] = {}
    for part in filter(None, (p.strip() for p in filled["human_fact_verdicts"].split(";"))):
        name, _, value = part.partition("=")
        try:
            verdicts[name.strip()] = FactVerdict(value.strip().lower())
        except ValueError:
            raise EvaluationError(f"{row.get('item_id')}: bad verdict {part!r}") from None
    abstained = filled["human_abstained"].lower()
    if abstained not in {"yes", "no"}:
        raise EvaluationError(f"{row.get('item_id')}: human_abstained must be yes or no")
    claims = tuple(
        c.strip()
        for c in filled["human_unsupported_claims"].replace("\n", "|").split("|")
        if c.strip()
    )
    try:
        grounded = int(filled["human_groundedness"])
        assessment = Assessment(
            evaluator_kind="human", evaluator_id=rater, fact_verdicts=verdicts,
            material_unsupported_claims=claims, groundedness=grounded,
            abstained=abstained == "yes", notes=filled["human_notes"][:500],
        )  # fmt: skip
    except ValueError as exc:
        raise EvaluationError(f"{row.get('item_id')}: {exc}") from None
    return assessment


# --------------------------------------------------------------------------- agreement


def raw_agreement(a: Sequence[object], b: Sequence[object]) -> float | None:
    if len(a) != len(b):
        raise ValueError("ratings must have equal length")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a) if a else None


def cohen_kappa(a: Sequence[object], b: Sequence[object]) -> float | None:
    """Unweighted Cohen's kappa for nominal labels; ``None`` when undefined."""
    if len(a) != len(b):
        raise ValueError("ratings must have equal length")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in set(ca) | set(cb)) / (n * n)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def weighted_kappa(
    a: Sequence[int], b: Sequence[int], levels: Sequence[int] = (0, 1, 2)
) -> float | None:
    """Quadratic-weighted kappa for ordinal scores; ``None`` when undefined."""
    if len(a) != len(b):
        raise ValueError("ratings must have equal length")
    n = len(a)
    if n == 0:
        return None
    k = len(levels)
    index = {level: i for i, level in enumerate(levels)}
    weight = [[((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)]
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[index[x]][index[y]] += 1
    row = [sum(r) for r in observed]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    obs_disagreement = sum(weight[i][j] * observed[i][j] for i in range(k) for j in range(k)) / n
    exp_disagreement = sum(weight[i][j] * row[i] * col[j] for i in range(k) for j in range(k)) / (
        n * n
    )
    if math.isclose(exp_disagreement, 0.0):
        return None
    return 1 - obs_disagreement / exp_disagreement


@dataclass(frozen=True)
class AgreementResult:
    n: int
    raw: float | None
    kappa: float | None
    kind: str

    def to_json(self) -> dict[str, object]:
        return {
            "n": self.n,
            "raw_agreement": self.raw,
            "kappa": self.kappa,
            "kappa_kind": self.kind,
        }


def agreement(
    a: Sequence[object], b: Sequence[object], *, ordinal: bool = False
) -> AgreementResult:
    kappa = weighted_kappa(a, b) if ordinal else cohen_kappa(a, b)  # type: ignore[arg-type]
    return AgreementResult(len(a), raw_agreement(a, b), kappa,
                           "quadratic-weighted" if ordinal else "cohen")  # fmt: skip
