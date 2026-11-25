"""Human verification of rule-scored system abstentions (separate from the judge-vs-human audit).

When the RAG system abstains without calling the model, no judge rates the response: the rubric
scores it by rule (answerable -> correctness 0; unanswerable -> 2; groundedness 2, no claims). The
rule *assumes* two things about the system's message - that it makes no repository-specific claim
and that it is a genuine abstention. This module samples those items into their **own** blinded-
ID worksheet so a human can confirm the assumption.

Its numbers are a *rule-item verification*. They are never pooled with, or reported as, the
judge-versus-human agreement of ``audit_agree``: different items, different question, different
statistic.
"""

from __future__ import annotations

import csv
import io
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from copilot.evaluation.generation.analysis import RepoInputs
from copilot.evaluation.generation.audit import AuditUnit, sample_audit

RULE_ITEM_PREFIX = "R-"
RULE_HUMAN_FIELDS = (
    "human_makes_no_repository_claims",  # yes / no
    "human_is_genuine_abstention",  # yes / no
    "human_notes",
)
RULE_KEY_FIELDS = ("item_id", "repository_id", "question_id")


@dataclass(frozen=True)
class RuleItem:
    repository_id: str
    question_id: str
    answerable: bool
    question: str
    reference: str
    system_message: str
    reason: str
    retrieved_count: int


def rule_items(inputs: Sequence[RepoInputs]) -> list[RuleItem]:
    """Every RAG system abstention (the model was not called) in the given runs."""
    out: list[RuleItem] = []
    for item in inputs:
        for entry in sorted(item.run.entries, key=lambda e: e.question_id):
            side = entry.rag
            if not side.system_insufficient:
                continue
            question = item.questions[entry.question_id]
            if question.is_answerable:
                reference = "Key facts the repository supports: " + " | ".join(
                    f.statement for f in question.key_facts
                )
            else:
                reference = (
                    f"Unanswerable ({question.unanswerable_rationale}). Appropriate answer: "
                    f"{question.expected_abstention}"
                )
            out.append(
                RuleItem(
                    item.run.repository_id,
                    entry.question_id,
                    question.is_answerable,
                    question.question,
                    reference,
                    side.answer or "",
                    side.insufficient_reason or "",
                    side.retrieved_count,
                )  # fmt: skip
            )
    return out


def build_rule_worksheet(
    items: Sequence[RuleItem], *, fraction: float, seed: int, status_label: str | None = None
) -> tuple[str, str, list[RuleItem]]:
    """``(csv_text, key_json, sampled_items)``; empty text when there is nothing to verify."""
    units = [AuditUnit(i.repository_id, i.question_id, i.answerable) for i in items]
    chosen = {
        (u.repository_id, u.question_id) for u in sample_audit(units, fraction=fraction, seed=seed)
    }
    sampled = [i for i in items if (i.repository_id, i.question_id) in chosen]
    if not sampled:
        return "", "[]\n", []
    order = sorted(sampled, key=lambda i: (i.repository_id, i.question_id))
    random.Random(f"rule-worksheet|{seed}").shuffle(order)
    rows, key = [], []
    for number, it in enumerate(order, start=1):
        item_id = f"{RULE_ITEM_PREFIX}{number:04d}"
        rows.append(
            {
                "item_id": item_id,
                **({"audit_status": status_label} if status_label else {}),
                "repository_id": it.repository_id,
                "question": it.question,
                "reference": it.reference,
                "system_message": it.system_message,
                "system_reason": it.reason,
                "retrieved_chunks": it.retrieved_count,
                **dict.fromkeys(RULE_HUMAN_FIELDS, ""),
            }
        )
        key.append({"item_id": item_id, "repository_id": it.repository_id,
                    "question_id": it.question_id})  # fmt: skip
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue(), json.dumps(key, indent=2) + "\n", sampled


def _yes_no(value: str, name: str, item_id: str) -> bool:
    text = (value or "").strip().lower()
    if text not in {"yes", "no"}:
        raise ValueError(f"{item_id}: {name} must be yes or no")
    return text == "yes"


def evaluate_rule_check(
    rows: Sequence[Mapping[str, str]], key: Sequence[Mapping[str, str]]
) -> dict[str, object]:
    """Counts for a filled rule worksheet. Nothing is skipped silently."""
    known = {e["item_id"] for e in key}
    seen: dict[str, int] = {}
    for row in rows:
        seen[row.get("item_id", "")] = seen.get(row.get("item_id", ""), 0) + 1
    duplicates = sorted(i for i, n in seen.items() if n > 1)
    unexpected = sorted(i for i in seen if i not in known)
    missing = sorted(known - set(seen))
    confirmed, disputed, unrated, invalid = [], [], [], []
    for row in rows:
        item_id = row.get("item_id", "")
        if item_id in duplicates or item_id not in known:
            continue
        values = [row.get(f, "").strip() for f in RULE_HUMAN_FIELDS[:2]]
        if not any(values):
            unrated.append(item_id)
            continue
        try:
            no_claims = _yes_no(values[0], RULE_HUMAN_FIELDS[0], item_id)
            genuine = _yes_no(values[1], RULE_HUMAN_FIELDS[1], item_id)
        except ValueError as exc:
            invalid.append({"item_id": item_id, "reason": str(exc)})
            continue
        (confirmed if no_claims and genuine else disputed).append(item_id)
    rated = len(confirmed) + len(disputed)
    return {
        "kind": "rule-item verification (NOT judge-human agreement)",
        "key_items": len(known),
        "worksheet_rows": len(rows),
        "rated": rated,
        "unrated": sorted(unrated),
        "invalid_ratings": invalid,
        "duplicate_item_ids": duplicates,
        "unexpected_item_ids": unexpected,
        "key_items_missing_from_worksheet": missing,
        "rule_confirmed": len(confirmed),
        "rule_disputed": sorted(disputed),
        "rule_confirmed_rate": len(confirmed) / rated if rated else None,
        "rule_confirmed_rate_note": None if rated else "no rated rows",
        "complete": not (invalid or duplicates or unexpected or missing or unrated),
    }
