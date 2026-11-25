"""Abstention and refusal metrics.

Whether an answer *abstains* is a separate question from whether it is *correct*:

* ``insufficient_evidence``  the answer says the repository/evidence/files do not show the
  thing asked
  (a specific, informative abstention), or the RAG system itself stopped because retrieval found no
  usable evidence (the model was never called);
* ``generic_refusal``        a refusal that is not about evidence ("I can't help with that");
* ``ambiguous``              the text both abstains and makes cited, substantive claims (a partial
  answer such as "[Source 2] shows X; the evidence does not show Y") - reported, never forced into
  either class;
* ``answered``               no abstention cue;
* ``no_response``            there is no answer to classify: the call failed, was blocked by the
  provider, was never attempted, or returned nothing. **A missing response is not a refusal** and is
  excluded from every rate.

The cue-based classifier is a deterministic first pass with known limits (paraphrased abstentions
are missed; an answer that merely *mentions* insufficiency can be mislabelled). **It is a
diagnostic only.** The primary abstention metrics label a generated response from the judge's
``response_type`` (prompt ``gen-judge-2``), a system abstention deterministically, and a response
with no valid judge rating as ``unrated`` - the cue rules are never substituted for a missing
rating. The cue-rule metrics, their disagreements with the judge and the unrated count are
reported next to the primary metrics (see ``analysis``).

Rates (each ``None`` when its denominator is zero):

* ``appropriate_abstention_rate``  unanswerable questions answered with ``insufficient_evidence``;
* ``generic_refusal_rate_on_unanswerable``  unanswerable questions met with a generic refusal;
* ``unsupported_answer_rate``      unanswerable questions that were ``answered``;
* ``false_refusal_rate``           answerable questions met with ``insufficient_evidence`` or a
  ``generic_refusal``;
* ``ambiguous`` is counted separately for both answerabilities.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from copilot.rag.citations import extract_source_numbers

Label = Literal[
    "answered", "insufficient_evidence", "generic_refusal", "ambiguous", "no_response", "unrated"
]
State = Literal["successful", "failed", "not_attempted"]

_INSUFFICIENT = [
    re.compile(p, re.I)
    for p in (
        r"\b(?:evidence|sources?|context|excerpts?|files?|code|repository|retrieved|provided|supplied)\b"
        r"[^.\n]{0,60}\b(?:does|do|did)\s+not\s+(?:show|contain|include|mention|provide|specify|state|cover|describe|reveal|say)",
        r"\b(?:no|insufficient|not\s+enough|limited)\s+(?:repository\s+)?(?:evidence|information)\b",
        r"\b(?:cannot|can't|can\s+not|unable\s+to|not\s+able\s+to|couldn't)\s+"
        r"(?:determine|tell|find|confirm|verify|answer|say|know|see)\b[^.\n]{0,100}"
        r"\b(?:repository|evidence|sources?|files?|context|provided|code)\b",
        r"\b(?:do\s+not|don't|does\s+not|doesn't)\s+have\s+(?:access\s+to|visibility\s+(?:into|of)|the)\b"
        r"[^.\n]{0,60}\b(?:repository|code|files?|source)\b",
        r"\bnot\s+(?:shown|present|found|specified|documented|mentioned|visible)\s+in\s+(?:the\s+)?"
        r"(?:repository|provided|supplied|retrieved|evidence|sources?|code|files?)\b",
        r"\bwithout\s+(?:seeing|access\s+to|the)\s+(?:the\s+)?(?:repository|code|files?|source)\b",
        r"\binsufficient\s+(?:repository\s+)?evidence\b",
        r"\bnothing\s+in\s+(?:the\s+)?(?:repository|code(?:base)?|files?|sources?|evidence|"
        r"retrieved\s+\w+|provided\s+\w+)\s+(?:shows?|says?|states?|mentions?|indicates?|"
        r"describes?|specif(?:y|ies)|reveals?)\b",
        r"\bno\s+(?:mention|indication|reference|record|trace|sign|information)\s+of\b[^.\n]{0,80}"
        r"\b(?:in|within)\s+(?:the\s+)?(?:repository|code(?:base)?|files?|sources?|evidence|"
        r"retrieved|provided)\b",
    )
]
_GENERIC_REFUSAL = [
    re.compile(p, re.I)
    for p in (
        r"\bI\s+(?:cannot|can't|can\s+not|am\s+unable\s+to|won't|will\s+not)\s+(?:help|assist|comply|do\s+that|answer\s+that|provide\s+that)\b",
        r"\bI'?m\s+sorry,?\s+(?:but\s+)?I\s+(?:cannot|can't)\b",
        r"\bagainst\s+my\s+(?:guidelines|policy|policies)\b",
    )
]


@dataclass(frozen=True)
class AbstentionLabel:
    label: Label
    source: Literal["system", "text", "none"]  # who decided: the RAG system, the cue rules, nothing


def classify_response(
    answer: str | None, *, state: State = "successful", system_insufficient: bool = False
) -> AbstentionLabel:
    """Classify one arm's response for one question (see the module docstring)."""
    if system_insufficient:
        return AbstentionLabel("insufficient_evidence", "system")
    if state != "successful" or answer is None or not answer.strip():
        return AbstentionLabel("no_response", "none")
    insufficient = any(p.search(answer) for p in _INSUFFICIENT)
    generic = any(p.search(answer) for p in _GENERIC_REFUSAL)
    if not insufficient and not generic:
        return AbstentionLabel("answered", "text")
    substantive = bool(extract_source_numbers(answer)) or bool(re.search(r"\.\w{1,4}:\d+", answer))
    if substantive and insufficient:
        return AbstentionLabel("ambiguous", "text")
    return AbstentionLabel("insufficient_evidence" if insufficient else "generic_refusal", "text")


@dataclass(frozen=True)
class AbstentionMetrics:
    answerable_total: int
    unanswerable_total: int
    answerable_evaluable: int
    unanswerable_evaluable: int
    no_response_answerable: int
    no_response_unanswerable: int
    appropriate_abstentions: int  # unanswerable + insufficient_evidence
    generic_refusals_on_unanswerable: int
    unsupported_answers: int  # unanswerable + answered
    false_refusals: int  # answerable + (insufficient_evidence | generic_refusal)
    ambiguous_answerable: int
    ambiguous_unanswerable: int
    unrated_answerable: int = 0  # text exists but no valid judge rating: never classified
    unrated_unanswerable: int = 0

    @staticmethod
    def _rate(num: int, den: int) -> float | None:
        return num / den if den else None

    @property
    def appropriate_abstention_rate(self) -> float | None:
        return self._rate(self.appropriate_abstentions, self.unanswerable_evaluable)

    @property
    def generic_refusal_rate_on_unanswerable(self) -> float | None:
        return self._rate(self.generic_refusals_on_unanswerable, self.unanswerable_evaluable)

    @property
    def unsupported_answer_rate(self) -> float | None:
        return self._rate(self.unsupported_answers, self.unanswerable_evaluable)

    @property
    def false_refusal_rate(self) -> float | None:
        return self._rate(self.false_refusals, self.answerable_evaluable)

    def to_json(self) -> dict[str, float | int | None]:
        data: dict[str, float | int | None] = dict(self.__dict__)
        for name in (
            "appropriate_abstention_rate",
            "generic_refusal_rate_on_unanswerable",
            "unsupported_answer_rate",
            "false_refusal_rate",
        ):
            data[name] = getattr(self, name)
        return data


def abstention_metrics(items: Iterable[tuple[bool, Label]]) -> AbstentionMetrics:
    """Aggregate ``(question_is_answerable, label)`` pairs into the metrics above.

    ``no_response`` and ``unrated`` items are counted but excluded from every rate denominator.
    """
    counts = dict.fromkeys(AbstentionMetrics.__dataclass_fields__, 0)
    for answerable, label in items:
        counts["answerable_total" if answerable else "unanswerable_total"] += 1
        if label == "no_response":
            counts["no_response_answerable" if answerable else "no_response_unanswerable"] += 1
            continue
        if label == "unrated":
            counts["unrated_answerable" if answerable else "unrated_unanswerable"] += 1
            continue
        counts["answerable_evaluable" if answerable else "unanswerable_evaluable"] += 1
        if label == "ambiguous":
            counts["ambiguous_answerable" if answerable else "ambiguous_unanswerable"] += 1
        elif answerable and label in ("insufficient_evidence", "generic_refusal"):
            counts["false_refusals"] += 1
        elif not answerable and label == "insufficient_evidence":
            counts["appropriate_abstentions"] += 1
        elif not answerable and label == "generic_refusal":
            counts["generic_refusals_on_unanswerable"] += 1
        elif not answerable and label == "answered":
            counts["unsupported_answers"] += 1
    return AbstentionMetrics(**counts)
