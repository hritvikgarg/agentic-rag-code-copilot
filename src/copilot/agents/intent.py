"""Deterministic intent router: which kind of repository question is this? (no LLM call)

Five intents. The rules are regular expressions over the question, evaluated in a **fixed
precedence order**; the first intent in that order with a matching rule wins::

    1. debugging      a reported or asked-about problem (errors, crashes, "why does X fail")
    2. symbol_lookup  a locational question ("where is X defined", "which files implement ...")
    3. architecture   structure and relationships ("architecture", "overview", "flow from A to B")
    4. code_explain   an explanation request ("explain ...", "how does X work")
    5. general        everything else

Why this order. Debugging and locational questions often *contain* explanation words ("explain why
it crashes", "where is X explained"), and the narrower, more actionable reading should win, so the
broad ``code_explain`` cues come last. ``architecture`` beats ``code_explain`` so that "Explain the
architecture of this repository" is routed as an architecture question.

Ambiguity. A question with no content word and no identifier ("What does this do?", "help") has no
subject to retrieve for, and a question with no matching rule has no recognised shape; both are
*ambiguous* and fall back to ``general``. ``IntentDecision.matched`` lists every intent that had
a matching rule, so an overlap is visible in the trace instead of silent.

Debugging routing only selects a retrieval strategy here; the evidence-based debugging assistant
is a later milestone.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from copilot.agents.terms import key_terms


class Intent(StrEnum):
    CODE_EXPLAIN = "code_explain"
    SYMBOL_LOOKUP = "symbol_lookup"
    ARCHITECTURE = "architecture"
    DEBUGGING = "debugging"
    GENERAL = "general"


class IntentDecision(BaseModel):
    """The router's output: the intent, the rule that fired, and what else matched."""

    model_config = ConfigDict(frozen=True)

    intent: Intent
    rule_id: str  # e.g. "debugging.problem-words", or "general.fallback" / "general.too-vague"
    matched: tuple[Intent, ...]  # every intent with a matching rule, in precedence order
    symbols: tuple[str, ...]  # identifier-like tokens found in the question (never invented)

    @property
    def symbol(self) -> str | None:
        return self.symbols[0] if self.symbols else None


def _rx(pattern: str, *, case_sensitive: bool = False) -> re.Pattern[str]:
    return re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)


# (rule id, pattern). Order inside an intent does not matter; precedence is between intents.
_RULES: tuple[tuple[Intent, tuple[tuple[str, re.Pattern[str]], ...]], ...] = (
    (
        Intent.DEBUGGING,
        (
            (
                "problem-words",
                _rx(
                    r"\b(traceback|stack ?trace|segfault|crash(?:es|ed|ing)?|bug(?:s|gy)?|"
                    r"broken|panic(?:s|ked)?|not working|doesn'?t work|does not work|"
                    r"fails?|failed|failing|failure|timed? ?out|hangs?|hanging|"
                    r"stuck|unexpected(?:ly)?)\b"
                ),
            ),
            (
                "reported-error",
                _rx(
                    r"\b(get(?:ting)?|got|see(?:ing)?|throws?|raises?|returns?|hits?|shows?|"
                    r"causes?)\s+(?:an?\s+|the\s+)?(?:\w+\s+)?(?:error|exception)\b"
                ),
            ),
            ("error-class", _rx(r"\b[A-Z][A-Za-z]*(?:Error|Exception)\b", case_sensitive=True)),
            ("http-status", _rx(r"\b(?:http\s*)?[45]\d\d\b")),
            (
                "why-wrong",
                _rx(
                    r"\bwhy\s+(?:does|is|do|did|might|would|am|are|can't|cannot|won't)\b.*"
                    r"\b(?:return|returns|raise|raises|throw|throws|wrong|empty|none|null|slow|"
                    r"missing|incorrect)\b"
                ),
            ),
        ),
    ),
    (
        Intent.SYMBOL_LOOKUP,
        (
            ("where-is", _rx(r"^\s*where(?:'s|\s+is|\s+are|\s+does|\s+do|\s+can|\s+should)\b")),
            (
                "where-verb",
                _rx(
                    r"\bwhere\b.*\b(?:defined|implemented|configured|declared|located|used|"
                    r"called|set up|setup|handled|stored|created|registered|loaded|initiali[sz]ed|"
                    r"found|imported)\b"
                ),
            ),
            (
                "which-files",
                _rx(
                    r"\bwhich\s+(?:files?|functions?|classes|class|modules?|methods?)\b.*"
                    r"\b(?:implement|define|handle|contain|use|call|configure|import|"
                    r"responsible)"
                ),
            ),
            (
                "find-symbol",
                _rx(
                    r"\b(?:find|locate|show me|list)\b.*\b(?:function|class|method|file|module|"
                    r"definition|usages?|implementation|callers?)\b"
                ),
            ),
            ("who-uses", _rx(r"\b(?:who|what)\s+(?:calls|uses|imports|references)\b")),
            ("definition-of", _rx(r"\bdefinition of\b")),
        ),
    ),
    (
        Intent.ARCHITECTURE,
        (
            (
                "architecture-words",
                _rx(
                    r"\b(?:architecture|architectural|high[- ]level|overview|big picture|"
                    r"system design|design of|(?:project|repository|repo|codebase|folder|"
                    r"directory) (?:structure|layout)|code ?base layout)\b"
                ),
            ),
            (
                "components-interact",
                _rx(
                    r"\b(?:components?|modules?|layers?|subsystems?|services?|packages?)\b.*"
                    r"\b(?:interact|communicate|relate|depend|fit together|organi[sz]ed|"
                    r"structured|connected)\b"
                ),
            ),
            (
                "flow-between",
                _rx(
                    r"\b(?:data flow|request flow|control flow|flow|pipeline|end[- ]to[- ]end)\b"
                    r".*\b(?:from|through|between|across|to)\b"
                ),
            ),
            (
                "how-organized",
                _rx(
                    r"\bhow\s+is\s+(?:the\s+)?(?:repo|repository|project|codebase|code base|"
                    r"application|app|system)\s+(?:organi[sz]ed|structured|laid out)\b"
                ),
            ),
        ),
    ),
    (
        Intent.CODE_EXPLAIN,
        (
            (
                "explain-words",
                _rx(
                    r"\b(?:explain|walk me through|walkthrough|describe|summari[sz]e|"
                    r"understand)\b"
                ),
            ),
            (
                "what-does",
                _rx(r"\bwhat\s+(?:does|is|do)\b.*\b(?:do|does|mean|for|purpose|return|returns)\b"),
            ),
            (
                "how-does",
                _rx(
                    r"\bhow\s+(?:does|do|is|are)\b.*\b(?:work|works|handle|handles|implemented|"
                    r"compute|computed|calculate|calculated|process|processed|built|made|"
                    r"generated|created)\b"
                ),
            ),
            ("purpose-of", _rx(r"\bpurpose of\b")),
        ),
    ),
)

PRECEDENCE: tuple[Intent, ...] = tuple(intent for intent, _ in _RULES) + (Intent.GENERAL,)

# Identifier-like tokens, most specific first. Only text from the question is ever matched.
_BACKTICKED = re.compile(r"`([^`\n]{1,60})`")
_FILENAME = re.compile(
    r"(?<![\w./-])[\w\-./]*[\w\-]\.(?:py|jsx?|tsx?|java|cpp|c|h|md|json|ya?ml)\b", re.IGNORECASE
)
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*)\(\)")
_SNAKE = re.compile(r"\b_{0,2}[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+_{0,2}\b")
_PASCAL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
_CAMEL = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b")
_DOTTED = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_MAX_SYMBOLS = 5


def extract_symbols(question: str) -> tuple[str, ...]:
    """Identifier-like tokens in the question: backticked, file names, ``f()``, snake/Pascal/camel
    case and dotted names. At most 5, unique, in that priority order. Nothing is invented."""
    found: dict[str, None] = {}

    def add(raw: str) -> None:
        token = raw.strip().rstrip("()").strip(".,;:!?")
        if 2 <= len(token) <= 60:
            found.setdefault(token, None)

    for match in _BACKTICKED.finditer(question):
        add(match.group(1))
    for match in _FILENAME.finditer(question):
        add(match.group(0))
    for match in _CALL.finditer(question):
        add(match.group(1))
    for pattern in (_SNAKE, _PASCAL, _CAMEL):
        for match in pattern.finditer(question):
            add(match.group(0))
    for match in _DOTTED.finditer(question):
        if not _FILENAME.fullmatch(match.group(0)):
            add(match.group(0))
    return tuple(found)[:_MAX_SYMBOLS]


def classify_intent(question: str) -> IntentDecision:
    """Route ``question`` (already validated and stripped); see the module docstring."""
    symbols = extract_symbols(question)
    matched: list[Intent] = []
    first_rule: dict[Intent, str] = {}
    for intent, rules in _RULES:
        for rule_id, pattern in rules:
            if pattern.search(question):
                matched.append(intent)
                first_rule[intent] = f"{intent.value}.{rule_id}"
                break
    if not key_terms(question) and not symbols:
        return IntentDecision(
            intent=Intent.GENERAL, rule_id="general.too-vague", matched=tuple(matched), symbols=()
        )
    if not matched:
        return IntentDecision(
            intent=Intent.GENERAL, rule_id="general.fallback", matched=(), symbols=symbols
        )
    winner = matched[0]  # _RULES is already in precedence order
    return IntentDecision(
        intent=winner, rule_id=first_rule[winner], matched=tuple(matched), symbols=symbols
    )
