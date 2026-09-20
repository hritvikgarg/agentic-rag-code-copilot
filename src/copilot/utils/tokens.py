"""Cheap, dependency-free token estimation.

**What:** an estimate of how many model tokens a piece of text would use.
**Why:** chunks must never exceed an embedding model's input limit, and we want a size measure
that does not require downloading a tokenizer (that arrives with the embedding model in
Milestone 4, where the estimate is validated against the real tokenizer).
**How:** count *ASCII word runs* (letters, digits, underscore) and *every other non-space
character individually* (punctuation, and each non-ASCII character, so CJK text or accented
identifiers are not badly under-counted). Real BPE/WordPiece tokenizers split long identifiers into
several pieces, so this is a middle estimate for identifiers and conservative for non-ASCII text.
It is a heuristic; the configured cap is a safety budget, not an exact guarantee.

The estimate is **additive over lines**: because a newline is whitespace and is never counted,
``estimate_tokens(a + "\\n" + b) == estimate_tokens(a) + estimate_tokens(b)``. The chunkers rely on
this to build windows from per-line counts in linear time.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def estimate_tokens(text: str) -> int:
    """Estimated token count of ``text`` (0 for empty or whitespace-only text)."""
    return sum(1 for _ in _TOKEN_RE.finditer(text))


def token_starts(text: str) -> Iterator[int]:
    """Yield the character offset at which each estimated token starts."""
    for match in _TOKEN_RE.finditer(text):
        yield match.start()
