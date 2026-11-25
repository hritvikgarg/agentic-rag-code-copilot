"""Conservative detection of references an answer makes that the repository does not support.

**What this is:** a screen for *invented repository references*: file paths, Python symbols and
dependency names that an answer presents as part of the repository but that the static inventory
(``inventory.py``) cannot find. Each reference gets one verdict:

* ``verified``       the repository contains it (path exists, name defined/used, import present);
* ``not_found``      it is presented as a repository fact and nothing in the inventory matches;
* ``ambiguous``      it matches several repository files (for example a bare ``config.py``);
* ``cannot_evaluate`` it is hypothetical, suggested, an example, or the inventory cannot judge it
  (no parsed Python for a symbol, a dependency claim without a usage cue);
* ``ignored``        a standard-library module, a Python builtin or a common generic word - never
  counted as a repository claim.

**What it deliberately is not:** a hallucination detector. It misses invented *behaviour*
(a real function described wrongly) and invented names that happen to occur elsewhere in the
repository (a name found only in a comment is ``verified``). It can wrongly flag a real name that
the repository builds dynamically, or a paraphrase in backticks. Suggested new files and example
code are not flagged: a sentence containing a cue such as "create", "add a new", "for example",
"such as", "you could" or "should" marks its references ``cannot_evaluate``. External dependencies
(pytest, numpy) are never flagged unless the sentence claims the *repository* uses them.

Nothing here executes or imports repository code.
"""

from __future__ import annotations

import builtins
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from copilot.evaluation.generation.inventory import RepoInventory

Verdict = Literal["verified", "not_found", "ambiguous", "cannot_evaluate", "ignored"]
Kind = Literal["path", "symbol", "dependency"]

_PATH_EXT = r"py|pyi|md|json|yaml|yml|toml|txt|js|jsx|ts|tsx|java|c|cpp|h|cfg|ini"
_PATH_RE = re.compile(
    rf"(?<![\w/.:-])((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:{_PATH_EXT}))(?![\w/-])"
)
_BACKTICK_RE = re.compile(r"`([^`\n]{1,120})`")
_IDENT_CHAIN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_CALL_RE = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\(")
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))", re.M
)
_PIP_RE = re.compile(r"\bpip3?\s+install\s+([A-Za-z0-9_.\-]+)")
_USES_RE = re.compile(
    r"\b(?:uses|imports|depends on|relies on|requires|built on)\s+(?:the\s+)?`([A-Za-z_][\w.\-]*)`",
    re.I,
)
_HYPOTHETICAL = re.compile(
    r"\b(create|creating|add(?:ing)? a new|new file|you could|you can|you should|should|would|"
    r"might|for example|e\.g\.|such as|suggest|consider|hypothetical|imagine|something like|"
    r"for instance|i would|if you)\b",
    re.I,
)
_GENERIC = frozenset(
    {"self", "cls", "args", "kwargs", "none", "true", "false", "init", "main", "test", "tests",
     "src", "data", "value", "values", "name", "key", "keys", "item", "items", "type", "id",
     "path", "file", "files", "text", "result", "results", "error", "errors", "default",
     "e", "x", "y"}
)  # fmt: skip
_STDLIB = frozenset(sys.stdlib_module_names)
_BUILTINS = frozenset(dir(builtins))
_FENCE_RE = re.compile(r"```.*?```", re.S)


class Reference(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Kind
    text: str
    verdict: Verdict
    reason: str


@dataclass(frozen=True)
class ReferenceReport:
    references: tuple[Reference, ...]

    def count(self, verdict: Verdict, kind: Kind | None = None) -> int:
        return sum(
            1 for r in self.references if r.verdict == verdict and (kind is None or r.kind == kind)
        )

    @property
    def counted(self) -> int:
        """References that took part in the rates (everything except ``ignored``)."""
        return sum(1 for r in self.references if r.verdict != "ignored")

    @property
    def not_found(self) -> int:
        return self.count("not_found")

    @property
    def verified_fraction(self) -> float | None:
        """verified / (verified + not_found + ambiguous); ``None`` when nothing was checkable."""
        denominator = self.count("verified") + self.count("not_found") + self.count("ambiguous")
        return self.count("verified") / denominator if denominator else None

    def summary(self) -> dict[str, int]:
        counter: Counter[str] = Counter(r.verdict for r in self.references)
        return {
            v: counter.get(v, 0)
            for v in ("verified", "not_found", "ambiguous", "cannot_evaluate", "ignored")
        }


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p for p in parts if p.strip()]


def _is_hypothetical(sentence: str) -> bool:
    return bool(_HYPOTHETICAL.search(sentence))


def _check_path(raw: str, inv: RepoInventory) -> tuple[Verdict, str]:
    path = raw[2:] if raw.startswith("./") else raw
    if path in inv.paths:
        return "verified", "path exists in the repository"
    base = path.rsplit("/", 1)[-1]
    matches = [p for p in inv.basenames.get(base, ()) if p.endswith(path) or "/" not in path]
    if len(matches) == 1:
        return "verified", f"unique file with this name: {matches[0]}"
    if len(matches) > 1:
        return "ambiguous", f"{len(matches)} files match this name"
    return "not_found", "no such file in the repository"


def _check_symbol(raw: str, inv: RepoInventory) -> tuple[Verdict, str]:
    text = raw.removesuffix("()")
    if text.startswith("self."):
        text = text[5:]
    parts = text.split(".")
    root = parts[0]
    if root.lower() in _GENERIC or root in _BUILTINS and len(parts) == 1:
        return "ignored", "generic word or builtin"
    if root in _STDLIB or (root in _BUILTINS and root not in inv.names):
        return "ignored", "standard-library or builtin name"
    if inv.parsed_python_files == 0:
        return "cannot_evaluate", "the repository has no parsed Python files"
    if text in inv.modules or ".".join(parts[-2:]) in inv.qualified or text in inv.qualified:
        return "verified", "module or Class.member defined in the repository"
    if all(p in inv.names or p in inv.text_tokens for p in parts):
        return "verified", "every part appears in the repository"
    missing = [p for p in parts if p not in inv.names and p not in inv.text_tokens]
    return "not_found", f"not in the repository: {', '.join(missing)}"


def _check_dependency(name: str, inv: RepoInventory, claimed_use: bool) -> tuple[Verdict, str]:
    root = name.split(".")[0].replace("-", "_")
    if root in _STDLIB:
        return "ignored", "standard-library module"
    if root in inv.imported_roots:
        return "verified", "imported by repository code"
    if root in inv.modules or any(m.startswith(root + ".") for m in inv.modules):
        return "verified", "a module of the repository itself"
    if claimed_use:
        return (
            "not_found",
            "the answer says the repository uses it, but no repository file imports it",
        )
    return "cannot_evaluate", "an external package mentioned without a repository-usage claim"


def find_references(answer: str, inventory: RepoInventory) -> ReferenceReport:
    """Every file path, symbol and dependency reference in ``answer`` with its verdict."""
    results: list[Reference] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: Kind, text: str, verdict: Verdict, reason: str) -> None:
        if (kind, text) in seen:
            return
        seen.add((kind, text))
        results.append(Reference(kind=kind, text=text, verdict=verdict, reason=reason))

    fenced = _FENCE_RE.findall(answer)
    prose = _FENCE_RE.sub(" ", answer)

    for sentence in _sentences(prose):
        hypothetical = _is_hypothetical(sentence)
        for match in _PATH_RE.finditer(sentence):
            raw = match.group(1).strip(".")
            if (
                raw.startswith(("http", "www."))
                or "://" in sentence[max(0, match.start() - 8) : match.start()]
            ):
                continue
            if hypothetical:
                add("path", raw, "cannot_evaluate", "hypothetical or suggested in this sentence")
            else:
                verdict, reason = _check_path(raw, inventory)
                add("path", raw, verdict, reason)
        candidates: list[str] = []
        for match in _BACKTICK_RE.finditer(sentence):
            token = match.group(1).strip()
            head = re.split(r"[\s(=\[:,]", token, maxsplit=1)[0]
            if _PATH_RE.fullmatch(head) or not _IDENT_CHAIN_RE.match(head):
                continue
            candidates.append(head)
        candidates += [
            m.group(1)
            for m in _CALL_RE.finditer(sentence)
            if "_" in m.group(1) or m.group(1)[:1].isupper()
        ]
        for token in candidates:
            if hypothetical:
                if (
                    token.split(".")[0] not in _STDLIB
                    and token.split(".")[0].lower() not in _GENERIC
                ):
                    add(
                        "symbol",
                        token,
                        "cannot_evaluate",
                        "hypothetical or suggested in this sentence",
                    )
                continue
            verdict, reason = _check_symbol(token, inventory)
            add("symbol", token, verdict, reason)
        for match in _USES_RE.finditer(sentence):
            verdict, reason = _check_dependency(
                match.group(1), inventory, claimed_use=not hypothetical
            )
            add("dependency", match.group(1), verdict, reason)

    for block in [*fenced, prose]:
        for match in _IMPORT_RE.finditer(block):
            module = match.group(1) or match.group(2)
            verdict, reason = _check_dependency(module, inventory, claimed_use=False)
            add("dependency", module.split(".")[0], verdict, reason)
        for match in _PIP_RE.finditer(block):
            verdict, reason = _check_dependency(match.group(1), inventory, claimed_use=False)
            add("dependency", match.group(1), verdict, reason)
    return ReferenceReport(tuple(results))
