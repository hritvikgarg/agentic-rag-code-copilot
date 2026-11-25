"""A static inventory of a repository: which files, modules and names really exist.

Built from the files that ingestion accepted, using only ``ast.parse`` and string splitting. **No
repository code is imported or executed.** The inventory answers questions such as "is there a
file called ``src/x/y.py``", "how many lines does it have", "is ``place_order`` defined or used
anywhere" and is the ground truth for the citation checker and the hallucinated-reference detector.

Limits (documented because they decide how the detector may be read): only Python is parsed for
symbols; names inside strings, comments and non-Python files are only seen as plain *tokens*
(``text_tokens``); dynamic features (``getattr``, generated classes, star imports) are invisible.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from copilot.chunking.windows import split_lines
from copilot.models.ingestion import SourceFile

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class RepoInventory:
    repository_id: str
    paths: frozenset[str]
    line_counts: dict[str, int]
    basenames: dict[str, tuple[str, ...]]  # "config.py" -> every path with that file name
    modules: frozenset[str]  # dotted module names and their dotted suffixes (>= 2 parts)
    names: frozenset[str]  # defined, used-attribute, parameter and variable names
    defined: dict[str, tuple[str, ...]]  # class/function/method name -> files defining it
    qualified: frozenset[str]  # "Class.method" and "Class.attr"
    text_tokens: frozenset[str]  # every identifier-like token in any ingested file
    imported_roots: frozenset[str]  # top-level modules imported by repository code
    parsed_python_files: int = 0
    unparsable: tuple[str, ...] = field(default_factory=tuple)

    def line_count(self, path: str) -> int | None:
        return self.line_counts.get(path)


def _module_names(path: str) -> list[str]:
    if not path.endswith(".py"):
        return []
    parts = path[:-3].split("/")
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return []
    names = []
    for i in range(len(parts)):
        suffix = parts[i:]
        if len(suffix) >= 2:
            names.append(".".join(suffix))
    return names


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.names: set[str] = set()
        self.defined: set[str] = set()
        self.qualified: set[str] = set()
        self.imports: set[str] = set()
        self._classes: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defined.add(node.name)
        self.names.add(node.name)
        self._classes.append(node.name)
        self.generic_visit(node)
        self._classes.pop()

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.defined.add(node.name)
        self.names.add(node.name)
        if self._classes:
            self.qualified.add(f"{self._classes[-1]}.{node.name}")
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            self.names.add(arg.arg)
        for extra in (node.args.vararg, node.args.kwarg):
            if extra is not None:
                self.names.add(extra.arg)
        self.generic_visit(node)

    visit_FunctionDef = _function  # noqa: N815
    visit_AsyncFunctionDef = _function  # noqa: N815

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)
        if isinstance(node.ctx, ast.Store) and self._classes:
            self.qualified.add(f"{self._classes[-1]}.{node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.names.add(node.attr)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name.split(".")[0])
            self.names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module:
            self.imports.add(node.module.split(".")[0])
        for alias in node.names:
            self.names.add(alias.asname or alias.name)


def build_inventory(files: Sequence[SourceFile], repository_id: str) -> RepoInventory:
    """Inventory of ``files`` (the ingested, accepted files of one repository)."""
    paths: set[str] = set()
    line_counts: dict[str, int] = {}
    basenames: dict[str, list[str]] = {}
    modules: set[str] = set()
    names: set[str] = set()
    defined: dict[str, set[str]] = {}
    qualified: set[str] = set()
    tokens: set[str] = set()
    imports: set[str] = set()
    unparsable: list[str] = []
    parsed = 0
    for file in files:
        path = file.relative_path
        paths.add(path)
        line_counts[path] = len(split_lines(file.content))
        basenames.setdefault(path.rsplit("/", 1)[-1], []).append(path)
        modules.update(_module_names(path))
        tokens.update(_TOKEN_RE.findall(file.content))
        if file.language != "python":
            continue
        try:
            tree = ast.parse(file.content)
        except (SyntaxError, ValueError, RecursionError):
            unparsable.append(path)
            continue
        parsed += 1
        visitor = _Visitor(path)
        visitor.visit(tree)
        names |= visitor.names
        qualified |= visitor.qualified
        imports |= visitor.imports
        for name in visitor.defined:
            defined.setdefault(name, set()).add(path)
    return RepoInventory(
        repository_id=repository_id,
        paths=frozenset(paths),
        line_counts=line_counts,
        basenames={k: tuple(sorted(v)) for k, v in basenames.items()},
        modules=frozenset(modules),
        names=frozenset(names),
        defined={k: tuple(sorted(v)) for k, v in defined.items()},
        qualified=frozenset(qualified),
        text_tokens=frozenset(tokens),
        imported_roots=frozenset(imports),
        parsed_python_files=parsed,
        unparsable=tuple(sorted(unparsable)),
    )
