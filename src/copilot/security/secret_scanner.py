"""Content-based secret scanner (the mandatory pre-LLM security gate's detector).

Filename filtering (ingestion) cannot see a key pasted into ``settings.py``. This module reads
the *content* of text that could be sent to an external LLM and reports potential secrets as
``SecretFinding`` objects that carry only safe metadata (rule id, relative path, line, column,
masked preview). The matched value itself is never stored, logged or raised.

Public API::

    scan_text(text, path)               -> list[SecretFinding]
    scan_source_file(source_file)       -> list[SecretFinding]
    scan_targets(targets)               -> SecretScanReport   (files, chunks, retrieved results)
    scan_repository(files)              -> SecretScanReport   (ingested SourceFile objects)

Scanning is deterministic: findings are sorted by ``(path, line, column, rule_id)`` and
overlapping matches on the same characters are reported once (the most specific rule wins).
This reduces risk; it cannot guarantee that every secret is found (see ``docs/security.md``).
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from copilot.models.ingestion import SourceFile
from copilot.security import patterns as p
from copilot.security.models import (
    ScanTarget,
    SecretFinding,
    SecretScanReport,
    Severity,
    redact,
)

logger = logging.getLogger(__name__)

MAX_LINE_CHARS = 20_000  # longer lines are scanned in overlapping windows (bounds regex work)
WINDOW_OVERLAP = 512  # larger than any token format, so a token on a window edge is still seen
_PRIVATE_KEY_LOOKAHEAD_LINES = 2


@dataclass(frozen=True)
class _Candidate:
    line_index: int  # 0-based index into the target's lines
    start: int  # 0-based column
    end: int
    priority: int  # lower wins when two candidates overlap
    rule_id: str
    severity: Severity
    reason: str
    value: str  # transient; only ever passed to ``redact``


def _split_lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _windows(line: str) -> Iterator[tuple[int, str]]:
    """Yield ``(offset, segment)``; a normal line is one segment."""
    if len(line) <= MAX_LINE_CHARS:
        yield 0, line
        return
    step = MAX_LINE_CHARS - WINDOW_OVERLAP
    for offset in range(0, len(line), step):
        yield offset, line[offset : offset + MAX_LINE_CHARS]
        if offset + MAX_LINE_CHARS >= len(line):
            break


def _has_key_material(lines: list[str], index: int, remainder: str) -> bool:
    if p.KEY_MATERIAL.search(remainder):
        return True  # e.g. a JSON string: "-----BEGIN ...-----\nMIIE..."
    for following in lines[index + 1 : index + 1 + _PRIVATE_KEY_LOOKAHEAD_LINES]:
        stripped = following.strip()
        if stripped.startswith(p.PEM_HEADER_FIELDS) or p.KEY_MATERIAL.fullmatch(stripped):
            return True
    return False


def _private_keys(lines: list[str], index: int, segment: str, offset: int) -> Iterator[_Candidate]:
    for match in p.PRIVATE_KEY_HEADER.finditer(segment):
        has_body = _has_key_material(lines, index, segment[match.end() :])
        rule_id = "private-key-block" if has_body else "private-key-header"
        severity, reason = p.PRIVATE_KEY_RULES[rule_id]
        yield _Candidate(
            index, offset + match.start(), offset + match.end(), 0, rule_id, severity, reason,
            match.group(0),
        )  # fmt: skip


def _patterns(index: int, segment: str, offset: int) -> Iterator[_Candidate]:
    for priority, rule in enumerate(p.PATTERN_RULES, start=1):
        for match in rule.regex.finditer(segment):
            group = rule.group or 0
            value = match.group(group)
            if not rule.accept(value):
                continue
            yield _Candidate(
                index, offset + match.start(group), offset + match.end(group), priority,
                rule.rule_id, rule.severity, rule.reason, value,
            )  # fmt: skip


def _assignment_candidate(
    index: int, offset: int, key: str, value: str, start: int, end: int
) -> _Candidate | None:
    if not p.is_secret_like_name(key.strip("\"'").lstrip("-")):
        return None
    kind = p.classify_secret_value(value)
    if kind is None:
        return None
    rule_id = "high-entropy-secret" if kind == "random" else "secret-assignment"
    severity, reason = p.ASSIGNMENT_RULES[rule_id]
    return _Candidate(index, offset + start, offset + end, 100, rule_id, severity, reason, value)


def _assignments(index: int, segment: str, offset: int, yaml: bool) -> Iterator[_Candidate]:
    for match in p.ASSIGNMENT.finditer(segment):
        candidate = _assignment_candidate(
            index, offset, match["key"], match["value"], match.start("value"), match.end("value")
        )
        if candidate:
            yield candidate
    if yaml:
        match = p.YAML_ASSIGNMENT.match(segment)
        if match:
            candidate = _assignment_candidate(
                index,
                offset,
                match["key"],
                match["value"],
                match.start("value"),
                match.end("value"),
            )
            if candidate:
                yield candidate


def _candidates(text: str, yaml: bool) -> list[_Candidate]:
    lines = _split_lines(text)
    found: list[_Candidate] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        for offset, segment in _windows(line):
            found.extend(_private_keys(lines, index, segment, offset))
            found.extend(_patterns(index, segment, offset))
            found.extend(_assignments(index, segment, offset, yaml))
    return found


def _resolve_overlaps(candidates: list[_Candidate]) -> list[_Candidate]:
    """Keep one candidate per span: the highest-priority (most specific) rule wins."""
    accepted: list[_Candidate] = []
    for cand in sorted(candidates, key=lambda c: (c.line_index, c.priority, c.start, c.rule_id)):
        clash = any(
            a.line_index == cand.line_index and cand.start < a.end and a.start < cand.end
            for a in accepted
        )
        if not clash:
            accepted.append(cand)
    return accepted


def _finding_sort_key(finding: SecretFinding) -> tuple[str, int, int, str]:
    return (finding.file_path, finding.line, finding.column, finding.rule_id)


def scan_text(text: str, path: str, *, first_line: int = 1) -> list[SecretFinding]:
    """Scan ``text`` (a file, or a chunk starting at ``first_line``) and return its findings.

    Args:
        text: The content to inspect. It is never logged or stored.
        path: Repository-relative POSIX path used in findings (validated; absolute or
            backslash paths are rejected so host paths cannot leak).
        first_line: 1-based file line number of the first line of ``text``.
    """
    target = ScanTarget(path=path, text=text, first_line=first_line)
    yaml = target.path.lower().endswith(p.YAML_SUFFIXES)
    findings = [
        SecretFinding(
            rule_id=c.rule_id,
            severity=c.severity,
            file_path=target.path,
            line=target.first_line + c.line_index,
            column=c.start + 1,
            reason=c.reason,
            preview=redact(c.value),
        )
        for c in _resolve_overlaps(_candidates(target.text, yaml))
    ]
    return sorted(findings, key=_finding_sort_key)


def scan_source_file(source_file: SourceFile) -> list[SecretFinding]:
    """Scan one ingested file. Findings carry its repository-relative path."""
    return scan_text(source_file.content, source_file.relative_path)


def scan_targets(targets: Iterable[ScanTarget]) -> SecretScanReport:
    """Scan many targets (whole files, chunks or retrieved results) into one report."""
    findings: list[SecretFinding] = []
    paths: set[str] = set()
    for target in targets:
        paths.add(target.path)
        findings.extend(scan_text(target.text, target.path, first_line=target.first_line))
    # Overlapping chunks of one file can report the same location twice; keep one.
    unique = {(f.file_path, f.line, f.column, f.rule_id): f for f in findings}
    ordered = tuple(sorted(unique.values(), key=_finding_sort_key))
    report = SecretScanReport(
        files_scanned=len(paths),
        findings=ordered,
        counts_by_rule=dict(sorted(Counter(f.rule_id for f in ordered).items())),
        counts_by_severity=dict(sorted(Counter(f.severity.value for f in ordered).items())),
    )
    logger.info(
        "Secret scan: files=%d findings=%d%s",
        report.files_scanned,
        report.findings_count,
        f" rules={report.counts_by_rule}" if report.findings else "",
    )
    return report


def scan_repository(files: Iterable[SourceFile]) -> SecretScanReport:
    """Scan every accepted file of an ingestion result."""
    return scan_targets(ScanTarget(path=f.relative_path, text=f.content) for f in files)
