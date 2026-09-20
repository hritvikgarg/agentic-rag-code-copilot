"""The fail-closed gate that must run before repository text reaches an external LLM.

Invariant for Milestone 6 and later::

    retrieved repository chunks -> assert_safe_for_external_llm(...) -> ONLY IF SAFE -> external LLM

Any finding raises ``RepositorySecretRiskError``. There is no ``force`` argument, environment
variable or setting that lets text through; a caller that cannot pass the gate must not call the
LLM. Text that is not one of the supported item types is refused with ``TypeError`` (a new kind of
context has to be adapted explicitly rather than skipping the scan).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from copilot.security.errors import RepositorySecretRiskError
from copilot.security.models import ScanTarget, SecretScanReport
from copilot.security.secret_scanner import scan_targets


def as_scan_target(item: Any) -> ScanTarget:
    """Adapt a ``ScanTarget``, ``SourceFile``, ``Chunk`` or ``RetrievalResult`` to a target.

    Raises:
        TypeError: for anything else (fail closed).
    """
    if isinstance(item, ScanTarget):
        return item
    path = getattr(item, "relative_path", None) or getattr(item, "file_path", None)
    text = getattr(item, "content", None)
    if text is None:
        text = getattr(item, "text", None)
    if isinstance(path, str) and isinstance(text, str):
        return ScanTarget(path=path, text=text, first_line=getattr(item, "start_line", 1))
    raise TypeError(
        f"cannot scan an object of type {type(item).__name__!r} for secrets; "
        "pass ScanTarget, SourceFile, Chunk or RetrievalResult objects"
    )


def assert_safe_for_external_llm(items: Iterable[Any]) -> SecretScanReport:
    """Scan everything that would be sent to an external LLM; raise if anything looks secret.

    Args:
        items: The exact texts that will form the LLM context (chunks, retrieved results,
            whole files or ``ScanTarget`` objects). Include user-supplied text (question, error
            log) as ``ScanTarget`` objects too.

    Returns:
        The (clean) ``SecretScanReport`` so the caller can log the counts.

    Raises:
        RepositorySecretRiskError: if any potential secret was found.
        TypeError: if an item cannot be scanned.
    """
    report = scan_targets(as_scan_target(item) for item in items)
    if report.blocking:
        raise RepositorySecretRiskError(report)
    return report
