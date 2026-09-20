"""Exceptions of the security package.

Messages contain rule ids, relative paths and line numbers only - never a matched value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from copilot.security.models import SecretScanReport

MAX_LISTED_FINDINGS = 10


class SecurityError(Exception):
    """Base class for security-gate failures."""


class RepositorySecretRiskError(SecurityError):
    """Repository text that may contain secrets was refused for external LLM use.

    There is deliberately no override: remove the secret (and rotate it if it was ever real),
    or leave the affected files out of the context.
    """

    def __init__(self, report: SecretScanReport) -> None:
        self.report = report
        listed = [
            f"{f.file_path}:{f.line} [{f.rule_id}]" for f in report.findings[:MAX_LISTED_FINDINGS]
        ]
        more = report.findings_count - len(listed)
        rules = ", ".join(
            f"{rule}={count}" for rule, count in sorted(report.counts_by_rule.items())
        )
        text = (
            "Repository context REFUSED for external LLM use: "
            f"{report.findings_count} potential secret(s) in {len(report.files_with_findings)} "
            f"file(s) (rules: {rules}). Findings: {'; '.join(listed)}"
            f"{f'; ... and {more} more' if more > 0 else ''}. "
            "Remove the secrets (and rotate any real credential that was committed) or leave the "
            "affected files out of the context. There is no override."
        )
        super().__init__(text)
