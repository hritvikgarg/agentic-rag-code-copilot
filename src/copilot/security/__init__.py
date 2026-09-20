"""Security gate for repository content (Milestone 5c).

``assert_safe_for_external_llm`` is the mandatory check before any repository text is sent to an
external LLM. No external LLM is called anywhere in this package. See ``docs/security.md``.
"""

from copilot.security.errors import RepositorySecretRiskError, SecurityError
from copilot.security.gate import as_scan_target, assert_safe_for_external_llm
from copilot.security.models import (
    ScanTarget,
    SecretFinding,
    SecretScanReport,
    Severity,
    redact,
)
from copilot.security.patterns import RULE_IDS
from copilot.security.secret_scanner import (
    scan_repository,
    scan_source_file,
    scan_targets,
    scan_text,
)

__all__ = [
    "RULE_IDS",
    "RepositorySecretRiskError",
    "ScanTarget",
    "SecretFinding",
    "SecretScanReport",
    "SecurityError",
    "Severity",
    "as_scan_target",
    "assert_safe_for_external_llm",
    "redact",
    "scan_repository",
    "scan_source_file",
    "scan_targets",
    "scan_text",
]
