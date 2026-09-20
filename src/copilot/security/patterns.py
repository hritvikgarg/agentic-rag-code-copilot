"""Detection rules and the value heuristics that keep them quiet.

Three families of rules, from most to least specific:

1. **Provider token formats** - a documented, distinctive prefix plus a long random body
   (GitHub, AWS, Google, Slack, Stripe, GitLab, Hugging Face, AI-provider ``sk-`` keys, ...).
2. **Structural rules** - private-key blocks, ``Bearer`` values, credentials embedded in URLs.
3. **Named assignments** - a *secret-like name* (``api_key``, ``password``, ``token``...) assigned
   a quoted literal that is not a placeholder, reference or template. A random-looking value on
   such a name is reported as ``high-entropy-secret``; other non-trivial values as
   ``secret-assignment``. Entropy alone never triggers a finding: a random string assigned to an
   ordinary name (a hash, a test vector) is ignored.

Everything is regular expressions plus small pure functions, so behaviour is easy to test and to
explain. Nothing here logs or returns a matched value except to the scanner, which masks it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from copilot.security.models import Severity

# --------------------------------------------------------------------------------------------
# Value heuristics
# --------------------------------------------------------------------------------------------


def shannon_entropy(value: str) -> float:
    """Shannon entropy in bits per character (0 for the empty string)."""
    if not value:
        return 0.0
    total = len(value)
    return -sum((n / total) * math.log2(n / total) for n in Counter(value).values())


_PLACEHOLDER_WORDS = frozenset(
    {
        "test", "tests", "testing", "demo", "mock", "foo", "bar", "baz", "value", "string",
        "secret", "password", "passwd", "token", "apikey", "api_key", "key", "none", "null",
        "nil", "undefined", "empty", "default", "admin", "user", "username", "localhost",
        "changeit", "password1", "secret123", "hunter2", "letmein", "tbd", "todo", "fixme",
    }
)  # fmt: skip
_PLACEHOLDER_SUBSTRINGS = (
    "your", "example", "sample", "dummy", "fake", "placeholder", "changeme", "change_me",
    "change-me", "redacted", "replace", "insert", "xxxx", "****", "todo", "fixme", "notreal",
    "not-a-real", "lorem", "123456", "abcdef", "<", ">",
)  # fmt: skip
_PLACEHOLDER_PREFIXES = ("test", "demo", "mock", "dummy", "fake", "example", "sample", "foo", "bar")
_MASK_ONLY = re.compile(r"[x*#\u2022.\-_]+", re.IGNORECASE)
# Values that are references or templates, not literals.
_TEMPLATE = re.compile(
    r"""^(?:\$\{?[A-Za-z_]|%[A-Za-z_(]|\{\{|\{[A-Za-z_]|<[^>]*>$|\[[A-Za-z_ ]+\]$|@[A-Za-z_])"""
)
_TEMPLATE_INSIDE = re.compile(r"\$\{|\{\{|%\(|%s|\{\}|\{[A-Za-z_]\w*\}")
_ENV_NAME = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")  # SOME_ENV_VARIABLE_NAME
_DOTTED_IDENTIFIER = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")  # module.attribute


def is_placeholder(value: str) -> bool:
    """True if ``value`` is an obvious placeholder, mask, template or empty value."""
    stripped = value.strip().strip("\"'`")
    if not stripped:
        return True
    lowered = stripped.lower()
    if lowered in _PLACEHOLDER_WORDS or _MASK_ONLY.fullmatch(lowered):
        return True
    if len(set(stripped)) == 1:  # aaaaaaaa
        return True
    if lowered.startswith(_PLACEHOLDER_PREFIXES):
        return True
    if _TEMPLATE.match(stripped) or _TEMPLATE_INSIDE.search(stripped):
        return True
    return any(marker in lowered for marker in _PLACEHOLDER_SUBSTRINGS)


def _character_classes(value: str) -> int:
    classes = (
        any(c.islower() for c in value),
        any(c.isupper() for c in value),
        any(c.isdigit() for c in value),
        any(not c.isalnum() and c not in "-_./" for c in value),
    )
    return sum(classes)


def classify_secret_value(value: str) -> str | None:
    """Judge a quoted literal assigned to a secret-like name.

    Returns ``"random"`` (long, random-looking), ``"generic"`` (non-trivial) or ``None`` (ignore).
    """
    if len(value) < 8 or is_placeholder(value):
        return None
    if _ENV_NAME.fullmatch(value) or _DOTTED_IDENTIFIER.fullmatch(value):
        return None  # the *name* of an environment variable or an attribute, not a value
    entropy = shannon_entropy(value)
    has_letter = any(c.isalpha() for c in value)
    has_digit = any(c.isdigit() for c in value)
    if len(value) >= 20 and entropy >= 3.5 and has_letter and has_digit:
        return "random"
    if entropy >= 2.5 and _character_classes(value) >= 2:
        return "generic"
    return None


# --------------------------------------------------------------------------------------------
# Secret-like names
# --------------------------------------------------------------------------------------------

_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_SECRET_WORDS = frozenset(
    {"password", "passwd", "pwd", "passphrase", "passcode", "secret", "token", "apikey",
     "credential", "credentials", "bearer"}
)  # fmt: skip
_SECRET_PAIRS = frozenset(
    {("api", "key"), ("access", "key"), ("private", "key"), ("secret", "key"),
     ("signing", "key"), ("encryption", "key"), ("master", "key"), ("auth", "key")}
)  # fmt: skip
# A name ending in one of these describes *something about* a secret, not the secret itself
# (``token_url``, ``api_key_env``, ``password_hash``, ``secret_id``).
_NON_SECRET_LAST_WORDS = frozenset(
    {"url", "uri", "path", "file", "filename", "dir", "name", "env", "var", "variable", "header",
     "field", "label", "prefix", "suffix", "type", "length", "len", "hash", "sha", "sha1",
     "sha256", "md5", "checksum", "digest", "id", "count", "min", "max", "ttl", "expires",
     "expiry", "format", "pattern", "regex", "description", "desc", "help", "message", "msg",
     "error", "endpoint", "host", "scheme", "location", "mode", "method", "kind", "style",
     "policy", "manager", "provider", "service", "class", "size", "limit", "timeout", "required",
     "enabled", "flag", "placeholder", "hint", "example"}
)  # fmt: skip


# A name containing one of these describes a header, environment variable or endpoint *about* a
# credential (``X_ACCESS_TOKEN_HEADER``, ``HEADER_API_KEY``, ``env_token``), not the credential.
_NON_SECRET_ANY_WORDS = frozenset({"header", "headers", "env", "environ", "endpoint"})


def name_words(name: str) -> list[str]:
    """Lower-case words of an identifier (snake_case, kebab-case, camelCase, dotted)."""
    return [w.lower() for part in re.split(r"[^A-Za-z0-9]+", name) for w in _WORD.findall(part)]


def is_secret_like_name(name: str) -> bool:
    """True for names such as ``api_key``, ``DB_PASSWORD``, ``clientSecret``, ``auth_token``.

    ``tokenizer``, ``max_tokens``, ``password_hash``, ``token_url`` and ``api_key_env`` are not.
    """
    words = name_words(name)
    if not words or words[-1] in _NON_SECRET_LAST_WORDS:
        return False
    if any(w in _NON_SECRET_ANY_WORDS for w in words):
        return False
    if any(w in _SECRET_WORDS for w in words):
        return True
    return any(pair in _SECRET_PAIRS for pair in zip(words, words[1:], strict=False))


# --------------------------------------------------------------------------------------------
# Rule table (provider formats and structural rules)
# --------------------------------------------------------------------------------------------


def _random_enough(value: str) -> bool:
    return not is_placeholder(value) and shannon_entropy(value) >= 3.0


def _has_digit_and_random(value: str) -> bool:
    return (
        not is_placeholder(value)
        and any(c.isdigit() for c in value)
        and shannon_entropy(value) >= 3.5
    )


def _bearer_ok(value: str) -> bool:
    return _has_digit_and_random(value)


def _url_password_ok(value: str) -> bool:
    return not is_placeholder(value) and len(value) >= 6


@dataclass(frozen=True)
class PatternRule:
    """A regex rule. ``group`` names the capture holding the sensitive value (default: whole)."""

    rule_id: str
    severity: Severity
    reason: str
    regex: re.Pattern[str]
    accept: Callable[[str], bool]
    group: str | None = None


def _rule(
    rule_id: str,
    severity: Severity,
    reason: str,
    pattern: str,
    accept: Callable[[str], bool],
    group: str | None = None,
    flags: int = 0,
) -> PatternRule:
    return PatternRule(rule_id, severity, reason, re.compile(pattern, flags), accept, group)


H, M = Severity.HIGH, Severity.MEDIUM

# Order is priority: when two rules match the same characters the earlier rule wins.
PATTERN_RULES: tuple[PatternRule, ...] = (
    _rule(
        "github-token", H, "GitHub token format",
        r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{50,255})"
        r"(?![A-Za-z0-9_])",
        _random_enough,
    ),
    _rule(
        "aws-access-key-id", H, "AWS access key id format",
        r"(?<![A-Z0-9])(?:AKIA|ASIA|ABIA|ACCA|AGPA|AIDA|AIPA|ANPA|ANVA|AROA)[A-Z0-9]{16}"
        r"(?![A-Z0-9])",
        lambda v: not is_placeholder(v),
    ),
    _rule(
        "google-api-key", H, "Google API key format",
        r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])",
        _random_enough,
    ),
    _rule(
        "slack-token", H, "Slack token format",
        r"(?<![A-Za-z0-9-])xox[abposr]-[0-9A-Za-z-]{20,}",
        _random_enough,
    ),
    _rule(
        "slack-webhook", H, "Slack incoming-webhook URL",
        r"https://hooks\.slack\.com/services/T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{20,}",
        lambda v: not is_placeholder(v),
    ),
    _rule(
        "stripe-live-key", H, "Stripe live key format",
        r"(?<![A-Za-z0-9_])[sr]k_live_[0-9A-Za-z]{20,}",
        _random_enough,
    ),
    _rule(
        "gitlab-token", H, "GitLab personal access token format",
        r"(?<![A-Za-z0-9_-])glpat-[0-9A-Za-z_-]{20,}",
        _random_enough,
    ),
    _rule(
        "huggingface-token", H, "Hugging Face token format",
        r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{34,}",
        _random_enough,
    ),
    _rule(
        "sendgrid-key", H, "SendGrid API key format",
        r"(?<![A-Za-z0-9_-])SG\.[0-9A-Za-z_-]{22}\.[0-9A-Za-z_-]{43}",
        _random_enough,
    ),
    _rule(
        "ai-provider-key", H, "AI-provider secret key format (sk-...)",
        r"(?<![A-Za-z0-9_-])sk-(?:ant-|proj-|svcacct-)?[A-Za-z0-9_-]{32,}",
        _has_digit_and_random,
    ),
    _rule(
        "bearer-token", M, "Bearer credential value",
        r"\b[Bb]earer\s+(?P<value>[A-Za-z0-9\-._~+/]{20,}={0,2})",
        _bearer_ok, group="value",
    ),
    _rule(
        "credentials-in-url", M, "password embedded in a URL",
        r"[A-Za-z][A-Za-z0-9+.\-]*://(?P<user>[^\s:/@'\"\[\]{}<>]+):(?!(?P=user)@)(?P<value>[^\s@/'\"\[\]{}<>]{6,})@"
        r"[^\s/'\"]+",
        _url_password_ok, group="value",
    ),
)  # fmt: skip

# ``-----BEGIN ... PRIVATE KEY-----`` (also "ENCRYPTED PRIVATE KEY" and "PGP PRIVATE KEY BLOCK").
PRIVATE_KEY_HEADER = re.compile(r"-{5}BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-{5}")
KEY_MATERIAL = re.compile(
    r"[A-Za-z0-9+/=]{40,}"
)  # a base64 body line (or the tail of a header line)
PEM_HEADER_FIELDS = ("Proc-Type:", "DEK-Info:")  # legacy encrypted PEM keys start with these

# ``name = "value"`` / ``"name": "value"`` / ``name: type = "value"`` / ``NAME='value'``.
ASSIGNMENT = re.compile(
    r"""
    (?P<key>-{0,2}(?:"[A-Za-z_][\w.\-]*"|'[A-Za-z_][\w.\-]*'|[A-Za-z_][\w.\-]*))\]?
    (?:\s*:\s*[A-Za-z_][\w\[\], .|]*?)?          # optional type annotation
    \s*(?::=|=>|=|:)\s*
    (?P<quote>["'])(?P<value>[^\s"']{8,512})(?P=quote)
    """,
    re.VERBOSE,
)
# Unquoted ``name: value`` lines, only used for YAML files.
YAML_ASSIGNMENT = re.compile(
    r"""^\s*(?:-\s+)?(?P<key>[A-Za-z_][\w.\-]*)\s*:\s+
    (?P<value>[^\s#'"!&*{\[<%$|>]\S{7,511}?)(?:\s+\#.*)?\s*$""",
    re.VERBOSE,
)
YAML_SUFFIXES = (".yaml", ".yml")

ASSIGNMENT_RULES: dict[str, tuple[Severity, str]] = {
    "high-entropy-secret": (H, "random-looking value assigned to a secret-like name"),
    "secret-assignment": (M, "literal value assigned to a secret-like name"),
}
PRIVATE_KEY_RULES: dict[str, tuple[Severity, str]] = {
    "private-key-block": (H, "private key block with key material"),
    "private-key-header": (H, "private key header without visible key material (may be truncated)"),
}

RULE_IDS: tuple[str, ...] = (
    tuple(sorted(PRIVATE_KEY_RULES))
    + tuple(rule.rule_id for rule in PATTERN_RULES)
    + tuple(sorted(ASSIGNMENT_RULES))
)
