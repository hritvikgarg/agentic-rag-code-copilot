"""Secret scanner: true positives, false positives, redaction, ordering, paths."""

import logging

import pytest
from pydantic import ValidationError

from copilot.security import (
    RULE_IDS,
    ScanTarget,
    SecretFinding,
    Severity,
    redact,
    scan_targets,
    scan_text,
)
from copilot.security import patterns as pat
from copilot.utils.paths import UnsafePathError
from tests.security import secret_helpers as h

PATH = "src/app/settings.py"


def rules(text: str, path: str = PATH) -> list[str]:
    return [f.rule_id for f in scan_text(text, path)]


# --------------------------------------------------------------------------------------------
# True positives
# --------------------------------------------------------------------------------------------

PROVIDER_CASES = [
    ("github-token", h.github_token()),
    ("github-token", "gho_" + h.synth("gho", 36)),
    ("github-token", "github_pat_" + h.synth("pat", 60, h.BASE62 + "_")),
    ("aws-access-key-id", h.aws_access_key()),
    ("google-api-key", h.google_api_key()),
    ("slack-token", h.slack_token()),
    ("slack-webhook", h.slack_webhook()),
    ("stripe-live-key", h.stripe_key()),
    ("gitlab-token", h.gitlab_token()),
    ("huggingface-token", h.huggingface_token()),
    ("sendgrid-key", h.sendgrid_key()),
    ("ai-provider-key", h.ai_provider_key()),
    ("ai-provider-key", h.anthropic_style_key()),
]


@pytest.mark.parametrize(("rule", "secret"), PROVIDER_CASES, ids=[c[0] for c in PROVIDER_CASES])
@pytest.mark.parametrize(
    "template",
    ['x = "{v}"', "# leaked {v} here", '{{"k": "{v}"}}', "CI_VALUE={v}"],
    ids=["assignment", "comment", "json", "bare"],
)
def test_provider_tokens_are_detected_with_exact_location(rule, secret, template):
    line = template.replace("{v}", secret).replace("{{", "{").replace("}}", "}")
    findings = scan_text("first line\n" + line + "\n", PATH)
    assert [f.rule_id for f in findings] == [rule]
    assert findings[0].line == 2
    assert findings[0].column == line.index(secret) + 1
    assert findings[0].severity is Severity.HIGH


@pytest.mark.parametrize("kind", ["RSA ", "", "OPENSSH ", "EC ", "DSA ", "ENCRYPTED "])
def test_private_key_blocks(kind):
    text = "def f():\n    pass\n\n" + h.private_key_text(kind)
    findings = scan_text(text, "keys/dev.py")
    assert [f.rule_id for f in findings] == ["private-key-block"]
    assert findings[0].line == 4
    assert findings[0].column == 1


def test_private_key_header_without_body_is_still_reported():
    header = "-----BEGIN " + "PRIVATE KEY-----"
    findings = scan_text(f"prefix\n{header}\nshort\n", "notes/pem.md")
    assert [f.rule_id for f in findings] == ["private-key-header"]


def test_private_key_pgp_block_and_json_one_liner():
    pgp = "-----BEGIN " + "PGP PRIVATE KEY BLOCK-----"
    assert rules(pgp + "\n" + h.synth("pgp", 64, h.BASE64)) == ["private-key-block"]
    body = h.synth("json", 64, h.BASE64)
    line = '{"private_key": "-----BEGIN ' + "PRIVATE KEY-----\\n" + body + '\\n-----END"}'
    assert rules(line, "cfg/sa.json") == ["private-key-block"]


def test_certificates_and_public_keys_are_not_secrets():
    cert = "-----BEGIN " + "CERTIFICATE-----\n" + h.synth("c", 64, h.BASE64) + "\n"
    pub = "-----BEGIN " + "PUBLIC KEY-----\n" + h.synth("p", 64, h.BASE64) + "\n"
    assert rules(cert + pub) == []


def test_bearer_token_header():
    value = h.random_value("bearer", 40)
    findings = scan_text(f'headers = {{"Authorization": "Bearer {value}"}}', PATH)
    assert [f.rule_id for f in findings] == ["bearer-token"]
    assert findings[0].severity is Severity.MEDIUM


def test_credentials_embedded_in_url():
    value = h.random_value("urlpw", 16)
    findings = scan_text(f'DATABASE = "postgresql://svc:{value}@db.internal:5432/app"', PATH)
    assert [f.rule_id for f in findings] == ["credentials-in-url"]


ASSIGNMENT_TEMPLATES = [
    'api_key = "{v}"',
    "API_KEY='{v}'",
    'self.client_secret = "{v}"',
    'DB_PASSWORD: "{v}"',
    '"token": "{v}"',
    'os.environ["API_KEY"] = "{v}"',
    'api_key: str = "{v}"',
    'connect(host, password="{v}")',
    'clientSecret: "{v}"',
    '--password="{v}"',
    "secret_key => '{v}'",
]


@pytest.mark.parametrize("template", ASSIGNMENT_TEMPLATES)
def test_high_entropy_value_on_secret_like_name(template):
    value = h.random_value("assign", 40)
    line = template.replace("{v}", value)
    findings = scan_text(line, PATH)
    assert [f.rule_id for f in findings] == ["high-entropy-secret"]
    assert findings[0].column == line.index(value) + 1


@pytest.mark.parametrize("value", ["Zq8!" + "vN3#rT5w", "Corr3ct" + "Horse9", "aB3" + "dE5fG7h"])
def test_non_trivial_literal_password_is_reported(value):
    for template in ['password = "{v}"', "secret: '{v}'", 'token = "{v}"']:
        line = template.replace("{v}", value)
        assert rules(line) == ["secret-assignment"], line


def test_yaml_unquoted_secret_is_reported_only_in_yaml_files():
    text = "db:\n  password: " + "Zq8!vN3#rT5w" + "\n"
    assert rules(text, "deploy/app.yml") == ["secret-assignment"]
    assert rules(text, "deploy/app.py") == []
    quoted_template = "api_key: " + h.random_value("yaml", 40) + "  # comment\n"
    assert rules(quoted_template, "ci/pipeline.yaml") == ["high-entropy-secret"]


def test_provider_rule_wins_over_assignment_and_is_reported_once():
    token = h.github_token("dup")
    findings = scan_text(f'github_token = "{token}"', PATH)
    assert [f.rule_id for f in findings] == ["github-token"]


def test_every_rule_id_is_covered_by_a_test():
    covered = {rule for rule, _ in PROVIDER_CASES} | {
        "private-key-block",
        "private-key-header",
        "bearer-token",
        "credentials-in-url",
        "high-entropy-secret",
        "secret-assignment",
    }
    assert covered == set(RULE_IDS)


# --------------------------------------------------------------------------------------------
# False positives
# --------------------------------------------------------------------------------------------

UUID = "123e4567-e89b-12d3-a456-426614174000"
HEX64 = h.synth("hex", 64, h.HEX)

QUIET_LINES = [
    # empty / short
    'api_key = ""',
    'password = "abc"',
    "api_key = None",
    # placeholders
    'api_key = "YOUR_API_KEY"',
    'GEMINI_API_KEY="your-key-here"',
    'password = "changeme"',
    'token = "example-token-value"',
    'secret = "dummy-secret-value"',
    'password = "test"',
    'token = "test-token-12345"',
    'token = "<token>"',
    'api_key = "redacted-value-here"',
    'password = "REDACTED"',
    'password = "xxxxxxxxxxxx"',
    'secret = "************"',
    'api_key = "${API_KEY}"',
    'password = "{{ db_password }}"',
    'token = "%(token)s"',
    # environment lookups and other non-literals
    'api_key = os.getenv("API_KEY")',
    'token = os.environ["TOKEN"]',
    'password = os.environ.get("DB_PASSWORD", "")',
    "const key = process.env.API_KEY;",
    "password = get_password()",
    "api_key = settings.api_key",
    "api_key: str | None = None",
    "self.token = token",
    # names of things, not values
    'GEMINI_API_KEY = "GEMINI_API_KEY"',
    'api_key_env = "GEMINI_API_KEY"',
    'token_url = "https://auth.example.org/oauth/token"',
    'HUGGINGFACE_HEADER_X_ACCESS_TOKEN = "X-Access-Token-Header"',
    'git_url = "git://[path-to-repo[:]][ref@]path/to/repo"',
    'tokenizer = "jinaai/jina-embeddings-v2-base-code"',
    "max_tokens = 4096",
    'password_hash = "' + HEX64 + '"',
    'secret_id = "' + h.random_value("sid", 40) + '"',
    'secret_name = "prod-database-credentials"',
    # hashes, checksums, UUIDs and random data on ordinary names
    'sha256 = "' + HEX64 + '"',
    'expected_checksum = "' + HEX64 + '"',
    'content_sha256 = "' + HEX64 + '"',
    'request_id = "' + UUID + '"',
    'user = {"id": "' + UUID + '"}',
    'payload = "' + h.random_value("payload", 48) + '"',
    'blob = "' + h.synth("blob", 64, h.BASE64) + '"',
    # one character class only (a slug or a word, not a credential)
    'token = "authorization-header-name"',
    # documentation examples that are clearly placeholders
    'export API_KEY="your-api-key-here"',
    "curl -H 'Authorization: Bearer <token>' https://api.example.org",
    "Authorization: Bearer {token}",
    "Authorization: Bearer YOUR_TOKEN_HERE_1234567890",
    "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
    "gh_token = ghp_" + "x" * 36,
    "postgres://user:password@localhost:5432/db",
    "postgres://postgres:postgres@localhost:5432/db",
    "https://user:${DB_PASSWORD}@host/db",
    "https://example.org/docs/authentication",
    # prose and ordinary code
    "A private key must never be committed to a repository.",
    "-----BEGIN " + "CERTIFICATE-----",
    "def authenticate(user, password):\n    return check(user, password)",
    "for token in tokens:\n    total += len(token)",
    "class TokenBucket:\n    rate = 5",
    "x = 1",
    "",
]


@pytest.mark.parametrize("line", QUIET_LINES, ids=[str(i) for i in range(len(QUIET_LINES))])
def test_placeholders_lookups_hashes_and_normal_code_are_quiet(line):
    assert scan_text(line, PATH) == [], line


def test_ordinary_high_entropy_code_strings_are_not_flagged():
    text = "\n".join(
        [
            f'DATA = "{h.random_value("d1", 80)}"',
            f'signature = "{h.synth("sig", 88, h.BASE64)}=="',
            f'salt_bytes = b"{h.random_value("d3", 32)}"',
            f'CACHE_KEY = "{h.random_value("d4", 40)}"',
        ]
    )
    assert scan_text(text, PATH) == []


def test_yaml_placeholders_are_quiet():
    text = "password: ${DB_PASSWORD}\ntoken: changeme-please\napi_key: !secret api_key\n"
    assert scan_text(text, "deploy/app.yml") == []


# --------------------------------------------------------------------------------------------
# Value heuristics
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("api_key", True), ("API_KEY", True), ("apiKey", True), ("APIKey", True),
        ("client_secret", True), ("clientSecret", True), ("DB_PASSWORD", True),
        ("auth-token", True), ("private_key", True), ("access_key", True),
        ("password", True), ("self.token", True), ("aws_secret_access_key", True),
        ("HEADER_ACCESS_TOKEN", False), ("env_token", False), ("tokenizer", False),
        ("max_tokens", False), ("token_url", False),
        ("password_hash", False), ("api_key_env", False), ("secret_id", False),
        ("key", False), ("cache_key", False), ("username", False), ("author", False),
    ],
)  # fmt: skip
def test_is_secret_like_name(name, expected):
    assert pat.is_secret_like_name(name) is expected


def test_entropy_of_known_strings():
    assert pat.shannon_entropy("") == 0.0
    assert pat.shannon_entropy("aaaa") == 0.0
    assert pat.shannon_entropy("abab") == pytest.approx(1.0)
    assert pat.shannon_entropy("abcd") == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (h.random_value("k", 40), "random"),
        ("Zq8!" + "vN3#rT5w", "generic"),
        ("short", None),
        ("changeme-now-please", None),
        ("SOME_ENV_VARIABLE", None),
        ("module.attribute.name", None),
        ("alllowercaseword", None),
    ],
)
def test_classify_secret_value(value, kind):
    assert pat.classify_secret_value(value) == kind


# --------------------------------------------------------------------------------------------
# Redaction: no complete secret in repr, dumps, logs or exceptions
# --------------------------------------------------------------------------------------------


def test_redact_masks_short_values_and_shows_at_most_six_characters_of_long_ones():
    assert redact("") == "****"
    assert redact("a" * 15) == "****"
    long = h.github_token()
    preview = redact(long)
    assert preview == f"{long[:4]}...{long[-2:]}"
    assert len(preview) - len("...") <= 6
    assert long not in preview


@pytest.mark.parametrize(("rule", "secret"), PROVIDER_CASES, ids=[c[0] for c in PROVIDER_CASES])
def test_secret_never_appears_in_finding_report_or_log(rule, secret, caplog):
    text = f'x = "{secret}"\n'
    with caplog.at_level(logging.DEBUG, logger="copilot"):
        report = scan_targets([ScanTarget(PATH, text)])
    (finding,) = report.findings
    everything = " ".join(
        [
            repr(finding),
            str(finding),
            finding.model_dump_json(),
            repr(report),
            str(report),
            report.model_dump_json(),
            repr(ScanTarget(PATH, text)),
            caplog.text,
        ]
    )
    assert not h.contains_secret(everything, secret)


def test_finding_has_no_field_that_could_hold_the_value():
    assert set(SecretFinding.model_fields) == {
        "rule_id", "severity", "file_path", "line", "column", "reason", "preview",
    }  # fmt: skip


# --------------------------------------------------------------------------------------------
# Paths, lines, ordering
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["/home/dev/app/a.py", "C:/Users/dev/app/a.py", "C:\\Users\\dev\\a.py", "src\\a.py", "../a.py"],
)
def test_absolute_and_windows_paths_are_rejected(bad):
    with pytest.raises(UnsafePathError):
        scan_text("x = 1", bad)
    with pytest.raises(ValidationError):
        SecretFinding(
            rule_id="r", severity=Severity.HIGH, file_path=bad, line=1, column=1, reason="x",
            preview="****",
        )  # fmt: skip


def test_findings_use_the_relative_posix_path_with_spaces_and_unicode():
    path = "my dir/módulo ü/config.py"
    (finding,) = scan_text(f'k = "{h.github_token()}"', path)
    assert finding.file_path == path
    assert finding.location == f"{path}:1:6"


def test_first_line_offsets_chunk_line_numbers_and_crlf_is_handled():
    token = h.github_token("crlf")
    text = f'a = 1\r\nb = "{token}"\r\n'
    (finding,) = scan_text(text, PATH, first_line=40)
    assert finding.line == 41
    with pytest.raises(ValueError, match="first_line"):
        scan_text("x", PATH, first_line=0)


def test_findings_are_ordered_deterministically_whatever_the_input_order():
    targets = [
        ScanTarget("b/second.py", f'x = "{h.github_token("o1")}"\ny = "{h.aws_access_key("o2")}"'),
        ScanTarget("a/first.py", f'x = "{h.stripe_key("o3")}"'),
        ScanTarget("b/second.py", "z = 0\n" * 3 + f'k = "{h.gitlab_token("o4")}"'),
    ]
    forward = scan_targets(targets)
    backward = scan_targets(list(reversed(targets)))
    assert forward == backward
    keys = [(f.file_path, f.line, f.column) for f in forward.findings]
    assert keys == sorted(keys)
    assert [f.file_path for f in forward.findings][0] == "a/first.py"


def test_report_counts_and_deduplicates_overlapping_chunks():
    token = h.github_token("chunks")
    chunk_a = ScanTarget(PATH, f'x = 1\nk = "{token}"\n', first_line=1)
    chunk_b = ScanTarget(PATH, f'k = "{token}"\nz = 2\n', first_line=2)  # overlaps chunk_a
    other = ScanTarget("src/other.py", f'k = "{h.aws_access_key("cnt")}"')
    report = scan_targets([chunk_a, chunk_b, other, ScanTarget("src/clean.py", "x = 1")])
    assert report.files_scanned == 3
    assert report.findings_count == 2  # the duplicate location is reported once
    assert report.counts_by_rule == {"aws-access-key-id": 1, "github-token": 1}
    assert report.counts_by_severity == {"high": 2}
    assert report.files_with_findings == [PATH, "src/other.py"]
    assert report.blocking


def test_clean_report_is_not_blocking():
    report = scan_targets([ScanTarget(PATH, "x = 1\n")])
    assert (report.files_scanned, report.findings_count, report.blocking) == (1, 0, False)
    assert report.counts_by_rule == {}


def test_empty_targets_give_an_empty_clean_report():
    report = scan_targets([])
    assert report.files_scanned == 0 and not report.blocking


# --------------------------------------------------------------------------------------------
# Very long lines
# --------------------------------------------------------------------------------------------


def test_secret_in_a_very_long_line_is_found_including_on_a_window_edge():
    from copilot.security import secret_scanner as scanner

    token = h.github_token("long")
    filler = "a " * 30_000
    assert [f.rule_id for f in scan_text(f"{filler}{token}", PATH)] == ["github-token"]
    edge = scanner.MAX_LINE_CHARS - 10  # token straddles the end of the first window
    line = "." * edge + token + "." * 30_000
    (finding,) = scan_text(line, PATH)
    assert finding.column == edge + 1


def test_scanning_is_fast_on_adversarial_input():
    import time

    text = ("api_key" + " = " + "'" * 5 + "a" * 300 + "\n") * 200 + "=" * 50_000
    start = time.perf_counter()
    scan_text(text, PATH)
    assert time.perf_counter() - start < 5
