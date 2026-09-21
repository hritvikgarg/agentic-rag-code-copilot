# Security: content-secret scanner and the external-LLM gate (Milestone 5c)

Status: **implemented and tested.** No external LLM or API is called anywhere in this repository
yet. The scanner and the gate exist so that Milestone 6 can call them *before* any repository text
leaves the machine. Secret scanning **reduces risk; it cannot guarantee that every secret is found.**

## Why filename filtering is not enough

Ingestion (`docs/ingestion.md`) rejects files by *name*: `.env`, `*.pem`, `credentials.json`, and so
on. That stops the obvious cases, but a credential is just text and it lands in ordinary files:

* a key pasted into `settings.py` or `config.yaml` while debugging;
* a token in a test fixture, a notebook-style script, a README example or a CI file;
* a private key embedded in a JSON string or a code comment.

Such a file has an innocent name, is accepted by ingestion, is chunked, and would be retrieved and
sent to an external LLM as "repository evidence". Once sent, the secret has left your control and
may be logged by the provider. Content scanning inspects what is *in* the file.

## What content scanning adds

`copilot.security` reads the text that could be sent and reports **potential secrets** as typed
findings. A finding carries only safe metadata (rule id, severity, relative path, line, column, a
static reason and a masked preview). The matched value is never stored, logged, printed by default
or placed in an exception. The gate (below) turns any finding into a refusal.

Data flow required from Milestone 6 on:

```
retrieved repository chunks (+ user question / error text)
        -> assert_safe_for_external_llm(...)      # raises RepositorySecretRiskError on any finding
        -> ONLY IF SAFE -> external LLM
```

## Rules implemented

Each rule has an id, a severity (`high` = provider-specific format, private key or random value on a
secret name; `medium` = generic assignment, bearer value or URL credential) and a static reason.
Severity is informational: **every finding blocks** (fail closed).

| Rule id | Detects |
|---|---|
| `private-key-block` | a `BEGIN ... PRIVATE KEY` header (RSA, EC, DSA, OpenSSH, PKCS#8, encrypted, PGP block) followed by key material, including the one-line JSON form |
| `private-key-header` | the same header with no visible key material (the key may be truncated or in the next chunk) |
| `github-token` | `ghp_`/`gho_`/`ghu_`/`ghs_`/`ghr_` tokens and `github_pat_` fine-grained tokens |
| `aws-access-key-id` | AWS access key ids (`AKIA`, `ASIA`, ... prefixes, 20 characters) |
| `google-api-key` | Google API keys (`AIza` + 35 characters) |
| `slack-token`, `slack-webhook` | Slack `xox?-` tokens; Slack incoming-webhook URLs |
| `stripe-live-key` | Stripe live secret/restricted keys |
| `gitlab-token` | GitLab personal access tokens (`glpat-`) |
| `huggingface-token` | Hugging Face tokens (`hf_`) |
| `sendgrid-key` | SendGrid API keys |
| `ai-provider-key` | AI-provider secret keys (`sk-`, `sk-ant-`, `sk-proj-` ... with a long random body) |
| `bearer-token` | `Bearer <value>` where the value is long, random-looking and contains a digit |
| `credentials-in-url` | `scheme://user:password@host` with a non-placeholder password |
| `high-entropy-secret` | a random-looking literal (>= 20 characters, entropy >= 3.5 bits/char, letters and digits) assigned to a **secret-like name** |
| `secret-assignment` | another non-trivial literal (>= 8 characters, entropy >= 2.5, at least two character classes) assigned to a secret-like name |

A **secret-like name** is judged word by word (snake_case, kebab-case, camelCase, dotted): it contains
`password`, `passwd`, `pwd`, `passphrase`, `secret`, `token`, `credential(s)`, `bearer`, or one of the
pairs `api key`, `access key`, `private key`, `secret key`, `signing key`, `encryption key`,
`master key`, `auth key`. So `api_key`, `DB_PASSWORD`, `clientSecret` and `auth-token` qualify;
`tokenizer`, `max_tokens` and `cache_key` do not. Unquoted values are considered only in `.yml`/`.yaml`
files (elsewhere an unquoted right-hand side is usually code such as `password = get_password()`).

**Entropy alone never triggers a finding.** A random 40-character string assigned to `payload`,
`checksum` or `CACHE_KEY` is ignored; the same string assigned to `api_key` is reported.

## False-positive strategy

The rules are deliberately specific, and a value is ignored when it is:

* empty, shorter than 8 characters, a single repeated character, a mask (`xxxx`, `****`), or contains a
  placeholder marker (`your`, `example`, `sample`, `dummy`, `fake`, `placeholder`, `changeme`,
  `redacted`, `replace`, `insert`, `todo`, `123456`, `abcdef`, ...), or starts with `test`/`demo`/`mock`/
  `foo`/`bar` (so the AWS documentation example key ending in `EXAMPLE` is quiet);
* a template or reference: `${VAR}`, `{{ var }}`, `%(var)s`, `<token>`, `$VAR`;
* not a literal at all: `os.getenv("API_KEY")`, `process.env.API_KEY`, `settings.api_key`,
  `get_password()` (a literal must be quoted right after the `=`/`:`);
* the *name* of something: an environment-variable name (`GEMINI_API_KEY`), a dotted identifier;
* assigned to a name that describes a secret rather than being one: names ending in `url`, `path`,
  `name`, `id`, `hash`, `sha256`, `checksum`, `type`, `env`, `header`, ... (`token_url`,
  `password_hash`, `api_key_env`, `secret_id`), or containing `header`/`env`/`endpoint`;
* hashes, checksums, UUIDs and random data on ordinary names (not secret-like, so never examined);
* a one-character-class word or slug (`authorization-header-name`);
* a passwords-in-URL example whose password equals the user (`postgres:postgres@`).

Overlapping matches are reported once: the most specific rule wins (a provider token in
`github_token = "..."` is `github-token`, not also `secret-assignment`).

### Measured false-positive rate (limited evidence)

The rules were checked against third-party code that has nothing to do with this project: the Python
packages installed in the development environment (2,673 ingested `.py/.md/.json/.yaml` files,
about 34 MB). After tuning, the scanner reported **10 findings, none of them a real credential**:
documentation and docstring examples such as a sample password in a pydantic docstring, a sample
JWT in a Hugging Face docstring and sample `user:password@` URLs. Two further hits on that corpus (a
header-name constant and a URL-syntax description) were fixed by rules, with tests. This is one
corpus, reviewed by hand; it is **not** a precision/recall measurement, and recall (missed secrets)
was not measured at all. Documentation examples that look like real credentials will be blocked;
that is the intended fail-closed trade-off.

## Redaction design

* `SecretFinding` has no field that holds the value: `rule_id`, `severity`, `file_path`, `line`,
  `column`, `reason`, `preview`. Its `repr`, `model_dump` and JSON therefore cannot leak it.
* `preview` is a mask: values shorter than 16 characters become `****`; longer values show at most the
  first 4 and last 2 characters (`ghp_...ab` style), so a preview cannot reconstruct a value.
* The CLI prints `path:line:column [severity] rule` only; `--show-preview` adds the mask.
* `ScanTarget.text` is excluded from `repr`; log lines contain counts and rule ids only; exception
  messages contain rule ids, relative paths and line numbers only. All of this is tested.
* Paths are validated repository-relative POSIX strings; absolute host paths, drive letters and
  backslashes are rejected, so a finding can never leak the user's directory layout.

## The fail-closed external-LLM gate

```python
from copilot.security import assert_safe_for_external_llm, RepositorySecretRiskError

report = assert_safe_for_external_llm(chunks_and_user_text)  # raises if anything looks secret
```

* Accepts `ScanTarget`, `SourceFile`, `Chunk` and `RetrievalResult` objects (chunk line numbers are
  offset by the chunk's `start_line`). Anything else raises `TypeError` rather than being skipped.
  Text that is not repository content but will be sent to the LLM (the question, an error log) must be
  passed in as a `ScanTarget`.
* Any finding raises `RepositorySecretRiskError`, whose message says the repository context is
  **REFUSED for external LLM use**, lists at most 10 `path:line [rule]` entries, and says to remove
  the secret (rotating it if it was ever real) or leave the affected files out. The error carries the
  safe `SecretScanReport` as `.report`.
* **There is no bypass:** no `force` argument, setting or environment variable. A caller that fails the
  gate must not call the LLM.
* Wiring (Milestone 6): `copilot.llm.GuardedLLMClient` calls the gate on the exact outbound strings
  (system instruction and user prompt, i.e. the question and the retrieved chunks) immediately
  before delegating to the provider client; if the gate raises, the provider client is never entered.
  The plain baseline is covered the same way. See [`llm.md`](llm.md).

## CLI

```bash
uv run python -m copilot.security scan path/to/repo --ignore-dir data
uv run python -m copilot.security scan path/to/repo --json
```

It reuses the existing safe ingestion (same ignore rules and limits; repeat the `--ignore-dir` flags of
your index build), scans the accepted files' content and prints the counts by rule and severity and one
line per finding. Exit status: `0` no findings, `1` potential secrets found, `2` error or **incomplete
scan** (invalid path, or ingestion stopped early on a limit, so nothing can be declared safe).

## Self-scan of this repository

`python -m copilot.security scan . --ignore-dir data --ignore-dir submission` reports **0 findings**
over 149 accepted files (source, tests, docs, benchmark). A test (`tests/security/test_secret_self_scan.py`)
enforces this. Test fixtures never contain a complete token-shaped literal: they are generated at run
time from a fixed seed (`tests/security/secret_helpers.py`) and are random strings, **not real
credentials**. The untracked `submission/` folder is not part of the repository and was not scanned.

## Limitations

* **Pattern-based.** New or unusual token formats, custom internal keys, short or low-entropy
  passwords, single-word passphrases, values shorter than 8 characters and values with spaces are missed.
* **Obfuscated or split secrets** (string concatenation, base64/encoded values, values assembled at
  run time, a key spread over several lines without a header) are missed.
* Unquoted assignments outside YAML (`export TOKEN=...`, `.properties`) are not examined.
* The placeholder heuristic can be fooled: a real secret that starts with `test` or contains `example`
  is treated as a placeholder.
* Chunk-level scanning sees only the chunk; a private key split across chunks is caught by its header
  (`private-key-header`) but a body-only chunk is not.
* Only the **working tree** is scanned. A secret deleted from a file but present in Git history is not
  found, and credentials are never checked for validity or rotated.
* Documentation examples that resemble real credentials will be blocked (fail closed).
* No allowlist or suppression comment exists by design (an in-repo marker would be controlled by the
  content being scanned). A deliberate, reviewed exception mechanism is a future decision.
* The scanner does not defend against prompt injection, and it only protects text that is passed to
  it: the caller must pass everything that goes to the LLM.
* Files never ingested (binary, oversized, unsupported extensions, sensitive names) are not scanned,
  because they are never indexed or sent.
