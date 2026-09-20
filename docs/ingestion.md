# Repository ingestion (Milestone 2)

Ingestion turns a directory into a list of typed `SourceFile` objects that later stages (chunking,
embedding) consume. It is **discovery and safe reading only**: no parsing, chunking, embeddings,
or LLM calls happen here, and file *contents are never inspected for secrets* (see
[Known limitations](#known-limitations)).

```python
from copilot.ingestion import ingest_repository

result = ingest_repository("path/to/repo")  # limits come from Settings / .env
for f in result.files:  # sorted by relative_path
    print(f.relative_path, f.language, f.size_bytes)  # f.content holds the decoded text
print(result.stats.summary())
```

Manual inspection (prints statistics and relative paths, never contents):

```bash
uv run python -m copilot.ingestion path/to/repo --skipped     # add --json for machine-readable
```

## Pipeline

```
resolve root -> walk (prune dirs; classify each entry by type, name, size)
             -> bounded read -> decode -> repository-wide limits -> SourceFile | SkippedFile
```

Checks run cheapest-first and before any file is opened: link/special-file type, sensitive name,
generated name, extension, size from `stat`. Only then is a file opened and read.

## Supported file types (V1)

| Extension | Language |
|---|---|
| `.py` | python |
| `.js`, `.jsx` | javascript |
| `.ts`, `.tsx` | typescript |
| `.java` | java |
| `.c` | c |
| `.cpp` | cpp |
| `.md` | markdown |
| `.json` | json |
| `.yaml`, `.yml` | yaml |

The mapping lives in one place (`copilot/ingestion/languages.py`). Matching is case-insensitive
on the last suffix (`SCRIPT.PY` is Python; `a.test.ts` is TypeScript). Extend it per run without
editing code: `IngestionPolicy().with_languages({".go": "go"})`. There are **no parsers** for these
languages yet; extension-based identification is all Milestone 2 does.

## Ignored directories (pruned, never entered)

`.git .hg .svn node_modules bower_components dist build out venv .venv site-packages __pycache__
.pytest_cache .ruff_cache .mypy_cache .tox .nox .eggs coverage htmlcov .coverage_html .idea .vscode
.vs .gradle .next .nuxt .cache .terraform`, plus the patterns `*.egg-info` and `cmake-build-*`.
Matching is by directory name, case-insensitively, at any depth. Pruned directories are counted in
`stats.pruned_directory_names`; their files are never listed, so they do not appear in
`files_discovered`. Add more with `policy.with_extra_ignored_directories("data")`.

Ingestion does **not** read `.gitignore`.

## Sensitive-file policy

Sensitive files are rejected **by name**, before they are opened, and reported as `sensitive`:

* `.env`, `.env.*` (including `.env.example`, deliberately conservative), `*.env`
* private-key/certificate stores: `*.pem *.key *.p12 *.pfx *.jks *.keystore *.ppk *.gpg *.pgp`,
  `id_rsa* id_dsa* id_ecdsa* id_ed25519*`
* credential files: `.npmrc .pypirc .netrc _netrc .htpasswd .git-credentials .pgpass .dockercfg`,
  `*.tfstate*`, `*.tfvars`, `kubeconfig`
* directories `.ssh .aws .gnupg .kube` are pruned as a whole
* **stem rule** for data-like or unknown extensions: a file whose name before the first dot is
  `credentials`, `secret(s)`, `token(s)` or `password(s)` (or starts with `client_secret`,
  `service-account`, `service_account`) is rejected, e.g. `secrets.yaml`, `credentials.json`,
  `token.json`.

The stem rule is **not** applied to code or documentation extensions, so source modules such as
`secrets.py` or `token.md` are kept, and `tokenizer.json` is kept because its stem is
`tokenizer`, not `token`. Files are never rejected because their *contents* mention "token" or
"secret". Generated/vendored files (`package-lock.json`, `pnpm-lock.yaml`, `*.min.js`,
`*.bundle.js`, `*_pb2.py`, ...) are skipped as `ignored_file`.

## Symlink policy

**Symlinks are never followed** (files or directories). Windows directory junctions are treated
the same way. A link is skipped and reported as:

* `unsafe_path` if its target resolves outside the repository root (an escape attempt), or
* `symlink` otherwise (skipped by policy, which also prevents duplicate content).

The root you pass in *may* itself be a symlink (you chose it); it is resolved once and all
containment checks use the resolved root. Just before opening each file the resolved path is
re-checked to be inside the root, and the file is opened with `O_NOFOLLOW` (where available) so a
path swapped for a symlink after discovery is refused. Broken and self-referential links are
reported, not followed. Special files (FIFOs, sockets, devices) are rejected without being opened.

## Limits

| Limit | Default | Setting | Behaviour when exceeded |
|---|---|---|---|
| Single file size | 500,000 bytes | `COPILOT_MAX_FILE_SIZE_BYTES` | file skipped as `oversized` (exactly at the limit is accepted) |
| Accepted files | 10,000 | `COPILOT_MAX_REPO_FILES` | scan **stops**; `stats.truncated = True`, `truncation_reason = "max_files=N"` |
| Accepted bytes | 200 MB | `COPILOT_MAX_REPO_TOTAL_MB` | scan **stops**; `truncation_reason = "max_total_bytes=N"` |

A truncated result is incomplete: callers must show that to the user rather than let the system
claim "no evidence" about files it never read. When truncated, `files_discovered` is a lower bound.
Reads are bounded to `limit + 1` bytes even if a file grows after it was measured.

## Encoding and binary files

* UTF-16 is accepted **only with a BOM** (Windows PowerShell 5 writes it); otherwise strict UTF-8,
  with an optional BOM that is stripped.
* A NUL byte means binary (`binary`). Invalid UTF-8 is `encoding_error`. There is intentionally no
  latin-1 fallback, because it never fails and would accept binary garbage as text.
* `CRLF` and lone `CR` are normalised to `LF`, so line numbers are the same everywhere. **Downstream
  code must split lines on `"\n"`**, not `str.splitlines()` (which also splits on form feed,
  vertical tab and Unicode separators that editors do not count as line breaks).
* Empty files are valid (`content == ""`).
* I/O errors (permissions, file vanished) are `unreadable`; one bad file never stops the scan.
  Unlistable directories are counted in `stats.directories_unreadable`.

## Output model (`copilot.models`)

* `SourceFile`: `repository_name`, `relative_path` (canonical POSIX, validated), `extension`,
  `language`, `size_bytes` (raw bytes on disk), `content`, `sha256` (of the raw bytes, for change
  detection and index manifests). `content` is excluded from `repr`.
* `SkippedFile`: `relative_path` and `SkipReason`. **Never** carries content.
* `IngestionStats`: `files_discovered/accepted/skipped`, `skip_reasons`, pruned-directory counts,
  bytes, per-language counts, `truncated`, `elapsed_seconds`.
* `IngestionResult`: `repository_name`, `files`, `skipped`, `stats` (all deterministic and sorted).

**No absolute host paths are stored anywhere.** They differ between Windows, Linux and deployment,
would leak the user's directory layout into indexes, and are not needed: to read a file later use
`safe_join(root, relative_path)` (`copilot.utils.paths`), which rejects `..`, absolute paths, drive
letters, NUL bytes and symlink escapes, treating both `/` and `\` as separators on every OS.

`skip_reasons` values: `unsupported_extension`, `ignored_file`, `sensitive`, `binary`, `oversized`,
`unreadable`, `encoding_error`, `unsafe_path`, `symlink`, `special_file`, `limit_exceeded`.

## Logging

One `INFO` summary per run; `WARNING` for symlinks/unsafe paths, unlistable directories and
truncation; per-file skips at `DEBUG` (relative path + reason). File contents, absolute paths and
environment variables are never logged.

## Known limitations

* **No content scanning.** Files are judged by name only. A `.py` file with a hard-coded API key is
  ingested. Because retrieved code will later be sent to an external LLM, a lightweight content
  scan is a **mandatory security task that must be completed before any repository context is sent
  to an external LLM in Milestone 6** (deliberately deferred; ingestion scope is not otherwise
  expanded).
* `.gitignore` is not honoured, so ingesting this project itself would include a populated `data/`
  directory unless it is added with `with_extra_ignored_directories("data")`.
* Name-based rules are heuristics: false positives (a legitimate `credentials.json` fixture) and
  false negatives (a secret in an oddly named `.yaml`) are possible.
* Windows behaviour is exercised only through `PureWindowsPath` semantics in tests; junction
  handling, reserved device names and long-path (`\\?\`) behaviour are **untested on real Windows**.
* Not covered by tests: an oversized file that grows between `fstat` and `read`, and hard links to
  files outside the root (a hard link is indistinguishable from a normal file).
* A directory tree with millions of unsupported files is still walked in full; there is no cap on
  discovered entries or directory depth (traversal is iterative, so it cannot overflow the stack).
* Only extension-based language detection; a `.h` header or extension-less script is unsupported.
