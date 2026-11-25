# M12 — Evidence-Based Debugging from Python Tracebacks (approved plan, revision 2)

Status: approved with corrections. Authorised so far: **S1 only** (traceback parser).
Baseline: HEAD `af6d6ec`. M12 is the final original core development milestone. M11B research
execution (hosted judge validation, primary generation experiment) is separate and is **not** a
dependency of M12.

## 1. Approved decisions

1. Additive `RagService.answer_from_evidence` keyword arguments are approved.
2. The LangGraph `DebugWorkflow` is kept.
3. Scope is Python tracebacks only.
4. Benchmark fixtures live in the repository under `debug-eval/`.
5. Offline evaluation first. Hosted debugging generation is deferred.
6. This plan is saved as `docs/m12-debugging-plan.md` and Project doc `claude/M12_PLAN.md`.

Implementation order: **S1 → S2 → S3 → S4 → S8 → S9 → S5 → S6 → S7 → S10**. Each slice needs its
own authorisation.

## 2. Product behaviour

Input: a Python traceback pasted by the user (plus an optional question of at most 2000
characters) and an opened, fingerprint-verified index of the repository.

Output (a deterministic report; only the hypotheses part may come from a model):

1. `REPOSITORY EVIDENCE` — hash-verified repository text with `file:line-range`, origin and
   provenance for every item.
2. `REPORTED BY TRACEBACK (user-supplied, unverified)` — exception type/message, frame order,
   claimed source lines.
3. `UNMATCHED / AMBIGUOUS FRAMES` — each frame that did not resolve, with the reason.
4. `EVIDENCE LIMITATIONS` — weak path matches, source-line mismatches, suspected version drift,
   split symbols, external-dependency frames, truncated fields.
5. `POSSIBLE CAUSES (HYPOTHESES)` — always rendered. If none were generated the section says why.

Always included: **"No repository code was executed and no fix was verified."**

Evidence availability (describes evidence, **not** confidence in a diagnosis; a valid source line
does not prove the root cause):

| Status | Meaning |
|---|---|
| `DIRECT_EVIDENCE_AVAILABLE` | the innermost repository frame resolved and its line is in range |
| `PARTIAL_EVIDENCE` | some evidence, but the innermost frame is external, a match is weak or ambiguous, a line is out of range, or version drift is suspected |
| `INSUFFICIENT_EVIDENCE` | no reliable repository evidence — **no hosted explanation call is made** |

## 3. Mandatory corrections (all adopted)

### 3.1 Backward compatibility
Ordinary RAG and `AgentWorkflow` behaviour is unchanged. The new `RagService` parameters default to
the exact existing rag-v1 construction. Pinned by tests: system-prompt hash, user-prompt
construction, exact outbound request bytes, prompt version, citation numbering, context packing and
secret-gate behaviour. No existing experimental configuration may change silently.

### 3.2 AST evidence boundaries
An AST chunk is **not** assumed to be a complete function. Evidence is typed:

- `complete_symbol` — exact AST node boundaries (`lineno`..`end_lineno`, decorators included);
- `partial_symbol_fragment` — an oversized symbol that was split; the fragment range is stated;
- `class_header` — class line(s) and docstring only;
- `fallback_window` — a numbered line window when no AST answer exists.

Existing `AstChunker` output is used where it is exact; exact `ast` node ranges are used when it is
not. Qualified names and exact source ranges are preserved. Static references are **name-based
possible usages**, never verified runtime call relationships.

### 3.3 Conservative frame resolution
- A basename-only match is a **weak candidate** (`weak_candidate`), never a resolution and never
  proof of repository identity.
- Suffix matches keep their heuristic provenance (`match_basis`).
- Ambiguous, unsafe, external and unmatched paths never resolve silently.
- No filesystem path supplied by a traceback is ever opened. Only hash-verified indexed files are
  read, through the existing safe-path boundaries.
- A truncated path field (`path_truncated`) is **never** a mapping candidate.

### 3.4 Evidence availability wording
Replaced SUFFICIENT/PARTIAL/INSUFFICIENT by the three statuses in section 2. Weak matches,
source-line mismatches and suspected version drift are always listed under EVIDENCE LIMITATIONS.

### 3.5 Secret and privacy protection
- Complete traceback fields (message, source lines, raw paths) are scanned **before** retrieval
  terms are extracted; scanning only extracted tokens is not sufficient.
- Related retrieval uses the existing **local** embedding provider; traceback-derived queries are
  never sent to an external embedding provider (asserted by a test).
- Both existing outbound secret-gate boundaries are preserved.
- Secret-bearing traceback content **and** repository evidence is masked in every rendered output.
- Raw tracebacks, absolute private paths, exception messages and source text are never persisted in
  logs or execution traces (counts, codes and short hashes only).
- Traceback content and repository source are untrusted data. Model-generated hypothesis fields are
  validated and rendered safely. Prompt injection cannot modify deterministic sections.

### 3.6 Evidence ordering
Direct traceback evidence precedes optional related retrieval. If deterministic evidence is adapted
into `RetrievalResult` with `score=0.0`, tests verify that `build_context(..., preserve_order=True)`
neither discards nor reorders it ahead of the budget. Synthetic scores are never shown as
similarities. Origin, range and provenance are kept per item. Hypothesis `evidence_sources` must
refer to the **packed** evidence (`BuiltContext.citations`), not discarded candidates.

### 3.7 Deterministic report
The five sections above are always rendered. Invalid JSON, invalid source references or invalid
fields from the model cannot corrupt deterministic sections. No autonomous code modification,
subprocess execution, arbitrary filesystem access or imports of repository code.

### 3.8 LangGraph
`DebugWorkflow` calls the same deterministic functions as the plain pipeline (no duplicated logic),
with bounded conditional related retrieval and optional explanation. A test asserts that pipeline
and graph give identical deterministic output. `AgentWorkflow` is unchanged.

### 3.9 Research artifact protection
Recorded **before** implementation (HEAD `af6d6ec`) in `docs/m12/protected-artifacts.sha256`:
601 SHA-256 hashes covering `generation-eval/` (protocols, benchmarks, judge controls, validation
spec/profile/results, frozen fixture repo), `benchmarks/`, and the git-ignored `data/` artifacts
(pilot generation, judge-validation, chunking experiment, both indexes, evaluation, snapshot).
The earlier **204 baseline hashes** are retained and re-verified. Frozen Copilot snapshot:
`a282ec471ebf1387cb4feafcfe732ad24a431c7a`; indexes `4aaeab9c650ab76c` and `d115dc06e63659a2` stay
read-only and are never rebuilt. `055a8d5` is the **original benchmark identifier** (it names the
benchmark files); the frozen repository snapshot is `a282ec4`. These are distinct.

## 4. Architecture

```
parse_traceback        (S1, pure)            parsing/
resolve_frames         (S2, pure, no I/O)    debugging/resolver.py
RepositoryView         (S2, hash-verified reads of indexed files only)
collect evidence       (S3)                  debugging/evidence.py, references.py
assess + pipeline      (S4)                  debugging/pipeline.py, report.py, CLI `analyze`
benchmark freeze/eval  (S8, S9)              debug-eval/, evaluation/debugging/
hosted explanation     (S5)                  debugging/prompts.py, hypotheses.py, explain.py
DebugWorkflow          (S6, LangGraph)       debugging/workflow.py
Streamlit Debug tab    (S7)
docs                   (S10)
```

Reuse: `ingest_repository`, `read_manifest`, `repository_fingerprint`, `StaleRepositoryError`,
`safe_join`, `validate_relative_posix`, `AstChunker`, `Retriever`, `merge_evidence`,
`arrange_evidence`, `build_context`, `Citation`, `RagService.answer_from_evidence`,
`GuardedLLMClient`, the secret scanner/gate, `FakeLLMClient`, `literal`/`escape_markdown`, and the
benchmark `Region`/`verify_regions` and hash-pinned protocol pattern.

Runtime files that will be **modified** (additive only; each in its own slice):
`parsing/__init__.py` (S1), `rag/service.py` (S5), `services/copilot_service.py` (S4/S5),
`services/presentation.py` and `ui/streamlit_app.py` (S7). Everything else is new.

## 5. Traceback parser (S1)

Deterministic, line-based, linear-time, string-operation parser. No `eval`, `exec`, `compile`,
imports of traceback content, subprocesses or filesystem access. Traceback text is data.

Interface: `parse_traceback(text, *, limits=DebugLimits()) -> ParseResult` and
`looks_like_traceback(text) -> bool` (a cheap hint, not a validator).

Typed output: `ParseResult` (status `complete|partial|failed`, ordered `segments`, `diagnostics`),
`TracebackSegment` (frames, exception, chain link), `StackFrame` (path, path style, line, function,
claimed source line, repeat count, truncation flags), `ExceptionInfo`, `SyntaxDetail`,
`ParseDiagnostic` (fixed code + severity + positions, **no raw text**).

Limits (all explicit; exceeding a limit is never silent): input 50,000 chars (rejected), 3,000
lines (rejected), line 2,000 chars, 200 frames per segment (first 10 + last 190 kept, omission
counted), 20 segments (last 20 kept), path 1,024, function 200, source line 300, type name 200,
message 500 chars / 20 lines, 50 diagnostics.

Supported: standard tracebacks, multiple frames, chained exceptions (`cause`/`context`), Windows,
POSIX and relative paths, SyntaxError/IndentationError/TabError (with or without header), missing
source lines, Python 3.11+ caret lines, recursion-elided output, pseudo-files (`<string>`,
`<frozen …>`, `<stdin>`), ANSI colour stripping, malformed/incomplete input, explicit
unsupported-format diagnostics. Not claimed: every Python version/format variant.

Unsupported (reported, not parsed): ExceptionGroup tracebacks, pytest-style output, IPython-style
output, JavaScript/Java traces, log-prefixed lines, tracebacks of other languages.

## 6. Slices

Each slice lists reuse, new files, modified files, interfaces, security boundaries, tests,
acceptance, risk and complexity (detail as in the approved proposal; refinements below).

- **S1 Parser** — new `parsing/models.py`, `parsing/traceback.py`; modified `parsing/__init__.py`;
  tests: golden, adversarial, limits, no-leak, no-exec, deterministic fuzz. Complexity M–H.
- **S2 Resolver + `RepositoryView`** — pure resolver (`resolved_exact`, `resolved_absolute_root`,
  `resolved_unique_suffix` with `match_basis`, `weak_candidate`, `ambiguous`, `not_in_index`,
  `external_dependency`, `pseudo_file`, `invalid_path`, `unsafe_path`, `path_truncated`,
  `invalid_line`, `line_out_of_range`); `RepositoryView` re-ingests and verifies the manifest
  fingerprint. Complexity M.
- **S3 Structural evidence** — typed evidence kinds (3.2), exact AST ranges, name-based static
  references, adapter to `RetrievalResult`, ordering tests (3.6). Complexity M–H.
- **S4 Assessment + pipeline + `analyze` CLI** — statuses of section 2, report renderer, local
  masking, whole-field secret scan before term extraction, local-embedder assertion. Complexity M.
- **S8 Benchmark freeze** — ten seeded bugs plus negative cases (ambiguous paths, unmatched files,
  unsafe paths, malformed tracebacks, external dependencies, version drift, secret-bearing input,
  prompt injection). Tracebacks come from running **team-authored fixtures** with a developer-side
  script outside the product package; arbitrary user code is never executed. Fixtures, tracebacks,
  expected regions and protocol are frozen (hash-pinned) before any evaluation. Debug indexes live
  in a separate git-ignored directory. Complexity M.
- **S9 Offline evaluation** — ordinary exception-line RAG vs traceback-aware evidence, identical
  snapshots/embeddings/index configuration, matched context-budget comparisons. Reports: parsing
  accuracy, frame mapping accuracy, buggy-file recall, failing-line recall, enclosing-symbol recall,
  evidence tokens, investigation latency, citation validity, incorrect-resolution count,
  appropriate abstention, deterministic evidence/hypothesis separation. **Descriptive at n=10**; no
  claim of verified fixes or hypothesis correctness. Complexity M.
- **S5 Hosted explanation** (deferred past S9) — `debug-v1` prompt, strict JSON hypotheses, pins of
  3.1. Complexity M.
- **S6 `DebugWorkflow`** — see 3.8. Complexity L–M.
- **S7 Streamlit Debug tab** — escaped rendering, size-capped input, no persistence, evidence-only
  default. Complexity L–M.
- **S10 Documentation and viva notes.** Complexity L.

## 7. How M12 leaves the frozen M11B work untouched

- New code lives in new packages/directories; nothing under `generation-eval/`, `benchmarks/`,
  `data/` or `submission/` is written by M12.
- The protected-artifact record (3.9) and the 204 baseline are re-verified at the end of every
  slice.
- Prompts, protocol constants and defaults used by M11 are not edited; S5 pins their bytes.
- Debug limits are a separate dataclass, not new `Settings` fields.
- Fixture indexes are built into a separate ignored directory, never into `data/indexes`.

## 8. Known limits

Python tracebacks only. Static references are name-based. A traceback from a different repository
version yields plausible but wrong evidence (detected by source-line comparison in many, not all,
cases). Hypotheses may be wrong; the benchmark measures structure and citation validity, not
hypothesis correctness. n=10 is descriptive only.
