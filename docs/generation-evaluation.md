# Plain LLM vs repository-aware RAG: answer-evaluation infrastructure (Milestone 11A)

Status: **infrastructure only.** Milestone 11A builds and tests everything needed to compare the
*answers* of a plain LLM and of repository-aware RAG, offline. **No hosted model was called, and no
answer-quality result exists.** Nothing in this document or in the tests claims that RAG answers are
better; the live comparison is Milestone 11B (commands at the end, **not executed**).

## What is compared

| | Arm A: plain LLM | Arm B: repository-aware RAG |
|---|---|---|
| Prompt | `plain-v2` (honesty rules, no evidence) | `rag-v1` (unchanged) |
| Evidence | none | top-5 chunks from the original line retriever |
| Held equal | same model and provider, temperature 0, `max_output_tokens` 8192, the question text (one benchmark field sent verbatim to both arms), the frozen repository |

The agentic arm is **deferred** (not part of this experiment). The protocol is a versioned file,
`generation-eval/protocol/primary-protocol.json`; its SHA-256 (which includes the confirmed model)
is stamped into every run, judgement and report, and a run can only resume under the same hash. The
generation model is deliberately **not** in the committed file: `run` requires
`--confirm-model PROVIDER/MODEL` and refuses a client that is not that model.

### Prompt fairness (decided before any generation)

`plain-v1` has no instruction to abstain or to label repository evidence, while `rag-v1` has, so a
RAG advantage could be an instruction effect. `plain-v2` adds the same honesty rules (do not state
repository specifics you cannot know; label "Repository evidence" vs "General knowledge, not from
this repository"; say explicitly when you cannot answer) with no evidence. `plain-v1` and `rag-v1`
are unchanged and pinned by SHA-256 (`compat.PINNED_PROMPT_DIGESTS`, tested). Residual asymmetry:
only RAG is told to cite `[Source n]`, so citation metrics apply to RAG (and to a plain answer only
if it writes explicit `file:line` locations).

### Retriever

The original line-window retriever is kept: 60 lines, 10 overlap, 512 tokens, Jina v2 base-code,
`prefixed` text, FAISS `IndexFlatIP`, top-k 5. Milestones 5b and 10 compared retrievers on
*retrieval* metrics; they did not show that any variant yields better *answers*, so no superiority
is claimed and nothing was tuned on this benchmark. Keeping it also keeps the verified historical RAG
responses comparable. The copilot reference index (`4aaeab9c650ab76c`) is used as is; it is not
rebuilt.

## Benchmarks (`generation-benchmark/1`)

Each question carries its answerability, key facts (each backed by sealed source regions), an
acceptable answer, and, for unanswerable questions, a rationale and the expected abstention.
`general_knowledge_sufficient` must be false for unanswerable questions (a general-programming
question is not an abstention test). Loader, sealing and drift checks reuse the retrieval-benchmark
machinery; `benchmark_digest` hashes the canonical text.

| Repository | Source | Questions | Answerable | Unanswerable | Key facts |
|---|---|---|---|---|---|
| `copilot` | frozen snapshot `a282ec47...` (the project itself) | 42 | 34 (the 34 preserved retrieval questions, same text and regions) | 8 | 118 |
| `inventory-orders` | new synthetic Python project, 30 ingested files, 54 tests | 30 | 24 | 6 | 74 |

Categories (copilot): algorithm 9, documentation 6, symbol-lookup 5, security 4, operations 4,
configuration 3, orchestration 3, persistence 2, process 2 and four singletons. Categories
(inventory-orders): implementation 6, cross-module 5, architecture 4, edge-case 4, symbol-lookup 2,
configuration 2, integration 2 and five singletons.

### The synthetic repository and leakage prevention

`generation-eval/repos/inventory-orders` was written for this study and frozen at commit `1cb147d`
(fingerprint `f633e7fe...d930522`, recorded in `inventory-orders.freeze.json` with
`frozen_before_questions_authored: true`) **before any question was written**. Its source is never
edited by the benchmark work. Protections: benchmarks and key facts live in
`generation-eval/benchmarks/`, outside any indexed repository root, and `.jsonl` is not an
ingestible extension; `tests/security/test_generation_leakage.py` asserts that no ingested file
contains an answer-key string and that the benchmark text matches the frozen repository. Unanswerable
questions were grep-checked against the frozen repositories (including the project docs).

Limits (also in the benchmark `.meta.json`): the same team wrote the code, the questions and the key
facts; the synthetic project is small, single-author and standard-library only. The copilot project
is the team's own, and its docs are in the index, so that arm measures retrieval-augmented use of a
repository a model has probably not seen, with author bias in question writing.

## Scoring

**Key-fact correctness (0-2, rubric `gen-rubric-1`)** - the primary measure. A judge (LLM or human)
gives a verdict per key fact (`supported` / `missing` / `contradicted`) and lists material
unsupported claims; `rubric.score_assessment` applies fixed rules:

* answerable: 0 if no fact is supported or more are contradicted than supported; 2 if all are
  supported, none contradicted and no unsupported claim; otherwise 1;
* unanswerable: 2 abstained with no unsupported claim; 1 abstained with unsupported claims; 0
  answered.

Keyword or string overlap is never used as correctness. A RAG system abstention (no evidence; the
model was never called) is scored by the same rules with `evaluator_kind="rule"` (no key fact is
conveyed), never by a judge.

**Programmatic checks** (no model involved): `citations.check_citations` (valid file and line range,
inside the retrieved evidence, overlapping the expected regions; `[Source n]` labels and explicit
`file:line` mentions are distinguished; answers without citations are "missing" for RAG and "not
applicable" for plain); `references.find_references` (conservative hallucinated-reference detector:
verified / not_found / ambiguous / cannot_evaluate / ignored; hypothetical sentences and external
dependencies are not flagged; inventory built with `ast` only, nothing is executed);
`refusal.classify_response` and `abstention_metrics` (answered / insufficient_evidence /
generic_refusal / ambiguous / no_response; appropriate-abstention, false-refusal, unsupported-answer
rates).

## The judge

`judge.LLMJudge` takes any `LLMClient` (provider-neutral; `FakeLLMClient` appears only in tests).
The request contains the question, the reference key facts with their source text, and the answer
with `[Source n]` labels replaced by `[cite]`. It contains no arm name, no model identity, no
retrieval scores. The answer and the repository text are fenced as untrusted data and marker
look-alikes are neutralised, which reduces but does not eliminate prompt-injection risk. Output is
strict JSON; **any** failure (provider error, blocked or unparsable output, unknown or missing fact
id, out-of-range grade) is recorded as a failed item with the raw output and retried on resume. It is
never defaulted. Every request passes the outbound secret gate.

Known limits: LLM judges can favour text from their own model family (a different family than the
generator is recommended; `same_model_as_generator` is recorded and reported), show position and
verbosity bias, and are sensitive to prompt wording (the judge prompt and the rubric are versioned;
a new version invalidates the cache). Their validity is checked, not assumed, by the human audit.

## Human audit

`audit-worksheet` draws a seeded sample (about 20%, stratified by repository x answerability, at
least one per stratum) from questions **both arms answered with text** (a system abstention would
reveal the arm), writes a blinded, shuffled worksheet (anonymous item ids, no arm or question id,
`[Source n]` redacted, human fields empty) and a separate key. `audit-agree` compares a filled
worksheet with the judge on the 0-2 correctness score (quadratic-weighted kappa), the abstention
flag and per-key-fact "supported" (Cohen's kappa) and always reports raw agreement; kappa is `null`
when a rater never varies (undefined). Partly filled rows are errors, not guesses. The audit shows
how much to trust the judge; it does not make the judge a ground truth.

## Execution design (resume, caches, failure handling)

* One run file per repository (`generation-<repo>.json`); every requested question is always
  present. Each side is `successful`, `failed` or `not_attempted`. Resume reuses successful sides
  unchanged and retries the rest, and **refuses before any call** if the protocol hash, benchmark
  digest, repository, index, model, settings, prompt versions or question set differ.
* Atomic checkpoint after every side; bounded exponential-backoff retry for 429/503/504; a circuit
  breaker shared across repositories stops after 3 consecutive hosted sides that end in 429 and marks
  the remainder `not_attempted` (never dropped).
* **Generation cache** (`ResponseCache`, under the secret gate, at the LLM boundary): key = provider,
  model, temperature, `max_output_tokens`, prompt version and the SHA-256 of the exact outbound text
  (question, system prompt and, for RAG, the retrieved evidence). A different model, setting, prompt
  or evidence can never be served another run's answer. Only successes are cached. **Judge cache**
  (`JudgeCache`): separate, key = judge provider/model/settings, judge prompt version, rubric version
  and input hash; successes only. Cache hits are not provider calls (they do not touch the 429 streak)
  and are excluded from latency analysis.
* Output from the fake provider is stamped `test_artifact` and written under
  `report-TEST-ARTIFACT.*`; reports label it and it is never a result.
* Run, judgement and cache files live under `data/generation-eval/` (git-ignored). They contain model
  answers and repository-relative paths only.

## Historical responses (Milestone 7.x)

`compat.check_historical_run` verifies, per side and read-only, schema, repository, index id,
provider, model, temperature, token budget, top-k, prompt version **and** that the prompt text still
hashes to its pinned digest, question text, side success, the index manifest's retriever spec, and for
RAG that every recorded source chunk exists in the index with the same file and lines. In
`data/evaluation/compare-8192-4aaeab9c650ab76c.json` there are 9 successful `plain-v1` and 8
successful `rag-v1` sides.

* The 9 `plain-v1` answers are **never** reusable in the `plain-v2` primary experiment (different
  system prompt); they remain a separate earlier experiment.
* The 8 RAG answers pass every check and are reusable **only** if the confirmed model equals the
  historical file's model and the protocol settings (8192 tokens, temperature 0, top-k 5) match.
  Otherwise none are. The other historical files (1024-token, quota-failed) have no reusable sides.
* Reused sides are marked `origin="historical"`, carry no token counts, and are excluded from latency.

## Statistical analysis

Primary endpoint (defined in the protocol before generation): the paired difference in mean key-fact
correctness, **RAG minus plain**, over answerable questions both arms answered and that were scored;
mean of per-question differences with a percentile bootstrap 95% CI (10,000 resamples, fixed seed,
shared indices; `copilot.evaluation.stats`), per repository and combined. Pairs with a missing,
failed or unscored side are **listed and counted, never dropped silently or imputed**; end-to-end
reliability (successful / failed / not attempted, judge failures) is reported separately from the
successful-answer analysis. Secondary: groundedness, completeness, citation correctness (where
applicable), hallucinated-reference counts, appropriate abstention, false refusal, retrieval hit@k
(RAG), latency (generated sides only) and input/output/reasoning tokens. The report states the
estimate, the interval and whether it excludes 0; it declares no winner. Every figure is recomputable
from the `rows` of the JSON report, which point at the raw files and carry their SHA-256 values.
Combining the two repositories does not replace showing each; with 34 + 24 answerable questions,
small differences are within noise.

## Estimated live call counts (upper bounds, from `plan`)

| | copilot | inventory-orders | total |
|---|---|---|---|
| Questions | 42 | 30 | 72 |
| Generation calls | 84 | 60 | **144** (136 if the 8 verified historical RAG sides are reused) |
| Judge calls | 84 | 60 | **144** (fewer: RAG system abstentions are rule-scored, and the judge cache can only lower it) |

## Commands (Windows PowerShell, from the project root)

Offline, safe to run any time:

```powershell
uv sync --group ui
uv run --group ui pytest -q
uv run python -m copilot.evaluation.generation plan --historical data/evaluation/compare-8192-4aaeab9c650ab76c.json
```

**Milestone 11B (NOT executed in 11A).** The inventory-orders index does not exist yet (the embedding
model is not cached where 11A ran). Build it with the same defaults as the copilot index, then
confirm its id is the one frozen in the protocol (`d115dc06e63659a2`); if it differs, stop and
re-freeze the protocol before generating:

```powershell
uv run python -m copilot.vectorstore build generation-eval/repos/inventory-orders --name inventory-orders
uv run python -m copilot.vectorstore info data/indexes/d115dc06e63659a2
```

Then, with the model chosen and confirmed by you (`.env` holds `COPILOT_LLM_MODEL` and the key):

```powershell
uv run python -m copilot.evaluation.generation plan --confirm-model gemini/<MODEL> --historical data/evaluation/compare-8192-4aaeab9c650ab76c.json
uv run python -m copilot.evaluation.generation run --confirm-model gemini/<MODEL> --historical data/evaluation/compare-8192-4aaeab9c650ab76c.json
uv run python -m copilot.evaluation.generation judge --confirm-model gemini/<MODEL> --judge-model <JUDGE_MODEL>
uv run python -m copilot.evaluation.generation report
uv run python -m copilot.evaluation.generation audit-worksheet
# rate data/generation-eval/primary/audit-worksheet.csv without opening audit-key.json, then:
uv run python -m copilot.evaluation.generation audit-agree --worksheet <filled.csv> --key data/generation-eval/primary/audit-key.json
```

`run` and `judge` are resumable (re-run the same command). Exit code 3 means the circuit breaker
stopped the run; completed sides are saved.

## Limitations

Single generation model; judge validity rests on a ~20% human audit by (at most) one or two raters;
both repositories were authored by the project team; the synthetic project is small and clean;
`plain-v2` cannot cite `[Source n]`; the programmatic reference checker is conservative and Python
focused (it misses claims it cannot verify and does not parse non-Python code); no claim about other repositories, languages or models is supported.
