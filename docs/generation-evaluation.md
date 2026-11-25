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
`generation-eval/protocol/primary-protocol-v1.1.json` (see "Protocol v1.1 amendment" below; the
original `primary-protocol.json` is kept unchanged); its SHA-256 (which includes the confirmed model)
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
`refusal.classify_response` (cue patterns; a **diagnostic only**, see "Abstention classification")
and `abstention_metrics` (answered / insufficient_evidence / generic_refusal / ambiguous /
no_response / unrated; appropriate-abstention, false-refusal, unsupported-answer rates).

### Abstention classification (exact definitions)

Whether a response abstains is classified, in this order:

1. **System abstention** (the RAG system stopped for lack of evidence; the model was never called):
   `insufficient_evidence`, label source `system`, scored by rule.
2. **Generated response with a valid judge rating** (judge prompt `gen-judge-2`): the label comes
   from the judge's required, closed `response_type`: `answered`, `insufficient_evidence`,
   `generic_refusal`, `mixed` (reported as `ambiguous`: part answer, part abstention) or
   `ambiguous`. The judge must also give a consistent `abstained` flag (`true` exactly for
   `insufficient_evidence`/`mixed`); an inconsistent output is a parse failure. `abstained=true`
   alone is never read as an *appropriate* abstention.
3. **Text but no valid judge rating** (judge failed, not run, or an older judge file without
   `response_type`): `unrated`. It is counted and reported but excluded from every rate
   denominator. The cue patterns are **never substituted**.
4. **No text** (call failed, blocked, not attempted): `no_response`; a missing response is not a
   refusal.

The cue-pattern classifier is reported separately as `abstention_regex_diagnostic` with its
disagreements against the judge, a confusion table, and the concordance counts with denominators
(judge-classified responses only; system abstentions and unrated responses are not compared). It
was widened to recognise phrasings such as "Nothing in the repository shows ...", "There is no
mention of this in the code" and "The retrieved sources do not say", but remains a heuristic.

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

**Eligibility and transparency.** A question can be audited only if both arms have model text and a
valid judge rating. `audit-worksheet` classifies **every** benchmark question and writes
`audit-population*.json`: total questions and responses, completed pairs, eligible and excluded
pairs, exclusions by reason (generation failed / not attempted / no text / rule-scored system
abstention / judge rating missing; primary counts add up to the excluded total), rule-scored system
abstentions, failed and not-attempted responses, questions with missing judge ratings, the selected
sample size, and the sample as a share of **both** the eligible pairs and the full benchmark. If no
question is eligible the command exits 2 with that diagnostic and writes no worksheet.

**Official vs provisional.** An official sample may be drawn only when the experiment is final: no
side not attempted, a judgement file for every repository, and a valid judge rating for every
auditable answer (generation failures that remain after retries are reported as exclusions, not
blockers). Otherwise the command refuses, unless `--provisional` is given: the artifacts are then
named `...-PROVISIONAL...` and every worksheet row carries `audit_status = PROVISIONAL`. Sampling is
unchanged (seeded, stratified by repository x answerability, at least one per stratum, ~20%) and
drawn from the eligible pairs; the worksheet is blinded and shuffled (anonymous item ids, no arm or
question id, `[Source n]` redacted, human fields empty) with a separate key.

**Judge vs human.** `audit-agree` compares a filled worksheet with the judge on the 0-2 correctness
score (quadratic-weighted kappa and unweighted Cohen's kappa), the abstention flag and per-key-fact
"supported", always with raw agreement; a kappa that is undefined is `null` with a stated reason.
Every worksheet row is accounted for: eligible, unrated, invalid rating, duplicate id, unexpected id
(including a rule-worksheet id), key item missing from the worksheet, missing/failed judge rating,
or another named exclusion. Invalid, duplicate, unexpected or missing rows make the command exit 3
(the report is still written); nothing is skipped silently. Partly filled rows are errors, not
guesses.

**Rule-scored system abstentions** are not judge ratings, so they are verified separately: the same
command writes `audit-rule-worksheet*.csv` (ids `R-0001`, ...), a seeded stratified sample of the
system abstentions with two yes/no fields (the message makes no repository claim; it is a genuine
abstention), and `audit-rule-check` reports confirmed/disputed counts. This is a *rule-item
verification*; it is never pooled with, or reported as, judge-human agreement.

The audit shows how much to trust the judge; it does not make the judge a ground truth.

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

## Groundedness: how to read it

Groundedness is reported in two forms with denominators: **overall** (every scored response,
including rule-scored system abstentions) and over **substantive generated answers** (judge-scored,
not abstaining), plus the per-arm split into judge-scored, rule-scored and missing ratings. A
rule-scored system abstention makes no claims, so it receives groundedness 2 by convention - on an
answerable question it raises overall groundedness while its correctness is 0. Groundedness must
therefore always be read with correctness, completeness and the false-refusal rate; a higher
overall groundedness caused by abstaining is not a better answer.

## Historical-response sensitivity

The eight verified historical RAG responses (compatible Gemini configuration) stay in the primary
analysis, marked `origin="historical"` (new responses are `origin="generated"`; cache hits
`origin="cache"`), and are never regenerated, overwritten or modified. The report repeats the
primary endpoint **excluding every question that has a reused side** and shows both results with
their sample sizes, the number of excluded pairs and the bootstrap intervals. Two limits cannot be
removed: the hosted model may have drifted since those responses were produced, and their packed
prompt was not recorded in full (only the pinned prompt-template digests, the question, the index
identity and the retrieved chunk ids are verified).

## Secret gate and caches

Every outbound request is scanned by the secret gate **before** any cache lookup and before any
provider call: `LLMJudge.judge` scans before consulting the judge cache (a blocked input yields a
failed record, zero provider calls and no cache entry, even if a stale or planted entry exists),
`MeteredLLM.generate` scans before the generation cache (and the services additionally wrap the
client in `GuardedLLMClient`), so a cache can never serve a response for input the gate would block.
Blocked input is not persisted in any new artifact. Tests: `test_generation_cache_gate.py`.

## Protocol v1.1 amendment

`primary-protocol-v1.1.json` supersedes `primary-protocol.json` (v1, kept byte-for-byte; its SHA-256
is pinned in `protocol.PROTOCOL_V1_FILE_SHA256` and tested). It carries an explicit `amendment`
record. It was made **before any real generation call**, after the acceptance review found that the
abstention and groundedness measures did not measure what their names suggest. Changed: judge
prompt `gen-judge-1` -> `gen-judge-2` (adds `response_type`; rubric `gen-rubric-1` and its scoring
are unchanged), the definitions of the abstention and groundedness secondary endpoints, a new
historical-sensitivity endpoint, one new limitation, and the audit procedure above. Unchanged: arms,
prompts and their pinned digests, retriever, top-k, temperature, token budget, repositories,
benchmarks, key facts, the primary endpoint and bootstrap settings, audit fraction and seed.

## Bounded pilot (isolated)

`pilot-v1.json` records four predefined questions - one answerable and one unanswerable per
repository - chosen by the smallest SHA-256 of `seed|repository|answerability|question id`. Only
ids and the answerability flag are read; no historical answer quality, retrieval success, judge
score or M10 outcome is used, and the ids were recorded before any live call. Selected:
`copilot`: `impl-faiss-persist` (answerable), `u-uptime` (unanswerable); `inventory-orders`:
`io-price-snapshot` (answerable), `io-u-payment` (unanswerable). The pilot uses the same resolved
protocol (model, prompts, top-k, temperature, token budget), judge and rubric as the primary run,
in its own directory (`data/generation-eval/pilot`), with runs stamped `scope=pilot` (primary and
pilot commands refuse each other's files), no historical reuse, no import into the primary
analysis, and reports named `report-PILOT*`. Upper bounds: 8 generation calls + 8 judge calls = 16
provider calls before retries (RAG system abstentions and judge-cache hits only lower this). Retries
stay bounded, the rate-limit breaker and model confirmation apply, and every request passes the
secret gate. The pilot is **not executed** here; it is far too small for inference.

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

**Pilot (NOT executed in 11A; run before the full experiment):**

```powershell
uv run python -m copilot.evaluation.generation plan --pilot --confirm-model gemini/<MODEL>
uv run python -m copilot.evaluation.generation run --pilot --confirm-model gemini/<MODEL>
uv run python -m copilot.evaluation.generation judge --pilot --confirm-model gemini/<MODEL> --judge-model <JUDGE_MODEL>
uv run python -m copilot.evaluation.generation report --pilot
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
# verify the rule-scored system abstentions (if any were sampled):
uv run python -m copilot.evaluation.generation audit-rule-check --worksheet <filled-rule.csv> --key data/generation-eval/primary/audit-rule-key.json
```

`run` and `judge` are resumable (re-run the same command). Exit code 3 means the circuit breaker
stopped the run; completed sides are saved.

## Limitations

Single generation model; judge validity rests on a ~20% human audit by (at most) one or two raters;
both repositories were authored by the project team; the synthetic project is small and clean;
`plain-v2` cannot cite `[Source n]`; the programmatic reference checker is conservative and Python
focused (it misses claims it cannot verify and does not parse non-Python code); no claim about other repositories, languages or models is supported.
