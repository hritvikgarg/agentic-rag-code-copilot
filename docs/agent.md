# Agentic RAG workflow (Milestone 8)

A LangGraph workflow that sits **beside** the baseline RAG path. It adds an explicit loop that
retrieves in a few bounded rounds, checks what it has found, and then either answers through the
existing guarded RAG service or abstains. The baseline (`python -m copilot.rag`, `RagService.answer`)
is unchanged and does not import this package.

> **No quality claim is made here.** Nothing in this milestone measures whether the agentic
> workflow answers better than baseline RAG. The offline tests use a fake retriever and a fake
> model; they establish that the *control flow* is correct and safe, not that a real model gives
> better answers. Measuring quality is the job of the evaluation milestones.

## Three systems, kept distinct

| | Plain LLM | Baseline RAG | Agentic RAG (this milestone) |
|---|---|---|---|
| Repository evidence | none | one retrieval, fixed top-k | up to N retrieval rounds, accumulated |
| Query | the question | the question | the question, then refined queries built from the question's own words |
| Decides when to stop | n/a | n/a | deterministic evidence heuristic + budgets |
| Intent-specific behaviour | none | none | top-k, per-file cap, ordering, refinement order |
| Citations | none | from the context that was sent | from the context that was sent (same code) |
| Prompt, model call, secret gate | plain prompt | RAG prompt `rag-v1`, guarded | the same RAG prompt `rag-v1`, same guarded client |
| Code | `answer_plain` | `RagService.answer` | `AgentWorkflow.run` -> `RagService.answer_from_evidence` |

## Graph

```mermaid
flowchart TD
    S([start]) --> C[classify_intent]
    C --> R[retrieve]
    R -- retrieval error --> A[abstain]
    R --> M[merge_evidence]
    M --> E[assess_evidence]
    E -- insufficient, round and fresh query left --> R
    E -- potentially sufficient --> G[generate]
    E -- exhausted, partial evidence --> G
    E -- empty, nothing left to try --> A
    G --> X([end])
    A --> X
```

`python -m copilot.agents graph` prints LangGraph's own rendering of the compiled graph, so the
diagram can be regenerated and compared with the code.

### Nodes

| Node | Responsibility | Uses an LLM? |
|---|---|---|
| `classify_intent` | Route the question (ordered regexes), extract key terms and identifiers, build the retrieval plan | no |
| `retrieve` | One search with the current query and the plan's `top_k`, via the existing `Retriever` (embedding model + FAISS) | no |
| `merge_evidence` | Add the round's results to the accumulated evidence (dedupe, caps) | no |
| `assess_evidence` | Lexical sufficiency check, decide refine / generate / abstain, pick the next query | no |
| `generate` | Order the evidence for the prompt and call `RagService.answer_from_evidence` | **the only node that can** |
| `abstain` | Terminal node when there is nothing to answer from | no |

### Edges

Fixed: `start -> classify_intent -> retrieve`, `merge_evidence -> assess_evidence`,
`generate -> end`, `abstain -> end`. Conditional (plain functions of the state, unit-tested):

* `route_after_retrieval`: failure -> `abstain`, otherwise `merge_evidence`.
* `route_after_assessment`: `potentially_sufficient` -> `generate`; `insufficient_refinable` ->
  `retrieve`; `empty` -> `retrieve` if a round and a fresh query remain, else `abstain`;
  `exhausted` -> `generate` (with the partial evidence).

## State

`AgentState` (`src/copilot/agents/state.py`) is a `TypedDict`. Every node returns only the keys it
changes; `trace` has an `operator.add` reducer so nodes append events.

| Field | Meaning |
|---|---|
| `query` | the validated user question, never modified |
| `config` | `AgentConfig` (limits, below) |
| `intent`, `decision` | the router's output and the rule that fired |
| `plan` | `RetrievalPlan` for the intent |
| `terms` | the question's content words |
| `current_query`, `current_kind` | the next/last search text and where it came from |
| `asked_queries` | normalised forms of every query issued (repeat detection) |
| `round`, `max_rounds` | attempt count and budget (the first retrieval is round 1) |
| `results` | the latest round's `RetrievalResult`s |
| `evidence` | accumulated evidence across rounds |
| `new_accepted` | chunks the latest round added |
| `assessment` | `Assessment` (outcome, reason, term coverage, missing terms, stop reason) |
| `termination` | why the run ended (`TerminationReason`) |
| `answer`, `citations` | the `RagAnswer` and its `Citation`s |
| `abstention_reason`, `failure` | why nothing was generated; a typed `FailureInfo` |
| `trace` | the structured trace |

Existing types are reused: `RetrievalResult`, `Citation`, `RagAnswer`, `GenerationInfo`.

## Intent routing (`intent.py`)

Deterministic and LLM-free. Intents are tested in this precedence order and the first with a
matching rule wins:

1. `debugging`: problem words ("crash", "fails", "bug", "traceback", "unexpected"), a reported
   error ("getting an error"), an error class (`KeyError`), an HTTP 4xx/5xx status, or "why does X
   return/raise ...".
2. `symbol_lookup`: "where is/are ...", "where ... defined/used/configured ...", "which files
   implement ...", "find/locate ... function/class ...", "who calls/uses ...".
3. `architecture`: "architecture", "overview", "high-level", "project/repository structure", a flow
   between components, "how is the repo organised".
4. `code_explain`: "explain", "walk me through", "describe", "how does X work", "what does X do",
   "purpose of".
5. `general`: everything else.

Why this order: debugging and locational questions often contain explanation words ("explain why
it crashes"), and the narrower reading should win; `architecture` beats `code_explain` so that
"Explain the architecture of this repository" is an architecture question. `IntentDecision.matched`
lists every intent that had a matching rule, so overlaps are visible in the trace.

Ambiguity falls back to `general`: a question with no recognised shape (`general.fallback`) or with
no content word and no identifier at all, such as "What does this function do?"
(`general.too-vague`).

`debugging` currently only selects a retrieval strategy. The evidence-based debugging assistant
(error-log handling, hypothesis separation) is a later milestone and is not built here.

### What the intent changes

| Intent | Chunks per round | Per-file cap | Order in the prompt | Refinement order |
|---|---|---|---|---|
| `symbol_lookup` | base | none | chunks whose symbol/path mention a question symbol or term first | symbol, missing terms, focused terms |
| `code_explain` | base | none | discovery order | missing terms, focused terms, symbol |
| `architecture` | 2x base (max 50) | 2 | one chunk per file first, then the rest | missing terms, focused terms |
| `debugging` | base | none | discovery order | missing terms, symbol, focused terms |
| `general` | base | none | discovery order | missing terms, focused terms |

Assessment also differs: `symbol_lookup` needs the named identifier to appear in the evidence;
`architecture` needs at least `COPILOT_AGENT_ARCHITECTURE_MIN_FILES` distinct files.

## Refinement (`planning.py`)

Round 1 always searches the user's exact question. A later round uses one of:

* **missing terms**: the question terms that the evidence does not mention yet (plus the first
  identifier of the question), so the next search targets what is absent;
* **focused terms**: the question's content words without the question boilerplate;
* **symbol**: the identifiers that appear in the question.

Each is followed by one fixed intent hint word ("definition", "implementation", "overview
structure", "error handling"). Refined queries are built **only from words in the question** plus
that hint. Nothing is taken from retrieved text, so repository content cannot steer the search, and
no file path or symbol that the user did not write can appear in a query. A candidate whose set of
words equals an earlier query's is skipped; if every candidate repeats, the loop stops
(`no_new_query`).

## Evidence assessment (`assessment.py`)

A **lexical heuristic, not a relevance judge.** All of these must hold for
`potentially_sufficient`:

* the fraction of question terms found (after crude suffix stripping) in the accumulated evidence's
  text, file paths and symbol names is at least `COPILOT_AGENT_MIN_TERM_COVERAGE` (default 0.5);
* `symbol_lookup`: a named identifier appears in the evidence;
* `architecture`: enough distinct files.

Outcomes: `empty`, `potentially_sufficient`, `insufficient_refinable`, `exhausted`.

Limits, stated plainly: it cannot tell whether the evidence is *correct* or *answers* the question;
common words pass trivially; synonyms the code never uses fail even when the right chunk was
retrieved; the stemmer is crude. There is deliberately **no similarity-score threshold**: cosine
scores are not calibrated and not comparable across queries, so any universal cutoff would be
arbitrary. The thresholds are configuration, not tuned constants, and have not been tuned on data.

## Termination

The loop ends when the first of these holds (recorded as `termination` and in the trace):

| Reason | Meaning |
|---|---|
| `evidence_potentially_sufficient` | the heuristic passed |
| `max_rounds_reached` | the round budget is used (default 3 including the first, max 5) |
| `no_new_evidence` | a refined round added no chunk that was not already known |
| `no_new_query` | every refinement would repeat an earlier query |
| `no_evidence` | nothing was retrieved and nothing more can be tried |
| `retrieval_failed` / `generation_failed` | a typed failure, kept as data |

When the heuristic was *not* satisfied but some evidence exists, generation still proceeds on that
partial evidence and the CLI prints a warning; the baseline RAG prompt itself tells the model to
state insufficient evidence. Only an empty evidence set (or a context that cannot fit any chunk)
abstains without a model call.

Evidence is preserved across rounds and never evicted; the total is capped by
`COPILOT_AGENT_MAX_EVIDENCE_CHUNKS` (default 15), and the prompt is still bounded by the baseline's
`COPILOT_RAG_CONTEXT_MAX_TOKENS`. Chunks are deduplicated by `chunk_id` and by
`(file, start line, end line)`. Metadata is untouched; only `rank` is rewritten (1..n) to carry the
intent-specific order into the baseline context builder, and `score` stays the cosine of the query
that found the chunk.

## Generation and security

* Generation calls `RagService.answer_from_evidence` (the second half of the baseline's `answer`):
  same context builder, same prompts and `rag-v1` version, same limits, same
  `GuardedLLMClient`. The exact question and every included chunk are secret-scanned, then the exact
  outbound request is scanned again; a finding raises before any provider call. There is no flag
  that disables it.
* Citations are the baseline's: built from retrieval metadata of the chunks that entered the prompt,
  never parsed from model text. A `[Source 9]` label the model invents is reported as unknown, not
  turned into a citation.
* No generation on abstention. Provider errors keep their type (`LLMRateLimitError`, ...), message
  and limits and are returned in `AgentResult.failure`.
* Retrieved text is untrusted data. It is matched lexically and placed in the delimited context;
  it never selects an intent, a query, a node or a tool.
* Read-only: the package imports nothing that runs commands, writes files, or reaches the network;
  the graph has no tool node. The only outward call is the guarded model call. (Both properties are
  asserted by tests that scan the package source.)

## Trace

Every node appends a `TraceEvent`: node, round, intent, duration, safe query metadata (kind,
character count, word count, a 12-hex hash, never the text), counts retrieved / newly accepted /
total, assessment outcome, term coverage, routing decision, termination reason, generation status.
It never contains the question, refined queries, source text, prompts, model output, provider error
bodies, keys, or scanner findings. Example (a scripted offline scenario, three rounds):

```
classify_intent: round=0 intent=code_explain query_terms=5 top_k=5 rule_id=code_explain.explain-words
retrieve: round=1 query_kind=original query_chars=68 query_sha256_12=5f30f9fb7573 top_k=5 retrieved=1
merge_evidence: round=1 retrieved=1 new_accepted=1 evidence_total=1
assess_evidence: round=1 assessment=insufficient_refinable term_coverage=0.2 missing_terms=4 routing=refine note=term_coverage_below_threshold
retrieve: round=2 query_kind=missing_terms query_chars=46 query_sha256_12=46a254644184 top_k=5 retrieved=1
merge_evidence: round=2 retrieved=1 new_accepted=1 evidence_total=2
assess_evidence: round=2 assessment=insufficient_refinable term_coverage=0.4 missing_terms=3 routing=refine
retrieve: round=3 query_kind=missing_terms query_chars=38 query_sha256_12=96c5a99217de top_k=5 retrieved=1
merge_evidence: round=3 retrieved=1 new_accepted=1 evidence_total=3
assess_evidence: round=3 assessment=potentially_sufficient term_coverage=0.6 routing=generate termination_reason=evidence_potentially_sufficient
generate: round=3 evidence_total=3 generation_status=answered
```

## Configuration

Agent settings are separate from the baseline's (`.env.example`):

| Variable | Default | Range | Meaning |
|---|---|---|---|
| `COPILOT_AGENT_MAX_RETRIEVAL_ROUNDS` | 3 | 1-5 | rounds including the first |
| `COPILOT_AGENT_MAX_EVIDENCE_CHUNKS` | 15 | 1-50 | accumulated evidence cap |
| `COPILOT_AGENT_MIN_TERM_COVERAGE` | 0.5 | 0-1 | heuristic threshold |
| `COPILOT_AGENT_ARCHITECTURE_MIN_FILES` | 2 | 1-10 | distinct files for architecture questions |

The per-round chunk count starts from the baseline's `COPILOT_RETRIEVAL_TOP_K`.

## CLI

```
python -m copilot.agents ask INDEX_DIR --repo PATH "question" [--max-rounds N] [--trace] [--fake-llm] [--ignore-dir D]
python -m copilot.agents graph
```

`--fake-llm` replaces only the model with a deterministic stand-in (no network, no key, no model
id); retrieval still uses the real local index and embedding model. Exit status: 0 answered or
abstained, 1 refused by the secret gate, 2 any other error. Without `--fake-llm` it behaves like
`copilot.rag ask` (configured Gemini model, secret gate).

Windows PowerShell, offline, against the frozen evaluation snapshot and its index (read-only: the
workflow never writes to either):

```powershell
cd C:\Users\divya\agentic-rag-code-copilot
uv run python -m copilot.agents ask data\indexes\4aaeab9c650ab76c --repo data\snapshots\a282ec471ebf1387cb4feafcfe732ad24a431c7a\tree --ignore-dir data --ignore-dir tests --ignore-dir submission --fake-llm --trace "Where is the secret scanner gate implemented?"
uv run python -m copilot.agents ask data\indexes\4aaeab9c650ab76c --repo data\snapshots\a282ec471ebf1387cb4feafcfe732ad24a431c7a\tree --ignore-dir data --ignore-dir tests --ignore-dir submission --fake-llm --max-rounds 2 "Explain the architecture of this repository"
uv run python -m copilot.agents graph
```

Use the same `--ignore-dir` flags as the index build (the evaluation commands in
`docs/evaluation.md`); the retriever refuses a repository that no longer matches the index.

## Testing

`tests/unit/test_agent_*.py` (offline, no model, no network): routing for every intent, overlap and
ambiguity; intent-specific planning, ordering and assessment; real multi-round iteration with a
scripted retriever; budget, repeated-query and no-new-evidence stops; dedupe; abstention; generation
through the guarded client with the baseline's exact prompt; citations from the supplied evidence;
provider failures; secret-gate blocking with zero provider calls; injection-like retrieved text;
trace contents; the read-only property; CLI argument validation, Windows-style paths and exit codes;
baseline and evaluation-artifact compatibility. **These tests do not establish real-model quality.**

## Limitations

* The sufficiency heuristic is lexical; see above. It was not tuned or evaluated on real data.
* Refinement can only re-use the question's own words; it cannot reformulate semantically (that
  would need a model, which this milestone deliberately avoids in the control path).
* Intent rules are English regexes written by hand; unusual phrasings fall back to `general`.
* Refinement uses more retrieval calls, so latency grows with rounds; embedding cost is local.
* There is no agentic-vs-baseline accuracy result. Do not cite one.
* The evidence-based debugging assistant, structure-aware chunking and the chunking experiment are
  later milestones.
