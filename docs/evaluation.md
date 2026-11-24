# Retrieval evaluation (Milestone 5b)

Status: the framework is implemented and tested, and the six-configuration matrix (chunk cap 512 /
768 / 1024 x `prefixed` / `raw`) has been **run with the real Jina model on native Windows**. Results
and the resulting default-configuration decision are below. Everything here applies to one
self-authored, non-independent 34-question benchmark; **nothing is claimed about statistical
significance or about other repositories.**

## What is measured, and why

Retrieval is evaluated *before* any LLM exists, so that a later answer-quality difference can be
attributed to retrieval or to generation. The question is: for a natural-language question, does the
top of the ranking contain a chunk that overlaps the code or documentation that answers it?

## Benchmark format (`retrieval-benchmark/1`)

`benchmarks/<name>.jsonl`, one JSON object per line:

```json
{"id": "impl-chunk-id", "question": "How are chunk IDs generated?",
 "relevant": [{"file_path": "src/copilot/models/chunk.py", "start_line": 45, "end_line": 89,
               "sha256": "<hash of those lines>"}], "notes": ""}
```

- Ground truth is **file path + line range**, never a chunk id, so the same benchmark works for any
  chunking strategy, chunk size or representation.
- A question may list several relevant regions; matching any one counts.
- `sha256` is the hash of the region's text (lines `start..end` of the newline-normalised file joined
  with `\n`). It pins the ground truth to the exact source (LF and CRLF checkouts hash alike).
- Optional `<name>.meta.json`: repository name, commit, ignore directories, `independent` flag,
  description and limitations.
- `python -m copilot.evaluation seal BENCH --repo PATH` writes the hashes; `verify` checks that every
  region still matches (missing file, out-of-range lines, unsealed or changed text are reported).
  `run` and `matrix` verify first and refuse to score a benchmark that no longer matches.

## Match rule

A retrieved chunk matches a region if it has the **same `file_path`** and its line range **overlaps the
region by at least one physical line** (inclusive). Adjacent but non-overlapping ranges do not match.

## Metrics

| Metric | Definition |
|---|---|
| Hit@k | fraction of questions whose first matching chunk has rank <= k (k = 1, 3, 5, 10) |
| MRR | mean of 1/rank of the first matching chunk; 0 when nothing matches within depth (depth 10) |
| mean lines per hit | mean physical line count of the *first matching* chunk, over questions that hit |
| mean retrieved lines | mean line count of all retrieved chunks |

**The chunk-size confound.** A larger chunk covers more lines and therefore overlaps a small
ground-truth region more easily. Higher Hit@k for a larger cap can reflect "bigger target", not "better
retrieval". That is why mean lines per hit is always reported and **no winner is chosen from Hit@k
alone**. Precision-style metrics per line are not computed. No significance test is used; with 34
questions one question is 2.9 percentage points.

## The included benchmark (`benchmarks/copilot_self_055a8d5.jsonl`)

**Pinned repository:** this project at git commit **`055a8d5`** (recorded in the `.meta.json`).
**Self-authored, not independent:** the benchmark is authored by the same person who wrote the code,
documentation and retrieval system (`independent: false`).

34 questions about this repository at commit `055a8d5`: 28 implementation questions (relevant regions
are code only, so a documentation hit counts as a miss) and 6 documentation questions (a doc or code
region counts). Regions were chosen by reading the source. The benchmark is pinned to a git commit;
verify it against an export of that commit (`git archive 055a8d5 | tar -x -C <dir>`), not against a
newer working tree, where the regions will legitimately have moved.

**Limitations (also in the `.meta.json`, `independent: false`):**

- **Not independent.** The same author wrote the code, the documentation and the questions, and knows
  the vocabulary. Results are optimistic and say nothing about unfamiliar repositories. A later
  milestone should add at least one unfamiliar open-source repository.
- Small (34 questions): one question changes Hit@k by about 2.94 percentage points. Uncertainty must be assessed with paired outcomes and uncertainty intervals (Milestone 10); no noise threshold is assumed.
- The index was built with `--ignore-dir data --ignore-dir tests --ignore-dir submission`, so tests
  (natural distractors) are not in the corpus.
- Documentation is in the corpus and ranks above code for many questions. Whether that counts as a
  failure depends on the user's need; the implementation questions score it as a miss.
- The ground-truth regions are function-sized, and the questions were written after seeing the code.

## Cross-check: default configuration measured on Linux (real model)

Configuration: chunk cap 512 estimated tokens, `prefixed` representation, 60-line windows, 10-line
overlap, `jinaai/jina-embeddings-v2-base-code`, FAISS `IndexFlatIP`, index `25e726f8fefb403b` (259
chunks), top-k depth 10, benchmark `copilot-self-055a8d5` (34 questions), repository export of commit
`055a8d5`. Command:

```
python -m copilot.evaluation run benchmarks/copilot_self_055a8d5.jsonl --repo <export of 055a8d5> \
    --index data/indexes/25e726f8fefb403b
```

| Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | mean lines per hit |
|---|---|---|---|---|---|
| 47.1% (16/34) | 70.6% (24/34) | 73.5% (25/34) | 82.4% (28/34) | 0.603 | 44.6 |

This run was made before the Windows matrix and is kept as a cross-check: the Windows matrix row for
cap 512 / `prefixed` (below) reports identical Hit@k, MRR and mean lines per hit.

Where the measurement was made: on a Linux VM, query embeddings from the real Jina model loaded
from the model cache downloaded on the author's Windows machine (offline); the index itself was built
on Windows and its repository fingerprint matched the export. One run, no repetition, no confidence
interval.

Misses within depth 10 (6 of 34): `impl-embed-config`, `impl-long-line`, `impl-fake-embedder`,
`impl-index-id`, `impl-skip-reasons`, `doc-query-doc`. Several implementation questions were answered
first by documentation (for example "Where is repository ingestion implemented?" returned
`docs/ingestion.md` at rank 1 and `loader.py:1-60`, which holds only the module header and
`resolve_repository_root`, at rank 4; the first chunk overlapping the function `ingest_repository`
(lines 68-171) appeared at rank 6).

## Manual spot checks

The six queries requested for Milestone 5b, top 5, same index. Relevant files were read to judge them.

| Query | Top result | Judgement |
|---|---|---|
| Where is repository ingestion implemented? | `docs/ingestion.md:129-161`; `ingestion/__init__.py`; `loader.py:1-60` at #4 | The topic is right, but the function `ingest_repository` is not in the top 5 |
| How are chunk IDs generated? | `docs/chunking.md:118-156` | Documentation about chunk ids; `chunking/base.py:51-105` (#3) *calls* `make_chunk_id`; the definition in `models/chunk.py` is not in the top 5 |
| Where is the embedding model configured? | `docs/embeddings.md:31-64` | `config/settings.py` (where `embedding_model` is set) is not in the top 5 |
| How is the FAISS vector index persisted? | `vectorstore/faiss_backend.py:46-76` | Wrong: that region holds search/reconstruct helpers, not persistence. `docs/vector-index.md` at #2-#5 is relevant; `index.py` `save` is not in the top 5 for this wording |
| Where are unsafe repository paths rejected? | `ingestion/loader.py:51-109` | Relevant (repository root validation); `utils/paths.py` at #2-#4 is the main implementation |
| How is logging configured? | `config/logging_setup.py:1-60` | Correct |

Only the last query is fully right at rank 1. The pattern is consistent with the benchmark: the model
finds the right *topic* but often prefers prose (docs) to the code that implements it.

## Matrix results (real model, native Windows)

Run by the author on Windows with the real `jinaai/jina-embeddings-v2-base-code` model:
`python -m copilot.evaluation matrix` over the pinned export of commit `055a8d5`, benchmark
`copilot-self-055a8d5` (34 questions), top-k depth 10, FAISS `IndexFlatIP`, 60-line windows, 10-line
overlap. Held constant: repository commit, model, benchmark questions, depth, index type, line size,
overlap and the match rule (same file, at least one overlapping line). Varied: the estimated-token cap
and the embedding representation. One run per configuration, no repetition. The cap-2048 option was
not run. These numbers were supplied from the Windows run (not re-executed in the development
environment); the 512 / `prefixed` row was independently reproduced on Linux (see above).

| Cap | Style | Chunks | Index KB | Embed s | Mean chunk lines | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Mean lines / hit | Mean lines retrieved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | prefixed | 259 | 899 | 88.4 | 31.6 | **47.1** | **70.6** | 73.5 | 82.4 | **0.603** | 44.6 | 39.2 |
| 512 | raw | 259 | 899 | 84.0 | 31.6 | 38.2 | 58.8 | 70.6 | 85.3 | 0.524 | 43.3 | 36.2 |
| 768 | prefixed | 192 | 667 | 98.0 | 39.7 | 41.2 | 64.7 | 76.5 | 82.4 | 0.553 | 49.5 | 46.8 |
| 768 | raw | 192 | 667 | 90.8 | 39.7 | 35.3 | 61.8 | 76.5 | 85.3 | 0.514 | 51.6 | 44.6 |
| 1024 | prefixed | 166 | 577 | 101.4 | 44.4 | 41.2 | 64.7 | 70.6 | 82.4 | 0.562 | 49.5 | 49.4 |
| 1024 | raw | 166 | 577 | 99.8 | 44.4 | 32.4 | 61.8 | 76.5 | 85.3 | 0.508 | 51.6 | 47.0 |

Hit@k and MRR in percent / fraction as reported by the runner. Bold marks the best value in a column
for Hit@1, Hit@3 and MRR. Embedding time depends on hardware and is informative only.

With 34 questions one question is 2.9 percentage points. The counts behind the table (questions
hit, out of 34):

| Config | Hit@1 | Hit@3 | Hit@5 | Hit@10 |
|---|---|---|---|---|
| 512 prefixed | 16 | 24 | 25 | 28 |
| 512 raw | 13 | 20 | 24 | 29 |
| 768 prefixed | 14 | 22 | 26 | 28 |
| 768 raw | 12 | 21 | 26 | 29 |
| 1024 prefixed | 14 | 22 | 24 | 28 |
| 1024 raw | 11 | 21 | 26 | 29 |

### What the table does and does not show

- **Larger chunks have a line-overlap advantage.** Mean chunk size grows from 31.6 to 39.7 to 44.4
  lines, and the mean size of the first matching chunk from about 44 to about 50 lines. A bigger chunk
  overlaps a small ground-truth region more easily, so Hit@k for larger caps is *inflated* relative to
  retrieval quality. Even with that advantage, 768 and 1024 did not beat 512 / `prefixed` at Hit@1,
  Hit@3 or MRR. Larger caps look better only at Hit@5 (768 `prefixed` 26 vs 25 questions; 768 and 1024
  `raw` 26 vs 24), a one-to-two-question difference that is consistent with the size advantage.
- **`raw` reached higher Hit@10 (85.3 vs 82.4, 29 vs 28 questions) at all three caps, but weaker
  Hit@1 and MRR at all three caps.** In other words `raw` found the answer somewhere in the top 10
  slightly more often (by one question), while `prefixed` put it nearer the top more often
  (Hit@1 by 2-3 questions, Hit@3 by 4 questions at 512 and 1 at 768 / 1024, MRR by 0.04-0.08).
  Hit@5 is mixed (`prefixed` +1 question at 512, equal at 768, `raw` +2 at 1024).
- The differences are one to three questions on a benchmark written by the code's author. They are
  **not statistically tested and should not be read as significant.**
- Cost: a larger cap gives fewer, larger chunks and a smaller index (899 -> 577 KB); embedding time
  did not change materially (84-101 s) at this repository size.

## Engineering decision: default dense retrieval configuration

**KEEP chunk cap 512 with the `prefixed` embedding representation as the default.** No configuration
change is required; this is the existing default (`chunk_max_tokens=512`, `embedding_text_style="prefixed"`).

Reasons, all from the table above:

1. Highest Hit@1 (47.1%, 16/34).
2. Highest Hit@3 (70.6%, 24/34).
3. Highest MRR (0.603).
4. Smaller chunks (mean 31.6 lines) than 768 / 1024, hence the smallest line-overlap advantage of the
   caps tested, so its lead is not explained by chunk size.
5. Downstream RAG puts only the first few retrieved chunks into the LLM context, so ranking quality at
   the top of the list matters more than whether the answer appears somewhere in the top 10.

What this decision is and is not:

- It is a **default for this project's dense retriever, based on one self-authored 34-question
  benchmark.** It does not claim that 512 or `prefixed` is better in general, on other repositories,
  or with other chunking strategies.
- No significance is claimed. `raw` reached higher Hit@10 at every cap; if a later use case cares
  about recall within a large top-k, the choice should be revisited.
- Revisit when an independent repository is added to the benchmark and when structure-aware chunking
  (Milestone 9) exists.

## Reproducing

```
python -m copilot.evaluation verify BENCH --repo EXPORT
python -m copilot.evaluation run    BENCH --repo EXPORT --index INDEX_DIR
python -m copilot.evaluation matrix BENCH --repo EXPORT --work-dir DIR --caps 512 768 1024 --output FILE
```

The matrix needs a machine with several GB of free RAM: on a 3.9 GB development VM the embedding
process was killed by the operating system, which is why the reported matrix was run on Windows.

Use the same `--ignore-dir` values as recorded in the `.meta.json` (they are the defaults). Reported
timings depend on hardware and are informative only.


# Plain LLM vs repository-aware RAG (Milestone 6)

Status: **the comparison framework is implemented and tested with fake models. A first live
34-question run was attempted and hit sustained Gemini rate limiting; it is not a valid
plain-vs-RAG result and none is claimed from it - see "The first live 34-question run was not
a valid comparison" below.** No comparison result of any kind exists yet, and quality
judgements still need a human reader.

## Question being asked

Given the same LLM, the same generation settings and the same questions, does giving the model
retrieved repository evidence (RAG, `rag-v1`) change the correctness, grounding, citation accuracy
and hallucination behaviour compared with sending the bare question (plain, `plain-v1`)?

## Design

* Questions: a small subset of the existing 34-question benchmark
  (`--limit N` = the first N, or `--question-id ID` repeated). Its limits apply in full: self-authored,
  not independent, small, and the plain model may have seen this project's public code.
* Both systems use the same client, model, temperature and token cap. Both go through the secret gate.
* Everything about answer *quality* is a **human judgement**. The LLM is not used as a judge, because
  a judge from the same model family can share the answerer's blind spots and hallucinations.
* The only automatic measurement is objective: whether an included source overlaps a ground-truth
  region (`first_relevant_source`), reusing the retrieval match rule. It says nothing about the answer.
* Rated entries only are aggregated; the summary states how many were rated and declares no winner.

## Workflow

```powershell
# 1. verify the benchmark against the repository (no LLM)
uv run python -m copilot.evaluation verify benchmarks\copilot_self_055a8d5.jsonl --repo .
# 2. collect paired answers for a subset (calls the hosted model after the secret gate)
uv run python -m copilot.evaluation compare benchmarks\copilot_self_055a8d5.jsonl --repo . `
    --index data\indexes\<index id> --limit 8 --delay-seconds 2 --output data\evaluation\compare-8.json
# 3. open the JSON, read each answer against the source at the pinned commit, fill in the ratings
# 4. aggregate what was rated
uv run python -m copilot.evaluation summarize data\evaluation\compare-8.json
```

`data/` is git-ignored: comparison files contain model answers about repository code and are not
committed. Each question is recorded with: the question, expected regions, the plain answer (or
error), the RAG status/answer/sources/cited and unknown source numbers/retrieval and total latency,
and empty `plain_rating` / `rag_rating` slots. A provider error or gate refusal on one question is
recorded and the run continues, retrying a transient provider failure (HTTP 429/503/504) with a bounded backoff first - see "Resuming an interrupted or rate-limited run" below for what counts as transient, the retry/pacing distinction, and `--resume`, which continues an existing `--output` file instead of starting over.

## Manual rubric (`answer-comparison/1`, also embedded in every file)

| Dimension | Scale | Meaning |
|---|---|---|
| `grounded_correctness` | 0-2 | Repository-specific claims checked against the source at the pinned commit: 0 mostly wrong/unsupported, 1 partly correct or vague, 2 correct and specific. |
| `citation_correctness` | 0-2, RAG only | Open each cited `file:lines`: 0 none support the claims, 1 some do, 2 all do. Leave null for plain. |
| `hallucinated_claims` | integer >= 0 | Count of distinct invented repository facts (files, functions, classes, parameters, line numbers). General programming knowledge does not count. |
| `completeness` | 0-2 | 0 misses the question, 1 partial, 2 covers what the reference regions show. |
| `appropriate_abstention` | true/false/null | true: it declined and the evidence really was missing (or the question is unanswerable); false: it declined despite available evidence, or answered confidently without support; null: not applicable. |

Rate blind if possible (hide which system produced which answer) and let a second person rate a
sample; with one rater and a few questions, treat the numbers as anecdotal. Because every
question in this benchmark has an answer in the repository, `appropriate_abstention` will rarely
apply; abstention is better tested with a few deliberately unanswerable questions, which do not
exist yet.

## What this experiment can and cannot show

* It can show, on a handful of questions, whether RAG answers cite real files and avoid invented
  ones more often than plain answers, and where RAG fails (retrieval missed the target).
* It cannot show significance (small n, one rater, non-independent benchmark, non-deterministic
  models), generalisation to other repositories, or the benefit of RAG over a prompt-only honest
  baseline (not run; the plain prompt has no abstention instruction).
* A low RAG score with `first_relevant_source = null` is a *retrieval* failure, not necessarily a
  generation failure; report the two separately.

## The first live 34-question run was not a valid comparison

A native-Windows run of `compare` against all 34 questions (index `4aaeab9c650ab76c`, model
`gemini-3.6-flash`, `top_k=5`, `temperature=0.0`, `max_output_tokens=1024`, `--delay-seconds 1`)
hit sustained Gemini rate/quota limiting:

| system | successful | failed | failure breakdown |
|---|---|---|---|
| plain | 3/34 | 31/34 | 30 x HTTP 429, 1 x HTTP 503 |
| RAG | 5/34 | 29/34 | 29 x HTTP 429 |

Only three questions (`impl-chunk-id`, `impl-embed-config`, `impl-symlinks`) had **both** sides
succeed. RAG additionally succeeded alone on `impl-ingest` and `impl-sensitive-files` (the plain
side of `impl-ingest` hit the one HTTP 503). Every question from `impl-read-decode` onward failed
on both sides with HTTP 429. **This is not a valid plain-vs-RAG result and is not summarized as
one**: with 31/34 plain answers and 29/34 RAG answers missing, the three (or five) surviving pairs
say nothing about whether RAG changes answer quality - they mostly say that 1 request/second was
not slow enough for the quota in effect at the time. No conclusion about RAG quality is drawn from
this run, and none should be until a resumed run actually completes both sides for a real subset.

The root cause was pacing, not the code path: a fixed `--delay-seconds 1` between requests
(`run_comparison`'s own `sleep` calls) does not adapt when the provider is already refusing
requests, and a single provider error was previously **recorded and moved on**, so a transient
429 was treated exactly like a permanent one - correct behaviour for a genuinely wrong request,
wrong behaviour for a provider asking to be tried again shortly. The `--resume`/retry/checkpoint
mechanism below exists to fix exactly this, without touching retrieval, chunking, embeddings, the
benchmark, the prompts, or any generation setting.

## Resuming an interrupted or rate-limited run

`compare --resume` continues an existing `--output` file instead of starting over:

* **Compatibility is checked before any hosted call.** The existing file's schema version,
  benchmark name, repository name, index id, LLM provider/model, plain/RAG prompt versions,
  `top_k`, `temperature`, `max_output_tokens`, and the exact set of question ids must all match
  the requested run, or `compare` refuses with `ComparisonMismatchError` and calls nothing. Two
  different experiments can never be silently merged into one file.
* **A side counts as successful when its `error` field is `null`** - `"answered"` and
  `"insufficient_evidence"` are both legitimate, non-error RAG outcomes and are reused exactly
  like a successful plain answer. A successful side (answer, error, latency, sources, citations,
  and any `plain_rating`/`rag_rating` already filled in) is reused byte-for-byte; only a side
  whose `error` is not `null` is re-asked. If a question's plain side succeeded and its RAG side
  failed, resuming makes exactly one new hosted call for that question (RAG only), and
  vice versa; if both already succeeded, resuming that question makes zero hosted calls.
* **Checkpointing is atomic and incremental.** After every hosted side (Milestone 7.1; it was
  once per question), the accumulated comparison run - always listing *every* requested question,
  with `not_attempted` placeholders for the ones not reached - is written to `--output` via `copilot.utils.atomic_file.write_text_atomically` (temp file in
  the same directory, `fsync`, then one `os.replace`) - the same "write, sync, then rename"
  pattern `copilot.vectorstore.atomic` already uses for index directories, applied to a single
  JSON file instead of a directory of files. A reader (including a later `--resume`) only ever
  sees the previous complete file or the new complete file, never a half-written one. Ctrl+C, a
  crash, or exhausting the retry budget loses at most the one request in flight - every side
  completed before it stays on disk exactly as it was, and `KeyboardInterrupt`
  is never caught and recorded as a fake model failure; it aborts the run immediately and `compare`
  reports where the checkpoint stopped.
* **Retry backoff is separate from request pacing.** `--delay-seconds` (unchanged default `0.0`)
  still pauses after *every* request, successful or not - it paces a well-behaved run. The new
  `--retry-*` flags govern what happens only after a transient provider failure: HTTP 429
  (`LLMRateLimitError`), HTTP 503 (`LLMProviderError` with `status_code == 503`) or HTTP 504
  (`LLMTimeoutError` with `status_code == 504`) - see
  `copilot.llm.retry` for the full reasoning. Nothing else is retried: a bad model name, bad
  credentials, a blocked/malformed response, or the secret gate's refusal are fixed conditions
  that another identical attempt cannot change, so they are recorded immediately, exactly as
  before. Defaults: 5s initial wait, x2 growth, capped at 60s, 5 attempts total (1 try + 4
  retries) - chosen because the run above kept hitting *sustained* 429s at 1-second spacing, so a
  short first wait clears a brief burst while the cap keeps one stuck question from blocking a
  resume for minutes. A Gemini `Retry-After` response header, when the SDK exposes one, extends
  (but never exceeds) the computed wait; in practice Gemini has not been observed to send it, so
  this is a best-effort improvement, not something the policy depends on.
* **Reporting.** `compare` always prints the total question count and, per system, how many
  succeeded (and how many of those were reused from the existing file, when resuming) versus how
  many still have an error after retries, whether the run is complete (zero remaining errors on
  both sides), and the output path. A question left with an error after retries keeps that typed
  error in the JSON so a later `--resume` can retry it again with a fresh backoff budget.

## Stopping early on sustained rate limiting (Milestone 7.1)

The second hosted attempt (`max_output_tokens=8192`) made 17 successful calls and then received
HTTP 429 for every one of the next 25 questions, plus two HTTP 504 timeouts. Retrying with backoff
cannot help when the provider keeps refusing, and each exhausted side cost about 75 seconds, so the
run spent about 80 minutes on 51 doomed sides. A **circuit breaker** now sits above the retry layer.

* **Side states.** Every side is `successful`, `failed` (a call was made and ended in an error) or
  `not_attempted` (no call was ever made; stored as `not_attempted: true` on a plain record and
  `status: "not_attempted"` on a RAG record). `compare` prints all three counts per system.
* **The rule.** When `--max-consecutive-rate-limit-failures` (default 3; `0` disables) hosted
  sides *in a row* have exhausted their retries with HTTP 429, the run stops, writes the checkpoint
  and exits with code **3**. The streak counts only hosted calls that ended in a 429. A hosted
  success resets it, and so does any other outcome, because the streak is *consecutive 429s*. A
  side reused from `--resume` is not a hosted call and neither extends nor resets the streak. A 429
  that a retry cleared is a success, not a failed side.
* **What the message says.** "Stopped after N consecutive hosted request(s) ended in HTTP 429 (rate
  or quota limit) after exhausting their retries. The provider did not say which limit was hit."
  It deliberately does not say "daily quota exhausted": Gemini's error text does not name the limit
  and this project does not guess. The reason is stored in the file as `stop_reason`.
* **Nothing is lost.** The file written at the stop lists all questions. Successful answers, their
  latencies and sources, and any ratings already entered are kept byte for byte. A later
  `--resume` asks only the `failed` and `not_attempted` sides, and clears `stop_reason`.
* **HTTP 504** (gateway timeout, `LLMTimeoutError` with `status_code == 504`) is now retried like
  429/503, with the same bounded backoff. A client-side timeout (no HTTP response), HTTP 408,
  authentication, configuration, model-not-found, blocked/malformed responses, other 4xx/5xx and
  the secret gate's refusal are never retried.
* **Benchmark drift is refused on resume.** If a question's text or expected regions in the file
  differ from the benchmark you are resuming with, `compare` refuses before any call: the stored
  retrieval hits would describe different regions.

## Evaluating a frozen snapshot while the application keeps changing (Milestone 7.1)

An index is valid only for the exact accepted source content it was built from (its
`repository_fingerprint`, see `copilot.vectorstore.fingerprint`). The benchmark indexes this
repository, so **every commit that touches an indexed file makes the live checkout stale** for the
existing index, and `--repo .` is refused with "the repository does not match the index". The
fix is to stop pointing the evaluation at the live checkout.

| Thing | Where it lives | Changes when |
|---|---|---|
| application under development | the repository checkout | every commit |
| repository being evaluated | `data/snapshots/<commit>/tree/` | never (frozen) |
| vector index | `data/indexes/<index_id>/` | never (new id for any new spec) |
| comparison results | `data/evaluation/<name>.json` | `--resume` fills it in |

A snapshot is `git archive <commit>` extracted into the git-ignored `data/` directory (a `git
worktree` would add branch/worktree state to `.git` and could be edited into a different state by
mistake). `snapshot create --index ...` extracts into a staging directory, recomputes the
fingerprint with the same ingestion the retriever uses, installs the tree only if it equals the
index's fingerprint, and otherwise installs nothing. Index manifests and comparison files are never
modified, and an existing snapshot is never overwritten. The metadata file `snapshot.json` sits next
to `tree/` rather than inside it because `.json` is an accepted source extension and would change
the fingerprint.

**Which snapshot belongs to the current comparison.** `data/evaluation/compare-8192-4aaeab9c650ab76c.json`
records index `4aaeab9c650ab76c`, whose manifest records fingerprint `e7709ab39fff91d4...`.
Recomputing the fingerprint for each of the 18 commits (ignore directories `data`, `tests`,
`submission`) matches exactly two: `a282ec4` and `166f48b`. They differ only in
`benchmarks/*.jsonl`, which is not an accepted source extension. The comparison was run with the
benchmark as re-anchored in `a282ec4`, so that is the snapshot to use. Between `a282ec4` and the
7.1 work nothing changed under `rag/`, `retrieval/`, `embeddings/`, `chunking/`, `ingestion/`,
`vectorstore/` or `config/`, and the 34 questions and regions in the benchmark equal those stored in
the comparison file.

```powershell
# once: freeze the commit and verify it against the index (nothing is installed on a mismatch)
uv run python -m copilot.evaluation snapshot create --commit a282ec4 `
    --index data\indexes\4aaeab9c650ab76c --ignore-dir data --ignore-dir tests --ignore-dir submission

# any time: re-check a snapshot
uv run python -m copilot.evaluation snapshot verify `
    --snapshot data\snapshots\a282ec471ebf1387cb4feafcfe732ad24a431c7a\tree `
    --index data\indexes\4aaeab9c650ab76c --ignore-dir data --ignore-dir tests --ignore-dir submission
```

Rules for Milestones 8-10 and beyond: always pass the snapshot's `tree` as `--repo` for
`compare`/`run`/`verify`; never rebuild `4aaeab9c650ab76c` or "re-anchor" the benchmark to a newer
commit while a comparison against it is in progress; a *new* experiment (different chunking, new
benchmark anchors) gets a new snapshot, a new index and a new comparison file, never a merge into
the old one. Ingestion, line chunking and the fingerprint formula must keep reproducing the old
chunks: a change there changes the fingerprint and `Retriever.open` will refuse the snapshot,
which is the intended alarm.

## Resuming the 8192-token comparison

Copy the file first if you want a backup (a resume rewrites `--output`), then:

```powershell
copy data\evaluation\compare-8192-4aaeab9c650ab76c.json data\evaluation\compare-8192-4aaeab9c650ab76c.before-resume.json
uv run python -m copilot.evaluation compare benchmarks\copilot_self_055a8d5.jsonl `
    --repo data\snapshots\a282ec471ebf1387cb4feafcfe732ad24a431c7a\tree `
    --index data\indexes\4aaeab9c650ab76c --output data\evaluation\compare-8192-4aaeab9c650ab76c.json `
    --resume --delay-seconds 6
```

The file currently holds 8 complete pairs and one plain-only success; `--resume` would ask the
other 51 sides. It stops after three consecutive 429s (exit code 3), so repeating this later costs
only the calls that can still succeed. The `.env` must still say `COPILOT_LLM_MAX_OUTPUT_TOKENS=8192`
and the same model, or the compatibility check refuses.

## Runs that are kept but are not results

* `compare-quota-failed-4aaeab9c650ab76c.json`: the first run (`max_output_tokens=1024`, mostly
  HTTP 429; 5 of its 8 successful answers end mid-sentence). Invalid.
* `compare.json` / `compare.backup.json`: a later all-429 run on a different index
  (`e62c923901fcde16`). Invalid.
* `compare-8192-4aaeab9c650ab76c.json`: the resumable run described above; not a result until it
  is complete and rated.
