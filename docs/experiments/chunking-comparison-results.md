# Results: controlled comparison of chunking strategies (Milestone 10)

This is a **retrieval** experiment on one repository and 34 questions. It says nothing about the
quality of generated answers. The design, hypotheses and analysis plan are in
[`chunking-comparison.md`](chunking-comparison.md) and were frozen (commit `b64da49`, config SHA-256
`680a87c5…`) before any query was embedded for Arms B, C or D. Every number below is read from
`experiments/chunking-comparison/results/*.json`; the complete generated tables are in
`experiments/chunking-comparison/report/`.

Confidence tags: **[Certain]** is a direct reading of a measured number, **[Likely]** a strong
inference from several numbers, **[Guessing]** an untested explanation.

## 1. Setup in one paragraph

Frozen snapshot `a282ec47…` of this repository (100 accepted files, 34 benchmark questions, 41
relevant regions), embedding model `jinaai/jina-embeddings-v2-base-code`, flat inner-product FAISS.
Documentation and non-Python files are chunked
identically in all four arms, so every difference comes from how Python files are cut.

| Arm | Chunker | Python chunks | Median / mean est. tokens | Index id |
|---|---|---|---|---|
| A | production line windows (60 lines, 10 overlap, 512 token cap) | 234 | 469 / 399 | `4aaeab9c650ab76c` |
| B | production AST (function / class / method, line fallback) | 490 | 122.5 / 161 | `e31fab15e045ea8c` |
| C | line windows, 15 lines, no overlap (size-matched to B) | 638 | 119 / 123 | `82e470046c96a49d` |
| D | AST units merged up to 512 tokens (no symbol metadata on merged chunks) | 233 | 369 / 339 | `4d89befefa68250f` |

Size matching worked for **C on the median** (−2.9 % against B, inside the 10 % tolerance) but not
on the mean (−23.5 %: B has a long tail up to 512 tokens that fixed 15-line windows do not).
**D was not adequately matched** to A (median −21 %, mean −15 %); the design recorded this in
advance (`matching.json`, `adequate: false`), so D cannot support a clean "structure at equal size"
claim.

## 2. Fixed-k retrieval

One extra success is 2.94 percentage points. Intervals are paired percentile bootstrap, seed
20261027, 20,000 resamples.

| Arm | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | Region coverage@5 | Mean retrieved tokens (top 10) |
|---|---|---|---|---|---|---|---|
| A | 19/34 | 24/34 | 25/34 | 27/34 | 0.647 | 0.622 | 4363 |
| B | 14/34 | 27/34 | 29/34 | 30/34 | 0.592 | 0.712 | 2677 |
| C | 10/34 | 23/34 | 28/34 | 28/34 | 0.500 | 0.507 | 2022 |
| D | 14/34 | 25/34 | 27/34 | 28/34 | 0.585 | 0.708 | 4054 |

## 3. Paired analysis

**Primary (confirmatory): B vs A, Hit@5.** B 29/34 vs A 25/34 = **+11.8 pp, 95 % CI [0.0, +26.5]**;
5 questions improved, 1 regressed, exact McNemar **p = 0.219**. The null is not rejected.
[Certain]

Other comparisons are exploratory; Holm correction covers all 44 non-primary binary tests.

| Comparison (X − Y), endpoint | Difference | 95 % CI | Discordant (X only / Y only) | Raw p | Holm p |
|---|---|---|---|---|---|
| B − A, Hit@1 | −14.7 pp | [−32.4, +2.9] | 3 / 8 | 0.227 | 1.000 |
| B − A, MRR@10 | −0.055 | [−0.188, +0.081] | n/a | n/a | n/a |
| B − A, coverage-hit@5 | +17.6 pp | [+2.9, +32.4] | 7 / 1 | 0.070 | 1.000 |
| B − A, hit within 2048-token budget | +14.7 pp | [0.0, +29.4] | 6 / 1 | 0.125 | 1.000 |
| C − A, Hit@5 | +8.8 pp | [−2.9, +20.6] | 4 / 1 | 0.375 | 1.000 |
| C − A, Hit@1 | −26.5 pp | [−44.1, −8.8] | 1 / 10 | 0.012 | 0.516 |
| C − A, MRR@10 | −0.148 | [−0.270, −0.022] | n/a | n/a | n/a |
| B − C, Hit@5 | +2.9 pp | [−8.8, +14.7] | 3 / 2 | 1.000 | 1.000 |
| B − C, region coverage@5 | +0.205 | [+0.081, +0.336] | n/a | n/a | n/a |
| B − C, coverage-hit@5 | +26.5 pp | [+8.8, +44.1] | 10 / 1 | 0.012 | 0.516 |
| D − A, Hit@5 | +5.9 pp | [0.0, +14.7] | 2 / 0 | 0.500 | 1.000 |
| D − B, Hit@5 | −5.9 pp | [−17.6, +5.9] | 1 / 3 | 0.625 | 1.000 |

No exploratory binary test survives Holm correction. The continuous endpoints (MRR, coverage) have
bootstrap intervals only and no multiplicity correction; several intervals exclude zero, so read
them as hypotheses for a larger benchmark, not findings.

### Fixed context budgets (production `build_context`, retrieval only)

Evidence hit means at least one relevant region overlaps a chunk that fits in the budget.

| Arm | 512 tokens | 1024 tokens | 2048 tokens | Region coverage at 2048 |
|---|---|---|---|---|
| A | 18/34 | 23/34 | 25/34 | 0.601 |
| B | 20/34 | 26/34 | 30/34 | 0.761 |
| C | 25/34 | 27/34 | 28/34 | 0.626 |
| D | 16/34 | 23/34 | 26/34 | 0.689 |

C's lead at 512 tokens (25 vs B's 20) reflects that tiny chunks fit the budget; C's region coverage
there is lower than B's (0.370 vs 0.433), so "hit" and "coverage" disagree about which arm is
better at the smallest budget.

## 4. Per-question changes (Hit@5)

* B improved over A (5): `impl-ingest`, `impl-chunk-id`, `impl-long-line`, `impl-index-id`,
  `impl-skip-reasons`. B regressed (1): `impl-faiss-persist`.
* D improved over A (2): `impl-ingest`, `impl-chunk-id`; D regressed: none.
* B improved over C (3): `impl-index-id`, `impl-skip-reasons`, `doc-atomic`. B regressed vs C (2):
  `impl-faiss-persist`, `impl-fake-embedder`.
* Partition of all 34 questions by outcome (two endpoints; see section 10 for the exact lists):
  at **Hit@5**, 23 questions are hit by every arm, 3 are missed by every arm and 8 differ between
  arms (23 + 3 + 8 = 34); at **Hit@10**, 25 are hit by every arm, 2 are missed by every arm and 7
  differ (25 + 2 + 7 = 34). Most of the benchmark therefore does not discriminate between arms.

## 5. Error analysis (done after evaluation; nothing was retuned)

1. **Where B's gains come from.** In all five gains the target is one function or class
   (`ingest_repository`, `make_chunk_id`, `split_long_line`, `compute_index_id`, `SkipReason`)
   inside a file that Arm A cut into ~500-token windows mixing several definitions. In A the top results were
   documentation or neighbouring windows; in B the exact definition became its own chunk and ranked
   first or second. The median size of the first relevant chunk is 47 lines for A and 19 for B.
   [Likely]
2. **Why B loses Hit@1.** In 5 of the 8 questions where A's top result is relevant and B's is not,
   B's top result is the *module header chunk* (imports and docstring) of the correct file and the
   relevant function is ranked second. The file is right, the region is wrong. B's top-1 types are
   15 function, 8 module, 4 method, 4 class and 3 line windows. [Certain for the counts; Likely for
   the interpretation that header chunks compete with definitions.]
3. **The one B regression, `impl-faiss-persist`**, has two relevant regions (`index.py` lines
   259-275 and `atomic.py` lines 60-82). A's single 490-token window of `index.py` covers the first;
   B's top 5 is the backend `serialize` helper and four documentation windows. Fragmentation can
   lose a multi-function answer. [Likely]
4. **Misses shared by all arms** (`impl-embed-config`, a 10-line settings block, and
   `doc-query-doc`, a 5-line documentation passage, are missed in the top 10 by every arm, and
   `impl-sensitive-files` is also missed in the top 5 by every arm). In these cases the top-ranked
   results were mostly documentation windows or generic definitions, so documentation crowding is
   an *observed potential contributor* to these failures. It was not tested as a cause (for example
   by excluding documentation from the index). Documentation is chunked identically in all arms,
   so these misses do not depend on the chunking strategy being compared.
5. **Merging (D)** keeps definitions that are too large to merge as their own chunks, and those
   hit (`impl-ingest`, `impl-chunk-id`). It loses `impl-long-line`, `impl-index-id` and
   `impl-skip-reasons`, where B's small single-definition chunk (with its symbol name) was the hit
   and D has absorbed the small definition into a larger merged chunk without symbol metadata that
   did not reach the top 5. [Likely; D was not size-matched, so this is not a clean structure test.]
6. **C vs A.** Small windows lose Hit@1 and MRR (the matching region is split across several
   15-line pieces, none of which dominates) while keeping Hit@5; the first relevant chunk
   has a median of 15 lines. A plausible mechanism is that one definition is split across several
   small pieces that compete with each other. [Guessing: the mechanism was not tested.]
7. **Redundancy and fragments.** Retrieved-set redundancy (chunk lines over distinct lines) in the
   top 10 averaged 1.108 for A (overlap), 1.039 for B, 1.026 for C, 1.048 for D. B and D returned
   0.35 repeated-symbol chunks per query (a long definition split in two); A and C none.

## 6. Efficiency (measured on a 2-CPU Linux sandbox)

| Arm | Chunks (all files) | Index bytes | Embedding time | Median query latency | p95 |
|---|---|---|---|---|---|
| A | 423 | 1,502,581 | 1060 s | 62.8 ms | 79.0 ms |
| B | 679 | 2,434,920 | 1320 s | 62.3 ms | 81.3 ms |
| C | 827 | 2,938,087 | 1000 s | 62.6 ms | 78.3 ms |
| D | 422 | 1,509,540 | 1037 s | 62.3 ms | 78.4 ms |

B costs 1.6× the chunks, +62 % index bytes and about +25 % embedding time over A. Observed query
latency is similar across the arms (medians 62.3 to 62.8 ms, 170 samples per arm, one machine);
the experiment was not designed to detect small latency differences. A plausible reason is that
the flat index is small and the per-query work does not depend on chunk count much
[Guessing: not profiled].
B returns 39 % fewer tokens for the same top-10 (2677 vs 4363), which is relevant to LLM context
cost. Embedding seconds are sums of measured batch times (batch size 8, resumable, sandbox CPU);
they are comparable between arms here but not with a Windows machine. [Certain]

## 7. Conclusions

**Supported**

1. [Certain] On this repository and benchmark, AST chunking retrieved a relevant chunk in the top 5
   for 4 more questions than line windows (5 gains, 1 loss). The difference is **not statistically
   distinguishable from zero** (exact McNemar p = 0.219; CI [0.0, +26.5] pp).
2. [Certain] AST chunking did not improve, and numerically lowered, Hit@1 and MRR (not
   significant).
3. [Likely] The size-matched line-based control recovered much of the observed Hit@5 difference
   (C 28/34 vs B 29/34 and A 25/34; B − C = +2.9 pp, CI [−8.8, +14.7]). This suggests that chunk
   size may contribute to the improvement, but the experiment does not isolate its causal
   contribution from structural boundaries, overlap, chunk count or other differences between the
   arms (for example, C has about 1.3× the Python chunks of B and no overlap, and A uses overlap).
4. [Guessing, exploratory] Structure may add coverage of the relevant region beyond size (B − C
   region coverage@5 +0.205, CI [+0.081, +0.336]; coverage-hit@5 +26.5 pp, raw p = 0.012). It does
   not survive Holm correction (p = 0.516), so it is a hypothesis for a larger benchmark, not a
   finding.
5. [Certain] AST chunking costs more chunks, disk and embedding time and returns less text per
   top-k; observed retrieval latency was similar across arms.

**Not supported**

* That AST chunking is better than line chunking. The only pre-registered test is not significant.
* That chunk size (or any single factor) *causes* the differences between arms.
* That documentation crowding *causes* the shared misses.
* That the production default should change. **The baseline line-window strategy remains the
  production default** (`Settings.chunking_strategy = "line"`); `ast` stays an opt-in strategy.
* That any arm produces better *answers*; no generation was run.
* That B and C (or B and D) are equivalent; failing to reject is not evidence of equivalence.
* Anything about D as a clean structure-at-equal-size test (not size-matched).
* Generalisation to other repositories, other languages or non-designer questions.
* That any exploratory p-value below 0.05 is a finding; after Holm none is.

## 8. Limitations

* 34 questions, 41 regions, one repository, authored by the system's designer; 23 questions are hit
  in the top 5 by every arm, leaving 8 questions on which the arms differ at all. Power to detect a moderate effect is low.
* The primary hit rule (same file, at least one shared line) favours large chunks; coverage
  measures favour whole-definition chunks. The two rules disagree in places; both are reported.
* Arm C's mean size is 23 % below B's; Arm D is not matched; overlap differs between A and the
  other arms.
* Only Python is structure-aware. Documentation and other languages use the same line chunker in
  every arm.
* The embedding step used the resumable path with batch size 8 on Linux. Arm A's `chunks.jsonl` and
  `index.faiss` are byte-identical to the existing Windows-built index (same SHA-256), which is
  consistent with batching and platform not having changed the vectors for that arm; the manifest
  differs (build metadata). This does not prove the same for B, C and D, which have no earlier
  index to compare with.
* Native Windows verification of Milestone 9 remains the user's open step.

## 9. Reproduction (Windows PowerShell, from the repository root)

```powershell
uv sync --group ui
# The design is frozen and committed; do not re-run `match` or `freeze` (they would overwrite it).
# 2. Four indexes, offline (model must already be cached; sets HF offline itself)
uv run python -m copilot.evaluation.chunking_experiment build
# 3. Retrieval, budgets, latency, statistics
uv run python -m copilot.evaluation.chunking_experiment run
# 4. Tables, plots, generated report
uv run python -m copilot.evaluation.chunking_experiment report
```

Outputs are written atomically and never overwrite existing results; add `--overwrite` only when
you intend to replace them, or point `--results-dir` / `--index-root` / `--out-dir` at new folders.
For a quick check that does not touch the committed results:

```powershell
uv run python -m copilot.evaluation.chunking_experiment build --index-root data\experiments\rerun\indexes --results-dir experiments\rerun\results --embedding-cache-root data\experiments\rerun\cache
uv run python -m copilot.evaluation.chunking_experiment run --index-root data\experiments\rerun\indexes --results-dir experiments\rerun\results
```

The sandbox that produced the committed results could not keep a process alive for the full build
(17 to 22 minutes of embedding per arm on 2 CPUs), so it used `COPILOT_EMBEDDING_BATCH_SIZE=8`,
`build --time-budget 125` repeated until it exited 0, and `run --phase retrieval`, `--phase latency
--repetition 0..4`, `--phase finalize`. On a normal machine the single commands above are enough.
Results should reproduce exactly for ranks and statistics given identical vectors; latency and
embedding seconds are machine-dependent.

## 10. Closeout verification of the reported numbers

Everything in this section is recomputed from the saved machine-readable results by an independent
script, `experiments/chunking-comparison/closeout/verify_results.py` (output:
`experiments/chunking-comparison/closeout/verification.json`; it reports no discrepancy). It does
not import the experiment package and changes no result.

**Counts and endpoints.** A question counts as a hit at k when the first relevant rank (any-overlap
rule, same file and at least one shared line) is at most k. All 34 question ids are unique and
appear in the same order in all four arms. The earlier summary "23 hit by all, 2 missed by all, 8
differ" mixed two endpoints and summed to 33; the consistent statements are:

| Endpoint | Hit by all four arms | Missed by all four arms | Arms differ | Total |
|---|---|---|---|---|
| Hit@5 | 23 | 3 (`impl-embed-config`, `impl-sensitive-files`, `doc-query-doc`) | 8 | 34 |
| Hit@10 | 25 | 2 (`impl-embed-config`, `doc-query-doc`) | 7 | 34 |

The 8 questions that differ at Hit@5 are `impl-ingest`, `impl-chunk-id`, `impl-faiss-persist`,
`impl-long-line`, `impl-fake-embedder`, `impl-index-id`, `impl-skip-reasons` and `doc-atomic`.

**Primary paired table (B vs A, Hit@5).** Both hit 24, only B hit 5, only A hit 1, neither hit 4
(total 34). The recomputed exact two-sided McNemar p-value is 0.21875 (= 2 × 7/64 for 5 vs 1
discordant pairs), equal to the saved value.

**Bootstrap interval.** Paired percentile bootstrap: the 34 questions are resampled with
replacement (seed 20261027, 20,000 resamples, `numpy.random.default_rng(seed).integers`), the same
resampled index matrix is used for every comparison, the statistic is the mean per-question
difference (B minus A) in percentage points, and the interval is the 2.5th to 97.5th percentile of
the resampled means (`numpy.quantile`, linear interpolation). Recomputed: +11.76 pp, [0.0, +26.47].
The lower limit of exactly 0.0 occurs because every resampled mean is a multiple of 1/34
(2.94 pp), so the interval lies on a coarse grid. It describes this benchmark, which is not a random sample of
questions, so it is not a population interval. The bootstrap interval and the McNemar test are
different procedures and need not agree about 0.

**Persisted index size.** The sum of the byte sizes of the three persisted index files
(`chunks.jsonl`, `index.faiss`, `manifest.json`); for Arm A 201,650 + 1,299,501 + 1,430 =
1,502,581. It excludes the embedding cache, the model cache and the snapshot. The recorded sizes
equal the sizes currently on disk.

**Embedding duration.** The sum of the wall-clock durations of the individual
`embed_documents` batch calls (batch size 8) that were recorded when each batch was first computed
in the arm's own cache. It therefore excludes waiting between resumable execution slices, model
loading (timed separately: 6.0 to 8.7 s per arm and recorded as `model_cold_start_seconds`), file
discovery, chunking and index writing. It includes tokenisation inside each batch and any
first-call runtime warm-up per slice, which was not separated out. Slices ran on a 2-CPU Linux sandbox, so
absolute seconds are comparable between arms in this experiment but not with another machine.

**Fixed-context candidate pool and selection policy.** For each question and arm the ranked pool
is the top 30 chunks of exact search. The production `build_context` walks the pool in rank
order: duplicates (same chunk id or same file and line span) are dropped, a chunk whose formatted
block (plus separator) does not fit in the remaining budget is dropped whole (never truncated) and
later, smaller chunks may still be included; formatting overhead counts towards the 512, 1024 or
2048 estimated-token budget. Design text: section 8 of `chunking-comparison.md`.
