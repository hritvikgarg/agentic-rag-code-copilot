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
* All four arms miss in the top 10: `impl-embed-config`, `doc-query-doc`.
* All four arms hit in the top 5: 23 questions, i.e. most of the benchmark does not discriminate.

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
4. **Misses shared by all arms** are not chunking problems: `impl-embed-config` (a 10-line settings
   block) and `doc-query-doc` (a 5-line documentation passage) are crowded out by other documents
   or by generic definitions in every arm. Documentation is chunked identically in all arms.
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

B costs 1.6× the chunks, +62 % index bytes and about +25 % embedding time over A; query latency is
indistinguishable (170 samples per arm; the flat index is small and the query embedding dominates).
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
3. [Likely] Much of the Hit@5 gain is explained by smaller chunks rather than by structure: a
   structure-blind arm of matched median size reached 28/34, one question short of B (B − C = +2.9 pp,
   CI [−8.8, +14.7]).
4. [Likely, exploratory] Structure adds *coverage* of the relevant region beyond size (B − C region
   coverage@5 +0.205, CI [+0.081, +0.336]; coverage-hit@5 +26.5 pp, raw p = 0.012). It does not
   survive Holm correction (p = 0.516) and should be tested on a larger benchmark.
5. [Certain] AST chunking costs more chunks, disk and embedding time and returns less text per
   top-k; retrieval latency is unaffected.

**Not supported**

* That AST chunking is better than line chunking. The only pre-registered test is not significant.
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
  evidence that batching and platform did not change the vectors; the manifest differs only in
  build metadata.
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
