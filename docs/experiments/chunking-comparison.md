# Experiment: controlled comparison of chunking strategies (Milestone 10)

**Status:** design frozen before any retrieval was run. Results, when they exist, are stored
separately under `experiments/chunking-comparison/results/` and discussed in
`docs/experiments/chunking-comparison-results.md`; they never edit this file.

This is an **offline retrieval experiment**. It does not call a hosted LLM and says nothing about
answer quality. A retrieval difference here does not establish an answer-quality difference.

## 1. Question

Milestone 9 measured how the structure-aware chunker (`ast`) cuts code differently from the
baseline (`line`): the median Python chunk is 122.5 estimated tokens for `ast` and 469 for
`line`. A retrieval difference between the two would therefore mix *structure* with *chunk size*
(and with overlap). This experiment separates them with two additional controls.

1. Does structure-aware chunking change repository retrieval on this benchmark?
2. Is any difference explained by chunk size (and overlap) rather than by structure?

## 2. Hypotheses (stated before looking at results)

* **H1 (primary, two-sided):** Hit@5 of Arm B (AST) differs from Hit@5 of Arm A (baseline).
  No direction is pre-committed. A decrease for B would be unsurprising, because the hit rule
  (section 6) favours large chunks and A's Python chunks are about four times larger.
* **H2 (size control):** if structure matters beyond size, Arm B differs from Arm C (line windows
  matched to B's Python size distribution, no overlap).
* **H3 (size effect):** if size alone matters, Arm C differs from Arm A although neither is
  structure-aware.
* **H4 (structure at large size):** Arm D (AST units merged up to the 512 token cap) differs from
  Arm A.
* **H5 (size within AST):** Arm D differs from Arm B.

Only H1 is a primary, confirmatory comparison. H2–H5 and every other endpoint are *exploratory*.
Failing to reject a null hypothesis is not evidence of equivalence.

## 3. Arms

| Arm | Strategy (registry id) | Python chunking | Non-Python chunking |
|---|---|---|---|
| A | `line` v1 (production baseline) | 60-line windows, overlap 10, 512 token cap | same |
| B | `ast` v1 (production, unchanged) | AST units, whole definitions ≤ 512 tokens, split inside oversized ones | baseline 60/10/512 |
| C | `line_sized` v1 (experimental) | line windows of **15** lines, overlap **0**, 512 cap | baseline 60/10/512 |
| D | `ast_merged` v1 (experimental) | Arm B's chunks, adjacent compatible ones merged up to **512** tokens | baseline 60/10/512 |

Non-Python files (documentation, JSON, YAML...) are cut by the same algorithm with the same
parameters in all four arms; the experiment verifies this (`design-check.json`:
`non_python_chunking_identical_across_arms`, 189 chunks). Differences between arms are therefore
confined to Python chunking. Unparseable Python files fall back to the baseline windows in B and D.

Arm C and D live in `src/copilot/chunking/experimental.py`. They are not listed in
`available_strategies()`, cannot be chosen through `Settings`, and are never a default. Their
chunk ids contain their own strategy name, version and parameters, so they cannot collide with the
production strategies.

**Merge rule (Arm D).** Starting from the unchanged `ast` chunks of a file, left to right, a chunk
is appended to the current group only if (a) both chunks have `segmentation == "ast"` and are
whole-line chunks (pieces of an oversized statement are never merged), (b) the chunk starts after
the group's last line (no overlap), and (c) the estimated tokens of the exact source slice from
the group's first line to the new chunk's last line stay ≤ `merge_max_tokens` (≤ 512, the global
cap). A multi-unit chunk has type `merged`, the exact source slice as text (blank separator lines
included), and **no** symbol metadata, because a single qualified name would be invented. A
group of one keeps its original type and symbols. A self-check per file verifies order, exact
text, budget and that no line was lost.

## 4. Controls held constant

Frozen repository snapshot `data/snapshots/a282ec471ebf1387cb4feafcfe732ad24a431c7a/tree`
(100 accepted files); the same ingestion rules and extra-ignored directories (`data`, `tests`,
`submission`); the same embedding model (`jinaai/jina-embeddings-v2-base-code`, 768-d, L2
normalised); the same embedding representation (`prefixed`, version 1: file path, language, line
range, blank line, text; **no AST symbols**); `IndexFlatIP` exact search (cosine); the same 34
questions and 41 regions; the same relevance rule; the same software environment. Chunk text is
embedded as a whole; the 512 estimated-token cap (≈ 560 real tokens) is far below the model's
context length.

## 5. Size-matching policy (fixed before retrieval; chunk statistics only)

Arms C and D were configured by looking only at descriptive chunk statistics, never at a retrieval
outcome. Policy:

* Only Python chunks are matched. Targets: Arm C → Arm B's Python distribution; Arm D → Arm A's.
* Compared statistics: median (primary), mean, quartiles/IQR, min/max, chunk count, overlap
  redundancy (line slots ÷ distinct lines), non-blank source-line coverage. Python-only and
  repository-wide profiles are both reported (`matching.json`, `design-check.json`).
* Search space (bounded): Arm C window size 4–60 lines, overlap fixed at 0 (Arm B has none, so
  redundancy is matched by construction); Arm D `merge_max_tokens` 128–512 in steps of 16.
* Selection: smallest |relative error of the median|; ties by smaller |relative error of the
  mean|; ties by smaller parameter. A match is *adequate* if |median error| ≤ 10 %; otherwise the
  closest feasible configuration is reported as such.

**Outcome (Python chunks, estimated tokens):**

| | chunks | min | p25 | median | mean | p75 | max | redundancy |
|---|---|---|---|---|---|---|---|---|
| A baseline | 234 | 16 | 331 | 469.0 | 398.7 | 504 | 512 | 1.163 |
| B ast | 490 | 9 | 51 | 122.5 | 161.1 | 237 | 512 | 1.002 |
| C (target B) | 638 | 1 | 87 | 119.0 | 123.3 | 155 | 395 | 1.000 |
| D (target A) | 233 | 16 | 250 | 369.0 | 338.8 | 455 | 512 | 1.002 |

* **Arm C** matches B's median to −2.9 % (adequate) but **not** its shape: a fixed window cannot
  reproduce B's heavy tail, so C's mean is 23.5 % below B's and its largest chunk is 395 tokens.
* **Arm D** is **not adequately matched**: the best feasible budget (512, the cap) leaves its median
  21.3 % below A's. Two causes: most definitions are too large to share a chunk, and A's mean is
  inflated by overlap (A repeats 16 % of its lines; D repeats none). D has 233 Python chunks vs
  A's 234, so the chunk *count* is matched but token size is not.
* Non-blank source-line coverage is 100 % in every arm; the arms differ in redundancy, not in what
  text exists.

Remaining confounds that matching cannot remove: A has overlap and B/C/D (nearly) do not; B's
metadata (symbols) is not embedded but its chunks are single definitions, which may embed
differently from arbitrary windows of the same size.

## 6. Benchmark and relevance (unchanged)

The existing 34-question, 41-region benchmark (`benchmarks/copilot_self_055a8d5.jsonl`, SHA-256
`8a0d0f4b…`) is used without adding, removing, rewording or excluding a question. Every region was
re-validated against the frozen snapshot before the freeze (sealed region hashes: all pass).
Arm A's chunk-id digest equals the existing index `4aaeab9c650ab76c`'s, so Arm A *is* the existing
baseline chunking. No symbol-lookup benchmark is added; AST chunk ids and symbols are never ground
truth.

**Primary relevance:** the existing rule — a chunk is a hit if it is in the same file and shares
at least one line with a ground-truth region. **This rule is biased towards larger chunks**: a
larger chunk overlaps a small region more easily. It is retained for continuity with all earlier
results and reported with chunk-size context. The questions were written by the project author,
who also designed the retrieval system; the benchmark is not independent.

**Supplementary (reported separately):** *region coverage@k* = fraction of the question's
ground-truth lines covered by the union of the top-k chunks; *coverage-hit@k* = coverage ≥ 0.5.
Coverage also rewards big chunks (a chunk that contains the whole region covers it), so neither
measure removes the size bias; the fixed-budget analysis (section 8) is the main size-aware view.

## 7. Fixed-k retrieval (per arm, per question)

Top-10 per question. Metrics: Hit@1/3/5/10; MRR@10 (reciprocal rank of the first relevant result
within the top 10, 0 if none); rank of the first relevant result; region coverage@5/10;
coverage-hit@5/10; retrieval latency; estimated tokens and distinct files retrieved. Saved per
question: chunk ids, paths, ranges, ranks, scores. Raw scores are **not** compared across indexes.

## 8. Fixed context budget (retrieval-only simulation)

For budgets of 512, 1024 and 2048 estimated tokens, each arm retrieves a pool of 30 ranked
chunks per question and the *production* `build_context` selects evidence within the budget:
rank order, duplicates dropped, a chunk that does not fit is dropped whole and later (smaller)
chunks may still fit; formatting overhead counts toward the budget. No arm gets different
truncation. Reported: evidence-hit@budget, region coverage, number of chunks, distinct files,
tokens consumed, budget utilisation. This simulates the context an answer generator would see;
**no LLM is called**.

## 9. Statistics

* **Primary:** H1, endpoint Hit@5, B vs A.
* **Binary endpoints (Hit@k, evidence-hit@budget, coverage-hit):** raw counts, absolute difference
  in percentage points, the 2×2 paired contingency table, number improved/regressed, **exact
  McNemar test** on the discordant pairs (two-sided binomial), and a **paired bootstrap 95 % CI**
  of the difference.
* **MRR@10 and coverage:** per-question paired differences, mean paired difference, paired
  bootstrap 95 % CI.
* **Bootstrap:** seed `20261027`, 20,000 resamples, percentile intervals; each resample draws
  questions with replacement *once* and applies the same draw to every metric and comparison.
* **Multiple comparisons:** the primary test is a single test at α = 0.05. All other McNemar tests
  are exploratory; Holm-adjusted p-values over that family are reported next to the raw values.
  Bootstrap intervals are not corrected.
* One question changes Hit@k by 2.94 percentage points. The 34 questions are not a random sample
  of anything, so intervals are *descriptive, conditional on this benchmark*; they do not support
  generalisation to other repositories or question distributions.

## 10. Efficiency

Index build is timed as ingest, chunk, embed, index-and-save separately; model cold start is
timed separately from indexing. Per arm: chunk count, vector count, index size on disk, retrieved
context size. **Latency:** one embedder shared by all arms; 3 warm-up queries (not recorded); 5
repetitions over all 34 questions; arm order rotated by one position each repetition; the timed
region is `Retriever.retrieve(query, 10)` (query embedding + exact search + chunk
materialisation) and excludes model loading, disk writes and report generation. Median and p95
over 170 samples per arm. Query embedding dominates and is the same for all arms, so latency
differences between arms are expected to be small.

## 11. Error analysis (after evaluation)

B-improved and B-regressed questions, all-miss questions, symbol-lookup failures,
documentation-dominated retrieval, fragmentation of oversized definitions, size and overlap
effects, with recorded examples. The benchmark and configurations are not modified afterwards;
any tuning is a separate experiment.

## 12. Limitations (known in advance)

* 34 questions, 41 regions, one repository, authored by the system's designer: not independent.
* Hit rule favours large chunks; coverage measures do too; matching is imperfect (section 5).
* Arm D is not size-matched to Arm A; overlap differs between A and the other arms.
* Only Python is structure-aware. Embeddings, FAISS and ONNX numerics may differ slightly between
  Linux and Windows; Arm A's results are compared with earlier Windows results as a sanity check,
  not as a gate.
* Retrieval quality is not answer quality.

## 13. Reproduction

```
uv run python -m copilot.evaluation.chunking_experiment match      # chunk statistics only
uv run python -m copilot.evaluation.chunking_experiment freeze     # validates, writes frozen-config.json
uv run python -m copilot.evaluation.chunking_experiment build      # four indexes (offline)
uv run python -m copilot.evaluation.chunking_experiment run        # fixed-k, budgets, latency, stats
uv run python -m copilot.evaluation.chunking_experiment report     # tables, plots, markdown
```

(PowerShell commands and flags are listed in the results document once the harness is complete.)

## 14. Freeze record

| Artifact | SHA-256 |
|---|---|
| `experiments/chunking-comparison/design/frozen-config.json` | `680a87c57a8e74e64f4d488a16f671eb259ff8a3a3c8e1fa013c4bf8324451d0` |
| `experiments/chunking-comparison/design/matching.json` | `a456f9c08aa77b03737727e150775befacf5b95c0349a083b15c6d9667d74516` |
| `experiments/chunking-comparison/design/design-check.json` | `3f4453a7dbc10b125fff852be4e8619e98c8246434ed7a66963698e69d5bc6a9` |
| `benchmarks/copilot_self_055a8d5.jsonl` | `8a0d0f4b880040540a5e09dc43dd1f005d883c26ed9821df7db3513a65a4f030` |

At the time of freezing, no query had been embedded or retrieved for Arms B, C or D. Arm A's
retrieval results on this benchmark were already known from Milestones 5b and 8 (the existing
index); those are not new evidence about the other arms.
