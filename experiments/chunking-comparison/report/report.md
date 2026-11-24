# Chunking comparison: generated report

Generated from `results/*.json`; descriptive only. Interpretation is in `docs/experiments/chunking-comparison-results.md`.

* Questions: 34 (benchmark SHA-256 `8a0d0f4b8800…`). One question = 2.94 pp.
* Bootstrap: seed 20261027, 20000 paired resamples, 95% percentile intervals.
* Primary comparison: B-A/hit@5 (exact McNemar, α = 0.05); all other tests exploratory, Holm family size 44.
* Index ids: A `4aaeab9c650ab76c`, B `e31fab15e045ea8c`, C `82e470046c96a49d`, D `4d89befefa68250f`

## Chunk sizes (estimated tokens)

| scope | arm | chunks | min | p25 | median | mean | p75 | max | redundancy | non_blank_coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| python | A | 234 | 16 | 331.0 | 469.0 | 398.7 | 504.0 | 512 | 1.163 | 1.0 |
| python | B | 490 | 9 | 51.0 | 122.5 | 161.1 | 237.0 | 512 | 1.002 | 1.0 |
| python | C | 638 | 1 | 87.0 | 119.0 | 123.3 | 155.0 | 395 | 1.0 | 1.0 |
| python | D | 233 | 16 | 250.0 | 369.0 | 338.8 | 455.0 | 512 | 1.002 | 1.0 |
| all | A | 423 | 16 | 423.0 | 492.0 | 435.4 | 505.0 | 512 | 1.261 | 1.0 |
| all | B | 679 | 9 | 70.0 | 204.0 | 250.1 | 470.0 | 512 | 1.146 | 1.0 |
| all | C | 827 | 1 | 98.0 | 138.0 | 205.0 | 237.0 | 512 | 1.136 | 1.0 |
| all | D | 422 | 16 | 333.0 | 463.5 | 402.4 | 498.0 | 512 | 1.14 | 1.0 |

## Fixed-k retrieval (top 10)

| arm | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | coverage@5 | coverage@10 | coverage-hit@5 | coverage-hit@10 | median first rank | mean retrieved lines | mean retrieved tokens (top10) | mean distinct files (top10) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 19/34 (55.9%) | 24/34 (70.6%) | 25/34 (73.5%) | 27/34 (79.4%) | 0.647 | 0.622 | 0.735 | 21/34 (61.8%) | 26/34 (76.5%) | 1 | 39.9 | 4363 | 5.03 |
| B | 14/34 (41.2%) | 27/34 (79.4%) | 29/34 (85.3%) | 30/34 (88.2%) | 0.592 | 0.712 | 0.804 | 27/34 (79.4%) | 29/34 (85.3%) | 2.0 | 23.1 | 2677 | 5.35 |
| C | 10/34 (29.4%) | 23/34 (67.6%) | 28/34 (82.4%) | 28/34 (82.4%) | 0.500 | 0.507 | 0.635 | 18/34 (52.9%) | 23/34 (67.6%) | 2.0 | 17.7 | 2022 | 4.41 |
| D | 14/34 (41.2%) | 25/34 (73.5%) | 27/34 (79.4%) | 28/34 (82.4%) | 0.585 | 0.708 | 0.775 | 25/34 (73.5%) | 27/34 (79.4%) | 1.5 | 36.1 | 4054 | 5.29 |

## Fixed context budget (retrieval-only simulation)

| arm | budget (est. tokens) | evidence hit | region coverage | mean chunks | mean distinct files | mean tokens used | mean utilisation |
|---|---|---|---|---|---|---|---|
| A | 512 | 18/34 (52.9%) | 0.350 | 1.41 | 1.41 | 437 | 0.853 |
| A | 1024 | 23/34 (67.6%) | 0.491 | 2.41 | 2.09 | 947 | 0.925 |
| A | 2048 | 25/34 (73.5%) | 0.601 | 4.62 | 3.18 | 1966 | 0.960 |
| B | 512 | 20/34 (58.8%) | 0.433 | 3.18 | 2.32 | 485 | 0.947 |
| B | 1024 | 26/34 (76.5%) | 0.627 | 4.59 | 3.24 | 990 | 0.967 |
| B | 2048 | 30/34 (88.2%) | 0.761 | 8.12 | 4.97 | 2019 | 0.986 |
| C | 512 | 25/34 (73.5%) | 0.370 | 3.15 | 2.06 | 480 | 0.937 |
| C | 1024 | 27/34 (79.4%) | 0.503 | 5.59 | 3.09 | 990 | 0.966 |
| C | 2048 | 28/34 (82.4%) | 0.626 | 9.88 | 4.53 | 2015 | 0.984 |
| D | 512 | 16/34 (47.1%) | 0.340 | 1.47 | 1.44 | 451 | 0.881 |
| D | 1024 | 23/34 (67.6%) | 0.556 | 2.71 | 2.21 | 967 | 0.945 |
| D | 2048 | 26/34 (76.5%) | 0.689 | 5.06 | 3.53 | 1987 | 0.970 |

## Paired comparison: binary endpoints

| comparison (X-Y) | endpoint | X count | Y count | diff pp | 95% CI pp | X only | Y only | both | neither | exact McNemar p | Holm p | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B-A | budget_hit@1024 | 26 | 23 | +8.8 | [-5.9, +23.5] | 5 | 2 | 21 | 6 | 0.453 | 1.000 | exploratory |
| B-A | budget_hit@2048 | 30 | 25 | +14.7 | [+0.0, +29.4] | 6 | 1 | 24 | 3 | 0.125 | 1.000 | exploratory |
| B-A | budget_hit@512 | 20 | 18 | +5.9 | [-14.7, +26.5] | 7 | 5 | 13 | 9 | 0.774 | 1.000 | exploratory |
| B-A | coverage_hit@10 | 29 | 26 | +8.8 | [+0.0, +20.6] | 3 | 0 | 26 | 5 | 0.250 | 1.000 | exploratory |
| B-A | coverage_hit@5 | 27 | 21 | +17.6 | [+2.9, +32.4] | 7 | 1 | 20 | 6 | 0.070 | 1.000 | exploratory |
| B-A | hit@1 | 14 | 19 | -14.7 | [-32.4, +2.9] | 3 | 8 | 11 | 12 | 0.227 | 1.000 | exploratory |
| B-A | hit@10 | 30 | 27 | +8.8 | [-2.9, +20.6] | 4 | 1 | 26 | 3 | 0.375 | 1.000 | exploratory |
| B-A | hit@3 | 27 | 24 | +8.8 | [-2.9, +20.6] | 4 | 1 | 23 | 6 | 0.375 | 1.000 | exploratory |
| B-A | hit@5 | 29 | 25 | +11.8 | [+0.0, +26.5] | 5 | 1 | 24 | 4 | 0.219 | - | primary |
| B-C | budget_hit@1024 | 26 | 27 | -2.9 | [-17.6, +11.8] | 3 | 4 | 23 | 4 | 1.000 | 1.000 | exploratory |
| B-C | budget_hit@2048 | 30 | 28 | +5.9 | [-8.8, +20.6] | 4 | 2 | 26 | 2 | 0.688 | 1.000 | exploratory |
| B-C | budget_hit@512 | 20 | 25 | -14.7 | [-32.4, +0.0] | 2 | 7 | 18 | 7 | 0.180 | 1.000 | exploratory |
| B-C | coverage_hit@10 | 29 | 23 | +17.6 | [+2.9, +32.4] | 7 | 1 | 22 | 4 | 0.070 | 1.000 | exploratory |
| B-C | coverage_hit@5 | 27 | 18 | +26.5 | [+8.8, +44.1] | 10 | 1 | 17 | 6 | 0.012 | 0.516 | exploratory |
| B-C | hit@1 | 14 | 10 | +11.8 | [+0.0, +26.5] | 5 | 1 | 9 | 19 | 0.219 | 1.000 | exploratory |
| B-C | hit@10 | 30 | 28 | +5.9 | [-8.8, +20.6] | 4 | 2 | 26 | 2 | 0.688 | 1.000 | exploratory |
| B-C | hit@3 | 27 | 23 | +11.8 | [+0.0, +26.5] | 5 | 1 | 22 | 6 | 0.219 | 1.000 | exploratory |
| B-C | hit@5 | 29 | 28 | +2.9 | [-8.8, +14.7] | 3 | 2 | 26 | 3 | 1.000 | 1.000 | exploratory |
| C-A | budget_hit@1024 | 27 | 23 | +11.8 | [+0.0, +26.5] | 5 | 1 | 22 | 6 | 0.219 | 1.000 | exploratory |
| C-A | budget_hit@2048 | 28 | 25 | +8.8 | [-2.9, +20.6] | 4 | 1 | 24 | 5 | 0.375 | 1.000 | exploratory |
| C-A | budget_hit@512 | 25 | 18 | +20.6 | [+2.9, +38.2] | 9 | 2 | 16 | 7 | 0.065 | 1.000 | exploratory |
| C-A | coverage_hit@10 | 23 | 26 | -8.8 | [-23.5, +5.9] | 2 | 5 | 21 | 6 | 0.453 | 1.000 | exploratory |
| C-A | coverage_hit@5 | 18 | 21 | -8.8 | [-26.5, +8.8] | 3 | 6 | 15 | 10 | 0.508 | 1.000 | exploratory |
| C-A | hit@1 | 10 | 19 | -26.5 | [-44.1, -8.8] | 1 | 10 | 9 | 14 | 0.012 | 0.516 | exploratory |
| C-A | hit@10 | 28 | 27 | +2.9 | [-5.9, +14.7] | 2 | 1 | 26 | 5 | 1.000 | 1.000 | exploratory |
| C-A | hit@3 | 23 | 24 | -2.9 | [-17.6, +11.8] | 3 | 4 | 20 | 7 | 1.000 | 1.000 | exploratory |
| C-A | hit@5 | 28 | 25 | +8.8 | [-2.9, +20.6] | 4 | 1 | 24 | 5 | 0.375 | 1.000 | exploratory |
| D-A | budget_hit@1024 | 23 | 23 | +0.0 | [-11.8, +11.8] | 2 | 2 | 21 | 9 | 1.000 | 1.000 | exploratory |
| D-A | budget_hit@2048 | 26 | 25 | +2.9 | [+0.0, +8.8] | 1 | 0 | 25 | 8 | 1.000 | 1.000 | exploratory |
| D-A | budget_hit@512 | 16 | 18 | -5.9 | [-23.5, +11.8] | 4 | 6 | 12 | 12 | 0.754 | 1.000 | exploratory |
| D-A | coverage_hit@10 | 27 | 26 | +2.9 | [+0.0, +8.8] | 1 | 0 | 26 | 7 | 1.000 | 1.000 | exploratory |
| D-A | coverage_hit@5 | 25 | 21 | +11.8 | [+2.9, +23.5] | 4 | 0 | 21 | 9 | 0.125 | 1.000 | exploratory |
| D-A | hit@1 | 14 | 19 | -14.7 | [-32.4, +2.9] | 3 | 8 | 11 | 12 | 0.227 | 1.000 | exploratory |
| D-A | hit@10 | 28 | 27 | +2.9 | [+0.0, +8.8] | 1 | 0 | 27 | 6 | 1.000 | 1.000 | exploratory |
| D-A | hit@3 | 25 | 24 | +2.9 | [-5.9, +11.8] | 2 | 1 | 23 | 8 | 1.000 | 1.000 | exploratory |
| D-A | hit@5 | 27 | 25 | +5.9 | [+0.0, +14.7] | 2 | 0 | 25 | 7 | 0.500 | 1.000 | exploratory |
| D-B | budget_hit@1024 | 23 | 26 | -8.8 | [-23.5, +5.9] | 2 | 5 | 21 | 6 | 0.453 | 1.000 | exploratory |
| D-B | budget_hit@2048 | 26 | 30 | -11.8 | [-26.5, +0.0] | 1 | 5 | 25 | 3 | 0.219 | 1.000 | exploratory |
| D-B | budget_hit@512 | 16 | 20 | -11.8 | [-29.4, +5.9] | 3 | 7 | 13 | 11 | 0.344 | 1.000 | exploratory |
| D-B | coverage_hit@10 | 27 | 29 | -5.9 | [-17.6, +5.9] | 1 | 3 | 26 | 4 | 0.625 | 1.000 | exploratory |
| D-B | coverage_hit@5 | 25 | 27 | -5.9 | [-17.6, +5.9] | 1 | 3 | 24 | 6 | 0.625 | 1.000 | exploratory |
| D-B | hit@1 | 14 | 14 | +0.0 | [-14.7, +14.7] | 3 | 3 | 11 | 17 | 1.000 | 1.000 | exploratory |
| D-B | hit@10 | 28 | 30 | -5.9 | [-20.6, +8.8] | 2 | 4 | 26 | 2 | 0.688 | 1.000 | exploratory |
| D-B | hit@3 | 25 | 27 | -5.9 | [-17.6, +5.9] | 1 | 3 | 24 | 6 | 0.625 | 1.000 | exploratory |
| D-B | hit@5 | 27 | 29 | -5.9 | [-17.6, +5.9] | 1 | 3 | 26 | 4 | 0.625 | 1.000 | exploratory |

## Paired comparison: continuous endpoints

| comparison (X-Y) | endpoint | mean X | mean Y | mean difference | 95% CI |
|---|---|---|---|---|---|
| B-A | budget_coverage@1024 | 0.627 | 0.491 | +0.137 | [-0.005, +0.283] |
| B-A | budget_coverage@2048 | 0.761 | 0.601 | +0.160 | [+0.031, +0.301] |
| B-A | budget_coverage@512 | 0.433 | 0.350 | +0.083 | [-0.081, +0.249] |
| B-A | coverage@10 | 0.804 | 0.735 | +0.069 | [-0.030, +0.185] |
| B-A | coverage@5 | 0.712 | 0.622 | +0.089 | [-0.050, +0.239] |
| B-A | mrr@10 | 0.592 | 0.647 | -0.055 | [-0.188, +0.081] |
| B-C | budget_coverage@1024 | 0.627 | 0.503 | +0.124 | [+0.009, +0.246] |
| B-C | budget_coverage@2048 | 0.761 | 0.626 | +0.135 | [+0.020, +0.260] |
| B-C | budget_coverage@512 | 0.433 | 0.370 | +0.063 | [-0.068, +0.195] |
| B-C | coverage@10 | 0.804 | 0.635 | +0.169 | [+0.048, +0.297] |
| B-C | coverage@5 | 0.712 | 0.507 | +0.205 | [+0.081, +0.336] |
| B-C | mrr@10 | 0.592 | 0.500 | +0.093 | [+0.001, +0.193] |
| C-A | budget_coverage@1024 | 0.503 | 0.491 | +0.012 | [-0.096, +0.121] |
| C-A | budget_coverage@2048 | 0.626 | 0.601 | +0.024 | [-0.069, +0.128] |
| C-A | budget_coverage@512 | 0.370 | 0.350 | +0.020 | [-0.113, +0.156] |
| C-A | coverage@10 | 0.635 | 0.735 | -0.099 | [-0.196, -0.001] |
| C-A | coverage@5 | 0.507 | 0.622 | -0.116 | [-0.228, -0.001] |
| C-A | mrr@10 | 0.500 | 0.647 | -0.148 | [-0.270, -0.022] |
| D-A | budget_coverage@1024 | 0.556 | 0.491 | +0.065 | [-0.027, +0.170] |
| D-A | budget_coverage@2048 | 0.689 | 0.601 | +0.088 | [+0.008, +0.185] |
| D-A | budget_coverage@512 | 0.340 | 0.350 | -0.010 | [-0.145, +0.121] |
| D-A | coverage@10 | 0.775 | 0.735 | +0.040 | [-0.013, +0.115] |
| D-A | coverage@5 | 0.708 | 0.622 | +0.086 | [+0.009, +0.180] |
| D-A | mrr@10 | 0.585 | 0.647 | -0.063 | [-0.172, +0.050] |
| D-B | budget_coverage@1024 | 0.556 | 0.627 | -0.071 | [-0.206, +0.051] |
| D-B | budget_coverage@2048 | 0.689 | 0.761 | -0.072 | [-0.203, +0.040] |
| D-B | budget_coverage@512 | 0.340 | 0.433 | -0.093 | [-0.243, +0.046] |
| D-B | coverage@10 | 0.775 | 0.804 | -0.029 | [-0.156, +0.086] |
| D-B | coverage@5 | 0.708 | 0.712 | -0.003 | [-0.127, +0.101] |
| D-B | mrr@10 | 0.585 | 0.592 | -0.008 | [-0.120, +0.097] |

## Efficiency

| arm | chunks | index bytes | chunking s | embedding s | build total s | latency median ms | latency p95 ms | latency samples |
|---|---|---|---|---|---|---|---|---|
| A | 423 | 1502581 | 0.10 | 1060.0 | 1063.0 | 62.8 | 79.0 | 170 |
| B | 679 | 2434920 | 0.18 | 1320.4 | 1323.9 | 62.3 | 81.3 | 170 |
| C | 827 | 2938087 | 0.10 | 999.5 | 1004.6 | 62.6 | 78.3 | 170 |
| D | 422 | 1509540 | 0.25 | 1037.4 | 1040.2 | 62.3 | 78.4 | 170 |

## Question groups (primary rule, Hit@5)

* **b_improved_over_a** (5): impl-ingest, impl-chunk-id, impl-long-line, impl-index-id, impl-skip-reasons
* **b_regressed_vs_a** (1): impl-faiss-persist
* **d_improved_over_a** (2): impl-ingest, impl-chunk-id
* **d_regressed_vs_a** (0): none
* **b_improved_over_c** (3): impl-index-id, impl-skip-reasons, doc-atomic
* **b_regressed_vs_c** (2): impl-faiss-persist, impl-fake-embedder
* **all_arms_miss_top10** (2): impl-embed-config, doc-query-doc
* **all_arms_hit_top5** (23): impl-unsafe-paths, impl-logging, impl-log-redact, impl-symlinks, impl-read-decode, impl-token-estimate, impl-windows, impl-normalize-vectors, impl-embed-text, impl-model-load, impl-fingerprint, impl-vector-order, impl-index-load, impl-build-pipeline, impl-chunker-select, impl-chunk-stats, impl-token-validation, impl-settings-load, impl-newlines, doc-sensitive, doc-cosine, doc-chunk-algo, doc-symlink

## Per-arm retrieval diagnostics

| arm | mean docs in top5 | misses with ≥3 docs in top5 | mean repeated-symbol chunks (top10) | mean retrieved redundancy (top10) |
|---|---|---|---|---|
| A | 1.74 | 3 | 0.00 | 1.108 |
| B | 1.15 | 3 | 0.35 | 1.039 |
| C | 0.71 | 1 | 0.00 | 1.026 |
| D | 2.00 | 3 | 0.35 | 1.048 |

Latency ranking mismatches vs the scoring pass: 0.
