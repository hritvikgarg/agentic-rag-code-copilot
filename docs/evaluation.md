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
- Small. Differences of a few questions are noise.
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
