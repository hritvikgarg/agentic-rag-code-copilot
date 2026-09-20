# Retrieval evaluation (Milestone 5b)

Status: the framework is implemented and tested. **One real-model measurement exists (default
configuration). The six-configuration matrix has been implemented but NOT yet run with the real
model** (see "Matrix status"). Nothing here is a claim of generalisation.

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

## Measured result: default configuration (real model)

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

## Matrix status: NOT RUN

The experiment matrix (chunk cap 512 / 768 / 1024 [/ 2048] x representation `prefixed` / `raw`) is
implemented (`python -m copilot.evaluation matrix`) and unit/integration tested with fake embedders,
but was **not executed with the real model** in this milestone: the development VM has about 3.9 GB of
RAM and the process was killed by the operating system while embedding (peak resident memory after
model load was 2.6 GB and grew during embedding). No numbers from the matrix are reported because none
exist. Run it on a machine with enough memory (the Windows machine used for Milestone 5a):

```
git archive 055a8d5 | tar -x -C ..\bench-055a8d5      # or extract the export any other way
uv run python -m copilot.evaluation matrix benchmarks/copilot_self_055a8d5.jsonl ^
    --repo ..\bench-055a8d5 --work-dir data\eval-work --caps 512 768 1024 2048 ^
    --output data\eval-work\result.json
```

Held constant across configurations: repository commit, model, benchmark questions, top-k depth,
FAISS index type, line size (60), overlap (10) and the match rule. Varied: token cap and
representation. It prints Hit@1/3/5/10, MRR, mean lines per hit, chunk count, index size and timing
per configuration and declares no winner. When reading the table, compare mean lines per hit and
chunk count next to Hit@k, as explained above.

## Reproducing

```
python -m copilot.evaluation verify BENCH --repo EXPORT
python -m copilot.evaluation run    BENCH --repo EXPORT --index INDEX_DIR
python -m copilot.evaluation matrix BENCH --repo EXPORT --work-dir DIR --caps 512 768 1024 --output FILE
```

Use the same `--ignore-dir` values as recorded in the `.meta.json` (they are the defaults). Reported
timings depend on hardware and are informative only.
