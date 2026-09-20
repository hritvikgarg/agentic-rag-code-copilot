# Baseline chunking (Milestone 3, Strategy A)

Strategy A (`"line"`) is a **true structure-blind baseline**. *Every* supported text input -
source code, Markdown, JSON and YAML - is cut by the same algorithm. Language and document
structure play no role in chunk boundaries; Markdown headings are ordinary text.

The structure-aware AST strategy (Strategy B, Milestone 9) will be a separate implementation
behind the same `Chunker` interface. `chunking_strategy="ast"` is a valid setting but is
*reserved*: selecting it raises `StrategyNotImplementedError` instead of silently falling back to
the baseline, which would corrupt the planned experiment.

## Usage

```python
from copilot.chunking import chunk_repository, create_chunker
from copilot.ingestion import ingest_repository

ingestion = ingest_repository("path/to/repo")
result = chunk_repository(ingestion, create_chunker())  # strategy/params come from Settings
print(result.stats.summary())
```

```bash
uv run python -m copilot.chunking path/to/repo --samples 5   # statistics + chunk metadata, never contents
```

## Algorithm

* **Primary boundary - configurable line windows.** A window is `chunk_size_lines` (default 60)
  whole lines. Consecutive windows share `chunk_overlap_lines` (default 10) lines.
* **Safety boundary - estimated token cap.** If a window would exceed `chunk_max_tokens`
  (default 512 *estimated* tokens) it is shortened to the longest run of whole lines that fits.
  The cap only counts tokens; it never inspects programming-language or document structure.
* **Progress rule.** The next window starts `overlap` lines before the previous end but always
  after the previous start, and a window is emitted only if it adds at least one not-yet-covered
  line (so a shortened window cannot cause a cascade of near-duplicates). When shortening leaves
  no room, the effective overlap is smaller than configured; around a fragmented line (below)
  there is no overlap.
* **Over-long single line.** A physical line that alone exceeds the cap cannot be shortened by
  removing lines, so it is split deterministically into contiguous **fragments** on token
  boundaries (see the fragment contract). Without this the safety limit could not be enforced.
* **Blank text.** Whitespace-only windows are dropped; an empty file yields no chunks.

Windows are built from per-line token counts (the estimate is additive over lines) in linear time.
The token estimator (`utils/tokens.py`) counts ASCII word runs and every other non-space
character individually. It is a **provisional heuristic**: it needs no tokenizer download, is
conservative for non-ASCII text, and will be checked against the real embedding tokenizer in
Milestone 4. The cap is therefore a safety budget, not an exact guarantee.

### Markdown

Markdown is windowed by lines like any other file. `chunking/markdown_sections.py` keeps a
heading-aware splitter (ATX headings, fenced-code aware) as a **standalone utility for a future
structure-aware strategy**. Nothing in the baseline imports it; a test asserts that importing the
baseline does not load the module and that Markdown boundaries are identical to those of the same
text in a `.py` file.

## Line semantics

Lines are those of the newline-normalised text (ingestion converts CRLF/CR to LF), numbered
**1-based and inclusive**: `start_line`/`end_line` name real lines. Text is split on `"\n"` only,
and a terminal newline *terminates* the last line rather than starting a new one, so it never
creates a phantom citation line:

| `content` | Physical lines | Chunks `(start, end, content)` |
|---|---|---|
| `""` | none | none |
| `"abc"` | `["abc"]` | `(1, 1, "abc")` |
| `"abc\n"` | `["abc"]` | `(1, 1, "abc")` |
| `"abc\n\n"` | `["abc", ""]` | `(1, 2, "abc\n")` (line 2 is a real blank line) |
| `"\n"` | `[""]` | none (whitespace only) |

For a chunk of whole lines, `content == "\n".join(lines[start_line-1:end_line])`.

## Long-line fragment contract

A fragment is a piece of one physical line, not a new source line.

* Every fragment keeps that line's own number: `start_line == end_line == <physical line>`.
* Fragments carry `fragment_index` (0-based) and `fragment_count`; both are `None` for chunks of
  whole lines (`Chunk.is_fragment`).
* Cuts are at estimated-token boundaries, so `"".join(fragment contents) == the original line`.
* Each fragment has at most `chunk_max_tokens` estimated tokens.
* A fragmented line is never also part of a whole-line window (it is excluded from windows).

## Chunk-ID contract

`chunk_id` is the first 16 hex characters (64 bits) of the SHA-256 of a canonical JSON string:

```text
text    = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
chunk_id = sha256(text.encode("utf-8")).hexdigest()[:16]

payload = [ "chunk-id/1",          # CHUNK_ID_SCHEMA
            repository_name,       # repository identity
            file_path,             # canonical repository-relative POSIX path
            source_sha256,         # SourceFile.sha256: SHA-256 of the file's raw bytes
            strategy,              # e.g. "line"
            strategy_version,      # algorithm version of the strategy, e.g. 1
            params,                # chunker parameter mapping (sorted by key)
            unit ]                 # position, see below

unit    = ["w", start_line, end_line]        # window of whole lines
        | ["f", line, fragment_index]        # fragment of one physical line
```

For the line strategy `params = {"max_tokens", "overlap_lines", "size_lines"}`. JSON quoting makes
the serialisation unambiguous: no path or name can forge a field boundary. Nothing random is used.
`source_sha256` already represents the content, so the chunk text is not hashed again (chunk
content is available separately as `Chunk.content_sha256`).

**What changes a chunk's identity:** the repository name, the file path, *any* edit to the file
(all chunk ids of that file change), the strategy name, the strategy `version`, any chunker
parameter, and the chunk's own position. Nothing else.

| Property | How it is met |
|---|---|
| A. Same source + strategy + config gives the same ids | pure function of the inputs above |
| B. Different repositories with the same path and content do not collide | `repository_name` is hashed |
| C. Changed source never reuses stale identity | `source_sha256` is hashed |
| D. Different strategies cannot collide | `strategy` (and its `version`) are hashed |
| E. Different relevant configuration cannot reuse ids | all `params` and the strategy `version` are hashed |
| F. Fragments have unique deterministic ids | `["f", line, fragment_index]` |

Consequences and limits:

* Because `source_sha256` covers the whole file, an edit anywhere changes **every** chunk id of that
  file, including chunks whose text did not change. Use `Chunk.content_sha256` to detect unchanged
  chunk text across versions.
* `repository_name` is the identity used. Two *different* repositories that share a directory name
  must be ingested with distinct `repository_name`s (an `ingest_repository` argument); otherwise
  they can only collide when path and bytes are identical, i.e. when the chunks are identical too.
* `strategy_version` must be bumped whenever the windowing algorithm **or the token estimator**
  changes output for the same input and parameters.
* 64 bits gives a collision probability of roughly n^2 / 2^65 (about 3e-8 for a million chunks);
  `chunk_repository` raises `ChunkingError` if a duplicate id is ever produced.
* `CHUNK_ID_SCHEMA` versions the formula itself. A golden-value test locks the current formula.

## Chunk metadata

`chunk_id`, `repository_name`, `file_path`, `source_sha256`, `language`, `chunk_type`
(`line_window`), `chunking_strategy`, `chunking_version`, `chunk_index`, `start_line`, `end_line`,
`content`, `content_sha256`, `token_estimate`, `fragment_index`, `fragment_count`, and the symbol
fields `symbol_name`, `qualified_name`, `parent_class` (always `None` for the baseline; filled in
Milestone 9).

Tested invariants: line ranges stay within the file; content equals the source slice (or a
fragment of one line); no chunk exceeds the token cap; ids are unique; no whole-line window is
contained in another; every non-blank line is covered by at least one chunk.

## Statistics contract

`ChunkStats` separates physical lines from chunk line slots:

| Field | Meaning |
|---|---|
| `source_lines_total` | physical lines in the chunked files |
| `unique_source_lines_represented` | distinct `(file, line)` covered by any chunk; a fragmented line counts **once** |
| `overlap_duplicated_lines` | lines counted more than once because of overlap: whole-line chunk line slots minus the distinct lines they cover. Fragments never contribute |
| `chunk_count` | `whole_line_chunk_count + fragment_chunk_count` |
| `fragment_chunk_count` / `fragmented_line_count` | fragment chunks / distinct lines that were fragmented |
| tokens mean/median/p95/max | over chunks, estimated |

## Known limitations

* Structure-blind by design: a function or Markdown section can be split across windows (a test
  documents this); that is what Strategy B is compared against.
* With the default 512-token cap, dense text reaches the cap before 60 lines, so effective windows
  are often shorter than 60 lines and overlap is reduced. The defaults are unmeasured starting
  points. On this repository (Milestone 3 measurement): 216 chunks, median 489 / p95 511 estimated
  tokens, 5240 source lines represented, 1453 lines duplicated by overlap.
* The token estimator is provisional until Milestone 4.
* JSON/YAML windows can cut mid-structure.
* Chunks do not yet include a file-path/context header for embedding (decided in Milestone 4).
* The id contract identifies repositories by `repository_name` only (see above).
