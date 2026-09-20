# Baseline chunking (Milestone 3)

Implements **Strategy A** only: structure-blind line windows for code and data files, heading
sections for Markdown. The structure-aware AST strategy (Strategy B, Milestone 9) is a separate
implementation behind the same `Chunker` interface; `chunking_strategy="ast"` is a valid setting
but is *reserved* and raises `StrategyNotImplementedError` rather than silently falling back to the
baseline (a silent fallback would corrupt the planned experiment).

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

| File kind | Rule | `chunk_type` |
|---|---|---|
| Code (`.py .js .jsx .ts .tsx .java .c .cpp`) | Overlapping windows of `chunk_size_lines` (60) lines, `chunk_overlap_lines` (10) overlap | `code_window` |
| JSON / YAML | Same windows | `config_window` |
| Markdown | One chunk per ATX heading section; oversized sections are windowed | `doc_section` |

Window rules (see `chunking/windows.py`):

1. A window is shortened, never truncated mid-line, if it would exceed `chunk_max_tokens` (512).
2. The next window starts `overlap` lines before the previous end, but only windows that add at
   least one new line are emitted (no near-duplicate cascades).
3. A single line above the token cap is split into contiguous **fragments** on token boundaries
   (`line_fragment=True`); concatenating them reproduces the line exactly.
4. Empty and whitespace-only files/windows produce no chunks.

Markdown: headings inside fenced code blocks are ignored; a heading with no body of its own is
merged into the next section; each chunk carries its heading path (`Guide > Install`).

## Chunk metadata

`chunk_id` (deterministic 16-hex hash of repo, strategy, path, position, lines and content),
`repository_name`, `file_path`, `language`, `chunk_type`, `chunking_strategy`, `chunk_index`,
`start_line`/`end_line` (1-based, inclusive, of the newline-normalised text), `content`,
`content_sha256`, `token_estimate`, `line_fragment`, `heading`, and the symbol fields
`symbol_name`, `qualified_name`, `parent_class` (always `None` for the baseline; filled in
Milestone 9).

Invariants (tested): `content` equals the source lines `start..end` joined with `"\n"` (or is a
piece of one line when `line_fragment`), no chunk exceeds the token cap, ids are unique, and every
non-blank line is covered by at least one chunk.

## Token estimate

`utils/tokens.py` counts ASCII word runs and every other non-space character. It needs no
tokenizer download, is additive over lines, and is deliberately conservative for non-ASCII text.
It is a **heuristic**: it is not validated against a real tokenizer until Milestone 4.

## Known limitations

* Structure-blind: a function can be split across windows (documented by a test; this is the
  weakness Strategy B is meant to address).
* With the default 512-token cap, dense code hits the token cap before the 60-line limit, so
  effective windows are often shorter than 60 lines. The defaults are unmeasured starting points.
* Token counts are estimates; the real embedding model's limit is checked in Milestone 4.
* Setext (`Title` / `====`) and HTML headings are not recognised as Markdown sections.
* JSON/YAML are windowed by lines and can be cut mid-structure.
* Chunks do not yet include a file-path/context header for embedding (decided in Milestone 4).
