# Streamlit interface and service layer (Milestone 7)

Run it (the UI libraries are an optional dependency group, so the core install stays small):

```powershell
uv sync --group ui
uv run --group ui streamlit run ui/streamlit_app.py
```

## What it does

1. **Sidebar:** type the repository folder (and the directories to ignore - they must match the
   ones used when the index was built), then *Build index* (local only: chunk + embed, the first
   run downloads the embedding model) or pick an existing index and *Open index*.
2. Opening an index re-ingests the repository and **verifies it still matches the index**
   (fingerprint check). If a file changed since the index was built, the app refuses with the
   retrieval error instead of citing stale line numbers.
3. **Ask:** the question and the retrieved chunks are sent to the hosted model configured in
   `.env`, after the content-secret gate. The page shows the answer, the numbered evidence
   (file, lines, score, the real cited text read from your repository) and plain-language
   warnings (insufficient evidence, uncited answer, unknown source numbers, truncated output).
4. Optional checkbox: show the **plain-LLM answer** (no repository) beneath it. It is labelled as
   ungrounded.

## Design

`ui/streamlit_app.py` contains no retrieval, prompting or LLM logic. It calls
`copilot.services.CopilotService` (build / open / ask / ask_plain / evidence_text) and
`copilot.services.presentation` (pure helpers: ignore-directory parsing, answer warnings, the
list of exceptions whose messages are safe to show). Heavy resources (embedding model, LLM client)
are created lazily and once; both factories are injectable, so the tests run the full flow with a
fake embedder and a fake LLM and no network. An exception that is not on the user-facing list is
reported by type only, never echoed (it could contain a host path).

## Literal file names (Milestone 7.1)

Streamlit renders most widget text as Markdown, so a path such as `src/pkg/__init__.py` showed up
as `src/pkg/` + **init** + `.py` (the double underscores read as bold) in the source expander title.
Checked by rendering real Streamlit 1.67 in a browser:

| Where | Markdown? | Fix |
|---|---|---|
| expander title, `st.success`, `st.caption` | yes | interpolated values go through `presentation.literal()` (an inline code span; the fence is longer than any backtick in the text) |
| `st.error` with an exception message | yes (also `:emoji:`, `$latex$`, `:red[..]`) | whole message through `presentation.escape_markdown()` (backslash before every ASCII punctuation character) |
| `st.selectbox` option labels | **no** | left alone - escaping would display the backslashes |
| `st.code` (the evidence text) | no | unchanged |
| `st.markdown(answer.answer)` | yes, on purpose | unchanged, see limitations |

`tests/unit/test_presentation.py` renders ten awkward paths (`__init__.py`, `**bold**.py`,
`file[1](x).py`, a backtick, `:smile:`, `$x$`, spaces, ...) through a CommonMark renderer and checks
that the path comes back byte for byte; `tests/unit/test_streamlit_app.py` checks the real app's
expander titles and the "Ready" banner.

## Viva notes

* **Why a service layer?** One tested entry point for the UI, the CLI and the LangGraph workflow.
  Without it the logic lives in Streamlit callbacks, where it cannot be unit-tested and gets copied.
* **Why verify the index against the repository on open?** Citations are line numbers. If the
  files moved, a "correct" retrieval would point at the wrong lines. The index stores metadata only,
  so the source text is re-read and every chunk is checked against its hash.
* **What is sent off the machine?** Only the question and the retrieved chunks, after the secret
  gate; indexing and retrieval are local.
* **A bug the tests found:** the index list was computed before a build finished, so a freshly built
  index was not selectable until the next click. The app now opens the new index automatically; the
  end-to-end `AppTest` covers it.

## Limitations

* One user, one process: indexes are chosen by folder; there is no upload or GitHub-URL input yet.
* The UI asks synchronously; a slow model call blocks the page (a spinner is shown).
* No conversation memory: each question is independent.
* The model's own answer is rendered as Markdown (its headings and lists are wanted). If the model
  writes `__init__.py` without backticks, Markdown will still bold it. The UI cannot fix that
  without rewriting the model's text; the "Retrieved sources" titles are the authoritative
  file names.
