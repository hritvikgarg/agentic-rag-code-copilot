"""Streamlit front-end: index a repository, ask questions, see the evidence.

Run:  uv run --group ui streamlit run ui/streamlit_app.py

This file contains NO retrieval, prompting or LLM logic: it only calls ``CopilotService`` and
formats what comes back. Questions and retrieved code chunks are sent to the hosted model
configured in ``.env`` (after the secret gate); building an index is local.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from copilot.rag.models import AnswerStatus
from copilot.services.copilot_service import CopilotService
from copilot.services.presentation import (
    USER_FACING_ERRORS,
    answer_warnings,
    parse_ignore_directories,
)

st.set_page_config(page_title="Repository Copilot", layout="wide")


def get_service() -> CopilotService:
    """One service per browser session (tests pre-seed ``st.session_state['service']``)."""
    if "service" not in st.session_state:
        st.session_state["service"] = CopilotService()
    return st.session_state["service"]


def show_error(exc: Exception) -> None:
    if isinstance(exc, USER_FACING_ERRORS):
        st.error(str(exc))
    else:  # a bug: never echo an unknown exception (it may contain host paths)
        st.error(f"Unexpected error ({type(exc).__name__}). See the server log.")


service = get_service()
settings = service.settings

st.title("Repository Copilot")
st.caption(
    "Ask questions about a repository. Answers are built from retrieved code and cite the files "
    "and lines they came from. When the evidence is not there, the app says so."
)

# --- sidebar: repository, index, settings --------------------------------------------------
with st.sidebar:
    st.header("Repository")
    repo_text = st.text_input("Repository folder", value=st.session_state.get("repo_text", ""))
    st.session_state["repo_text"] = repo_text
    ignore_text = st.text_input(
        "Ignore directories (comma separated)",
        value=st.session_state.get("ignore_text", ""),
        help="Must match the directories ignored when the index was built.",
    )
    st.session_state["ignore_text"] = ignore_text
    ignore_dirs = parse_ignore_directories(ignore_text)

    st.header("Index")
    indexes = service.list_indexes()
    labels = [
        f"{i.repository_name} | {i.index_id} | {i.chunk_count} chunks | {i.created_at}"
        for i in indexes
    ]
    choice = (
        st.selectbox(
            "Existing indexes", options=range(len(indexes)), format_func=lambda n: labels[n]
        )
        if indexes
        else None
    )
    if not indexes:
        st.info("No index yet. Build one below (local; nothing is sent anywhere).")

    col_open, col_build = st.columns(2)
    if col_open.button("Open index", disabled=choice is None or not repo_text):
        try:
            with st.spinner("Loading index and verifying the repository..."):
                st.session_state["opened"] = service.open_index(
                    indexes[choice].path, repo_text, ignore_directories=ignore_dirs
                )
            st.session_state.pop("answer", None)
        except Exception as exc:  # noqa: BLE001 - routed through show_error
            show_error(exc)
    if col_build.button("Build index", disabled=not repo_text):
        if not Path(repo_text).is_dir():
            st.error("That folder does not exist.")
        else:
            try:
                with st.spinner(
                    "Chunking and embedding locally (first run downloads the model)..."
                ):
                    report = service.build_index(repo_text, ignore_directories=ignore_dirs)
                    # Open the index we just built so the user can ask immediately (the index
                    # list above was computed before the build, so it cannot show it yet).
                    st.session_state["opened"] = service.open_index(
                        report.index_path, repo_text, ignore_directories=ignore_dirs
                    )
                st.session_state.pop("answer", None)
                st.session_state["notice"] = (
                    f"Indexed {report.files} files into {report.chunks} chunks "
                    f"in {report.seconds_total:.1f}s (index {report.index_id})."
                    + (
                        " WARNING: ingestion hit a repository limit; the index is incomplete."
                        if report.ingestion_truncated
                        else ""
                    )
                )
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                show_error(exc)

    st.header("Settings")
    top_k = st.slider("Chunks retrieved (top-k)", 1, 20, settings.retrieval_top_k)
    compare_plain = st.checkbox("Also show the plain-LLM answer (no repository)", value=False)
    st.caption(f"Model: {settings.llm_provider}/{settings.llm_model or 'NOT SET'}")
    st.caption(f"Embeddings: {settings.embedding_model} (local)")

notice = st.session_state.pop("notice", None)
if notice:
    st.sidebar.success(notice)
opened = st.session_state.get("opened") or service.opened
if opened is not None:
    s = opened.summary
    st.success(
        f"Ready: {s.repository_name} - index {s.index_id} ({s.chunk_count} chunks from "
        f"{s.source_file_count} files, {s.chunking_strategy})"
    )
else:
    st.info("Open or build an index in the sidebar to start asking questions.")

# --- ask -----------------------------------------------------------------------------------
question = st.text_area("Your question", placeholder="Where is repository ingestion implemented?")
if st.button("Ask", type="primary", disabled=opened is None or not question.strip()):
    try:
        with st.spinner("Retrieving evidence and asking the model..."):
            st.session_state["answer"] = service.ask(question, top_k)
            st.session_state["plain"] = service.ask_plain(question) if compare_plain else None
    except Exception as exc:  # noqa: BLE001
        st.session_state.pop("answer", None)
        show_error(exc)

answer = st.session_state.get("answer")
if answer is not None:
    for warning in answer_warnings(answer):
        (st.warning if answer.status is AnswerStatus.ANSWERED else st.info)(warning)
    if answer.status is AnswerStatus.ANSWERED:
        st.subheader("Answer")
        st.markdown(answer.answer or "")
    st.subheader("Retrieved sources")
    if not answer.sources:
        st.write("No evidence was supplied.")
    for source in answer.sources:
        cited = (
            " (cited in the answer)" if source.source_number in answer.cited_source_numbers else ""
        )
        with st.expander(f"[{source.label}] {source.location} - score {source.score:.3f}{cited}"):
            st.code(service.evidence_text(source), language=None)
    gen = answer.generation
    if gen is not None:
        st.caption(
            f"{gen.provider}/{gen.model} - tokens {gen.total_tokens} - "
            f"retrieval {answer.retrieval_seconds}s, generation {answer.generation_seconds}s"
        )

plain = st.session_state.get("plain")
if plain is not None:
    st.subheader("Plain LLM (no repository - NOT grounded)")
    st.warning(
        "This answer had no access to the repository; file names or functions may be invented."
    )
    st.markdown(plain.answer)
