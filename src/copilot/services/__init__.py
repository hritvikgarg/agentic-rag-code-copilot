"""Application services: the single entry point used by the UI (and later the workflow)."""

from copilot.services.copilot_service import (
    CopilotService,
    IndexSummary,
    OpenIndex,
    ServiceStateError,
    summarize_index,
)

__all__ = [
    "CopilotService",
    "IndexSummary",
    "OpenIndex",
    "ServiceStateError",
    "summarize_index",
]
