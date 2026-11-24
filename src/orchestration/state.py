"""Shared state and separate results for Research and Data."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResearchResult(BaseModel):
    """Research returns an answer, its sources, and any failure information."""

    status: Literal["success", "partial", "failed"]
    citation_checks: list[dict[str, Any]] = Field(default_factory=list)
    answer: str = ""
    sources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    error: str | None = None  # Safe description, never a raw API error or key.


class DataResult(BaseModel):
    """Query or aggregate results with an explanation."""

    status: Literal["success", "partial", "failed"]
    rows: list[dict[str, Any]] = Field(default_factory=list)
    explanation: str
    sources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    error: str | None = None  # Safe description, never raw SQL/API errors.


class DocumentRequest(BaseModel):
    """Use page=None for full text; PDF pages are one-based physical pages."""
    doc_id: str
    page: int | None = Field(default=None, ge=1)


class SummaryResult(BaseModel):
    retryable: bool = True
    status: Literal["success", "partial", "failed"]
    answer: str = ""
    sources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    citation_checks: list[dict[str, Any]] = Field(default_factory=list)
    provider: str = ""
    route_reason: str = ""
    error: str | None = None


class State(BaseModel):
    agent_plan: list[Literal["research", "data", "summary"]] = Field(default_factory=list)
    plan_index: int = 0
    repair_count: int = 0
    query: str = Field(min_length=1)
    research_task: str  # Supervisor's instruction to Research.
    research_query: str = ""  # Short search query, separate from writing instructions.
    research_use_web: bool = False  # Search current web information when needed.
    research_result: ResearchResult | None = None  # None means not run yet.
    data_task: str  # Supervisor's instruction to Data.
    data_result: DataResult | None = None  # None means not run yet.
    summary_mode: Literal["documents", "shared"] = "documents"
    summary_task: str = ""
    summary_documents: list[DocumentRequest] = Field(default_factory=list)
    summary_use_web: bool = False
    summary_result: SummaryResult | None = None
    next_agent: Literal["research", "data", "summary", "end"] = "research"
    reason: str  # Why Supervisor chose the next step.
    step_count: int = Field(default=0, ge=0)  # Limit the number of loops
    final_answer: str | None = None
