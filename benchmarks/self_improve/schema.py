"""Normalized trajectory schema shared across all three benchmark ingestion
modules (aider_polyglot, gaia, harbor/tb). See TDD_SPEC.md for the shapes each
ingest/*.py module is expected to produce."""
from typing import Literal

from pydantic import BaseModel, Field


class ComponentUsage(BaseModel):
    pred_name: str
    invocation_count: int = Field(default=0, ge=0)
    was_error_context: bool = False


class NormalizedTrajectory(BaseModel):
    benchmark: Literal["aider_polyglot", "gaia", "harbor", "tb"]
    task_id: str
    success: bool
    stop_reason: str
    turn_count: int
    partial_score: float | None = Field(default=None, ge=0.0, le=1.0)
    components_used: list[ComponentUsage] = Field(default_factory=list)
    failure_signals: list[str] = Field(default_factory=list)
    summarized_transcript: str = ""
    raw_paths: dict[str, str] = Field(default_factory=dict)
