"""Typed models for the GEPA shadow harness (Phase 0).

A ``GepaCandidate`` is a bounded prompt-profile snapshot proposal. A
``GepaEvaluationResult`` records how RedThread scored it, keeping the GEPA-accept,
RedThread-accept, and promotion decisions explicitly separate. Nothing here writes
production state; these are research-plane artifacts only.
"""

from __future__ import annotations

from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class GepaSplit(str, Enum):
    """Which evaluation split an objective belongs to."""

    TRAIN = "train"
    VAL = "val"
    CONTROL = "control"


class GepaCandidate(BaseModel):
    """One bounded prompt-profile snapshot proposed by a GEPA proposer."""

    candidate_id: str = Field(default_factory=lambda: f"gepa-{uuid4().hex[:8]}")
    parent_candidate_ids: list[str] = Field(default_factory=list)
    mutation_surface: str = "prompt_profile_snapshot"
    components: dict[str, str] = Field(default_factory=dict)
    touched_fields: list[str] = Field(default_factory=list)
    budget_used: int = 0
    snapshot_ref: str | None = None


class GepaObjectiveScore(BaseModel):
    """One axis of a candidate's Pareto vector: normalized score on one objective."""

    slug: str
    split: GepaSplit
    objective_score: float


class GepaEvaluationResult(BaseModel):
    """RedThread's scoring of a candidate, with authority levels kept separate."""

    candidate_id: str
    redthread_score_source: str = "composite_score"
    scalar_score_for_optimizer: float = 0.0
    objective_scores: list[GepaObjectiveScore] = Field(default_factory=list)
    control_gate_passed: bool = False
    accepted_by_gepa: bool = False
    accepted_by_redthread_supervisor: bool = False
    promotion_status: str = "not_promoted"
    side_info_ref: str | None = None
