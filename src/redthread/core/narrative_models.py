"""Narrative adaptation models for Crescendo's phase-aware strategy layer.

These are pure data containers — no behavior.
Policy logic lives in narrative_policy.py.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class NarrativePhase(str, Enum):
    """Ordered phases of a Crescendo narrative arc."""

    RAPPORT = "rapport"
    LEGITIMACY = "legitimacy"
    PRESSURE = "pressure"
    ASK = "ask"

    def next(self) -> NarrativePhase:
        """Advance to the next phase (capped at ASK)."""
        order = list(NarrativePhase)
        idx = order.index(self)
        return order[min(idx + 1, len(order) - 1)]


class NarrativeState(BaseModel):
    """Mutable narrative state tracked across Crescendo turns.

    Updated by NarrativeAdaptationPolicy.update() after each accepted turn.
    Serialized into trace.metadata["narrative_state"] at end of run.
    """

    phase: NarrativePhase = NarrativePhase.RAPPORT
    used_strategies: list[str] = Field(default_factory=list)
    successful_strategies: list[str] = Field(default_factory=list)
    failed_strategies: list[str] = Field(default_factory=list)
    pivot_count: int = 0
    consecutive_failures: int = 0
    last_refusal_signal: str | None = None
    turn_count: int = 0


class NarrativeGuidance(BaseModel):
    """Policy output: what Crescendo should do on the next turn.

    Injected as a ## Narrative Guidance section in the attacker prompt.
    """

    recommended_strategy: str
    phase_action: Literal["hold", "escalate", "pivot"]
    framing_hint: str
    avoid_patterns: list[str]
