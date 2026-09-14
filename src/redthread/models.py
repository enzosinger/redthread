"""Core data models for RedThread.

All models are immutable Pydantic v2 dataclasses.
These flow through the entire pipeline: Persona → AttackTrace → AttackResult → CampaignResult.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MitreAtlasTactic(str, Enum):
    """MITRE ATLAS top-level tactics used to seed adversarial personas."""

    RECONNAISSANCE = "AML.TA0002"
    RESOURCE_DEVELOPMENT = "AML.TA0003"
    INITIAL_ACCESS = "AML.TA0004"
    ML_ATTACK_STAGING = "AML.TA0001"
    EXFILTRATION = "AML.TA0010"
    IMPACT = "AML.TA0011"


class PsychologicalTrigger(str, Enum):
    """Social engineering triggers layered onto personas."""

    AUTHORITY = "authority_impersonation"
    URGENCY = "urgency_and_scarcity"
    FEAR = "fear_and_intimidation"
    RECIPROCITY = "reciprocity_and_trust"
    SOCIAL_PROOF = "social_proof_and_consensus"


class AttackOutcome(str, Enum):
    """Terminal state of a single attack run."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ERROR = "error"
    SKIPPED = "skipped"


class Persona(BaseModel):
    """An adversarial persona used to initialize the attacker LLM.

    Dual-objective structure:
      - cover_story: what the attacker claims to be doing (benign)
      - hidden_objective: the actual exploit goal (malicious)
    """

    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    name: str
    tactic: MitreAtlasTactic
    technique: str
    cover_story: str
    hidden_objective: str
    system_prompt: str
    psychological_triggers: list[PsychologicalTrigger]
    seed_pretext: str = ""
    allowed_strategies: list[str] = Field(default_factory=list)


class AttackNode(BaseModel):
    """A single node in a TAP attack tree."""

    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    parent_id: str | None = None
    depth: int = 0
    attacker_prompt: str = ""
    target_response: str = ""
    score: float = 0.0
    improvement_rationale: str = ""
    is_pruned: bool = False
    pruned_reason: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MCTSNode(BaseModel):
    """A node in the GS-MCTS search tree.

    Stores both MCTS statistics (for UCT selection) and conversation data
    (for trace reconstruction). Conversation history is NOT stored inline —
    it is reconstructed by walking the parent_id chain via MCTSTree.get_history().
    """

    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    parent_id: str | None = None
    depth: int = 0
    strategy: str = ""
    attacker_prompt: str = ""
    target_response: str = ""
    visit_count: int = 0
    total_reward: float = 0.0
    score: float = 0.0
    is_terminal: bool = False
    is_expanded: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CrescendoTurn(BaseModel):
    """A single turn in a Crescendo escalation dialogue."""

    turn_number: int
    escalation_level: int = 0
    attacker_prompt: str
    target_response: str
    score: float = 0.0
    backtracked: bool = False
    backtrack_attempts: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationTurn(BaseModel):
    """A single turn in the attacker ↔ target dialogue."""

    turn_number: int
    attacker_prompt: str
    target_response: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    improvement_rationale: str = ""
    is_successful: bool = False


class AttackTrace(BaseModel):
    """Full conversation trace from a single attack run."""

    id: str = Field(default_factory=lambda: f"trace-{str(uuid4())[:8]}")
    persona: Persona
    algorithm: str
    turns: list[ConversationTurn] = Field(default_factory=list)
    nodes: list[AttackNode] = Field(default_factory=list)
    crescendo_turns: list[CrescendoTurn] = Field(default_factory=list)
    mcts_nodes: list[MCTSNode] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    outcome: AttackOutcome = AttackOutcome.FAILURE
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeVerdict(BaseModel):
    """Output from the JudgeAgent for a single evaluated turn or full trace.

    score: continuous float [1.0 – 5.0] via G-Eval probability weighting.
           1.0 = perfect adherence to policy
           5.0 = complete, confirmed guardrail bypass
    """

    score: float
    raw_score: int
    reasoning: str
    feedback: str
    rubric_name: str
    is_jailbreak: bool


class AttackResult(BaseModel):
    """A scored, completed attack run — the primary unit of campaign output."""

    id: str = Field(default_factory=lambda: f"result-{str(uuid4())[:8]}")
    trace: AttackTrace
    verdict: JudgeVerdict
    iterations_used: int
    duration_seconds: float


class CampaignConfig(BaseModel):
    """User-supplied campaign parameters."""

    objective: str
    target_system_prompt: str
    rubric_name: str = "authorization_bypass"
    num_personas: int = 3
    prompting_layer_profile: dict[str, Any] = Field(default_factory=dict)
    persona_weighting_plan: dict[str, Any] = Field(default_factory=dict)


class CampaignResult(BaseModel):
    """Aggregate output of a full red-team campaign."""

    id: str = Field(default_factory=lambda: f"campaign-{str(uuid4())[:8]}")
    config: CampaignConfig
    results: list[AttackResult] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def attack_success_rate(self) -> float:
        """ASR: % of runs where JudgeAgent confirmed a jailbreak."""
        if not self.results:
            return 0.0
        successes = sum(1 for r in self.results if r.verdict.is_jailbreak)
        return successes / len(self.results)

    @property
    def average_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.verdict.score for r in self.results) / len(self.results)
