"""Guardrail Loader — Phase 4.5 Stabilization.

Bridging the gap between defense synthesis (`MEMORY.md`) and the live session.
At the start of a campaign, the target LLM's scoped guardrails are loaded
from the `MemoryIndex` and structurally injected into the configuration.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from redthread.config.settings import RedThreadSettings
from redthread.core.defense_assets import append_guardrails_to_system_prompt
from redthread.core.defense_models import DeploymentRecord
from redthread.memory.index import MemoryIndex
from redthread.models import CampaignConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardrailInjectionAudit:
    """Non-secret proof of one runtime guardrail injection decision."""

    target_model: str
    prompt_hash: str
    action: str
    active_guardrail_count: int
    active_trace_ids: list[str] = field(default_factory=list)
    clause_hashes: list[str] = field(default_factory=list)
    source: str = "structured_deployments"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_event(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "target_model": self.target_model,
            "prompt_hash": self.prompt_hash,
            "action": self.action,
            "active_guardrail_count": self.active_guardrail_count,
            "active_trace_ids": self.active_trace_ids,
            "clause_hashes": self.clause_hashes,
            "source": self.source,
        }


class GuardrailLoader:
    """Dynamically loads and injects targeted guardrails into a campaign.
    
    Guardrails are scoped by `target_model` and the base `system_prompt`.
    """

    def __init__(self, settings: RedThreadSettings, memory_index: MemoryIndex | None = None) -> None:
        self.settings = settings
        self.memory = memory_index or MemoryIndex(settings)
        self.last_audit: GuardrailInjectionAudit | None = None

    def _compute_prompt_hash(self, prompt: str) -> str:
        normalized = (prompt or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def get_scoped_clauses(self, target_model: str, base_system_prompt: str) -> list[str]:
        """Fetch all validated guardrail clauses from MEMORY.md for this scope."""
        prompt_hash = self._compute_prompt_hash(base_system_prompt)
        return self.memory.load_scoped_guardrails(target_model, prompt_hash)

    def inject_guardrails(self, config: CampaignConfig) -> CampaignConfig:
        """Return a new CampaignConfig with guardrails appended to target_system_prompt.
        
        If no guardrails match the current target model and base system prompt,
        returns the original config unmodified.
        """
        prompt_hash = self._compute_prompt_hash(config.target_system_prompt)
        records = self.memory.load_scoped_guardrail_records(self.settings.target_model, prompt_hash)
        clauses = [record.guardrail_clause for record in records]
        source = "structured_deployments"
        if not clauses and not self.memory.iter_deployments():
            clauses = self.get_scoped_clauses(self.settings.target_model, config.target_system_prompt)
            source = "legacy_markdown"
        audit = self._build_audit(self.settings.target_model, prompt_hash, records, clauses, source)

        if not clauses:
            logger.info("🛡️ GuardrailLoader | No active guardrails found for target scope.")
            self._log_audit_event(audit)
            return config

        logger.info(
            "🛡️ GuardrailLoader | Injected %d active guardrail(s) into target system prompt.",
            len(clauses)
        )
        self._log_audit_event(audit)

        injected_prompt = append_guardrails_to_system_prompt(config.target_system_prompt, clauses)
        return config.model_copy(update={"target_system_prompt": injected_prompt})

    def _build_audit(
        self,
        target_model: str,
        prompt_hash: str,
        records: list[DeploymentRecord],
        clauses: list[str],
        source: str,
    ) -> GuardrailInjectionAudit:
        return GuardrailInjectionAudit(
            target_model=target_model,
            prompt_hash=prompt_hash,
            action="INJECT" if clauses else "SKIP",
            active_guardrail_count=len(clauses),
            active_trace_ids=[record.trace_id for record in records],
            clause_hashes=[self._compute_prompt_hash(clause) for clause in clauses],
            source=source,
        )

    def _log_audit_event(self, audit: GuardrailInjectionAudit) -> None:
        """Write non-secret structured audit proof to logs/guardrail_audit.jsonl."""
        audit_path = self.settings.log_dir / "guardrail_audit.jsonl"
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        self.last_audit = audit
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(audit.as_event()) + "\n")
