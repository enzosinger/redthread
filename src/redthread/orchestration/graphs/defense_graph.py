"""DefenseGraph — LangGraph worker node for post-evaluation defense synthesis.

Receives a judged AttackResult, passes it through the DefenseSynthesisEngine
(Isolate → Classify → Generate → Validate), and writes validated candidate
guardrail evidence to MEMORY.md via MemoryIndex.

Only called for confirmed jailbreaks (is_jailbreak=True from JudgeWorker).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, NotRequired

from typing_extensions import TypedDict

from redthread.core.defense_status import candidate_flags

logger = logging.getLogger(__name__)


class DefenseWorkerState(TypedDict):
    """State schema for the defense synthesis worker node."""

    settings_dict: dict[str, Any]
    result_dict: dict[str, Any]
    validated_candidate: bool
    defense_deployed: bool
    guardrail_clause: str | None
    record_dict: NotRequired[dict[str, Any]]
    error: str | None


async def run_defense_worker(state: DefenseWorkerState) -> dict[str, Any]:
    """Run defense synthesis and index validated candidates, not active controls.

    Called by the LangGraph supervisor only for confirmed jailbreaks.
    """
    from redthread.config.settings import RedThreadSettings
    from redthread.core.defense_synthesis import DefenseSynthesisEngine
    from redthread.memory.index import MemoryIndex
    from redthread.models import AttackResult

    try:
        settings = RedThreadSettings.model_validate(state["settings_dict"])
        result = AttackResult.model_validate(state["result_dict"])

        logger.info(
            "🛡️  DefenseWorker | trace=%s | synthesizing candidate_defense...",
            result.trace.id,
        )

        engine = DefenseSynthesisEngine(settings)
        record = await engine.run(result)

        if record.validation.passed:
            index = MemoryIndex(settings)
            index.append(record, guardrail_status="validated_candidate")
            logger.info(
                "✅ DefenseWorker | validated_candidate indexed | trace=%s | category=%s",
                record.trace_id,
                record.classification.category,
            )
            return {
                **state,
                **candidate_flags(True),
                "guardrail_clause": record.guardrail_clause,
                "record_dict": asdict(record),
                "error": None,
            }
        else:
            logger.warning(
                "⚠️  DefenseWorker | validation failed | trace=%s | replay_score=%.2f",
                record.trace_id,
                record.validation.judge_score,
            )
            return {
                **state,
                **candidate_flags(False),
                "guardrail_clause": record.guardrail_clause,
                "record_dict": asdict(record),
                "error": None,
            }

    except Exception as exc:
        logger.exception("DefenseWorker failed: %s", exc)
        return {
            **state,
            **candidate_flags(False),
            "guardrail_clause": None,
            "error": str(exc),
        }
