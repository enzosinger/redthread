"""AttackGraph — LangGraph worker node for executing attack algorithms.

Each AttackGraph instance runs ONE persona through the configured algorithm
(PAIR, TAP, Crescendo, or MCTS) in an isolated context, so multiple can be
fanned-out in parallel by the supervisor via LangGraph's Send API.
"""

from __future__ import annotations

import logging
from typing import Any

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class AttackWorkerState(TypedDict):
    """State schema for a single attack worker node."""

    settings_dict: dict[str, Any]
    persona_dict: dict[str, Any]
    target_system_prompt: str
    rubric_name: str
    result_dict: dict[str, Any] | None
    error: str | None


async def run_attack_worker(state: AttackWorkerState) -> AttackWorkerState:
    """Executes a single attack run for one persona.

    Called by the LangGraph supervisor as a worker node. Deserializes inputs,
    dispatches to the appropriate algorithm, and serializes the result back
    into state for collection by the supervisor's collector node.
    """

    from redthread.config.settings import RedThreadSettings
    from redthread.models import Persona

    try:
        settings = RedThreadSettings.model_validate(state["settings_dict"])
        persona = Persona.model_validate(state["persona_dict"])

        logger.info(
            "⚔️  AttackWorker | persona=%s | algorithm=%s",
            persona.name,
            settings.algorithm.value,
        )

        from redthread.core.attack_runner import build_default_attack_runner_registry

        attacker = build_default_attack_runner_registry().create(settings.algorithm, settings)

        result = await attacker.run(
            persona=persona,
            target_system_prompt=state.get("target_system_prompt", ""),
            rubric_name=state["rubric_name"],
        )

        return {
            **state,
            "result_dict": result.model_dump(mode="json"),
            "error": None,
        }

    except Exception as exc:
        logger.exception("AttackWorker failed: %s", exc)
        return {
            **state,
            "result_dict": None,
            "error": str(exc),
        }
