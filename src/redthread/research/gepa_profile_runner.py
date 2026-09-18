"""Prompt-profile runner for the Phase 1 GEPA spike."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from redthread.config.settings import AlgorithmType, RedThreadSettings
from redthread.research.baseline import run_batch
from redthread.research.gepa_allowlist import ALLOWED_COMPONENT_FIELDS, assert_allowlisted
from redthread.research.models import ObjectiveResult, ResearchObjective
from redthread.research.prompt_profiles import load_prompt_profiles
from redthread.research.workspace import ResearchWorkspace


def seed_candidate_from_profiles(
    profiles: dict[str, Any],
    fields: tuple[str, ...] = tuple(sorted(ALLOWED_COMPONENT_FIELDS)),
) -> dict[str, str]:
    """Flatten allowlisted prompt-profile fields into GEPA string components."""
    candidate: dict[str, str] = {}
    for field in fields:
        section, key = field.split(".", maxsplit=1)
        value = profiles.get(section, {}).get(key)
        if value is None:
            continue
        candidate[field] = "\n".join(value) if isinstance(value, list) else str(value)
    assert_allowlisted(candidate)
    return candidate


def apply_candidate_to_profiles(
    profiles: dict[str, Any],
    candidate: dict[str, str],
) -> dict[str, Any]:
    """Return a prompt-profile copy with an allowlisted GEPA candidate applied."""
    assert_allowlisted(candidate)
    updated = deepcopy(profiles)
    for field, value in candidate.items():
        section, key = field.split(".", maxsplit=1)
        current = updated.setdefault(section, {}).get(key)
        updated[section][key] = _as_profile_value(value, current)
    return updated


def _as_profile_value(value: str, current: Any) -> Any:
    """Preserve list-valued profile fields such as ``tap.strategies``."""
    if isinstance(current, list):
        items = [line.strip(" -\t") for line in value.splitlines()]
        return [item for item in items if item]
    return value


class PromptProfileBatchRunner:
    """Run one objective with a GEPA candidate scoped to a runtime snapshot."""

    def __init__(
        self,
        settings: RedThreadSettings,
        workspace: ResearchWorkspace,
        *,
        algorithm_override: AlgorithmType | None = None,
    ) -> None:
        self.settings = settings
        self.workspace = workspace
        self.algorithm_override = algorithm_override
        self.workspace.ensure_layout()

    def __call__(self, objective: ResearchObjective, candidate: dict[str, str]) -> ObjectiveResult:
        candidate_dir = self._write_candidate_profiles(candidate)
        scoped_settings = self.settings.model_copy(
            update={"research_runtime_dir": candidate_dir, "memory_dir": self.workspace.research_memory_dir}
        )
        summary = asyncio.run(
            run_batch(
                scoped_settings,
                [objective],
                mode="gepa_eval",
                lane="gepa",
                algorithm_override=self.algorithm_override,
            )
        )
        if summary.objective_results:
            return summary.objective_results[0]
        return ObjectiveResult(
            slug=objective.slug,
            campaign_id=summary.campaign_ids[0] if summary.campaign_ids else summary.run_id,
            attack_success_rate=summary.average_asr,
            average_score=summary.average_score,
            confirmed_jailbreaks=summary.confirmed_jailbreaks,
            near_misses=summary.near_misses,
        )

    def _write_candidate_profiles(self, candidate: dict[str, str]) -> Path:
        candidate_id = f"phase1-{uuid4().hex[:8]}"
        candidate_dir = self.workspace.gepa_candidates_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        base = load_prompt_profiles(self.workspace.prompt_profiles_path)
        updated = apply_candidate_to_profiles(base, candidate)
        (candidate_dir / "prompt_profiles.json").write_text(
            json.dumps(updated, indent=2),
            encoding="utf-8",
        )
        return candidate_dir
