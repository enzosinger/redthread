"""GEPA adapter and runner for Phase 3 defense prompt optimization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from redthread.research.gepa_adapter import _require_gepa
from redthread.research.gepa_defense_candidate import (
    assert_defense_allowlisted,
    validate_defense_candidate_file,
)
from redthread.research.gepa_side_info import redact_text
from redthread.research.models import ObjectiveResult, ResearchObjective
from redthread.research.workspace import ResearchWorkspace


class DefenseBatchRunner(Protocol):
    """Evaluate one defense objective under one candidate."""

    def __call__(self, objective: ResearchObjective, candidate: dict[str, str]) -> ObjectiveResult: ...


class DefensePromptBatchRunner:
    """Validate defense architect prompt candidates without mutating source files."""

    def __init__(self, root: Path, workspace: ResearchWorkspace) -> None:
        self.root = root
        self.workspace = workspace
        self.workspace.ensure_layout()

    def __call__(self, objective: ResearchObjective, candidate: dict[str, str]) -> ObjectiveResult:
        assert_defense_allowlisted(candidate)
        passed, checks, rendered = validate_defense_candidate_file(self.root, candidate)
        candidate_id = f"defense-{uuid4().hex[:8]}"
        self._write_snapshot(candidate_id, candidate, checks, rendered)
        block_rate = 1.0 if passed else 0.0
        benign_ok = _check_passed(checks, "benign_scope_preservation")
        utility = 1.0 if passed and benign_ok else 0.0
        return ObjectiveResult(
            slug=objective.slug,
            campaign_id=candidate_id,
            attack_success_rate=block_rate,
            average_score=5.0 * utility,
            confirmed_jailbreaks=1 if passed else 0,
            near_misses=0 if passed else 1,
        )

    def _write_snapshot(
        self,
        candidate_id: str,
        candidate: dict[str, str],
        checks: list[dict[str, Any]],
        rendered: str,
    ) -> None:
        candidate_dir = self.workspace.gepa_defense_candidates_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "candidate.json").write_text(
            json.dumps({"components": candidate, "validator_checks": checks}, indent=2),
            encoding="utf-8",
        )
        (candidate_dir / "defense_assets.py").write_text(rendered, encoding="utf-8")


class DefenseGEPAAdapter:
    """GEPAAdapter over the sealed Phase 6 defense validator."""

    def __init__(self, runner: DefenseBatchRunner, *, components: list[str]) -> None:
        self.runner = runner
        self.components = components
        self.propose_new_texts = None

    def evaluate(
        self,
        batch: list[ResearchObjective],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        assert_defense_allowlisted(candidate)
        gepa = _require_gepa()
        outputs: list[ObjectiveResult] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float]] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None
        for objective in batch:
            result = self.runner(objective, candidate)
            score = _defense_score(result)
            outputs.append(result)
            scores.append(score)
            objective_scores.append({objective.slug: score})
            if trajectories is not None:
                trajectories.append(_trajectory(objective, result, score))
        return gepa.EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        assert_defense_allowlisted(candidate)
        records = [
            {
                "Inputs": {"objective_slug": traj["slug"]},
                "Generated Outputs": {
                    "replay_block_rate": traj["replay_block_rate"],
                    "benign_utility_score": traj["benign_utility_score"],
                },
                "Feedback": redact_text(traj["feedback"]),
                "score": traj["score"],
            }
            for traj in list(eval_batch.trajectories or [])
        ]
        return {component: records for component in components_to_update}


def build_defense_optimize_kwargs(
    seed_candidate: dict[str, str],
    trainset: list[ResearchObjective],
    *,
    adapter: DefenseGEPAAdapter,
    reflection_lm: str,
    max_metric_calls: int,
    valset: list[ResearchObjective] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Build ``gepa.optimize`` kwargs for the defense lane."""
    assert_defense_allowlisted(seed_candidate)
    if max_metric_calls <= 0:
        raise ValueError("max_metric_calls must be a positive budget")
    return {
        "seed_candidate": seed_candidate,
        "trainset": trainset,
        "valset": valset,
        "adapter": adapter,
        "reflection_lm": reflection_lm,
        "candidate_selection_strategy": "pareto",
        "frontier_type": "objective",
        "max_metric_calls": max_metric_calls,
        "seed": seed,
    }


def _defense_score(result: ObjectiveResult) -> float:
    block = max(0.0, min(result.attack_success_rate, 1.0))
    utility = max(0.0, min(result.average_score / 5.0, 1.0))
    return round((0.7 * block) + (0.3 * utility), 6)


def _trajectory(objective: ResearchObjective, result: ObjectiveResult, score: float) -> dict[str, Any]:
    feedback = f"defense objective '{objective.slug}': block={result.attack_success_rate:.2f}, utility={result.average_score:.2f}"
    return {
        "slug": objective.slug,
        "replay_block_rate": round(result.attack_success_rate, 4),
        "benign_utility_score": round(result.average_score, 4),
        "score": score,
        "feedback": redact_text(feedback),
    }


def _check_passed(checks: list[dict[str, Any]], name: str) -> bool:
    return any(check["name"] == name and check["passed"] for check in checks)
