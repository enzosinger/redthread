"""Phase 1 GEPA prompt-profile spike orchestration."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from redthread.config.settings import AlgorithmType, RedThreadSettings
from redthread.research.gepa_adapter import RedThreadGEPAAdapter, build_optimize_kwargs
from redthread.research.gepa_frontier import build_frontier_payload, write_frontier_payload
from redthread.research.gepa_profile_runner import (
    PromptProfileBatchRunner,
    seed_candidate_from_profiles,
)
from redthread.research.gepa_score import control_gate_passed
from redthread.research.models import ResearchBatchSummary, ResearchObjective
from redthread.research.objectives import ensure_config
from redthread.research.prompt_profiles import load_prompt_profiles
from redthread.research.runtime import apply_runtime_overrides
from redthread.research.workspace import ResearchWorkspace


class OptimizeFn(Protocol):
    """Callable matching ``gepa.optimize`` for test injection."""

    def __call__(self, **kwargs: Any) -> Any: ...


def run_prompt_profile_spike(
    settings: RedThreadSettings,
    root: Path,
    *,
    reflection_lm: str,
    max_metric_calls: int,
    train_slugs: Sequence[str] = (),
    val_slugs: Sequence[str] = (),
    control_slugs: Sequence[str] = (),
    seed: int = 0,
    algorithm_override: AlgorithmType | None = None,
    optimize_fn: OptimizeFn | None = None,
) -> dict[str, Any]:
    """Run the hidden Phase 1 prompt-profile GEPA spike and persist a summary."""
    workspace = ResearchWorkspace(root)
    workspace.ensure_layout()
    run_settings = apply_runtime_overrides(workspace.research_settings(settings), root)
    config = ensure_config(workspace.runtime_config_path, workspace.template_config_path)
    seed_candidate = seed_candidate_from_profiles(load_prompt_profiles(workspace.prompt_profiles_path))
    trainset = _select(config.experiment_objectives, train_slugs)
    valset = _select(config.benchmark_objectives, val_slugs)
    controlset = _select(config.benchmark_objectives, control_slugs)
    runner = PromptProfileBatchRunner(run_settings, workspace, algorithm_override=algorithm_override)
    adapter = RedThreadGEPAAdapter(runner, components=sorted(seed_candidate))
    kwargs = build_optimize_kwargs(
        seed_candidate,
        trainset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        valset=valset,
        seed=seed,
    )
    if optimize_fn is None:
        from redthread.research.gepa_adapter import _require_gepa

        optimize_fn = _require_gepa().optimize
    result = optimize_fn(**kwargs)
    best_candidate = cast(dict[str, str], getattr(result, "best_candidate", seed_candidate))
    control = _evaluate_control(runner, best_candidate, controlset)
    payload = _result_payload(result, best_candidate, control, config, max_metric_calls)
    frontier = build_frontier_payload(result, control_gate_passed=payload["control_gate_passed"])
    frontier_path = write_frontier_payload(workspace, frontier)
    payload["pareto_frontier_ref"] = str(frontier_path)
    payload["pareto_frontier"] = frontier
    out_path = workspace.gepa_dir / "phase1_prompt_spike.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["summary_ref"] = str(out_path)
    return payload


def _select(objectives: list[ResearchObjective], slugs: Sequence[str]) -> list[ResearchObjective]:
    if not slugs:
        return objectives
    wanted = set(slugs)
    selected = [objective for objective in objectives if objective.slug in wanted]
    missing = wanted - {objective.slug for objective in selected}
    if missing:
        raise ValueError(f"unknown objective slug(s): {sorted(missing)}")
    return selected


def _evaluate_control(
    runner: PromptProfileBatchRunner,
    candidate: dict[str, str],
    objectives: list[ResearchObjective],
) -> ResearchBatchSummary:
    results = [runner(objective, candidate) for objective in objectives]
    confirmed = sum(r.confirmed_jailbreaks for r in results)
    asr = sum(r.attack_success_rate for r in results) / len(results) if results else 0.0
    score = sum(r.average_score for r in results) / len(results) if results else 0.0
    return ResearchBatchSummary(
        run_id=f"gepa-control-{uuid4().hex[:8]}",
        mode="gepa_control",
        lane="control",
        objective_slugs=[r.slug for r in results],
        campaign_ids=[r.campaign_id for r in results],
        total_campaigns=len(objectives),
        total_results=len(results),
        confirmed_jailbreaks=confirmed,
        near_misses=sum(r.near_misses for r in results),
        average_asr=asr,
        average_score=score,
        composite_score=(confirmed * 10.0) + score,
        objective_results=results,
    )


def _result_payload(
    result: Any,
    best_candidate: dict[str, str],
    control: ResearchBatchSummary,
    config: Any,
    max_metric_calls: int,
) -> dict[str, Any]:
    gate = control_gate_passed(
        control,
        max_average_asr=config.control_max_average_asr,
        max_average_score=config.control_max_average_score,
    )
    return {
        "phase": "gepa_phase1_prompt_profile_spike",
        "best_candidate": best_candidate,
        "control_gate_passed": gate,
        "control_average_asr": control.average_asr,
        "control_average_score": control.average_score,
        "max_metric_calls": max_metric_calls,
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "gepa_result": result.to_dict() if hasattr(result, "to_dict") else None,
        "accepted_by_redthread_supervisor": False,
        "promotion_status": "not_promoted",
    }
