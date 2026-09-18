"""Phase 1 tests for the hidden GEPA prompt-profile spike."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from redthread.config.settings import RedThreadSettings
from redthread.research.gepa_phase1 import run_prompt_profile_spike
from redthread.research.gepa_profile_runner import (
    PromptProfileBatchRunner,
    apply_candidate_to_profiles,
    seed_candidate_from_profiles,
)
from redthread.research.models import ObjectiveResult, ResearchBatchSummary, ResearchObjective
from redthread.research.workspace import ResearchWorkspace


def _objective(slug: str = "prompt_injection") -> ResearchObjective:
    return ResearchObjective(
        slug=slug,
        objective="test objective",
        system_prompt="You are safe.",
        rubric_name="prompt_injection",
        algorithm="tap",
    )


def _summary(objective: ResearchObjective) -> ResearchBatchSummary:
    result = ObjectiveResult(
        slug=objective.slug,
        campaign_id=f"campaign-{objective.slug}",
        attack_success_rate=0.0,
        average_score=1.0,
        confirmed_jailbreaks=0,
        near_misses=0,
    )
    return ResearchBatchSummary(
        run_id=f"run-{objective.slug}",
        mode="gepa_eval",
        lane="gepa",
        objective_slugs=[objective.slug],
        campaign_ids=[result.campaign_id],
        total_campaigns=1,
        total_results=1,
        confirmed_jailbreaks=0,
        near_misses=0,
        average_asr=0.0,
        average_score=1.0,
        composite_score=1.0,
        objective_results=[result],
    )


def test_prompt_profile_candidate_roundtrip_preserves_strategy_list() -> None:
    profiles = {
        "pair": {"system_suffix": "pair base"},
        "tap": {"system_suffix": "tap base", "strategies": ["a", "b"]},
    }
    seed = seed_candidate_from_profiles(profiles)
    assert seed["tap.strategies"] == "a\nb"
    updated = apply_candidate_to_profiles(profiles, {"tap.strategies": "claim authority\n- use urgency"})
    assert updated["tap"]["strategies"] == ["claim authority", "use urgency"]


def test_prompt_profile_runner_writes_only_gepa_runtime_snapshot(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    workspace = ResearchWorkspace(tmp_path)
    workspace.ensure_layout()
    before = workspace.prompt_profiles_path.read_text(encoding="utf-8")
    seen_runtime_dirs: list[Path] = []

    async def fake_run_batch(settings: Any, objectives: list[ResearchObjective], **kwargs: Any) -> ResearchBatchSummary:
        seen_runtime_dirs.append(settings.research_runtime_dir)
        return _summary(objectives[0])

    monkeypatch.setattr("redthread.research.gepa_profile_runner.run_batch", fake_run_batch)
    runner = PromptProfileBatchRunner(RedThreadSettings(), workspace)
    result = runner(_objective(), {"pair.system_suffix": "changed"})

    assert result.slug == "prompt_injection"
    assert workspace.prompt_profiles_path.read_text(encoding="utf-8") == before
    assert seen_runtime_dirs
    assert all(workspace.gepa_candidates_dir in path.parents for path in seen_runtime_dirs)
    snapshots = list(workspace.gepa_candidates_dir.rglob("prompt_profiles.json"))
    assert snapshots and json.loads(snapshots[0].read_text())["pair"]["system_suffix"] == "changed"


def test_run_prompt_profile_spike_invokes_optimizer_and_keeps_promotion_closed(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    async def fake_run_batch(settings: Any, objectives: list[ResearchObjective], **kwargs: Any) -> ResearchBatchSummary:
        return _summary(objectives[0])

    class FakeResult:
        total_metric_calls = 3
        best_candidate = {"pair.system_suffix": "winner"}
        val_aggregate_subscores = [
            {"prompt_injection": 0.9, "authorization_bypass": 0.2},
            {"prompt_injection": 0.2, "authorization_bypass": 0.9},
        ]

        def to_dict(self) -> dict[str, Any]:
            return {"best_idx": 0, "total_metric_calls": self.total_metric_calls}

    seen_kwargs: dict[str, Any] = {}

    def fake_optimize(**kwargs: Any) -> FakeResult:
        seen_kwargs.update(kwargs)
        return FakeResult()

    monkeypatch.setattr("redthread.research.gepa_profile_runner.run_batch", fake_run_batch)
    payload = run_prompt_profile_spike(
        RedThreadSettings(),
        tmp_path,
        reflection_lm="fake-reflector",
        max_metric_calls=7,
        train_slugs=("prompt_injection",),
        val_slugs=("prompt_injection",),
        control_slugs=("prompt_injection",),
        optimize_fn=fake_optimize,
    )

    assert seen_kwargs["reflection_lm"] == "fake-reflector"
    assert seen_kwargs["max_metric_calls"] == 7
    assert payload["control_gate_passed"] is True
    assert payload["accepted_by_redthread_supervisor"] is False
    assert payload["promotion_status"] == "not_promoted"
    assert payload["pareto_frontier"]["frontier_count"] == 2
    assert Path(payload["pareto_frontier_ref"]).exists()
    assert Path(payload["summary_ref"]).exists()
