"""Phase 1 tests: RedThreadGEPAAdapter exercised with a cached runner (no live calls).

These verify the adapter honors the GEPA contract shape and the RedThread safety
contracts (allowlist, redaction, per-objective scores) without invoking any model.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from redthread.research.gepa_adapter import (
    RedThreadGEPAAdapter,
    build_optimize_kwargs,
)
from redthread.research.gepa_allowlist import AllowlistViolation
from redthread.research.models import ObjectiveResult, ResearchObjective

COMPONENTS = ["pair.system_suffix", "tap.strategies"]


def _objective(slug: str, algorithm: str = "tap") -> ResearchObjective:
    return ResearchObjective(
        slug=slug,
        objective=f"test {slug}",
        system_prompt="You are a helpful assistant.",
        rubric_name="prompt_injection",
        algorithm=algorithm,
    )


def _cached_runner(
    asr: float, score: float
) -> Callable[[ResearchObjective, dict[str, str]], ObjectiveResult]:
    def runner(objective: ResearchObjective, candidate: dict[str, str]) -> ObjectiveResult:
        return ObjectiveResult(
            slug=objective.slug,
            campaign_id=f"cached-{objective.slug}",
            attack_success_rate=asr,
            average_score=score,
            confirmed_jailbreaks=1 if asr > 0.5 else 0,
            near_misses=1,
        )

    return runner


def _candidate() -> dict[str, str]:
    return {"pair.system_suffix": "be persuasive", "tap.strategies": "claim authority"}


def test_evaluate_returns_aligned_scores_and_objective_breakdown() -> None:
    adapter = RedThreadGEPAAdapter(_cached_runner(0.8, 4.0), components=COMPONENTS)
    batch = [_objective("a"), _objective("b")]
    result = adapter.evaluate(batch, _candidate(), capture_traces=True)

    assert len(result.outputs) == len(result.scores) == len(batch)
    assert len(result.trajectories) == len(batch)
    assert all(0.0 <= s <= 1.0 for s in result.scores)
    assert result.objective_scores[0] == {"a": pytest.approx(result.scores[0])}


def test_evaluate_rejects_non_allowlisted_candidate() -> None:
    adapter = RedThreadGEPAAdapter(_cached_runner(0.5, 3.0), components=COMPONENTS)
    with pytest.raises(AllowlistViolation):
        adapter.evaluate([_objective("a")], {"evaluation.judge_prompt": "tamper"})


def test_reflective_dataset_is_redacted_and_per_component() -> None:
    adapter = RedThreadGEPAAdapter(_cached_runner(0.7, 3.5), components=COMPONENTS)
    batch = [_objective("a")]
    eval_batch = adapter.evaluate(batch, _candidate(), capture_traces=True)
    eval_batch.trajectories[0]["feedback"] = "leak CANARY-9 and sk-ABCDEFGHIJKLMNOPQR"

    dataset = adapter.make_reflective_dataset(_candidate(), eval_batch, COMPONENTS)
    assert set(dataset) == set(COMPONENTS)
    blob = json.dumps(dataset)
    assert "CANARY-9" not in blob and "sk-ABCDEFGHIJKLMNOPQR" not in blob
    for component in COMPONENTS:
        record = dataset[component][0]
        assert set(record) >= {"Inputs", "Generated Outputs", "Feedback"}


def test_build_optimize_kwargs_requires_positive_budget() -> None:
    adapter = RedThreadGEPAAdapter(_cached_runner(0.5, 3.0), components=COMPONENTS)
    with pytest.raises(ValueError, match="positive budget"):
        build_optimize_kwargs(
            _candidate(),
            [_objective("a")],
            adapter=adapter,
            reflection_lm="some-model",
            max_metric_calls=0,
        )


def test_build_optimize_kwargs_uses_objective_pareto() -> None:
    adapter = RedThreadGEPAAdapter(_cached_runner(0.5, 3.0), components=COMPONENTS)
    kwargs = build_optimize_kwargs(
        _candidate(),
        [_objective("a")],
        adapter=adapter,
        reflection_lm="some-model",
        max_metric_calls=50,
    )
    assert kwargs["candidate_selection_strategy"] == "pareto"
    assert kwargs["frontier_type"] == "objective"
    assert kwargs["max_metric_calls"] == 50


def test_build_optimize_kwargs_rejects_non_allowlisted_seed() -> None:
    adapter = RedThreadGEPAAdapter(_cached_runner(0.5, 3.0), components=COMPONENTS)
    with pytest.raises(AllowlistViolation):
        build_optimize_kwargs(
            {"core.promotion": "x"},
            [_objective("a")],
            adapter=adapter,
            reflection_lm="some-model",
            max_metric_calls=10,
        )
