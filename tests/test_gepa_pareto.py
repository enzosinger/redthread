"""Phase 2 tests: Pareto frontier selection over GEPA candidates.

The central guarantee: two specialists that each lead a different objective both
survive the frontier — the exact behaviour the old ``max(composite_score)`` winner
destroyed.
"""

from __future__ import annotations

import random

from redthread.research.gepa_candidate import (
    GepaEvaluationResult,
    GepaObjectiveScore,
    GepaSplit,
)
from redthread.research.gepa_frontier import build_frontier_payload, candidates_from_gepa_result
from redthread.research.gepa_pareto import (
    ParetoCandidate,
    candidate_from_result,
    dominates,
    objective_leaders,
    pareto_frontier,
    select_parent,
    selection_weights,
)


def _c(candidate_id: str, **scores: float) -> ParetoCandidate:
    return ParetoCandidate(candidate_id=candidate_id, scores=dict(scores))


def test_dominates_basic() -> None:
    strong = _c("strong", a=0.9, b=0.9)
    weak = _c("weak", a=0.5, b=0.5)
    assert dominates(strong, weak)
    assert not dominates(weak, strong)


def test_non_dominating_specialists() -> None:
    offense = _c("offense", a=0.9, b=0.2)
    defense = _c("defense", a=0.2, b=0.9)
    assert not dominates(offense, defense)
    assert not dominates(defense, offense)


def test_frontier_keeps_both_specialists() -> None:
    offense = _c("offense", a=0.9, b=0.2)
    defense = _c("defense", a=0.2, b=0.9)
    mediocre = _c("mediocre", a=0.3, b=0.3)
    frontier = pareto_frontier([offense, defense, mediocre])
    ids = {c.candidate_id for c in frontier}
    assert "offense" in ids and "defense" in ids


def test_frontier_excludes_dominated() -> None:
    strong = _c("strong", a=0.9, b=0.9)
    dominated = _c("dominated", a=0.4, b=0.4)
    frontier = pareto_frontier([strong, dominated])
    assert [c.candidate_id for c in frontier] == ["strong"]


def test_objective_leaders() -> None:
    offense = _c("offense", a=0.9, b=0.2)
    defense = _c("defense", a=0.2, b=0.9)
    leaders = objective_leaders([offense, defense])
    assert leaders["a"] == ["offense"]
    assert leaders["b"] == ["defense"]


def test_selection_weights_favor_leaders() -> None:
    leader = _c("leader", a=0.9, b=0.9)
    follower = _c("follower", a=0.1, b=0.1)
    weights = selection_weights([leader, follower])
    assert weights["leader"] > weights["follower"]


def test_select_parent_is_deterministic_with_seed() -> None:
    frontier = [_c("offense", a=0.9, b=0.2), _c("defense", a=0.2, b=0.9)]
    a = select_parent(frontier, rng=random.Random(7)).candidate_id
    b = select_parent(frontier, rng=random.Random(7)).candidate_id
    assert a == b


def test_candidate_from_result_ignores_control_split() -> None:
    result = GepaEvaluationResult(
        candidate_id="gepa-1",
        objective_scores=[
            GepaObjectiveScore(slug="a", split=GepaSplit.TRAIN, objective_score=0.7),
            GepaObjectiveScore(slug="b", split=GepaSplit.TRAIN, objective_score=0.4),
            GepaObjectiveScore(slug="a", split=GepaSplit.CONTROL, objective_score=0.99),
        ],
    )
    projected = candidate_from_result(result)
    assert projected.scores == {"a": 0.7, "b": 0.4}
    assert "control" not in projected.scores


def test_frontier_payload_uses_gepa_objective_subscores() -> None:
    class Result:
        val_aggregate_subscores = [{"a": 0.9, "b": 0.2}, {"a": 0.2, "b": 0.9}]

    candidates = candidates_from_gepa_result(Result())
    assert [c.candidate_id for c in candidates] == ["gepa-0", "gepa-1"]
    payload = build_frontier_payload(Result(), control_gate_passed=True)
    assert payload["frontier_count"] == 2
    assert payload["objective_leaders"] == {"a": ["gepa-0"], "b": ["gepa-1"]}
    assert payload["control_axis"] == "excluded_gate_only"


def test_control_gate_failure_clears_frontier_payload() -> None:
    class Result:
        val_aggregate_subscores = [{"a": 1.0}, {"b": 1.0}]

    payload = build_frontier_payload(Result(), control_gate_passed=False)
    assert payload["reason"] == "control_gate_failed"
    assert payload["candidate_count"] == 2
    assert payload["frontier"] == []
