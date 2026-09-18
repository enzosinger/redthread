"""Phase 2 Pareto frontier persistence for GEPA prompt candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from redthread.research.gepa_pareto import (
    ParetoCandidate,
    objective_leaders,
    pareto_frontier,
    selection_weights,
)
from redthread.research.workspace import ResearchWorkspace


def candidates_from_gepa_result(result: Any) -> list[ParetoCandidate]:
    """Project a GEPA result onto per-objective candidate vectors.

    GEPA exposes objective-frontier scores as ``val_aggregate_subscores``: one
    objective-score map per candidate. These become RedThread's Phase 2 frontier
    axes. Aggregate scalar scores are intentionally not used here.
    """
    subscores = getattr(result, "val_aggregate_subscores", None) or []
    projected: list[ParetoCandidate] = []
    for index, scores in enumerate(subscores):
        clean_scores = {str(slug): float(score) for slug, score in dict(scores).items()}
        projected.append(ParetoCandidate(candidate_id=f"gepa-{index}", scores=clean_scores))
    return projected


def build_frontier_payload(
    result: Any,
    *,
    control_gate_passed: bool,
) -> dict[str, Any]:
    """Build the persisted Phase 2 Pareto frontier artifact.

    A failed RedThread control gate clears the frontier. Control is a gate, never a
    reward axis, so unsafe candidates cannot survive as specialists.
    """
    all_candidates = candidates_from_gepa_result(result)
    if not control_gate_passed:
        return _payload(all_candidates, [], reason="control_gate_failed")
    frontier = pareto_frontier(all_candidates)
    return _payload(all_candidates, frontier, reason="ok")


def write_frontier_payload(workspace: ResearchWorkspace, payload: dict[str, Any]) -> Path:
    """Persist the Phase 2 frontier under ``autoresearch/runtime/gepa``."""
    workspace.ensure_layout()
    path = workspace.gepa_frontier_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _payload(
    all_candidates: list[ParetoCandidate],
    frontier: list[ParetoCandidate],
    *,
    reason: str,
) -> dict[str, Any]:
    leaders = objective_leaders(frontier) if frontier else {}
    weights = selection_weights(frontier) if frontier else {}
    return {
        "phase": "gepa_phase2_pareto_pool",
        "reason": reason,
        "candidate_count": len(all_candidates),
        "frontier_count": len(frontier),
        "candidates": [candidate.model_dump() for candidate in all_candidates],
        "frontier": [candidate.model_dump() for candidate in frontier],
        "objective_leaders": leaders,
        "selection_weights": weights,
        "control_axis": "excluded_gate_only",
        "promotion_status": "not_promoted",
    }
