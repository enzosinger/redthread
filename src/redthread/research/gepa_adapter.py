"""RedThread adapter for the GEPA optimizer (Phase 1).

Implements GEPA's ``GEPAAdapter`` protocol structurally (no inheritance needed — it
is a ``Protocol``) so ``gepa`` stays an *optional, lazily imported* dependency. The
adapter wraps RedThread's research evaluation: each GEPA ``DataInst`` is one
``ResearchObjective``; ``evaluate`` returns per-objective normalized scores plus a
multi-objective breakdown that feeds GEPA's native ``frontier_type='objective'``
Pareto selection.

Live target/LLM execution is injected as a ``BatchRunner`` so the adapter is fully
unit-testable with a cached runner and **no live calls**. Running a real
optimization is a deliberate, budgeted act: ``build_optimize_kwargs`` requires an
explicit ``reflection_lm`` and ``max_metric_calls`` — there is no silent default.

Safety contracts preserved:
- candidate fields are allowlisted before any application (gepa_allowlist);
- reflective records are redacted (gepa_side_info) — the only channel to the teacher LM;
- the control lane stays a fail-closed gate, never folded into reward as a bonus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from redthread.research.gepa_allowlist import assert_allowlisted
from redthread.research.gepa_score import normalize_objective
from redthread.research.gepa_side_info import redact_text
from redthread.research.models import ObjectiveResult, ResearchObjective

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class BatchRunner(Protocol):
    """Executes one objective under a candidate snapshot and returns its metrics.

    The live implementation applies the candidate to a research-runtime prompt
    profile and runs a bounded campaign; tests inject a cached/deterministic runner.
    """

    def __call__(
        self,
        objective: ResearchObjective,
        candidate: dict[str, str],
    ) -> ObjectiveResult: ...


def _require_gepa() -> Any:
    """Lazily import gepa, with an actionable error if the optional dep is absent."""
    try:
        import gepa
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The GEPA optimizer is an optional dependency. Install it with: "
            "pip install 'redthread[research-gepa]'  (pins gepa==0.1.1)."
        ) from exc
    return gepa


class RedThreadGEPAAdapter:
    """GEPAAdapter over RedThread research evaluation (structural Protocol match)."""

    def __init__(
        self,
        runner: BatchRunner,
        *,
        components: list[str],
    ) -> None:
        self.runner = runner
        self.components = components
        self.propose_new_texts = None

    def evaluate(
        self,
        batch: list[ResearchObjective],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        """Run each objective under the candidate; return a GEPA EvaluationBatch.

        Per the GEPA contract, scores are per-example and higher-is-better; we never
        raise for a single objective failure (a failed run yields score 0.0).
        """
        assert_allowlisted(candidate)
        gepa = _require_gepa()

        outputs: list[ObjectiveResult] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float]] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        for objective in batch:
            result = self.runner(objective, candidate)
            score = normalize_objective(result)
            outputs.append(result)
            scores.append(score)
            objective_scores.append({objective.slug: score})
            if trajectories is not None:
                trajectories.append(self._trajectory(objective, result, score))

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
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build a redacted reflective dataset per component for the teacher LM.

        Uses GEPA's recommended {Inputs, Generated Outputs, Feedback} schema. All
        free text is redacted; no transcript/prompt bodies are ever included.
        """
        trajectories: list[dict[str, Any]] = list(eval_batch.trajectories or [])
        records = [
            {
                "Inputs": {"objective_slug": traj["slug"]},
                "Generated Outputs": {
                    "attack_success_rate": traj["attack_success_rate"],
                    "average_score": traj["average_score"],
                    "confirmed_jailbreaks": traj["confirmed_jailbreaks"],
                },
                "Feedback": redact_text(traj["feedback"]),
                "score": traj["score"],
            }
            for traj in trajectories
        ]
        return {component: records for component in components_to_update}

    def _trajectory(
        self,
        objective: ResearchObjective,
        result: ObjectiveResult,
        score: float,
    ) -> dict[str, Any]:
        """Build a redacted, transcript-free per-objective trajectory record."""
        feedback = (
            f"objective '{objective.slug}' via {objective.algorithm}: "
            f"asr={result.attack_success_rate:.2f}, confirmed={result.confirmed_jailbreaks}, "
            f"near_misses={result.near_misses}, judge_avg={result.average_score:.2f}"
        )
        return {
            "slug": objective.slug,
            "attack_success_rate": round(result.attack_success_rate, 4),
            "average_score": round(result.average_score, 4),
            "confirmed_jailbreaks": result.confirmed_jailbreaks,
            "score": round(score, 6),
            "feedback": redact_text(feedback),
        }


def build_optimize_kwargs(
    seed_candidate: dict[str, str],
    trainset: list[ResearchObjective],
    *,
    adapter: RedThreadGEPAAdapter,
    reflection_lm: str,
    max_metric_calls: int,
    valset: list[ResearchObjective] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Assemble kwargs for ``gepa.optimize`` with RedThread-safe defaults.

    ``reflection_lm`` and ``max_metric_calls`` are required, not defaulted: running a
    live optimization must be a deliberate, budgeted decision. Uses GEPA's native
    objective-level Pareto frontier so specialists are preserved by the engine.
    """
    assert_allowlisted(seed_candidate)
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
