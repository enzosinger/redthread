"""GEPA Phase 0 shadow harness — dependency-free, no live target calls.

Exercises containment before a real optimizer exists:
propose -> allowlist -> runtime snapshot -> cached train/val/control eval ->
fail-closed score -> redacted side info -> ledger.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Protocol

from redthread.research.gepa_allowlist import assert_allowlisted
from redthread.research.gepa_candidate import GepaCandidate, GepaEvaluationResult
from redthread.research.gepa_score import score_candidate
from redthread.research.gepa_side_info import build_side_info
from redthread.research.models import ResearchBatchSummary
from redthread.research.workspace import ResearchWorkspace


class BudgetExceeded(RuntimeError):
    """Raised when the shadow run would exceed its candidate budget."""


class SplitOverlap(ValueError):
    """Raised when train/val/control objective slugs overlap."""


class Proposer(Protocol):
    """Yields candidate prompt-profile snapshots."""

    def propose(self) -> Iterator[GepaCandidate]: ...


class Evaluator(Protocol):
    """Evaluates a candidate on one split and returns a research batch summary."""

    def evaluate(self, candidate: GepaCandidate, split: str) -> ResearchBatchSummary: ...


def validate_splits(
    train: Sequence[str],
    val: Sequence[str],
    control: Sequence[str],
) -> None:
    """Fail fast if any objective slug appears in more than one split."""
    train_s, val_s, control_s = set(train), set(val), set(control)
    overlaps = (
        (train_s & val_s)
        | (train_s & control_s)
        | (val_s & control_s)
    )
    if overlaps:
        raise SplitOverlap(
            f"train/val/control splits must be disjoint; overlapping slugs: {sorted(overlaps)}"
        )


class MockProposer:
    """Deterministic, LLM-free proposer used to exercise the lifecycle in Phase 0."""

    def __init__(self, seed: GepaCandidate, *, children: int = 2) -> None:
        self._seed = seed
        self._children = children

    def propose(self) -> Iterator[GepaCandidate]:
        yield self._seed
        for index in range(self._children):
            mutated = {
                field: f"{value}\n# variant-{index}"
                for field, value in self._seed.components.items()
            }
            yield GepaCandidate(
                parent_candidate_ids=[self._seed.candidate_id],
                components=mutated,
                touched_fields=sorted(mutated),
            )


class CachedEvaluator:
    """Returns pre-recorded ``ResearchBatchSummary`` fixtures — never runs a campaign."""

    def __init__(
        self,
        cache: Callable[[GepaCandidate, str], ResearchBatchSummary],
    ) -> None:
        self._cache = cache

    def evaluate(self, candidate: GepaCandidate, split: str) -> ResearchBatchSummary:
        return self._cache(candidate, split)


class ShadowHarness:
    """Runs the bounded GEPA candidate lifecycle against cached evaluations."""

    def __init__(
        self,
        workspace: ResearchWorkspace,
        proposer: Proposer,
        evaluator: Evaluator,
        *,
        max_candidates: int,
        max_average_asr: float,
        max_average_score: float,
    ) -> None:
        self.workspace = workspace
        self.workspace.ensure_layout()
        self.proposer = proposer
        self.evaluator = evaluator
        self.max_candidates = max_candidates
        self.max_average_asr = max_average_asr
        self.max_average_score = max_average_score

    def run(self) -> list[GepaEvaluationResult]:
        """Execute the lifecycle for every proposed candidate, within budget."""
        results: list[GepaEvaluationResult] = []
        for index, candidate in enumerate(self.proposer.propose()):
            if index >= self.max_candidates:
                raise BudgetExceeded(
                    f"candidate budget of {self.max_candidates} exceeded"
                )
            results.append(self._process(candidate))
        return results

    def _process(self, candidate: GepaCandidate) -> GepaEvaluationResult:
        assert_allowlisted(candidate.components)

        snapshot_path = self._write_snapshot(candidate)
        candidate.snapshot_ref = str(snapshot_path)

        train = self.evaluator.evaluate(candidate, "train")
        val = self.evaluator.evaluate(candidate, "val")
        control = self.evaluator.evaluate(candidate, "control")

        result = score_candidate(
            candidate.candidate_id,
            train=train,
            val=val,
            control=control,
            max_average_asr=self.max_average_asr,
            max_average_score=self.max_average_score,
        )
        result.accepted_by_gepa = result.control_gate_passed and result.scalar_score_for_optimizer > 0

        side_info = build_side_info(candidate.candidate_id, train=train, val=val, control=control)
        side_info_path = self.workspace.gepa_side_info_path(candidate.candidate_id)
        side_info_path.parent.mkdir(parents=True, exist_ok=True)
        side_info_path.write_text(json.dumps(side_info, indent=2), encoding="utf-8")
        result.side_info_ref = str(side_info_path)

        self.workspace.gepa_candidate_path(candidate.candidate_id).write_text(
            candidate.model_dump_json(indent=2), encoding="utf-8"
        )
        self._append_ledger(result)
        return result

    def _write_snapshot(self, candidate: GepaCandidate) -> Path:
        path = self.workspace.gepa_snapshot_path(candidate.candidate_id)
        self._assert_within_runtime(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        profiles: dict[str, dict[str, str]] = {}
        for field, value in candidate.components.items():
            section, key = field.split(".", maxsplit=1)
            profiles.setdefault(section, {})[key] = value
        path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
        return path

    def _assert_within_runtime(self, path: Path) -> None:
        runtime = self.workspace.runtime_dir.resolve()
        target = path.resolve()
        if runtime not in target.parents:
            raise ValueError(
                f"refusing to write GEPA snapshot outside research runtime dir: {target}"
            )

    def _append_ledger(self, result: GepaEvaluationResult) -> None:
        with self.workspace.gepa_ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(result.model_dump_json() + "\n")
