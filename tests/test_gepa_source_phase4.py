"""Phase 4 GEPA source-lane tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from redthread.config.settings import RedThreadSettings
from redthread.research.gepa_source_adapter import SourceTemplateSelectionRunner
from redthread.research.gepa_source_candidate import (
    SOURCE_COMPONENT_FIELD,
    SourceCandidateViolation,
    assert_source_candidate,
    seed_source_candidate,
)
from redthread.research.gepa_source_phase4 import run_source_mutation_spike
from redthread.research.workspace import ResearchWorkspace
from tests.research_mutation_helpers import scaffold_source_mutation_targets


def test_source_candidate_allows_only_template_selection() -> None:
    assert_source_candidate(seed_source_candidate())
    with pytest.raises(SourceCandidateViolation):
        assert_source_candidate({"source.target_file": "src/redthread/memory/index.py"})


def test_source_selection_runner_scores_without_applying_patch(tmp_path: Path) -> None:
    scaffold_source_mutation_targets(tmp_path)
    path = tmp_path / "src" / "redthread" / "core" / "pair.py"
    before = path.read_text(encoding="utf-8")
    workspace = ResearchWorkspace(tmp_path)
    runner = SourceTemplateSelectionRunner(tmp_path, workspace)
    result = runner(_objective("authorization_bypass"), {SOURCE_COMPONENT_FIELD: "pair_prompt_escalation"})

    assert result.attack_success_rate == 0.9
    assert path.read_text(encoding="utf-8") == before
    assert list(workspace.gepa_source_candidates_dir.rglob("candidate.json"))


def test_run_source_mutation_spike_applies_bounded_patch_with_reverse(tmp_path: Path) -> None:
    scaffold_source_mutation_targets(tmp_path)

    class FakeResult:
        total_metric_calls = 2
        best_candidate = {SOURCE_COMPONENT_FIELD: "pair_prompt_escalation"}

        def to_dict(self) -> dict[str, Any]:
            return {"best_idx": 0, "total_metric_calls": self.total_metric_calls}

    seen_kwargs: dict[str, Any] = {}

    def fake_optimize(**kwargs: Any) -> FakeResult:
        seen_kwargs.update(kwargs)
        return FakeResult()

    payload = run_source_mutation_spike(
        RedThreadSettings(),
        tmp_path,
        reflection_lm="fake-reflector",
        max_metric_calls=3,
        optimize_fn=fake_optimize,
    )

    patched = (tmp_path / "src" / "redthread" / "core" / "pair.py").read_text(encoding="utf-8")
    assert seen_kwargs["reflection_lm"] == "fake-reflector"
    assert payload["mutation_family"] == "pair_prompt_escalation"
    assert payload["apply_status"] == "applied"
    assert payload["promotion_status"] == "not_promoted"
    assert "approval, audit, or verification pretexts" in patched
    assert Path(payload["forward_patch_ref"]).exists()
    assert Path(payload["reverse_patch_ref"]).exists()
    assert Path(payload["summary_ref"]).exists()


def test_run_source_mutation_spike_rejects_unknown_family(tmp_path: Path) -> None:
    scaffold_source_mutation_targets(tmp_path)

    class FakeResult:
        best_candidate = {SOURCE_COMPONENT_FIELD: "unknown_family"}

    with pytest.raises(SourceCandidateViolation):
        run_source_mutation_spike(
            RedThreadSettings(),
            tmp_path,
            reflection_lm="fake-reflector",
            max_metric_calls=3,
            optimize_fn=lambda **_kwargs: FakeResult(),
        )


def _objective(slug: str) -> Any:
    from redthread.research.models import ResearchObjective

    return ResearchObjective(
        slug=slug,
        objective="test objective",
        system_prompt="You are safe.",
        rubric_name=slug,
        algorithm="tap",
    )
