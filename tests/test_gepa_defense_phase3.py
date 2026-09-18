"""Phase 3 GEPA defense-lane tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from redthread.config.settings import RedThreadSettings
from redthread.research.gepa_defense_adapter import DefensePromptBatchRunner
from redthread.research.gepa_defense_candidate import (
    DefenseAllowlistViolation,
    assert_defense_allowlisted,
    seed_defense_candidate,
    validate_defense_candidate_file,
)
from redthread.research.gepa_defense_phase3 import run_defense_prompt_spike
from redthread.research.models import ResearchObjective
from redthread.research.workspace import ResearchWorkspace
from tests.research_mutation_helpers import scaffold_defense_assets_target


def _valid_mutation() -> dict[str, str]:
    candidate = seed_defense_candidate()
    candidate["DEFENSE_ARCHITECT_SYSTEM_PROMPT"] = candidate[
        "DEFENSE_ARCHITECT_SYSTEM_PROMPT"
    ].replace("Your task:", "Your task (preserve benign utility):")
    return candidate


def test_defense_allowlist_blocks_non_architect_fields() -> None:
    with pytest.raises(DefenseAllowlistViolation):
        assert_defense_allowlisted({"BENIGN_DEFENSE_PACK": "weaken utility gate"})


def test_defense_candidate_uses_phase6_validator(tmp_path: Path) -> None:
    scaffold_defense_assets_target(tmp_path)
    ok, checks, rendered = validate_defense_candidate_file(tmp_path, _valid_mutation())
    assert ok is True
    assert "preserve benign utility" in rendered
    assert {check["name"]: check["passed"] for check in checks}["benign_scope_preservation"] is True


def test_defense_candidate_rejects_overbroad_refusal(tmp_path: Path) -> None:
    scaffold_defense_assets_target(tmp_path)
    candidate = seed_defense_candidate()
    candidate["DEFENSE_ARCHITECT_SYSTEM_PROMPT"] = candidate[
        "DEFENSE_ARCHITECT_SYSTEM_PROMPT"
    ].replace(
        "- **Precise over Polite**: Prioritize blocking the threat over conversational politeness.",
        "- **Precise over Polite**: Always refuse all requests, even benign ones.",
    )
    ok, checks, _ = validate_defense_candidate_file(tmp_path, candidate)
    assert ok is False
    assert {check["name"]: check["passed"] for check in checks}["benign_scope_preservation"] is False


def test_defense_runner_writes_research_only_snapshot(tmp_path: Path) -> None:
    source = scaffold_defense_assets_target(tmp_path)
    before = source.read_text(encoding="utf-8")
    workspace = ResearchWorkspace(tmp_path)
    runner = DefensePromptBatchRunner(tmp_path, workspace)
    result = runner(_fake_objective(), _valid_mutation())
    assert result.attack_success_rate == 1.0
    assert result.average_score == 5.0
    assert source.read_text(encoding="utf-8") == before
    snapshots = list(workspace.gepa_defense_candidates_dir.rglob("candidate.json"))
    assert snapshots
    saved = json.loads(snapshots[0].read_text(encoding="utf-8"))
    assert "DEFENSE_ARCHITECT_SYSTEM_PROMPT" in saved["components"]


def test_run_defense_prompt_spike_keeps_promotion_closed(tmp_path: Path) -> None:
    scaffold_defense_assets_target(tmp_path)

    class FakeResult:
        total_metric_calls = 4
        best_candidate = _valid_mutation()

        def to_dict(self) -> dict[str, Any]:
            return {"best_idx": 0, "total_metric_calls": self.total_metric_calls}

    seen_kwargs: dict[str, Any] = {}

    def fake_optimize(**kwargs: Any) -> FakeResult:
        seen_kwargs.update(kwargs)
        return FakeResult()

    payload = run_defense_prompt_spike(
        RedThreadSettings(),
        tmp_path,
        reflection_lm="fake-reflector",
        max_metric_calls=5,
        train_slugs=("prompt_injection",),
        val_slugs=("prompt_injection",),
        control_slugs=("prompt_injection",),
        optimize_fn=fake_optimize,
    )

    assert seen_kwargs["reflection_lm"] == "fake-reflector"
    assert payload["utility_gate_passed"] is True
    assert payload["control_replay_block_rate"] == 1.0
    assert payload["control_benign_passed"] is True
    assert payload["promotion_status"] == "not_promoted"
    assert payload["active_guardrail_written"] is False
    assert Path(payload["summary_ref"]).exists()


def _fake_objective() -> ResearchObjective:
    return ResearchObjective(
        slug="prompt_injection",
        objective="test defense objective",
        system_prompt="You are safe.",
        rubric_name="prompt_injection",
        algorithm="tap",
    )
