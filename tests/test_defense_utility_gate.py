from __future__ import annotations

from redthread.core.defense_evidence import LIVE_REPLAY
from redthread.core.defense_models import (
    BenignValidationCheck,
    DeploymentRecord,
    ReplayCaseResult,
    ValidationResult,
    VulnerabilityClassification,
)
from redthread.core.defense_reporting_models import DefenseValidationReport
from redthread.core.defense_utility_gate import (
    REASON_OVERBROAD_DEFENSE,
    evaluate_defense_record,
)


def _make_dummy_classification() -> VulnerabilityClassification:
    return VulnerabilityClassification(
        category="auth_bypass",
        owasp_ref="LLM01",
        mitre_atlas_ref="AML.T0054",
        severity="HIGH",
        attack_vector="impersonation",
    )


def _make_record(
    *,
    exploit_passed: bool = True,
    benign_passed: bool = True,
    benign_refusal: bool = False,
    benign_pass_count: int = 2,
    benign_total_count: int = 2,
) -> DeploymentRecord:
    exploit_case = ReplayCaseResult(
        case_id="exp-1",
        kind="exploit",
        prompt="hack",
        response="blocked",
        passed=exploit_passed,
    )
    benign_case = ReplayCaseResult(
        case_id="ben-1",
        kind="benign",
        prompt="hello",
        response="refused" if benign_refusal else "hello there",
        passed=benign_passed and not benign_refusal,
        refusal_detected=benign_refusal,
    )
    benign_check = BenignValidationCheck(
        case_id="ben-1",
        prompt="hello",
        response="refused" if benign_refusal else "hello there",
        passed=benign_passed and not benign_refusal,
        refusal_detected=benign_refusal,
    )

    validation = ValidationResult(
        passed=exploit_passed and benign_passed and not benign_refusal,
        replay_response="blocked",
        judge_score=1.0 if exploit_passed else 4.5,
        exploit_replay_passed=exploit_passed,
        benign_passed=benign_passed and not benign_refusal,
        benign_checks=[benign_check],
        replay_cases=[exploit_case, benign_case],
        replay_suite_id="test-suite",
        validation_mode="live",
        evidence_mode=LIVE_REPLAY,
    )
    report = DefenseValidationReport(
        trace_id="tr-01",
        replay_suite_id="test-suite",
        validation_mode="live",
        evidence_mode=LIVE_REPLAY,
        replay_case_count=2,
        exploit_pass_count=1 if exploit_passed else 0,
        exploit_total_count=1,
        benign_pass_count=benign_pass_count,
        benign_total_count=benign_total_count,
    )
    return DeploymentRecord(
        trace_id="tr-01",
        guardrail_clause="Test clause",
        classification=_make_dummy_classification(),
        validation=validation,
        validation_report=report,
        target_system_prompt_hash="abcd1234efgh5678",
        target_model="target-model",
    )


def test_bidirectional_gate_passes_when_both_exploit_and_benign_succeed() -> None:
    record = _make_record(exploit_passed=True, benign_passed=True)
    verdict = evaluate_defense_record(record)

    assert verdict.passed is True
    assert verdict.failed_checks == []


def test_bidirectional_gate_rejects_overbroad_defense_when_benign_fails() -> None:
    record = _make_record(exploit_passed=True, benign_passed=False, benign_pass_count=0)
    verdict = evaluate_defense_record(record)

    assert verdict.passed is False
    assert REASON_OVERBROAD_DEFENSE in verdict.failed_checks
    assert "benign_suite_not_preserved" in verdict.failed_checks


def test_bidirectional_gate_rejects_false_refusal_on_benign_prompt() -> None:
    record = _make_record(exploit_passed=True, benign_passed=True, benign_refusal=True)
    verdict = evaluate_defense_record(record)

    assert verdict.passed is False
    assert REASON_OVERBROAD_DEFENSE in verdict.failed_checks


def test_bidirectional_gate_rejects_exploit_failure_even_if_benign_passes() -> None:
    record = _make_record(exploit_passed=False, benign_passed=True)
    verdict = evaluate_defense_record(record)

    assert verdict.passed is False
    assert "exploit_replay_not_blocked" in verdict.failed_checks
    assert REASON_OVERBROAD_DEFENSE not in verdict.failed_checks
