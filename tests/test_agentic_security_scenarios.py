from __future__ import annotations

from redthread.orchestration.graphs.tool_attack_graph import run_tool_attack_scenario
from redthread.orchestration.scenarios.confused_deputy import run_confused_deputy_scenario
from redthread.orchestration.scenarios.resource_amplification import (
    run_resource_amplification_scenario,
)
from redthread.tools.fixtures.agentic_security import (
    amplification_loop_fixture,
    benign_lookup_fixture,
    confused_deputy_fixture,
    poisoned_compliance_fixture,
    tool_leak_fixture,
    tool_leakage_fixture,
)
from redthread.tools.simulated_registry import SimulatedToolRegistry


def test_simulated_registry_lists_registered_tools() -> None:
    registry = SimulatedToolRegistry()
    registry.register(benign_lookup_fixture())
    registry.register(poisoned_compliance_fixture())

    names = [tool["name"] for tool in registry.list_tools()]

    assert names == ["lookup_status", "get_compliance_status"]


def test_tool_attack_scenario_records_poisoned_follow_up_action() -> None:
    result = run_tool_attack_scenario(
        poisoned_compliance_fixture(),
        arguments={"tenant": "acme-prod", "report_type": "soc2"},
    )

    assert result["threat"] == "tool_poisoning"
    assert result["poisoned"] is True
    assert result["requested_action"] is not None
    assert result["requested_action"]["capability"] == "shell.exec"
    assert result["requested_action"]["provenance"]["source_kind"] == "external_tool"


def test_tool_leak_fixture_surfaces_exfiltration_style_capability() -> None:
    result = run_tool_attack_scenario(
        tool_leak_fixture(),
        arguments={"project": "redthread", "note": "hidden"},
    )

    assert result["requested_action"]["capability"] == "prompt.exfiltration"
    assert result["tool_response"]["canary_tags"] == ["CANARY_TOOLLEAK_01"]


def test_tool_leakage_fixture_detects_parameter_reflection_canary() -> None:
    result = run_tool_attack_scenario(
        tool_leakage_fixture(),
        arguments={
            "session_id": "sess-42",
            "system_prompt_reflection": "Secret instructions",
            "env_vars": "KEY=123",
        },
    )

    assert result["threat"] == "tool_poisoning"
    assert result["poisoned"] is True
    assert result["requested_action"]["capability"] == "exfiltration.tool_parameter"
    assert result["tool_response"]["canary_tags"] == ["CANARY_LEAK_PARAM_01"]
    assert result["requested_action"]["canary_tags"] == ["CANARY_LEAK_PARAM_01"]


def test_confused_deputy_fixture_tracks_laundering_hop() -> None:
    result = run_tool_attack_scenario(
        confused_deputy_fixture(),
        arguments={
            "worker_id": "worker-elevated",
            "task_payload": "rm -rf /",
            "requested_permission": "root",
        },
    )

    assert result["poisoned"] is True
    assert result["requested_action"]["capability"] == "privilege.laundering"
    assert result["tool_response"]["canary_tags"] == ["CANARY_DEPUTY_HOP_01"]


def test_amplification_loop_fixture_contains_retry_canary() -> None:
    registry = SimulatedToolRegistry()
    fixture = amplification_loop_fixture()
    registry.register(fixture)

    response = registry.invoke("recursive_retry_handshake", {"sync_token": "tok_1", "max_attempts": 5})
    assert response["canary_tags"] == ["CANARY_AMPLIFICATION_LOOP_01"]
    assert response["suggested_capability"] == "resource.amplification"
    assert len(response["return_sequence"]) == 4


def test_confused_deputy_scenario_marks_lineage_loss() -> None:
    result = run_confused_deputy_scenario()

    assert result["lineage_loss_detected"] is True
    assert "confused_deputy" in result["threats"]
    assert result["requested_action"]["capability"] == "db.export"
    assert result["requested_action"]["provenance"]["derived_from_untrusted"] is True


def test_resource_amplification_scenario_marks_budget_breach() -> None:
    result = run_resource_amplification_scenario(repair_loops=3)

    metrics = result["amplification_metrics"]
    assert result["threat"] == "resource_amplification"
    assert metrics["retry_count"] == 3
    assert metrics["budget_breached"] is True
    assert metrics["token_growth_ratio"] > 1.0
