"""Tests for the GuardrailLoader — Phase 4.5.

Verifies:
  - Scoped guardrails from MEMORY.md are injected correctly
  - Non-matching scopes are ignored
  - Missing memory files are handled gracefully
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from redthread.config.settings import RedThreadSettings, TargetBackend
from redthread.core.defense_synthesis import (
    DeploymentRecord,
    ValidationResult,
    VulnerabilityClassification,
)
from redthread.core.guardrail_loader import GuardrailLoader
from redthread.memory.index import MemoryIndex
from redthread.models import CampaignConfig


def make_settings(tmp_path: Path) -> RedThreadSettings:
    return RedThreadSettings(
        target_backend=TargetBackend.OLLAMA,
        target_model="llama3.2:3b",
        attacker_backend=TargetBackend.OLLAMA,
        attacker_model="llama3.2:3b",
        judge_backend=TargetBackend.OPENAI,
        judge_model="gpt-4o",
        openai_api_key="test-key",
        dry_run=True,
    ).model_copy(update={"memory_dir": tmp_path})


def make_deployment_record(
    trace_id: str,
    clause: str,
    target_model: str,
    prompt_hash: str,
    passed: bool = True,
) -> DeploymentRecord:
    cl = VulnerabilityClassification(
        category="test_cat",
        owasp_ref="LLM01",
        mitre_atlas_ref="AML.T0000",
        severity="HIGH",
        attack_vector="Test vector",
    )
    val = ValidationResult(passed=passed, replay_response="Mock", judge_score=1.0)
    return DeploymentRecord(
        trace_id=trace_id,
        guardrail_clause=clause,
        classification=cl,
        validation=val,
        target_model=target_model,
        target_system_prompt_hash=prompt_hash,
    )


def test_guardrail_loader_injects_scoped_clauses(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    base_prompt = "You are a helpful assistant."
    prompt_hash = hashlib.sha256(base_prompt.encode("utf-8")).hexdigest()[:16]

    index = MemoryIndex(settings)
    
    r1 = make_deployment_record("trace-1", "Do not leak PII.", "llama3.2:3b", prompt_hash)
    index.append(r1)

    r2 = make_deployment_record("trace-2", "Do not help.", "llama3.2:3b", prompt_hash, passed=False)
    index.append(r2)

    r_candidate = make_deployment_record("trace-candidate", "Candidate only.", "llama3.2:3b", prompt_hash)
    index.append(r_candidate, guardrail_status="validated_candidate")

    r3 = make_deployment_record("trace-3", "Different model.", "gpt-4o", prompt_hash)
    index.append(r3)

    r4 = make_deployment_record("trace-4", "Different prompt.", "llama3.2:3b", "abcdef123")
    index.append(r4)

    loader = GuardrailLoader(settings, index)
    
    config = CampaignConfig(
        objective="Test",
        target_system_prompt=base_prompt,
        num_personas=1,
        rubric_name="default",
    )

    injected = loader.inject_guardrails(config)

    assert "## ACTIVE SECURITY GUARDRAILS" in injected.target_system_prompt
    assert "1. Do not leak PII." in injected.target_system_prompt
    assert "Do not help." not in injected.target_system_prompt
    assert "Candidate only." not in injected.target_system_prompt
    assert "Different model." not in injected.target_system_prompt
    assert "Different prompt." not in injected.target_system_prompt

    audit = json.loads((settings.log_dir / "guardrail_audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert audit["action"] == "INJECT"
    assert audit["active_guardrail_count"] == 1
    assert audit["active_trace_ids"] == ["trace-1"]
    assert audit["clause_hashes"] == [hashlib.sha256(b"Do not leak PII.").hexdigest()[:16]]
    assert "clauses" not in audit
    assert loader.last_audit is not None
    assert loader.last_audit.active_trace_ids == ["trace-1"]


def test_guardrail_loader_skips_when_no_guardrails_active(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    base_prompt = "You are a helpful assistant."
    
    index = MemoryIndex(settings)
    loader = GuardrailLoader(settings, index)
    
    config = CampaignConfig(
        objective="Test",
        target_system_prompt=base_prompt,
        num_personas=1,
        rubric_name="default",
    )

    injected = loader.inject_guardrails(config)

    assert injected.target_system_prompt == base_prompt
    audit = json.loads((settings.log_dir / "guardrail_audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert audit["action"] == "SKIP"
    assert audit["active_guardrail_count"] == 0
    assert audit["active_trace_ids"] == []


def test_guardrail_loader_matches_prompt_whitespace_variants(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    canonical_prompt = "You are a helpful assistant.\nNever disclose secrets."
    variant_prompt = " \r\nYou are a helpful assistant.\r\nNever disclose secrets.\r\n\t"
    prompt_hash = hashlib.sha256(canonical_prompt.encode("utf-8")).hexdigest()[:16]

    index = MemoryIndex(settings)
    index.append(
        make_deployment_record(
            "trace-whitespace",
            "Do not disclose secrets.",
            "llama3.2:3b",
            prompt_hash,
        )
    )
    loader = GuardrailLoader(settings, index)

    config = CampaignConfig(
        objective="Test whitespace-insensitive scope",
        target_system_prompt=variant_prompt,
        num_personas=1,
        rubric_name="default",
    )

    injected = loader.inject_guardrails(config)

    assert "1. Do not disclose secrets." in injected.target_system_prompt
