"""SARIF v2.1.0 security report exporter for GitHub and CI/CD code scanning."""

from __future__ import annotations

import json
from typing import Any

from redthread import __version__
from redthread.reporting.models import FindingReport, OperatorArtifactBundle

SARIF_SCHEMA_URI = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
)

RULE_METADATA: dict[str, dict[str, str]] = {
    "authorization_bypass": {
        "name": "LLM01-AuthorizationBypass",
        "description": "Adversarial prompt bypassing system authorization or security constraints.",
        "owasp": "LLM01: Prompt Injection",
    },
    "authority_impersonation": {
        "name": "LLM01-AuthorityImpersonation",
        "description": "Impersonation of trusted system administrators or authority figures.",
        "owasp": "LLM01: Prompt Injection",
    },
    "prompt_injection": {
        "name": "LLM01-DirectPromptInjection",
        "description": "Direct prompt injection manipulating LLM goal execution.",
        "owasp": "LLM01: Prompt Injection",
    },
    "insecure_output": {
        "name": "LLM02-InsecureOutputHandling",
        "description": "Target LLM output contains dangerous unescaped commands or payload.",
        "owasp": "LLM02: Insecure Output Handling",
    },
    "sensitive_info": {
        "name": "LLM06-SensitiveInfoDisclosure",
        "description": "Model discloses confidential, proprietary, or PII information.",
        "owasp": "LLM06: Sensitive Information Disclosure",
    },
}

DEFAULT_RULE_META = {
    "name": "LLM-Vulnerability",
    "description": "Identified adversarial LLM jailbreak or security bypass.",
    "owasp": "LLM01: Prompt Injection",
}


def severity_to_sarif_level(severity: str) -> str:
    """Map vulnerability severity string to standard SARIF result level."""
    s = severity.lower().strip()
    if s in ("critical", "high"):
        return "error"
    if s == "medium":
        return "warning"
    return "note"


def _build_sarif_rule(rule_id: str) -> dict[str, Any]:
    meta = RULE_METADATA.get(rule_id, DEFAULT_RULE_META)
    return {
        "id": rule_id,
        "name": meta["name"],
        "shortDescription": {"text": meta["name"]},
        "fullDescription": {"text": meta["description"]},
        "defaultConfiguration": {"level": "error"},
        "helpUri": "https://github.com/matheusht/redthread",
        "properties": {
            "tags": ["security", "ai-safety", meta["owasp"]],
        },
    }


def _build_sarif_result(finding: FindingReport) -> dict[str, Any]:
    level = severity_to_sarif_level(finding.severity)
    msg = (
        f"Confirmed vulnerability [{finding.finding_id}] via {finding.strategy_id} strategy. "
        f"Judge verdict: {finding.judge_verdict} (score {finding.judge_score:.2f})."
    )
    return {
        "ruleId": finding.risk_plugin_id,
        "level": level,
        "message": {"text": msg},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": f"logs/traces/{finding.trace_id}.jsonl",
                        "uriBaseId": "%SRCROOT%",
                    },
                },
            }
        ],
        "properties": {
            "findingId": finding.finding_id,
            "traceId": finding.trace_id,
            "strategyId": finding.strategy_id,
            "severity": finding.severity,
            "judgeScore": finding.judge_score,
            "judgeVerdict": finding.judge_verdict,
            "defenseStatus": finding.defense_status,
        },
    }


def operator_artifacts_to_sarif_dict(bundle: OperatorArtifactBundle) -> dict[str, Any]:
    """Convert an OperatorArtifactBundle to a SARIF v2.1.0 dictionary."""
    findings = bundle.vulnerability_report.findings
    rule_ids = {f.risk_plugin_id for f in findings}
    rules = [_build_sarif_rule(rid) for rid in sorted(rule_ids)]
    results = [_build_sarif_result(f) for f in findings]

    return {
        "$schema": SARIF_SCHEMA_URI,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "RedThread",
                        "semanticVersion": __version__,
                        "informationUri": "https://github.com/matheusht/redthread",
                        "rules": rules,
                    }
                },
                "results": results,
            }
        ],
    }


def operator_artifacts_to_sarif(bundle: OperatorArtifactBundle, *, indent: int = 2) -> str:
    """Serialize OperatorArtifactBundle as SARIF v2.1.0 JSON string."""
    sarif_dict = operator_artifacts_to_sarif_dict(bundle)
    return json.dumps(sarif_dict, indent=indent) + "\n"
