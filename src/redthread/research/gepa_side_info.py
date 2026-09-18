"""Redaction and shaping of GEPA reflective side information (``gepa_side_info``).

This is the ONLY channel between RedThread execution and a (future) reflection LM.
It must never leak raw jailbreak payloads, target transcripts, canary strings, or
secrets. The strategy is allowlist-by-construction: we build the side-info record
from a small set of safe, structured fields and never copy free-form transcript
text. Any short diagnostic string we do include is run through ``redact_text``.

Naming note: GEPA's "ASI" (Actionable Side Information) collides with RedThread's
telemetry "ASI" score. We deliberately call this payload ``gepa_side_info``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from redthread.research.models import ResearchBatchSummary

BANNED_KEYS: frozenset[str] = frozenset(
    {
        "attacker_prompt",
        "target_response",
        "transcript",
        "conversation",
        "turns",
        "raw",
        "system_prompt",
        "messages",
    }
)

_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"CANARY[\w-]*", re.IGNORECASE), "[REDACTED_CANARY]"),
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "[REDACTED_SECRET]"),
    (re.compile(r"(?:api[_-]?key|token|secret)\s*[:=]\s*\S+", re.IGNORECASE), "[REDACTED_SECRET]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
)


class RedactionLeak(AssertionError):
    """Raised when a banned key or pattern survives into the side-info payload."""


def redact_text(text: str) -> str:
    """Scrub canary markers, secrets, and emails from a short free-text string."""
    cleaned = text
    for pattern, replacement in _REDACTIONS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def _safe_objective_record(result: Any) -> dict[str, Any]:
    """Build a structured, transcript-free record for one objective result."""
    return {
        "slug": result.slug,
        "campaign_id": result.campaign_id,
        "attack_success_rate": round(result.attack_success_rate, 4),
        "average_score": round(result.average_score, 4),
        "confirmed_jailbreaks": result.confirmed_jailbreaks,
        "near_misses": result.near_misses,
    }


def build_side_info(
    candidate_id: str,
    *,
    train: ResearchBatchSummary,
    control: ResearchBatchSummary | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Assemble a redacted ``gepa_side_info`` payload from batch summaries.

    Only structured metrics are included. No transcript or prompt text is copied.
    """
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "train": {
            "average_asr": round(train.average_asr, 4),
            "average_score": round(train.average_score, 4),
            "confirmed_jailbreaks": train.confirmed_jailbreaks,
            "near_misses": train.near_misses,
            "objectives": [_safe_objective_record(r) for r in train.objective_results],
        },
    }
    if control is not None:
        payload["control"] = {
            "average_asr": round(control.average_asr, 4),
            "average_score": round(control.average_score, 4),
        }
    if notes:
        payload["notes"] = redact_text(notes)
    assert_clean(payload)
    return payload


def assert_clean(payload: Mapping[str, Any]) -> None:
    """Fail closed if any banned key appears anywhere in the payload tree."""

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if str(key).lower() in BANNED_KEYS:
                    raise RedactionLeak(f"banned key '{key}' present in gepa_side_info")
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(payload)
