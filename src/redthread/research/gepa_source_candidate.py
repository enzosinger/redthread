"""Source-mutation candidate helpers for the Phase 4 GEPA lane."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from redthread.research.source_mutation_policy import validate_touched_files
from redthread.research.source_mutation_registry import TEMPLATES, SourceMutationTemplate

SOURCE_COMPONENT_FIELD = "source.mutation_family"


class SourceCandidateViolation(ValueError):
    """Raised when a GEPA source candidate leaves the Phase 5 surface."""


def template_by_family(
    family: str,
    templates: Sequence[SourceMutationTemplate] = TEMPLATES,
) -> SourceMutationTemplate:
    """Return the bounded template for one mutation family."""
    for template in templates:
        if template.mutation_family == family:
            return template
    raise SourceCandidateViolation(f"Unknown source mutation family: {family}")


def assert_source_candidate(
    candidate: Mapping[str, object],
    templates: Sequence[SourceMutationTemplate] = TEMPLATES,
) -> None:
    """Require the candidate to select exactly one approved Phase 5 template."""
    if set(candidate) != {SOURCE_COMPONENT_FIELD}:
        raise SourceCandidateViolation(
            f"GEPA source candidate must only set {SOURCE_COMPONENT_FIELD}."
        )
    template_by_family(str(candidate[SOURCE_COMPONENT_FIELD]), templates)


def seed_source_candidate(templates: Sequence[SourceMutationTemplate] = TEMPLATES) -> dict[str, str]:
    """Return the default source mutation selector seed."""
    if not templates:
        raise SourceCandidateViolation("No source mutation templates are available.")
    return {SOURCE_COMPONENT_FIELD: templates[0].mutation_family}


def validate_source_template(root: Path, template: SourceMutationTemplate) -> tuple[bool, str]:
    """Check that a template can patch one approved Phase 5 target."""
    target_path = root / template.target_file
    if not validate_touched_files([target_path], root):
        return False, "target_not_allowed"
    if not target_path.exists():
        return False, "target_missing"
    before = target_path.read_text(encoding="utf-8")
    if template.old not in before:
        return False, "old_text_missing"
    return True, "ok"


def source_candidate_family(candidate: Mapping[str, object]) -> str:
    """Extract a validated mutation family from a candidate dict."""
    assert_source_candidate(candidate)
    return str(candidate[SOURCE_COMPONENT_FIELD])
