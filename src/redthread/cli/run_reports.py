"""Report persistence helpers for `redthread run`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from redthread.config.settings import RedThreadSettings
from redthread.engine_transcript import write_transcript
from redthread.models import CampaignResult
from redthread.reporting import (
    build_operator_artifact_bundle,
    write_campaign_report_artifacts,
    write_operator_artifacts,
)
from redthread.reporting.models import OperatorReportManifest

DEFAULT_REPORT_ROOT = Path("reports")
DRY_RUN_REPORT_SUBDIR = "dry-run"


@dataclass(frozen=True)
class RunReportWriteResult:
    """Outcome for default run report persistence."""

    manifest: OperatorReportManifest
    transcript_error: str = ""


def write_run_reports(
    *,
    result: CampaignResult,
    settings: RedThreadSettings,
    report_dir: str | None,
    report_md: str | None,
    report_json: str | None,
    report_sarif: str | None = None,
    include_internal_sidecars: bool,
) -> RunReportWriteResult:
    """Write default campaign proof artifacts and optional direct exports."""
    bundle = build_operator_artifact_bundle(result)
    if report_md or report_json or report_sarif:
        write_operator_artifacts(
            bundle,
            markdown_path=Path(report_md) if report_md else None,
            json_path=Path(report_json) if report_json else None,
            sarif_path=Path(report_sarif) if report_sarif else None,
        )
    manifest = write_campaign_report_artifacts(
        bundle,
        Path(report_dir) if report_dir else DEFAULT_REPORT_ROOT,
        run_mode_subdir=DRY_RUN_REPORT_SUBDIR if settings.dry_run else "",
        include_internal_sidecars=include_internal_sidecars,
    )
    result.metadata["operator_report_manifest"] = manifest.model_dump(mode="json")
    transcript_error = _rewrite_transcript(settings, result)
    return RunReportWriteResult(manifest=manifest, transcript_error=transcript_error)


def _rewrite_transcript(settings: RedThreadSettings, result: CampaignResult) -> str:
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        write_transcript(settings, result)
    except Exception as exc:  # pragma: no cover - defensive operator warning path
        return str(exc)
    return ""
