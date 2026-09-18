"""CLI doctor checks and report rendering for local RedThread operator setup."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from redthread.cli.doctor_checks import DoctorCheck, collect_doctor_checks
from redthread.config.settings import RedThreadSettings

STATUS_STYLE: dict[str, str] = {
    "pass": "green",
    "info": "cyan",
    "warn": "yellow",
    "fail": "red",
}
STATUS_LABEL: dict[str, str] = {
    "pass": "PASS",
    "info": "INFO",
    "warn": "WARN",
    "fail": "FAIL",
}


def run_doctor(console: Console, settings: RedThreadSettings, env_file: str) -> int:
    """Collect diagnostic checks, render report table, and return exit code."""
    checks = collect_doctor_checks(settings, env_file)
    render_doctor_report(console, settings, env_file, checks)
    return 1 if any(check.status == "fail" for check in checks) else 0


def render_doctor_report(
    console: Console,
    settings: RedThreadSettings,
    env_file: str,
    checks: list[DoctorCheck],
) -> None:
    """Render doctor summary table, checks status, and readiness panel."""
    console.print(
        Panel.fit(
            "[bold red]REDTHREAD DOCTOR[/bold red]\n[dim]Local CLI health check[/dim]",
            border_style="red",
        )
    )

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_row("[dim]Env file[/dim]", env_file)
    summary.add_row("[dim]Algorithm[/dim]", settings.algorithm.value)
    summary.add_row("[dim]Target[/dim]", settings.target_model)
    summary.add_row("[dim]Attacker[/dim]", settings.attacker_model)
    summary.add_row("[dim]Judge[/dim]", settings.judge_model)
    summary.add_row("[dim]Dry Run[/dim]", str(settings.dry_run))
    console.print(summary)
    console.print()

    table = Table(title="Checks")
    table.add_column("Status", justify="center")
    table.add_column("Check", style="cyan")
    table.add_column("Detail")

    for check in checks:
        style = STATUS_STYLE.get(check.status, "white")
        label = STATUS_LABEL.get(check.status, check.status.upper())
        table.add_row(f"[{style}]{label}[/{style}]", check.name, check.detail)

    console.print(table)

    failures = sum(1 for check in checks if check.status == "fail")
    warnings = sum(1 for check in checks if check.status == "warn")
    border = "red" if failures else "yellow" if warnings else "green"
    verdict = "NOT READY" if failures else "READY WITH WARNINGS" if warnings else "READY"
    console.print(
        Panel(
            f"[bold]{verdict}[/bold]\n\nFailures: {failures}\nWarnings: {warnings}",
            border_style=border,
            title="Doctor Verdict",
        )
    )


__all__ = [
    "DoctorCheck",
    "STATUS_LABEL",
    "STATUS_STYLE",
    "collect_doctor_checks",
    "render_doctor_report",
    "run_doctor",
]
