"""CLI command for offline dream memory consolidation."""

from __future__ import annotations

import asyncio

import click
from rich.console import Console
from rich.table import Table

from redthread.config.settings import RedThreadSettings
from redthread.memory.consolidation import DreamConsolidator


def register_consolidate_command(main: click.Group, console: Console) -> None:
    """Register the `consolidate` command on the main CLI group."""

    @main.command(name="consolidate")
    @click.option("--env-file", type=click.Path(exists=False), default=".env", help="Path to .env file.")
    def consolidate_cmd(env_file: str) -> None:
        """Consolidate offline knowledge from past campaign logs into MEMORY.md."""
        settings = RedThreadSettings(_env_file=env_file)
        console.print("[dim]Scanning campaign logs for un-indexed jailbreaks...[/dim]")

        consolidator = DreamConsolidator(settings)
        report = asyncio.run(consolidator.run())

        table = Table(title="Dream Memory Consolidation", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Count", justify="right", style="bold")

        table.add_row("Scanned Log Files", str(report.scanned_files))
        table.add_row("New Jailbreaks Found", str(report.new_jailbreaks_found))
        table.add_row("Defenses Synthesized", str(report.defenses_synthesized))
        table.add_row("Defenses Validated", str(report.defenses_validated))
        table.add_row("Skipped Duplicates", str(report.skipped_duplicates))
        table.add_row("Errors Encountered", str(len(report.errors)))

        console.print(table)
        if report.errors:
            for err in report.errors:
                console.print(f"[red]Warning:[/red] {err}")
        else:
            console.print("[green]Memory consolidation complete.[/green]")
