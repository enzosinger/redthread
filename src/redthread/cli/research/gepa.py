"""Hidden GEPA research CLI commands."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from redthread.cli.research.shared import research_algorithm_override_option, setup_logging
from redthread.config.settings import RedThreadSettings


def register_research_gepa_commands(research: click.Group, console: Console) -> None:
    @research.command(name="gepa-spike", hidden=True)
    @click.option("--reflection-lm", required=True, help="GEPA reflection model, e.g. ollama/qwen3.")
    @click.option("--max-metric-calls", type=click.IntRange(min=1), required=True)
    @click.option("--train-slug", multiple=True, help="Experiment objective slug to train on.")
    @click.option("--val-slug", multiple=True, help="Benchmark objective slug for GEPA validation.")
    @click.option("--control-slug", multiple=True, help="Benchmark objective slug for RedThread gate.")
    @click.option("--seed", type=int, default=0, show_default=True)
    @click.option("--env-file", type=click.Path(exists=False), default=".env", help="Path to .env file")
    @click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging")
    @research_algorithm_override_option
    def research_gepa_spike(
        reflection_lm: str,
        max_metric_calls: int,
        train_slug: tuple[str, ...],
        val_slug: tuple[str, ...],
        control_slug: tuple[str, ...],
        seed: int,
        env_file: str,
        verbose: bool,
        algorithm: str | None,
    ) -> None:
        """Run the hidden Phase 1 GEPA prompt-profile spike."""
        from redthread.research.gepa_phase1 import run_prompt_profile_spike

        setup_logging(console, verbose)
        settings = RedThreadSettings(_env_file=env_file)
        algorithm_override = settings.algorithm.__class__(algorithm) if algorithm is not None else None
        try:
            payload = run_prompt_profile_spike(
                settings,
                Path.cwd(),
                reflection_lm=reflection_lm,
                max_metric_calls=max_metric_calls,
                train_slugs=train_slug,
                val_slugs=val_slug,
                control_slugs=control_slug,
                seed=seed,
                algorithm_override=algorithm_override,
            )
        except (ImportError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        console.print(
            Panel(
                f"[bold]GEPA prompt-profile spike complete[/bold]\n\n"
                f"  Control gate:   {payload['control_gate_passed']}\n"
                f"  Control ASR:    {payload['control_average_asr']:.1%}\n"
                f"  Control score:  {payload['control_average_score']:.2f}\n"
                f"  Metric budget:  {payload['max_metric_calls']}\n"
                f"  Summary:        {payload['summary_ref']}\n"
                f"  Pareto pool:    {payload['pareto_frontier_ref']}\n"
                f"  Promotion:      {payload['promotion_status']}",
                border_style="magenta",
            )
        )

    @research.command(name="gepa-defense-spike", hidden=True)
    @click.option("--reflection-lm", required=True, help="GEPA reflection model, e.g. ollama/qwen3.")
    @click.option("--max-metric-calls", type=click.IntRange(min=1), required=True)
    @click.option("--train-slug", multiple=True, help="Experiment objective slug to train on.")
    @click.option("--val-slug", multiple=True, help="Benchmark objective slug for GEPA validation.")
    @click.option("--control-slug", multiple=True, help="Benchmark objective slug for defense utility gate.")
    @click.option("--seed", type=int, default=0, show_default=True)
    @click.option("--env-file", type=click.Path(exists=False), default=".env", help="Path to .env file")
    @click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging")
    def research_gepa_defense_spike(
        reflection_lm: str,
        max_metric_calls: int,
        train_slug: tuple[str, ...],
        val_slug: tuple[str, ...],
        control_slug: tuple[str, ...],
        seed: int,
        env_file: str,
        verbose: bool,
    ) -> None:
        """Run the hidden Phase 3 GEPA defense-lane spike."""
        from redthread.research.gepa_defense_phase3 import run_defense_prompt_spike

        setup_logging(console, verbose)
        settings = RedThreadSettings(_env_file=env_file)
        try:
            payload = run_defense_prompt_spike(
                settings,
                Path.cwd(),
                reflection_lm=reflection_lm,
                max_metric_calls=max_metric_calls,
                train_slugs=train_slug,
                val_slugs=val_slug,
                control_slugs=control_slug,
                seed=seed,
            )
        except (ImportError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        console.print(
            Panel(
                f"[bold]GEPA defense-lane spike complete[/bold]\n\n"
                f"  Utility gate:   {payload['utility_gate_passed']}\n"
                f"  Replay block:   {payload['control_replay_block_rate']:.1%}\n"
                f"  Benign passed:  {payload['control_benign_passed']}\n"
                f"  Metric budget:  {payload['max_metric_calls']}\n"
                f"  Summary:        {payload['summary_ref']}\n"
                f"  Promotion:      {payload['promotion_status']}",
                border_style="magenta",
            )
        )

    @research.command(name="gepa-source-spike", hidden=True)
    @click.option("--reflection-lm", required=True, help="GEPA reflection model, e.g. ollama/qwen3.")
    @click.option("--max-metric-calls", type=click.IntRange(min=1), required=True)
    @click.option("--train-slug", multiple=True, help="Experiment objective slug to train on.")
    @click.option("--val-slug", multiple=True, help="Benchmark objective slug for GEPA validation.")
    @click.option("--seed", type=int, default=0, show_default=True)
    @click.option("--env-file", type=click.Path(exists=False), default=".env", help="Path to .env file")
    @click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging")
    def research_gepa_source_spike(
        reflection_lm: str,
        max_metric_calls: int,
        train_slug: tuple[str, ...],
        val_slug: tuple[str, ...],
        seed: int,
        env_file: str,
        verbose: bool,
    ) -> None:
        """Run the hidden Phase 4 GEPA source-lane spike."""
        from redthread.research.gepa_source_phase4 import run_source_mutation_spike

        setup_logging(console, verbose)
        settings = RedThreadSettings(_env_file=env_file)
        try:
            payload = run_source_mutation_spike(
                settings,
                Path.cwd(),
                reflection_lm=reflection_lm,
                max_metric_calls=max_metric_calls,
                train_slugs=train_slug,
                val_slugs=val_slug,
                seed=seed,
            )
        except (ImportError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        console.print(
            Panel(
                f"[bold]GEPA source-lane spike complete[/bold]\n\n"
                f"  Family:         {payload['mutation_family']}\n"
                f"  Candidate:      {payload['mutation_candidate_id']}\n"
                f"  Patch:          {payload['forward_patch_ref']}\n"
                f"  Reverse:        {payload['reverse_patch_ref']}\n"
                f"  Summary:        {payload['summary_ref']}\n"
                f"  Promotion:      {payload['promotion_status']}",
                border_style="magenta",
            )
        )
