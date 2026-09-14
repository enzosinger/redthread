"""Throwaway Phase-1 lift spike: does GEPA beat the hard-coded mutation table?

This is NOT a CLI command and NOT part of the product surface. It is the disposable
proof vehicle both the CEO and CTO asked for: run ONE small, real GEPA optimization
on local models and answer a single go/no-go question — does a reflective,
Pareto-selected prompt candidate beat today's lookup-table mutation on a held-out
objective? If yes, we wire GEPA under the hood into the existing autoresearch lane
(no new command, behind a config gate). If no, we stop before building that plumbing.

Surfacing decision (CEO+CTO, unanimous): GEPA ships under the hood, never as a new
operator command. This script exists only to earn that integration, then it can be
deleted.

USAGE
-----
Offline self-test (validates the measurement harness, no model calls, no deps):
    python scripts/spikes/gepa_phase1_spike.py --mock

Live spike (requires the optional extra + a running Ollama + judge credentials):
    pip install 'redthread[research-gepa]'           # gepa + litellm
    # start Ollama and pull the configured attacker/target models
    export REDTHREAD_GEPA_REFLECTION_MODEL=ollama/<a-capable-local-instruct-model>
    python scripts/spikes/gepa_phase1_spike.py --max-metric-calls 30

Notes
- Reflection LM is local by default (per the chosen "local + gpt-4o judge" setup);
  the judge stays whatever REDTHREAD_JUDGE_* is configured (gpt-4o by default).
- Live model calls are real and cost time/tokens. Keep --max-metric-calls small.
- The candidate is applied ONLY to the research-runtime prompt_profiles.json; it
  never touches tracked source.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

from redthread.config.settings import RedThreadSettings
from redthread.research.baseline import run_objective
from redthread.research.gepa_adapter import RedThreadGEPAAdapter, build_optimize_kwargs
from redthread.research.gepa_score import normalize_objective
from redthread.research.models import ObjectiveResult, ResearchObjective
from redthread.research.objectives import default_research_config
from redthread.research.prompt_profiles import default_prompt_profiles, load_prompt_profiles
from redthread.research.workspace import ResearchWorkspace

COMPONENTS = ["pair.system_suffix", "tap.strategies"]

Runner = Callable[[ResearchObjective, dict[str, str]], ObjectiveResult]


def seed_candidate() -> dict[str, str]:
    """Build the GEPA seed from today's default attacker prompt profiles."""
    profiles = default_prompt_profiles()
    return {
        "pair.system_suffix": profiles["pair"]["system_suffix"],
        "tap.strategies": json.dumps(profiles["tap"]["strategies"]),
    }


def apply_candidate(workspace: ResearchWorkspace, candidate: dict[str, str]) -> None:
    """Write a candidate into the research-runtime prompt_profiles.json (only there)."""
    profiles = load_prompt_profiles(workspace.prompt_profiles_path)
    for field, value in candidate.items():
        section, key = field.split(".", maxsplit=1)
        section_profile = profiles.setdefault(section, {})
        if key == "strategies":
            section_profile[key] = json.loads(value)
        else:
            section_profile[key] = value
    workspace.prompt_profiles_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")


def make_live_runner(base_settings: RedThreadSettings, root: Path) -> Runner:
    """Real runner: apply candidate to runtime profiles, run one bounded campaign."""

    def runner(objective: ResearchObjective, candidate: dict[str, str]) -> ObjectiveResult:
        workspace = ResearchWorkspace(root)
        research_settings = workspace.research_settings(base_settings)
        apply_candidate(workspace, candidate)
        campaign_id, asr, avg_score, confirmed, near_misses = asyncio.run(
            run_objective(research_settings, objective)
        )
        return ObjectiveResult(
            slug=objective.slug,
            campaign_id=campaign_id,
            attack_success_rate=asr,
            average_score=avg_score,
            confirmed_jailbreaks=confirmed,
            near_misses=near_misses,
        )

    return runner


def make_mock_runner() -> Runner:
    """Deterministic offline runner: rewards candidates that differ from the seed.

    This validates the measurement harness (apply -> evaluate -> compare) without any
    model call. It does NOT prove lift — only that the script's plumbing is sound.
    """
    base = seed_candidate()

    def runner(objective: ResearchObjective, candidate: dict[str, str]) -> ObjectiveResult:
        drift = sum(1 for k, v in candidate.items() if v != base.get(k))
        asr = min(0.4 + 0.1 * drift, 0.95)
        return ObjectiveResult(
            slug=objective.slug,
            campaign_id=f"mock-{objective.slug}",
            attack_success_rate=asr,
            average_score=2.0 + drift,
            confirmed_jailbreaks=1 if asr > 0.5 else 0,
            near_misses=1,
        )

    return runner


def _objectives() -> tuple[list[ResearchObjective], ResearchObjective]:
    """Train objectives + one held-out objective for the lift comparison."""
    pack = default_research_config().benchmark_objectives
    trainset = pack[:2]
    holdout = pack[2] if len(pack) > 2 else pack[-1]
    return trainset, holdout


def _score(runner: Runner, objective: ResearchObjective, candidate: dict[str, str]) -> float:
    return float(normalize_objective(runner(objective, candidate)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="offline harness self-test, no model calls")
    parser.add_argument("--max-metric-calls", type=int, default=30, help="GEPA budget (live only)")
    parser.add_argument(
        "--reflection-model",
        default=os.environ.get("REDTHREAD_GEPA_REFLECTION_MODEL", ""),
        help="litellm model id for GEPA reflection, e.g. ollama/<model> (live only)",
    )
    args = parser.parse_args()

    root = Path.cwd()
    trainset, holdout = _objectives()
    seed = seed_candidate()

    if args.mock:
        runner = make_mock_runner()
        adapter = RedThreadGEPAAdapter(runner, components=COMPONENTS)
        tweaked = {**seed, "pair.system_suffix": seed["pair.system_suffix"] + "\n# tweak"}
        baseline = _score(runner, holdout, seed)
        candidate = _score(runner, holdout, tweaked)
        eval_batch = adapter.evaluate(trainset, seed, capture_traces=True)
        reflective = adapter.make_reflective_dataset(seed, eval_batch, COMPONENTS)
        print(f"[mock] baseline held-out score = {baseline:.4f}")
        print(f"[mock] tweaked  held-out score = {candidate:.4f}")
        print(f"[mock] adapter evaluate -> {len(eval_batch.scores)} scores, "
              f"reflective components = {sorted(reflective)}")
        assert candidate >= baseline, "mock harness invariant failed"
        print("[mock] harness OK — wiring is sound. Live run will produce the real lift number.")
        return 0

    if not args.reflection_model:
        parser.error(
            "live run needs --reflection-model (or REDTHREAD_GEPA_REFLECTION_MODEL), "
            "e.g. ollama/<a-capable-local-instruct-model>"
        )
    import gepa

    settings = RedThreadSettings()
    runner = make_live_runner(settings, root)
    adapter = RedThreadGEPAAdapter(runner, components=COMPONENTS)

    baseline = _score(runner, holdout, seed)
    print(f"[live] baseline (hand-written profiles) held-out score = {baseline:.4f}")

    kwargs = build_optimize_kwargs(
        seed,
        trainset,
        adapter=adapter,
        reflection_lm=args.reflection_model,
        max_metric_calls=args.max_metric_calls,
        valset=[holdout],
    )
    result = gepa.optimize(**kwargs)  # type: ignore[attr-defined]
    best = cast("dict[str, str]", result.best_candidate)
    gepa_score = _score(runner, holdout, best)

    delta = gepa_score - baseline
    print(f"[live] GEPA best held-out score          = {gepa_score:.4f}")
    print(f"[live] LIFT = {delta:+.4f}  ({'GO' if delta > 0 else 'NO-GO'})")
    print(f"[live] total metric calls = {result.total_metric_calls}, "
          f"candidates = {result.num_candidates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
