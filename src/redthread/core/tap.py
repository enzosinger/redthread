"""TAP — Tree of Attacks with Pruning algorithm.

Implements the tree-based closed-loop black-box jailbreaking algorithm from:
  "Tree of Attacks: Jailbreaking Black-Box LLMs Automatically"
  Mehdi et al. (2023) — https://arxiv.org/abs/2312.02119
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from redthread.config.settings import RedThreadSettings
from redthread.core.attack_execution import attack_execution_metadata
from redthread.core.attack_utils import extract_improvement, extract_prompt
from redthread.core.tap_branching import (
    DEFAULT_STRATEGIES,
    build_dry_run_tap_result,
    build_tap_branch_input,
    expand_dry_run_tree,
    finalize_tap_trace,
    resolve_tap_system_prompt,
)
from redthread.core.tap_pruning import (
    prune_by_score,
    prune_off_topic,
    reconstruct_path,
    score_tap_node,
)
from redthread.evaluation.judge import JudgeAgent
from redthread.models import AttackNode, AttackResult, AttackTrace, Persona
from redthread.pyrit_adapters.targets import (
    RedThreadTarget,
    build_attacker,
    build_target,
    send_with_execution_metadata,
)
from redthread.research.prompt_profiles import load_prompt_profiles, resolve_prompt_profiles_path

logger = logging.getLogger(__name__)


class TAPAttack:
    """TAP closed-loop adversarial attack algorithm."""

    def __init__(
        self,
        settings: RedThreadSettings,
        attacker: RedThreadTarget | None = None,
        target: RedThreadTarget | None = None,
        judge: JudgeAgent | None = None,
    ) -> None:
        self.settings = settings
        self._attacker = attacker
        self._target = target
        self._judge = judge or JudgeAgent(settings)
        profiles = load_prompt_profiles(resolve_prompt_profiles_path(settings))
        self._tap_profile = profiles.get("tap", {})
        self.strategies = self._tap_profile.get("strategies", DEFAULT_STRATEGIES)

    def _get_attacker(self) -> RedThreadTarget:
        if self._attacker is None:
            self._attacker = build_attacker(self.settings)
        return self._attacker

    def _get_target(self) -> RedThreadTarget:
        if self._target is None:
            self._target = build_target(self.settings)
        return self._target

    async def run(
        self,
        persona: Persona,
        target_system_prompt: str = "",
        rubric_name: str = "authorization_bypass",
    ) -> AttackResult:
        """Execute the full TAP loop for a given persona."""
        start_time = time.monotonic()
        trace = AttackTrace(
            persona=persona,
            algorithm="tap",
            started_at=datetime.now(timezone.utc),
            metadata={"target_system_prompt": target_system_prompt} if target_system_prompt else {},
        )
        logger.info(
            "🌲 TAP attack starting | persona=%s | target=%s | D=%d, B=%d, W=%d",
            persona.name,
            self.settings.target_model,
            self.settings.tree_depth,
            self.settings.branching_factor,
            self.settings.tree_width,
        )
        attacker_sys_prompt = resolve_tap_system_prompt(persona, self._tap_profile)
        root_node = AttackNode(depth=0, score=0.0)
        trace.nodes.append(root_node)
        active_leaves = [root_node]
        jailbreak_node: AttackNode | None = None

        for depth in range(1, self.settings.tree_depth + 1):
            logger.info("🌳 Tree Depth %d/%d (Leaves: %d)", depth, self.settings.tree_depth, len(active_leaves))
            if self.settings.dry_run:
                active_leaves = expand_dry_run_tree(
                    active_leaves, trace, depth, self.settings.branching_factor, self.settings.tree_width
                )
                continue

            new_nodes = await self._branch(active_leaves, persona, attacker_sys_prompt, depth, trace.id)
            trace.nodes.extend(new_nodes)
            surviving = prune_off_topic(new_nodes, persona.hidden_objective)
            if not surviving:
                logger.warning("All branches pruned at depth %d due to topic drift.", depth)
                break

            for node in surviving:
                node.target_response = await send_with_execution_metadata(
                    self._get_target(),
                    prompt=node.attacker_prompt,
                    conversation_id=f"tap-target-{trace.id}-{node.id}",
                    execution_metadata=attack_execution_metadata(
                        algorithm="tap",
                        lane="target",
                        trace_id=trace.id,
                        metadata={"depth": depth, "node_id": node.id},
                    ),
                )
                node.score = score_tap_node(self._judge, node.attacker_prompt, node.target_response, rubric_name)
                if node.score >= self.settings.success_threshold:
                    jailbreak_node = node
                    logger.info("✅ Jailbreak found at depth %d! Score: %.1f", depth, node.score)
                    break

            if jailbreak_node:
                break
            active_leaves = prune_by_score(surviving, self.settings.tree_width)

        trace.ended_at = datetime.now(timezone.utc)
        if self.settings.dry_run:
            return build_dry_run_tap_result(trace, rubric_name)

        return await finalize_tap_trace(
            trace, jailbreak_node, self._judge, persona.name, rubric_name, start_time
        )

    async def _branch(
        self, leaves: list[AttackNode], persona: Persona, system_prompt: str, depth: int, trace_id: str
    ) -> list[AttackNode]:
        new_nodes: list[AttackNode] = []
        for leaf in leaves:
            for branch_idx in range(self.settings.branching_factor):
                strategy = self.strategies[branch_idx % len(self.strategies)]
                prompt_input = build_tap_branch_input(leaf, persona, strategy)
                raw_out = await send_with_execution_metadata(
                    self._get_attacker(),
                    prompt=f"[SYSTEM]: {system_prompt}\n\n[USER]: {prompt_input}",
                    conversation_id=f"tap-attacker-{trace_id}-{leaf.id}-{branch_idx}",
                    execution_metadata=attack_execution_metadata(
                        algorithm="tap",
                        lane="attacker",
                        trace_id=trace_id,
                        metadata={"depth": depth, "parent_id": leaf.id, "branch": branch_idx},
                    ),
                )
                new_nodes.append(
                    AttackNode(
                        parent_id=leaf.id,
                        depth=depth,
                        attacker_prompt=extract_prompt(raw_out),
                        improvement_rationale=extract_improvement(raw_out),
                    )
                )
        return new_nodes

    def _prune_off_topic(self, nodes: list[AttackNode], objective: str) -> list[AttackNode]:
        return prune_off_topic(nodes, objective)

    def _score_node(self, prompt: str, response: str, rubric_name: str) -> float:
        return score_tap_node(self._judge, prompt, response, rubric_name)

    def _prune_by_score(self, nodes: list[AttackNode], max_width: int) -> list[AttackNode]:
        return prune_by_score(nodes, max_width)

    def reconstruct_path(self, end_node: AttackNode, all_nodes: list[AttackNode]) -> list[AttackNode]:
        return reconstruct_path(end_node, all_nodes)

    def _extract_prompt(self, raw: str) -> str:
        return extract_prompt(raw)

    def _extract_improvement(self, raw: str) -> str:
        return extract_improvement(raw)
