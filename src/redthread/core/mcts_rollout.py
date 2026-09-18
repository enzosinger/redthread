"""Rollout simulation and branch expansion execution for GS-MCTS."""

from __future__ import annotations

from typing import TYPE_CHECKING

from redthread.core.attack_execution import attack_execution_metadata
from redthread.core.mcts_helpers import (
    MCTSTree,
    build_expansion_prompt,
    build_rollout_prompt,
    format_mcts_history,
)
from redthread.core.mcts_phases import compile_mcts_target_prompt
from redthread.models import MCTSNode, Persona
from redthread.pyrit_adapters.targets import (
    RedThreadTarget,
    send_with_usage_and_execution_metadata,
)

if TYPE_CHECKING:
    from redthread.evaluation.judge import JudgeAgent


async def expand_mcts_branch(
    leaf: MCTSNode,
    strategy: str,
    turn_number: int,
    history_text: str,
    history: list[tuple[str, str]],
    persona: Persona,
    attacker_system: str,
    target_system_prompt: str,
    attacker: RedThreadTarget,
    target: RedThreadTarget,
    trace_id: str,
) -> tuple[MCTSNode, int]:
    """Execute attacker generation and target response for a single strategy branch."""
    prompt = f"[SYSTEM]: {attacker_system}\n\n[USER]: " + build_expansion_prompt(
        persona, strategy, history_text, turn_number
    )
    attacker_msg, a_tokens = await send_with_usage_and_execution_metadata(
        attacker,
        prompt=prompt,
        conversation_id=f"mcts-expand-{trace_id}-d{turn_number}",
        execution_metadata=attack_execution_metadata(
            algorithm="mcts",
            lane="attacker",
            trace_id=trace_id,
            metadata={"depth": turn_number, "strategy": strategy},
        ),
    )
    target_input = compile_mcts_target_prompt(history, attacker_msg.strip(), target_system_prompt)
    target_resp, t_tokens = await send_with_usage_and_execution_metadata(
        target,
        prompt=target_input,
        conversation_id=f"mcts-target-{trace_id}-d{turn_number}",
        execution_metadata=attack_execution_metadata(
            algorithm="mcts",
            lane="target",
            trace_id=trace_id,
            metadata={"depth": turn_number, "strategy": strategy},
        ),
    )
    child = MCTSNode(
        parent_id=leaf.id,
        depth=turn_number,
        strategy=strategy,
        attacker_prompt=attacker_msg.strip(),
        target_response=target_resp,
    )
    return child, a_tokens + t_tokens


async def simulate_mcts_rollout(
    node: MCTSNode,
    tree: MCTSTree,
    persona: Persona,
    target_system_prompt: str,
    rubric_name: str,
    rollout_attacker: RedThreadTarget,
    target: RedThreadTarget,
    judge: JudgeAgent,
    max_rollout_turns: int,
    max_depth: int,
) -> tuple[float, int]:
    """CoT rollout: simulate rapid trajectory using token-constrained attacker."""
    sim_history = list(tree.get_history(node))
    current_depth = node.depth
    consumed_tokens = 0

    for _ in range(max_rollout_turns):
        if current_depth >= max_depth:
            break
        turn_number = current_depth + 1
        prompt = build_rollout_prompt(
            persona, format_mcts_history(sim_history), turn_number, max_depth
        )
        rollout_msg, tokens = await send_with_usage_and_execution_metadata(
            rollout_attacker,
            prompt=prompt,
            conversation_id=f"mcts-rollout-{node.id}-t{turn_number}",
            execution_metadata=attack_execution_metadata(
                algorithm="mcts",
                lane="rollout_attacker",
                trace_id=node.id,
                metadata={"depth": turn_number, "strategy": node.strategy},
            ),
        )
        consumed_tokens += tokens
        target_input = compile_mcts_target_prompt(
            sim_history, rollout_msg.strip(), target_system_prompt
        )
        target_resp, t_tokens = await send_with_usage_and_execution_metadata(
            target,
            prompt=target_input,
            conversation_id=f"mcts-rollout-tgt-{node.id}-t{turn_number}",
            execution_metadata=attack_execution_metadata(
                algorithm="mcts",
                lane="target",
                trace_id=node.id,
                metadata={"rollout": True, "depth": turn_number, "strategy": node.strategy},
            ),
        )
        consumed_tokens += t_tokens
        sim_history.append((rollout_msg.strip(), target_resp))
        current_depth += 1

    reward = 0.0
    if sim_history:
        last_a, last_t = sim_history[-1]
        reward = judge.evaluate_turn_raw(last_a, last_t, rubric_name)
    return reward, consumed_tokens
