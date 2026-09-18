"""Pruning and path reconstruction heuristics for TAP attack tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

from redthread.models import AttackNode, ConversationTurn

if TYPE_CHECKING:
    from redthread.evaluation.judge import JudgeAgent


def prune_off_topic(nodes: list[AttackNode], objective: str) -> list[AttackNode]:
    """Pre-Query Prune phase using heuristic keyword search."""
    tokens = [
        t.lower()
        for t in objective.replace("(", "").replace(")", "").split()
        if len(t) > 3
    ]
    if not tokens:
        return nodes

    surviving: list[AttackNode] = []
    for node in nodes:
        prompt_lower = node.attacker_prompt.lower()
        hit = any(token in prompt_lower for token in tokens)
        if not hit and "ignore previous" not in prompt_lower and len(prompt_lower.split()) > 30:
            node.is_pruned = True
            node.pruned_reason = "off_topic"
        else:
            surviving.append(node)
    return surviving


def prune_by_score(nodes: list[AttackNode], max_width: int) -> list[AttackNode]:
    """Rank by score, keep only top W nodes."""
    nodes.sort(key=lambda n: n.score, reverse=True)
    top_w = nodes[:max_width]
    for pruned_node in nodes[max_width:]:
        pruned_node.is_pruned = True
        pruned_node.pruned_reason = "low_score"
    return top_w


def reconstruct_path(end_node: AttackNode, all_nodes: list[AttackNode]) -> list[AttackNode]:
    """Iterative walk back up to root."""
    node_map = {node.id: node for node in all_nodes}
    path: list[AttackNode] = []
    current: AttackNode | None = end_node
    while current:
        path.append(current)
        if not current.parent_id:
            break
        current = node_map.get(current.parent_id)

    path.reverse()
    return path


def score_tap_node(judge: JudgeAgent, prompt: str, response: str, rubric_name: str) -> float:
    """Score a node via Judge heuristic inline."""
    turn = ConversationTurn(
        turn_number=1,
        attacker_prompt=prompt,
        target_response=response,
    )
    return judge.evaluate_turn(turn, rubric_name)
