from __future__ import annotations

from typing import Sequence

from ..knowledge_space.space import KnowledgeSpace
from ..knowledge_space.tasks import TaskUniverse
from ..knowledge_space.state import KnowledgeState


def generate_chain_world(task_ids: Sequence[str]) -> KnowledgeSpace:
    universe = TaskUniverse(task_ids)
    ordered_tasks = universe.task_ids
    valid_states = []
    current = set[str]()

    for task_id in ordered_tasks:
        current.add(task_id)
        valid_states.append(KnowledgeState(current))

    valid_states = [KnowledgeState(())] + valid_states
    prerequisites = {
        ordered_tasks[i]: [ordered_tasks[i - 1]] for i in range(1, len(ordered_tasks))
    }

    return KnowledgeSpace(
        tasks=universe,
        valid_states=valid_states,
        generator_rules={
            "type": "chain",
            "prerequisites": prerequisites,
        },
        metadata={"ordered_tasks": ordered_tasks},
    )
