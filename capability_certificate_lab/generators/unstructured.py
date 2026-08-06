from __future__ import annotations

from ..knowledge_space.space import KnowledgeSpace
from ..knowledge_space.tasks import TaskUniverse
from ..knowledge_space.state import KnowledgeState


def generate_unstructured_world(task_ids) -> KnowledgeSpace:
    universe = TaskUniverse(task_ids)
    task_list = universe.task_ids

    valid_states = []
    for mask in range(1 << len(task_list)):
        selected = [task_list[i] for i in range(len(task_list)) if (mask >> i) & 1]
        valid_states.append(KnowledgeState(selected))

    return KnowledgeSpace(
        tasks=universe,
        valid_states=valid_states,
        generator_rules={
            "type": "unstructured",
            "prerequisites": {},
        },
    )
