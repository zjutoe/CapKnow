from __future__ import annotations

from collections.abc import Sequence

from ..knowledge_space.tasks import TaskUniverse
from ..knowledge_space.state import KnowledgeState


def simulate_response(state: KnowledgeState, task: str) -> int:
    return int(task in state.tasks)

def simulate_response_matrix(
    states: Sequence[KnowledgeState],
    tasks: Sequence[str] | TaskUniverse,
) -> list[list[int]]:
    task_ids = tasks.task_ids if isinstance(tasks, TaskUniverse) else list(tasks)
    return [[simulate_response(state, task) for task in task_ids] for state in states]
