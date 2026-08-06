from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Callable

from ..knowledge_space.tasks import TaskUniverse
from ..knowledge_space.state import KnowledgeState


def validate_state(
    state: KnowledgeState,
    *,
    task_universe: TaskUniverse,
    prerequisites: Mapping[str, Sequence[str] | set[str]] | None = None,
    composition_constraints: Iterable[Callable[[KnowledgeState], bool]] | None = None,
) -> bool:
    for task_id in state.tasks:
        if task_id not in task_universe:
            return False

    if prerequisites is None:
        prerequisites = {}

    for task_id in state.tasks:
        required = prerequisites.get(task_id, ())
        for req in required:
            if req not in state.tasks:
                return False

    if composition_constraints is None:
        return True
    for check in composition_constraints:
        if not check(state):
            return False
    return True
