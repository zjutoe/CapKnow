from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import List

from ..knowledge_space.space import KnowledgeSpace
from ..knowledge_space.tasks import TaskUniverse
from ..knowledge_space.state import KnowledgeState
from ..validation.validator import validate_state


def _normalize_tree(edges: Sequence[tuple[str, str]] | Mapping[str, Sequence[str]]) -> List[tuple[str, str]]:
    # Risk note: mapping inputs with a parent that has no children are not included in the edge list.
    # This implementation assumes such empty-child parents are not present in tree inputs.
    if isinstance(edges, Mapping):
        normalized: List[tuple[str, str]] = []
        for parent, children in edges.items():
            for child in children:
                normalized.append((str(parent), str(child)))
        return normalized
    return [(str(parent), str(child)) for parent, child in edges]


def generate_tree_world(tree_edges: Sequence[tuple[str, str]] | Mapping[str, Sequence[str]]) -> KnowledgeSpace:
    pairs = _normalize_tree(tree_edges)
    task_ids = []
    for parent, child in pairs:
        task_ids.append(parent)
        task_ids.append(child)
    task_ids = list(dict.fromkeys(task_ids))

    universe = TaskUniverse(task_ids)
    prerequisites = defaultdict(list[str])
    for parent, child in pairs:
        prerequisites[child].append(parent)

    valid_states = []
    task_list = universe.task_ids
    for mask in range(1 << len(task_list)):
        selected = [task_list[i] for i in range(len(task_list)) if (mask >> i) & 1]
        state = KnowledgeState(selected)
        if validate_state(state, task_universe=universe, prerequisites=prerequisites):
            valid_states.append(state)

    return KnowledgeSpace(
        tasks=universe,
        valid_states=sorted(
            valid_states,
            key=lambda state: (len(state.tasks), state.as_tuple(task_list)),
        ),
        generator_rules={
            "type": "tree",
            "prerequisites": {k: tuple(v) for k, v in prerequisites.items()},
        },
    )
