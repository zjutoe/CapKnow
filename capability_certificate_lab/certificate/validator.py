from __future__ import annotations

from collections.abc import Callable, Sequence

from ..knowledge_space.space import KnowledgeSpace
from ..knowledge_space.state import KnowledgeState
from ..validation.identifiability.core import response_signature, Signature


def _ordered_valid_states(knowledge_space: KnowledgeSpace):
    task_ids = list(knowledge_space.tasks.task_ids)
    valid_states = [
        state for state in knowledge_space.valid_states if knowledge_space.is_valid_state(state)
    ]
    return sorted(valid_states, key=lambda state: state.as_tuple(task_ids)), task_ids


def validate_certificate(
    knowledge_space: KnowledgeSpace,
    certificate: Sequence[str],
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature] = response_signature,
) -> bool:
    states, _ = _ordered_valid_states(knowledge_space)
    if len(states) <= 1:
        return True

    task_set = set(knowledge_space.tasks.task_ids)
    for task_id in certificate:
        if task_id not in task_set:
            raise ValueError(f"Task '{task_id}' not in knowledge space")

    selected_tasks = list(certificate)
    signatures = [response_signature_fn(state, selected_tasks) for state in states]
    return len(set(signatures)) == len(states)
