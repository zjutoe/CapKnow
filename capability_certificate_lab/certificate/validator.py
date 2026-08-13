from __future__ import annotations

from collections.abc import Callable, Sequence

from ..knowledge_space.space import KnowledgeSpace
from ..knowledge_space.state import KnowledgeState
from ..validation.identifiability.core import (
    response_signature,
    Signature,
    stable_state_id,
    validate_response_signature,
)


def _ordered_valid_states(knowledge_space: KnowledgeSpace):
    task_ids = list(knowledge_space.tasks.task_ids)
    declared_states = list(knowledge_space.valid_states)
    seen: set[KnowledgeState] = set()
    duplicates: list[str] = []
    for state in declared_states:
        if state in seen:
            duplicates.append(stable_state_id(state, task_ids))
        seen.add(state)
    if duplicates:
        raise ValueError(f"Duplicate declared states are not allowed: {duplicates}")

    invalid = [
        stable_state_id(state, task_ids)
        for state in declared_states
        if not knowledge_space.is_valid_state(state)
    ]
    if invalid:
        raise ValueError(f"Declared states violate world rules: {invalid}")

    return sorted(declared_states, key=lambda state: state.as_tuple(task_ids)), task_ids


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
    signatures: list[Signature] = []
    for state in states:
        signature = validate_response_signature(
            response_signature_fn(state, selected_tasks),
            len(selected_tasks),
            "Response signature",
        )
        signatures.append(signature)
    return len(set(signatures)) == len(states)
