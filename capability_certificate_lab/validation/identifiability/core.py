from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from typing import Sequence

from ...knowledge_space.space import KnowledgeSpace
from ...knowledge_space.state import KnowledgeState
from ...simulator.response import simulate_response_matrix


Signature = tuple[int, ...]


@dataclass(frozen=True)
class IdentifiabilityReport:
    num_tasks: int
    num_states: int
    num_unique_signatures: int
    identifiable: bool
    collision_count: int
    collision_groups: list[list[str]]
    compression_ratio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "num_tasks": self.num_tasks,
            "num_states": self.num_states,
            "num_unique_signatures": self.num_unique_signatures,
            "identifiable": self.identifiable,
            "collision_count": self.collision_count,
            "collision_groups": self.collision_groups,
            "compression_ratio": self.compression_ratio,
        }


def stable_state_id(state: KnowledgeState, task_ids: Sequence[str]) -> str:
    ordered_known = [task_id for task_id in task_ids if task_id in state]
    known_set = set(ordered_known)
    extras = sorted(task_id for task_id in state.tasks if task_id not in known_set)
    return json.dumps(ordered_known + extras, separators=(",", ":"))


def response_signature(state: KnowledgeState, task_ids: Sequence[str]) -> Signature:
    matrix = simulate_response_matrix([state], task_ids)
    return tuple(matrix[0])


def validate_response_signature(
    signature: Sequence[int],
    expected_length: int,
    context: str = "Response signature",
) -> Signature:
    normalized = tuple(signature)
    if len(normalized) != expected_length:
        raise ValueError(
            f"{context} length must equal queried task count: "
            f"expected {expected_length}, got {len(normalized)}."
        )
    if any(value not in (0, 1) for value in normalized):
        raise ValueError(f"{context} must contain only binary 0/1 values.")
    return normalized


def check_identifiability(
    knowledge_space: KnowledgeSpace,
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature] = response_signature,
) -> IdentifiabilityReport:
    task_ids = list(knowledge_space.tasks.task_ids)
    declared_states = list(knowledge_space.valid_states)
    if not declared_states:
        raise ValueError("Declared valid_states must not be empty for identifiability audit.")

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

    ordered_states = sorted(declared_states, key=lambda state: state.as_tuple(task_ids))

    buckets: dict[Signature, list[str]] = {}
    for state in ordered_states:
        signature = validate_response_signature(
            response_signature_fn(state, task_ids),
            len(task_ids),
            "Identifiability response signature",
        )
        buckets.setdefault(signature, []).append(stable_state_id(state, task_ids))

    collision_groups = [group for group in buckets.values() if len(group) > 1]
    num_states = len(ordered_states)
    num_unique_signatures = len(buckets)
    collision_count = len(collision_groups)
    compression_ratio = 0.0 if num_states == 0 else num_unique_signatures / num_states

    return IdentifiabilityReport(
        num_tasks=len(task_ids),
        num_states=num_states,
        num_unique_signatures=num_unique_signatures,
        identifiable=collision_count == 0,
        collision_count=collision_count,
        collision_groups=collision_groups,
        compression_ratio=compression_ratio,
    )
