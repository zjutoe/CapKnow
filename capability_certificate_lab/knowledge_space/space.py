from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Sequence

from .state import KnowledgeState
from .tasks import TaskUniverse
from ..validation.validator import validate_state


def _to_composition_constraint_checker(
    constraint: object,
) -> Callable[[KnowledgeState], bool]:
    if callable(constraint):
        return constraint

    if not isinstance(constraint, Mapping):
        raise TypeError(
            "composition_constraints entries must be either callables or mapping objects"
        )

    left = constraint.get("left")
    right = constraint.get("right")
    result = constraint.get("result")

    if not all(isinstance(item, str) for item in (left, right, result)):
        raise ValueError(
            "composition constraint mapping must define string fields: left, right, result"
        )

    def _check(local_state: KnowledgeState, *, left=left, right=right, result=result) -> bool:
        return not (
            left in local_state
            and right in local_state
            and result not in local_state
        )

    return _check


def _normalize_composition_constraints(
    composition_constraints: Iterable[object] | None,
) -> tuple[Callable[[KnowledgeState], bool], ...] | None:
    if composition_constraints is None:
        return None

    return tuple(
        _to_composition_constraint_checker(constraint)
        for constraint in composition_constraints
    )


@dataclass
class KnowledgeSpace:
    tasks: TaskUniverse
    valid_states: Sequence[KnowledgeState] = ()
    generator_rules: MutableMapping[str, object] = field(default_factory=dict)
    metadata: MutableMapping[str, object] = field(default_factory=dict)

    def is_valid_state(self, state: KnowledgeState) -> bool:
        prerequisites = self.generator_rules.get("prerequisites", {})
        composition_constraints = self.generator_rules.get("composition_constraints")
        normalized_constraints = _normalize_composition_constraints(
            composition_constraints
        )
        return (
            validate_state(
                state,
                task_universe=self.tasks,
                prerequisites=prerequisites,
                composition_constraints=normalized_constraints,
            )
        )

    def to_dict(self) -> Mapping[str, object]:
        return {
            "tasks": self.tasks.to_dict(),
            "valid_states": [state.to_dict() for state in self.valid_states],
            "generator_rules": dict(self.generator_rules),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "KnowledgeSpace":
        tasks = TaskUniverse.from_dict(payload["tasks"])  # type: ignore[index]
        valid_states = [KnowledgeState.from_dict(item) for item in payload.get("valid_states", ())]
        return cls(
            tasks=tasks,
            valid_states=valid_states,
            generator_rules=payload.get("generator_rules", {}),
            metadata=payload.get("metadata", {}),
        )
