from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping, Sequence

from .state import KnowledgeState
from .tasks import TaskUniverse
from ..validation.validator import validate_state


@dataclass
class KnowledgeSpace:
    tasks: TaskUniverse
    valid_states: Sequence[KnowledgeState] = ()
    generator_rules: MutableMapping[str, object] = field(default_factory=dict)
    metadata: MutableMapping[str, object] = field(default_factory=dict)

    def is_valid_state(self, state: KnowledgeState) -> bool:
        prerequisites = self.generator_rules.get("prerequisites", {})
        return validate_state(
            state,
            task_universe=self.tasks,
            prerequisites=prerequisites,
            composition_constraints=self.generator_rules.get("composition_constraints"),
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
