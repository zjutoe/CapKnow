from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class KnowledgeState:
    """A set-like representation of known capabilities."""

    tasks: frozenset[str]

    def __init__(self, tasks: Iterable[str] = ()):  # type: ignore[no-redef]
        object.__setattr__(self, "tasks", frozenset(str(task) for task in tasks))

    def __contains__(self, task_id: str) -> bool:
        return task_id in self.tasks

    def to_dict(self) -> Mapping[str, list[str]]:
        return {"tasks": sorted(self.tasks)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Sequence[str]]) -> "KnowledgeState":
        return cls(payload.get("tasks", ()))

    def as_tuple(self, task_order: Sequence[str] | None = None) -> tuple[str, ...]:
        if task_order is None:
            return tuple(sorted(self.tasks))
        order_map = {task_id: i for i, task_id in enumerate(task_order)}
        return tuple(sorted(self.tasks, key=lambda task_id: order_map[task_id]))
