from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Mapping, MutableMapping


@dataclass(frozen=True)
class Task:
    id: str

    def to_dict(self) -> Mapping[str, str]:
        return {"id": self.id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, str]) -> "Task":
        return cls(payload["id"])


class TaskUniverse:
    """A sequence-like collection of unique tasks with index access."""

    def __init__(self, tasks: Iterable[str | Task]):
        normalized: List[Task] = []
        seen = set()
        for task in tasks:
            item = task if isinstance(task, Task) else Task(str(task))
            if item.id in seen:
                raise ValueError(f"Duplicate task id: {item.id}")
            seen.add(item.id)
            normalized.append(item)
        self._tasks: List[Task] = normalized
        self._index: MutableMapping[str, int] = {
            task.id: i for i, task in enumerate(self._tasks)
        }

    @property
    def tasks(self) -> List[Task]:
        return list(self._tasks)

    @property
    def task_ids(self) -> List[str]:
        return [task.id for task in self._tasks]

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, item: str | Task) -> bool:
        task_id = item.id if isinstance(item, Task) else str(item)
        return task_id in self._index

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks)

    def get(self, task_id: str) -> Task:
        return self._tasks[self._index[task_id]]

    def index(self, task_id: str) -> int:
        return self._index[task_id]

    def to_dict(self) -> Mapping[str, list[Mapping[str, str]]]:
        return {"tasks": [task.to_dict() for task in self._tasks]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Iterable[Mapping[str, str]]]) -> "TaskUniverse":
        return cls([Task.from_dict(item) for item in payload["tasks"]])
