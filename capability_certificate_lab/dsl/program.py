from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .primitives import PrimitiveOperation


def _validate_loop_max_iterations(max_iterations: object) -> None:
    if type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError("Loop max_iterations must be a positive integer.")


@dataclass(frozen=True)
class Program:
    """Base interface for DSL nodes."""

    def required_primitive_ids(self) -> set[str]:
        raise NotImplementedError

    def to_task_id(self) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class PrimitiveNode(Program):
    op: PrimitiveOperation
    args: tuple[str, ...] = ()

    def required_primitive_ids(self) -> set[str]:
        return {self.op.op_id}

    def to_task_id(self) -> str:
        return self.op.op_id

    def to_dict(self) -> dict[str, Any]:
        return {"type": "primitive", "op": self.op.to_dict(), "args": list(self.args)}


@dataclass(frozen=True)
class SequenceNode(Program):
    steps: tuple[Program, ...]

    def required_primitive_ids(self) -> set[str]:
        required: set[str] = set()
        for step in self.steps:
            required |= step.required_primitive_ids()
        return required

    def to_task_id(self) -> str:
        if not self.steps:
            return "SEQ_EMPTY"
        return f"SEQ[{','.join(step.to_task_id() for step in self.steps)}]"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "sequence", "steps": [step.to_dict() for step in self.steps]}


@dataclass(frozen=True)
class ConditionNode(Program):
    condition: Program
    then: Program
    otherwise: Program | None = None

    def required_primitive_ids(self) -> set[str]:
        required: set[str] = {"CONDITION"}
        required |= self.condition.required_primitive_ids()
        required |= self.then.required_primitive_ids()
        if self.otherwise is not None:
            required |= self.otherwise.required_primitive_ids()
        return required

    def to_task_id(self) -> str:
        if self.otherwise is None:
            return f"IF[{self.condition.to_task_id()}]"
        return f"IF[{self.condition.to_task_id()}|{self.otherwise.to_task_id()}|{self.then.to_task_id()}]"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "condition",
            "condition": self.condition.to_dict(),
            "then": self.then.to_dict(),
        }
        if self.otherwise is not None:
            payload["otherwise"] = self.otherwise.to_dict()
        return payload


@dataclass(frozen=True)
class LoopNode(Program):
    body: Program
    max_iterations: int

    def required_primitive_ids(self) -> set[str]:
        _validate_loop_max_iterations(self.max_iterations)
        return {"LOOP"} | self.body.required_primitive_ids()

    def to_task_id(self) -> str:
        return f"LOOP[{self.max_iterations}x{self.body.to_task_id()}]"

    def to_dict(self) -> dict[str, Any]:
        _validate_loop_max_iterations(self.max_iterations)
        return {
            "type": "loop",
            "max_iterations": self.max_iterations,
            "body": self.body.to_dict(),
        }
