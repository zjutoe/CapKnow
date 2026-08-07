from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class PrimitiveOperation:
    """Executable primitive with minimal static metadata."""

    op_id: str
    input_type: str
    output_type: str
    difficulty: float = 1.0
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "op_id": self.op_id,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "difficulty": self.difficulty,
            "requires": list(self.requires),
        }


def _primitive_id(op_id: str) -> PrimitiveOperation:
    return PrimitiveOperation(op_id=op_id, input_type="any", output_type="any")


PRIMITIVES: dict[str, PrimitiveOperation] = {
    "ADD": _primitive_id("ADD"),
    "COMPARE": _primitive_id("COMPARE"),
    "MEMORY": _primitive_id("MEMORY"),
    "SEARCH": _primitive_id("SEARCH"),
    "FILTER": _primitive_id("FILTER"),
    "LOOP": _primitive_id("LOOP"),
    "CONDITION": _primitive_id("CONDITION"),
}


__all__ = [
    "PrimitiveOperation",
    "PRIMITIVES",
    "PRIMITIVE_TASKS",
    "get_primitive",
]


PRIMITIVE_TASKS: tuple[str, ...] = tuple(PRIMITIVES.keys())


def get_primitive(op_id: str) -> PrimitiveOperation:
    if op_id not in PRIMITIVES:
        raise ValueError(f"Unknown primitive operation '{op_id}'.")
    return PRIMITIVES[op_id]


def list_primitives() -> Sequence[PrimitiveOperation]:
    return tuple(PRIMITIVES.values())

