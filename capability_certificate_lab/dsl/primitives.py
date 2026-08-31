from __future__ import annotations

from dataclasses import dataclass


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


def _primitive_id(op_id: str, input_type: str, output_type: str) -> PrimitiveOperation:
    return PrimitiveOperation(op_id=op_id, input_type=input_type, output_type=output_type)


PRIMITIVES: dict[str, PrimitiveOperation] = {
    "ADD": _primitive_id("ADD", "numeric_pair", "number"),
    "COMPARE": _primitive_id("COMPARE", "comparison_pair", "bool"),
    "MEMORY": _primitive_id("MEMORY", "memory_lookup", "memory_context"),
    "SEARCH": _primitive_id("SEARCH", "search_context", "bool"),
    "FILTER": _primitive_id("FILTER", "filter_context", "list"),
    "LOOP": _primitive_id("LOOP", "loop_value", "loop_value"),
    "CONDITION": _primitive_id("CONDITION", "condition_value", "bool"),
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
