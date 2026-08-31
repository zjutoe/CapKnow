from __future__ import annotations

from dataclasses import dataclass

from .program import Program, SequenceNode


@dataclass(frozen=True)
class CompositionRule:
    left: str
    right: str
    result: str

    def to_dict(self) -> dict[str, str]:
        return {"left": self.left, "right": self.right, "result": self.result}


DEFAULT_COMPOSITION_RULES: tuple[CompositionRule, ...] = (
    CompositionRule("MEMORY", "SEARCH", "RETRIEVAL"),
    CompositionRule("RETRIEVAL", "CONDITION", "PLANNING"),
)


def build_composite_program(rule: CompositionRule, left_program: Program, right_program: Program) -> Program:
    if left_program is None or right_program is None:
        raise ValueError("left_program and right_program must be provided for composition.")
    if not isinstance(left_program, Program) or not isinstance(right_program, Program):
        raise TypeError("left_program and right_program must be DSL programs.")
    return SequenceNode((left_program, right_program))
