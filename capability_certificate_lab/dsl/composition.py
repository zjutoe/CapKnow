from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from .program import Program, SequenceNode


@dataclass(frozen=True)
class CompositionRule:
    left: str
    right: str
    result: str

    def to_dict(self) -> dict[str, str]:
        return {"left": self.left, "right": self.right, "result": self.result}


@dataclass(frozen=True)
class CapabilityGraph:
    rules: tuple[CompositionRule, ...]

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        return {"rules": [rule.to_dict() for rule in self.rules]}


DEFAULT_COMPOSITION_RULES: tuple[CompositionRule, ...] = (
    CompositionRule("MEMORY", "SEARCH", "RETRIEVAL"),
    CompositionRule("RETRIEVAL", "CONDITION", "PLANNING"),
)


def build_task_name(rule: CompositionRule) -> str:
    return rule.result


def build_composite_program(rule: CompositionRule, left_program: Program, right_program: Program) -> Program:
    if left_program is None or right_program is None:
        raise ValueError("left_program and right_program must be provided for composition.")
    if not isinstance(left_program, Program) or not isinstance(right_program, Program):
        raise TypeError("left_program and right_program must be DSL programs.")
    return SequenceNode((left_program, right_program))


def resolve_composition(rules: tuple[CompositionRule, ...] = DEFAULT_COMPOSITION_RULES) -> Callable[[str, str], str | None]:
    lookup: dict[tuple[str, str], str] = {}
    for rule in rules:
        key = (rule.left, rule.right)
        if key in lookup and lookup[key] != rule.result:
            raise ValueError(
                f"Conflicting composition rules for {rule.left}+{rule.right}: "
                f"{lookup[key]} vs {rule.result}"
            )
        lookup[key] = rule.result

    def _resolver(a: str, b: str) -> str | None:
        return lookup.get((a, b))

    return _resolver
