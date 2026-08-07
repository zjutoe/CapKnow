from __future__ import annotations

from collections.abc import Mapping

from .program import ConditionNode, LoopNode, PrimitiveNode, Program, SequenceNode
from .primitives import get_primitive


class InvalidProgramError(ValueError):
    """Raised when a DSL program violates execution preconditions."""


def execute(program: Program, state_has: Mapping[str, bool], input_context: object | None = None) -> int:
    _ = input_context
    if not isinstance(state_has, Mapping):
        raise ValueError("state_has must be a mapping from task name to boolean.")

    if isinstance(program, PrimitiveNode):
        _validate_primitive(program)
        return 1 if program.op.op_id in state_has and state_has[program.op.op_id] else 0

    if isinstance(program, SequenceNode):
        if len(program.steps) == 0:
            return 1
        if any(step is None for step in program.steps):
            raise InvalidProgramError("Sequence steps cannot be None.")
        for step in program.steps:
            if execute(step, state_has, input_context=input_context) == 0:
                return 0
        return 1

    if isinstance(program, ConditionNode):
        if program.condition is None or program.then is None:
            raise InvalidProgramError("ConditionNode requires condition and then branch.")
        cond = execute(program.condition, state_has, input_context=input_context)
        if cond:
            return execute(program.then, state_has, input_context=input_context)
        if program.otherwise is None:
            return 1
        return execute(program.otherwise, state_has, input_context=input_context)

    if isinstance(program, LoopNode):
        if program.max_iterations <= 0:
            raise InvalidProgramError("Loop max_iterations must be a positive integer.")
        for _ in range(program.max_iterations):
            if execute(program.body, state_has, input_context=input_context) == 0:
                return 0
        return 1

    raise InvalidProgramError(f"Unsupported program node type '{type(program).__name__}'.")


def _validate_primitive(node: PrimitiveNode) -> None:
    _ = get_primitive(node.op.op_id)
