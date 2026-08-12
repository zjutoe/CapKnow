from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .program import ConditionNode, LoopNode, PrimitiveNode, Program, SequenceNode
from .primitives import get_primitive


class InvalidProgramError(ValueError):
    """Raised when a DSL program violates execution preconditions."""


class MissingCapabilityError(InvalidProgramError):
    """Raised when a structurally valid program needs an unavailable primitive."""


@dataclass(frozen=True)
class ExecutionResult:
    value: object
    output_type: str


def execute(
    program: Program,
    state_has: Mapping[str, bool],
    input_context: object | None = None,
) -> ExecutionResult:
    if not isinstance(state_has, Mapping):
        raise ValueError("state_has must be a mapping from task name to boolean.")
    _validate_program(program)
    return _execute_validated(program, state_has, input_context)


def _execute_validated(
    program: Program,
    state_has: Mapping[str, bool],
    input_context: object | None = None,
) -> ExecutionResult:
    if isinstance(program, PrimitiveNode):
        _require_capability(program.op.op_id, state_has)
        primitive = get_primitive(program.op.op_id)
        return ExecutionResult(
            value=_execute_primitive(program.op.op_id, input_context),
            output_type=primitive.output_type,
        )

    if isinstance(program, SequenceNode):
        current_input = input_context
        result: ExecutionResult | None = None
        for step in program.steps:
            result = _execute_validated(step, state_has, input_context=current_input)
            current_input = result.value
        if result is None:
            raise InvalidProgramError("SequenceNode requires at least one step.")
        return result

    if isinstance(program, ConditionNode):
        _require_capability("CONDITION", state_has)
        cond = _execute_validated(program.condition, state_has, input_context=input_context)
        if not isinstance(cond.value, bool):
            raise InvalidProgramError("ConditionNode condition must produce a bool.")
        if cond.value:
            return _execute_validated(program.then, state_has, input_context=input_context)
        if program.otherwise is None:
            return ExecutionResult(value=False, output_type="bool")
        return _execute_validated(program.otherwise, state_has, input_context=input_context)

    if isinstance(program, LoopNode):
        _require_capability("LOOP", state_has)
        current_input = input_context
        result: ExecutionResult | None = None
        for _ in range(program.max_iterations):
            result = _execute_validated(program.body, state_has, input_context=current_input)
            current_input = result.value
        if result is None:
            return ExecutionResult(value=current_input, output_type="loop_value")
        return result

    raise InvalidProgramError(f"Unsupported program node type '{type(program).__name__}'.")


def _validate_program(program: Program) -> None:
    if isinstance(program, PrimitiveNode):
        _validate_primitive(program)
        return

    if isinstance(program, SequenceNode):
        if len(program.steps) == 0:
            raise InvalidProgramError("SequenceNode requires at least one step.")
        if any(step is None for step in program.steps):
            raise InvalidProgramError("Sequence steps cannot be None.")
        for step in program.steps:
            _validate_program(step)
        return

    if isinstance(program, ConditionNode):
        if program.condition is None or program.then is None:
            raise InvalidProgramError("ConditionNode requires condition and then branch.")
        _validate_program(program.condition)
        _validate_program(program.then)
        if program.otherwise is not None:
            _validate_program(program.otherwise)
        return

    if isinstance(program, LoopNode):
        if type(program.max_iterations) is not int or program.max_iterations <= 0:
            raise InvalidProgramError("Loop max_iterations must be a positive integer.")
        if program.body is None:
            raise InvalidProgramError("LoopNode requires a body.")
        _validate_program(program.body)
        return

    raise InvalidProgramError(f"Unsupported program node type '{type(program).__name__}'.")


def _validate_primitive(node: PrimitiveNode) -> None:
    canonical = get_primitive(node.op.op_id)
    if node.op != canonical:
        raise InvalidProgramError(
            f"Primitive metadata for '{node.op.op_id}' does not match the registry."
        )


def _require_capability(op_id: str, state_has: Mapping[str, bool]) -> None:
    if op_id not in state_has:
        raise MissingCapabilityError(f"Missing required primitive capability '{op_id}'.")
    value = state_has[op_id]
    if type(value) is not bool:
        raise InvalidProgramError(f"Capability value for '{op_id}' must be a boolean.")
    if not value:
        raise MissingCapabilityError(f"Missing required primitive capability '{op_id}'.")


def _read_pair(input_context: object, op_id: str) -> tuple[object, object]:
    if isinstance(input_context, Mapping):
        if "left" in input_context and "right" in input_context:
            return input_context["left"], input_context["right"]
    if (
        isinstance(input_context, Sequence)
        and not isinstance(input_context, (str, bytes))
        and len(input_context) == 2
    ):
        return input_context[0], input_context[1]
    raise InvalidProgramError(f"{op_id} requires a two-value input.")


def _read_mapping(input_context: object, op_id: str) -> Mapping[str, object]:
    if not isinstance(input_context, Mapping):
        raise InvalidProgramError(f"{op_id} requires a mapping input.")
    return input_context


def _read_items(input_context: Mapping[str, object], op_id: str) -> Sequence[object]:
    items = input_context.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise InvalidProgramError(f"{op_id} requires an 'items' sequence.")
    return items


def _target_from_context(input_context: Mapping[str, object], op_id: str) -> object:
    if "target" in input_context:
        return input_context["target"]
    if "value" in input_context:
        return input_context["value"]
    raise InvalidProgramError(f"{op_id} requires 'target' or 'value'.")


def _execute_primitive(op_id: str, input_context: object) -> object:
    if op_id == "ADD":
        left, right = _read_pair(input_context, op_id)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise InvalidProgramError("ADD requires numeric inputs.")
        return left + right

    if op_id == "COMPARE":
        left, right = _read_pair(input_context, op_id)
        return left == right

    if op_id == "MEMORY":
        context = _read_mapping(input_context, op_id)
        memory = context.get("memory")
        key = context.get("key")
        if not isinstance(memory, Mapping):
            raise InvalidProgramError("MEMORY requires a 'memory' mapping.")
        if key not in memory:
            raise InvalidProgramError("MEMORY key must exist in memory.")
        next_context = dict(context)
        next_context["value"] = memory[key]
        next_context.setdefault("target", memory[key])
        return next_context

    if op_id == "SEARCH":
        context = _read_mapping(input_context, op_id)
        items = _read_items(context, op_id)
        target = _target_from_context(context, op_id)
        return target in items

    if op_id == "FILTER":
        context = _read_mapping(input_context, op_id)
        items = _read_items(context, op_id)
        target = _target_from_context(context, op_id)
        return [item for item in items if item == target]

    if op_id == "LOOP":
        return input_context

    if op_id == "CONDITION":
        if isinstance(input_context, Mapping) and "condition" in input_context:
            return bool(input_context["condition"])
        return bool(input_context)

    raise InvalidProgramError(f"Unsupported primitive operation '{op_id}'.")
