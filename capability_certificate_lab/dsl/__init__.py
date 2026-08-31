"""Executable DSL primitives and task world utilities."""

from .composition import (
    CompositionRule,
    DEFAULT_COMPOSITION_RULES,
    build_composite_program,
)
from .executor import ExecutionResult, InvalidProgramError, MissingCapabilityError, execute
from .primitives import PrimitiveOperation, PRIMITIVES, PRIMITIVE_TASKS, get_primitive
from .program import ConditionNode, LoopNode, PrimitiveNode, Program, SequenceNode
from .simulator import default_input_context, make_dsl_response_signature, simulate_task_response
from .task_generator import generate_dsl_world

__all__ = [
    "CompositionRule",
    "DEFAULT_COMPOSITION_RULES",
    "build_composite_program",
    "ExecutionResult",
    "InvalidProgramError",
    "MissingCapabilityError",
    "execute",
    "PrimitiveOperation",
    "PRIMITIVES",
    "PRIMITIVE_TASKS",
    "get_primitive",
    "ConditionNode",
    "LoopNode",
    "PrimitiveNode",
    "Program",
    "SequenceNode",
    "default_input_context",
    "make_dsl_response_signature",
    "simulate_task_response",
    "generate_dsl_world",
]
