from __future__ import annotations

from collections.abc import Mapping, Sequence
from random import Random

from ..knowledge_space.state import KnowledgeState
from ..validation.identifiability.core import Signature
from ..probabilistic.response_model import ResponseNoiseModel, simulate_probabilistic_response
from .executor import InvalidProgramError, MissingCapabilityError, execute
from .program import Program


def default_input_context() -> dict[str, object]:
    return {
        "left": 1,
        "right": 1,
        "memory": {"needle": "needle"},
        "key": "needle",
        "items": ["needle", "other"],
        "target": "needle",
        "condition": True,
    }


def simulate_task_response(
    state: KnowledgeState,
    program: Program,
    input_context: object | None = None,
    noise: ResponseNoiseModel | None = None,
    rng: Random | None = None,
) -> int:
    state_has = {task_id: task_id in state for task_id in state.tasks}
    try:
        execute(
            program,
            state_has,
            input_context=default_input_context() if input_context is None else input_context,
        )
    except MissingCapabilityError:
        base = 0
    except InvalidProgramError:
        raise
    else:
        base = 1
    if noise is None:
        return base
    return simulate_probabilistic_response(
        state_has_task=bool(base == 1),
        noise=noise,
        attempts=1,
        rng=rng,
    )


def make_dsl_response_signature(
    task_programs: Mapping[str, Program],
    input_context: object | None = None,
):
    if not task_programs:
        raise ValueError("No DSL tasks defined for signature generation.")

    def _signature(state: KnowledgeState, ordered_task_ids: Sequence[str]) -> Signature:
        ordered = list(ordered_task_ids)
        missing = [task_id for task_id in ordered if task_id not in task_programs]
        if missing:
            raise ValueError(f"Missing task programs for tasks: {missing}")
        return tuple(
                simulate_task_response(
                    state,
                    task_programs[task_id],
                    input_context=input_context,
                )
                for task_id in ordered
        )

    return _signature
