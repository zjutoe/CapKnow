from __future__ import annotations

from collections.abc import Mapping, Sequence
from random import Random

from ..knowledge_space.state import KnowledgeState
from ..validation.identifiability.core import Signature
from ..probabilistic.response_model import ResponseNoiseModel, simulate_probabilistic_response
from .executor import execute
from .program import Program


def simulate_task_response(
    state: KnowledgeState,
    program: Program,
    noise: ResponseNoiseModel | None = None,
    rng: Random | None = None,
) -> int:
    state_has = {task_id: task_id in state for task_id in state.tasks}
    base = execute(program, state_has)
    if noise is None:
        return int(base)
    return simulate_probabilistic_response(
        state_has_task=bool(base == 1),
        noise=noise,
        attempts=1,
        rng=rng,
    )


def make_dsl_response_signature(
    task_programs: Mapping[str, Program],
    noise: ResponseNoiseModel | None = None,
    rng: Random | None = None,
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
                noise=noise,
                rng=rng,
            )
            for task_id in ordered
        )

    return _signature
