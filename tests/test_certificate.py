from __future__ import annotations

from capability_certificate_lab.certificate import (
    solve_exact_certificate,
    solve_greedy_certificate,
    solve_random_certificate,
    validate_certificate,
)
from capability_certificate_lab.generators import generate_chain_world, generate_unstructured_world
from capability_certificate_lab.knowledge_space.space import KnowledgeSpace
from capability_certificate_lab.knowledge_space.state import KnowledgeState
from capability_certificate_lab.knowledge_space.tasks import TaskUniverse
from collections.abc import Sequence


def _collapsed_signature(state: KnowledgeState, task_ids: Sequence[str]) -> tuple[int, ...]:
    return (0,) if len(task_ids) >= 2 else ()


def test_chain_exact_certificate_is_minimal():
    space = generate_chain_world(["A", "B", "C", "D"])
    result = solve_exact_certificate(space)

    assert result.valid
    assert result.task_count == 4
    assert result.certificate_size == 4
    assert len(result.selected_tasks) == result.certificate_size
    assert validate_certificate(space, result.selected_tasks)


def test_independent_tasks_certificate_reaches_task_count():
    space = generate_unstructured_world(["Q0", "Q1", "Q2"])
    result = solve_exact_certificate(space)

    assert result.valid
    assert result.certificate_size == result.task_count == 3
    assert validate_certificate(space, result.selected_tasks)


def test_structured_space_certificate_is_smaller_than_task_count():
    tasks = TaskUniverse(["A", "B", "C", "D"])
    space = KnowledgeSpace(
        tasks=tasks,
        valid_states=[
            KnowledgeState(()),
            KnowledgeState(("D",)),
            KnowledgeState(("C",)),
            KnowledgeState(("C", "D")),
            KnowledgeState(("B",)),
            KnowledgeState(("B", "D")),
        ],
        metadata={"type": "structured"},
    )
    result = solve_exact_certificate(space)

    assert result.valid
    assert result.certificate_size < result.task_count
    assert result.certificate_size == 3


def test_non_identifiable_input_is_rejected():
    world = KnowledgeSpace(
        tasks=TaskUniverse(["A", "B"]),
        valid_states=[KnowledgeState(("A",)), KnowledgeState(("B",))],
    )
    result = solve_exact_certificate(world, response_signature_fn=_collapsed_signature)

    assert not result.valid
    assert result.certificate_size == 0


def test_greedy_and_random_are_valid_but_not_better_than_exact():
    space = generate_chain_world(["A", "B", "C"])
    exact = solve_exact_certificate(space)
    greedy = solve_greedy_certificate(space)
    random = solve_random_certificate(space, seed=123)

    assert exact.valid
    assert greedy.valid
    assert random.valid
    assert exact.certificate_size <= greedy.certificate_size
    assert exact.certificate_size <= random.certificate_size
