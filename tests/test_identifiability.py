from __future__ import annotations

from typing import Sequence

from capability_certificate_lab.generators import (
    generate_chain_world,
    generate_tree_world,
    generate_unstructured_world,
)
from capability_certificate_lab.knowledge_space.space import KnowledgeSpace
from capability_certificate_lab.knowledge_space.state import KnowledgeState
from capability_certificate_lab.knowledge_space.tasks import TaskUniverse
from capability_certificate_lab.validation.identifiability import (
    check_identifiability,
    response_signature,
)


def test_chain_space_is_identifiable():
    space = generate_chain_world(["A", "B", "C", "D"])
    report = check_identifiability(space)

    assert report.identifiable
    assert report.num_tasks == 4
    assert report.num_states == 5
    assert report.num_unique_signatures == 5
    assert report.collision_count == 0
    assert report.compression_ratio == 1.0


def _collapsed_signature(state: KnowledgeState, task_ids: Sequence[str]) -> tuple[int, ...]:
    return (1,) if task_ids else ()


def test_artificial_collision_is_detected():
    world = KnowledgeSpace(
        tasks=TaskUniverse(["A", "B"]),
        valid_states=[KnowledgeState(("A",)), KnowledgeState(("B",))],
        metadata={"type": "artificial_collision"},
    )
    report = check_identifiability(world, response_signature_fn=_collapsed_signature)

    assert not report.identifiable
    assert report.collision_count == 1
    assert report.collision_groups == [["{A}", "{B}"]]


def test_tree_space_is_identifiable():
    space = generate_tree_world({"A": ["B", "C"]})
    report = check_identifiability(space)

    assert report.identifiable
    assert report.collision_count == 0


def test_unstructured_state_signatures_are_unique():
    world = generate_unstructured_world(["Q0", "Q1", "Q2", "Q3", "Q4"])
    report = check_identifiability(world, response_signature)

    assert report.identifiable
    assert report.num_states == 32
    assert report.num_unique_signatures == 32
    assert report.compression_ratio == 1.0
