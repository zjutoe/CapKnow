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
from capability_certificate_lab.validation.identifiability.core import stable_state_id
import pytest


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
    return (1,) * len(task_ids)


def _too_short_signature(state: KnowledgeState, task_ids: Sequence[str]) -> tuple[int, ...]:
    return (1,) * max(0, len(task_ids) - 1)


def _too_long_signature(state: KnowledgeState, task_ids: Sequence[str]) -> tuple[int, ...]:
    return (1,) * (len(task_ids) + 1)


def _non_binary_signature(state: KnowledgeState, task_ids: Sequence[str]) -> tuple[int, ...]:
    return tuple(2 for _ in task_ids)


def test_artificial_collision_is_detected():
    world = KnowledgeSpace(
        tasks=TaskUniverse(["A", "B"]),
        valid_states=[KnowledgeState(("A",)), KnowledgeState(("B",))],
        metadata={"type": "artificial_collision"},
    )
    report = check_identifiability(world, response_signature_fn=_collapsed_signature)

    assert not report.identifiable
    assert report.collision_count == 1
    assert report.collision_groups == [['["A"]', '["B"]']]


def test_invalid_empty_and_duplicate_declared_states_raise():
    tasks = TaskUniverse(["A", "B"])

    with pytest.raises(ValueError, match="must not be empty"):
        check_identifiability(KnowledgeSpace(tasks=tasks, valid_states=[]))

    invalid_world = KnowledgeSpace(
        tasks=tasks,
        valid_states=[KnowledgeState(("Z",))],
    )
    with pytest.raises(ValueError, match="violate world rules"):
        check_identifiability(invalid_world)

    duplicate_world = KnowledgeSpace(
        tasks=tasks,
        valid_states=[KnowledgeState(("A",)), KnowledgeState(("A",))],
    )
    with pytest.raises(ValueError, match="Duplicate declared states"):
        check_identifiability(duplicate_world)


def test_malformed_response_signatures_raise():
    world = KnowledgeSpace(
        tasks=TaskUniverse(["A", "B"]),
        valid_states=[KnowledgeState(("A",)), KnowledgeState(("B",))],
    )

    with pytest.raises(ValueError, match="length"):
        check_identifiability(world, response_signature_fn=_too_short_signature)
    with pytest.raises(ValueError, match="length"):
        check_identifiability(world, response_signature_fn=_too_long_signature)
    with pytest.raises(ValueError, match="binary"):
        check_identifiability(world, response_signature_fn=_non_binary_signature)


def test_stable_state_id_is_unambiguous_for_special_task_ids():
    task_ids = ["A,B", "A", "B", "{brace}", "brace{}", 'quote"', "slash\\"]

    assert stable_state_id(KnowledgeState(("A,B",)), task_ids) != stable_state_id(
        KnowledgeState(("A", "B")),
        task_ids,
    )
    assert stable_state_id(KnowledgeState(("{brace}",)), task_ids) == '["{brace}"]'
    assert stable_state_id(KnowledgeState(("brace{}",)), task_ids) == '["brace{}"]'
    assert stable_state_id(KnowledgeState(("A,B", "{brace}")), task_ids) != stable_state_id(
        KnowledgeState(("A", "B", "{brace}")),
        task_ids,
    )
    assert stable_state_id(KnowledgeState(('quote"',)), task_ids) == '["quote\\""]'
    assert stable_state_id(KnowledgeState(("slash\\",)), task_ids) == '["slash\\\\"]'


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
