from __future__ import annotations

import json
import math

import pytest

from capability_certificate_lab.certificate import solve_exact_certificate
from capability_certificate_lab.generators import (
    generate_independent_block_world,
    generate_prefix_block_world,
)
from capability_certificate_lab.knowledge_space.space import KnowledgeSpace
from capability_certificate_lab.knowledge_space.state import KnowledgeState
from capability_certificate_lab.knowledge_space.tasks import TaskUniverse
from scripts.phase7_structural_compressibility import (
    GRID,
    adaptive_diagnostics,
    block_coverage_diagnostics,
    canonical_no_compression_controls,
    canonical_state_signatures,
    exhaustive_structured_population_closure,
    fixed_metric_diagnostics,
    generate_matched_control_world,
    response_column_diagnostics,
    uniform_blocks,
    validate_fixed_certificate_independent,
    validate_matched_control_population,
)


def _expected_independent_states(blocks: list[list[str]]) -> list[KnowledgeState]:
    states = []
    for mask in range(1 << len(blocks)):
        selected = []
        for block_index, block in enumerate(blocks):
            if (mask >> block_index) & 1:
                selected.extend(block)
        states.append(KnowledgeState(selected))
    return states


def _expected_prefix_states(blocks: list[list[str]]) -> list[KnowledgeState]:
    states = [KnowledgeState(())]
    selected = []
    for block in blocks:
        selected = [*selected, *block]
        states.append(KnowledgeState(selected))
    return states


@pytest.mark.parametrize(
    "generator",
    [generate_independent_block_world, generate_prefix_block_world],
)
def test_block_generators_reject_empty_and_duplicate_inputs(generator):
    with pytest.raises(ValueError, match="must not be empty"):
        generator([])
    with pytest.raises(ValueError, match="must not be empty"):
        generator([["A"], []])
    with pytest.raises(ValueError, match="Duplicate task id"):
        generator([["A", "A"]])
    with pytest.raises(ValueError, match="Duplicate task id"):
        generator([["A"], ["B", "A"]])


def test_block_generators_preserve_task_block_and_state_ordering():
    blocks = [["A0", "A1"], ["B0", "B1"], ["C0"]]

    independent = generate_independent_block_world(blocks)
    assert independent.tasks.task_ids == ["A0", "A1", "B0", "B1", "C0"]
    assert independent.metadata["blocks"] == blocks
    assert independent.valid_states == _expected_independent_states(blocks)
    json.dumps(independent.to_dict())

    prefix = generate_prefix_block_world(blocks)
    assert prefix.tasks.task_ids == ["A0", "A1", "B0", "B1", "C0"]
    assert prefix.metadata["blocks"] == blocks
    assert prefix.valid_states == _expected_prefix_states(blocks)
    json.dumps(prefix.to_dict())


def test_block_state_populations_closure_and_representative_illegal_states():
    for block_count, block_size in GRID:
        blocks = uniform_blocks(block_count, block_size)
        independent = generate_independent_block_world(blocks)
        assert independent.valid_states == _expected_independent_states(blocks)
        assert len(independent.valid_states) == 1 << block_count
        assert all(independent.is_valid_state(state) for state in independent.valid_states)
        assert not independent.is_valid_state(KnowledgeState([blocks[0][0]]))
        independent_closure = exhaustive_structured_population_closure(independent)
        assert independent_closure["matches"], independent_closure

        prefix = generate_prefix_block_world(blocks)
        assert prefix.valid_states == _expected_prefix_states(blocks)
        assert len(prefix.valid_states) == block_count + 1
        assert all(prefix.is_valid_state(state) for state in prefix.valid_states)
        assert not prefix.is_valid_state(KnowledgeState([blocks[0][0]]))
        assert not prefix.is_valid_state(KnowledgeState(blocks[1]))
        prefix_closure = exhaustive_structured_population_closure(prefix)
        assert prefix_closure["matches"], prefix_closure


def test_response_column_equivalence_classes_are_exactly_blocks():
    for block_count, block_size in GRID:
        blocks = uniform_blocks(block_count, block_size)
        for world in (
            generate_independent_block_world(blocks),
            generate_prefix_block_world(blocks),
        ):
            diagnostics = response_column_diagnostics(world)
            assert diagnostics["equivalence_class_count"] == block_count
            assert [item["tasks"] for item in diagnostics["equivalence_classes"]] == blocks


def test_exact_fixed_size_and_block_coverage_formulas_across_grid():
    for block_count, block_size in GRID:
        blocks = uniform_blocks(block_count, block_size)
        task_count = block_count * block_size
        for world in (
            generate_independent_block_world(blocks),
            generate_prefix_block_world(blocks),
        ):
            exact = solve_exact_certificate(world)
            fixed_validation = validate_fixed_certificate_independent(
                world,
                exact.selected_tasks,
            )
            coverage = block_coverage_diagnostics(blocks, exact.selected_tasks)
            metrics = fixed_metric_diagnostics(
                exact.certificate_size,
                task_count,
                len(world.valid_states),
            )

            assert exact.valid
            assert exact.certificate_size == block_count
            assert fixed_validation["valid"]
            assert coverage["all_blocks_covered_once"]
            assert metrics["fixed_task_ratio"] == 1 / block_size
            assert metrics["fixed_task_savings"] == task_count - block_count


def test_independent_block_adaptive_depth_formula_across_grid():
    for block_count, block_size in GRID:
        world = generate_independent_block_world(uniform_blocks(block_count, block_size))
        for policy in ("entropy", "balanced"):
            diagnostics = adaptive_diagnostics(world, policy)
            reconstructed = diagnostics["reconstructed_metrics"]
            assert diagnostics["solver"]["valid"]
            assert diagnostics["public_validator_valid"]
            assert diagnostics["independent_validator_valid"]
            assert diagnostics["solver_metrics_agree_with_reconstruction"]
            assert reconstructed["worst_case_depth"] == block_count
            assert reconstructed["average_depth"] == block_count


def test_prefix_block_adaptive_tree_gates_across_grid():
    for block_count, block_size in GRID:
        world = generate_prefix_block_world(uniform_blocks(block_count, block_size))
        fixed = solve_exact_certificate(world)
        for policy in ("entropy", "balanced"):
            diagnostics = adaptive_diagnostics(world, policy)
            reconstructed = diagnostics["reconstructed_metrics"]
            expected_worst = math.ceil(math.log2(block_count + 1))
            expected_leaves = block_count + 1

            assert diagnostics["solver"]["valid"]
            assert diagnostics["public_validator_valid"]
            assert diagnostics["independent_validator_valid"]
            assert diagnostics["solver_metrics_agree_with_reconstruction"]
            assert reconstructed["leaf_count"] == expected_leaves
            assert reconstructed["node_count"] == 2 * expected_leaves - 1
            assert reconstructed["worst_case_depth"] == expected_worst
            assert reconstructed["average_depth"] < fixed.certificate_size
            if block_count == 2:
                assert reconstructed["worst_case_depth"] == fixed.certificate_size
            else:
                assert reconstructed["worst_case_depth"] < fixed.certificate_size


def test_canonical_no_compression_controls_have_single_coordinate_witnesses():
    controls = canonical_no_compression_controls()
    assert [control["name"] for control in controls] == [
        "chain_n4",
        "chain_n6",
        "chain_n8",
        "accepted_tree_A_BC_BD",
        "unstructured_n4",
        "unstructured_n6",
        "unstructured_n8",
    ]
    for control in controls:
        assert control["no_compression_oracles_passed"]
        assert control["single_coordinate_witnesses"]["all_tasks_indispensable"]
        assert control["exact"]["certificate_size"] == control["exact"]["task_count"]
        assert control["independent_fixed_validator"]["valid"]


def test_matched_control_generation_is_reproducible_and_enforces_constraints():
    first_world, first_attempt, first_columns = generate_matched_control_world(
        task_count=6,
        state_count=4,
        seed=7,
    )
    second_world, second_attempt, second_columns = generate_matched_control_world(
        task_count=6,
        state_count=4,
        seed=7,
    )

    assert first_attempt == second_attempt
    assert first_columns == second_columns
    assert canonical_state_signatures(first_world) == canonical_state_signatures(second_world)
    diagnostics = validate_matched_control_population(first_world)
    assert diagnostics["non_constant_columns"]
    assert diagnostics["distinct_columns"]
    assert diagnostics["distinct_state_rows"]
    assert diagnostics["identifiability"]["identifiable"]
    assert first_world.generator_rules == {}


def test_malformed_or_duplicate_row_controls_are_rejected():
    duplicate_row = KnowledgeSpace(
        tasks=TaskUniverse(["A", "B"]),
        valid_states=[KnowledgeState(["A"]), KnowledgeState(["A"])],
        generator_rules={},
    )
    with pytest.raises(ValueError, match="Duplicate declared states"):
        validate_matched_control_population(duplicate_row)

    duplicate_column = KnowledgeSpace(
        tasks=TaskUniverse(["A", "B"]),
        valid_states=[KnowledgeState(()), KnowledgeState(["A", "B"])],
        generator_rules={},
    )
    with pytest.raises(ValueError, match="duplicate task columns"):
        validate_matched_control_population(duplicate_column)
