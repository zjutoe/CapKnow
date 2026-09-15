from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from capability_certificate_lab.certificate import (
    solve_adaptive_certificate,
    solve_exact_certificate,
    validate_adaptive_certificate,
)
from capability_certificate_lab.generators import (
    generate_chain_world,
    generate_independent_block_world,
    generate_prefix_block_world,
    generate_tree_world,
    generate_unstructured_world,
)
from capability_certificate_lab.knowledge_space.space import KnowledgeSpace
from capability_certificate_lab.knowledge_space.state import KnowledgeState
from capability_certificate_lab.knowledge_space.tasks import TaskUniverse
from capability_certificate_lab.validation.identifiability.core import (
    check_identifiability,
    stable_state_id,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "artifacts" / "phase7_structural_compressibility"
RESULT_FILENAME = "structural_compressibility.json"
MANIFEST_FILENAME = "manifest.json"
COMMAND = "PYTHONPATH=. python scripts/phase7_structural_compressibility.py"
BASELINE_COMMIT = "02380501e5def5d9f624578a93bc0580de1e030a"
IMPLEMENTATION_START_COMMIT = "a727c5e9dd8619e0d8135b8530af691d5a9e6694"
HANDOFF_PATH = "docs/Capability_Certificate_Task_Handoff_009_Phase7_Structural_Compressibility.md"
GRID = ((2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (2, 3), (3, 3), (4, 3))
MATCHED_CONTROL_SEEDS = tuple(range(20))
ABS_TOL = 1e-12


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _git(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def source_binding_before_run(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if output_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing output root: {_relative(output_root)}")

    status = _git(["status", "--porcelain", "--untracked-files=all"])
    if status:
        raise RuntimeError(
            "Formal Phase 7 execution requires clean committed source before "
            f"creating outputs. Dirty status:\n{status}"
        )

    return {
        "baseline_commit": BASELINE_COMMIT,
        "implementation_start_commit": IMPLEMENTATION_START_COMMIT,
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_status_porcelain_before_run": status,
        "clean_worktree_before_run": True,
        "source_binding": "git_commit",
        "handoff_path": HANDOFF_PATH,
        "script_path": "scripts/phase7_structural_compressibility.py",
        "command": COMMAND,
    }


def verify_source_after_publish(source: Mapping[str, Any], output_root: Path = OUTPUT_ROOT) -> str:
    current_head = _git(["rev-parse", "HEAD"])
    if current_head != source["git_head"]:
        raise RuntimeError("Source commit changed during Phase 7 formal execution.")

    output_path = _relative(output_root)
    status = _git(
        [
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            ".",
            f":(exclude){output_path}",
            f":(exclude){output_path}/**",
        ]
    )
    if status:
        raise RuntimeError(
            "Only the newly created Phase 7 output root may differ after execution. "
            f"Dirty status:\n{status}"
        )
    return status


def ordered_declared_states(knowledge_space: KnowledgeSpace) -> list[KnowledgeState]:
    task_ids = list(knowledge_space.tasks.task_ids)
    declared_states = list(knowledge_space.valid_states)
    if not declared_states:
        raise ValueError("Declared valid_states must not be empty.")

    seen: set[KnowledgeState] = set()
    duplicates: list[str] = []
    for state in declared_states:
        if state in seen:
            duplicates.append(stable_state_id(state, task_ids))
        seen.add(state)
    if duplicates:
        raise ValueError(f"Duplicate declared states are not allowed: {duplicates}")

    invalid = [
        stable_state_id(state, task_ids)
        for state in declared_states
        if not knowledge_space.is_valid_state(state)
    ]
    if invalid:
        raise ValueError(f"Declared states violate world rules: {invalid}")

    return sorted(declared_states, key=lambda state: state.as_tuple(task_ids))


def canonical_state_signatures(knowledge_space: KnowledgeSpace) -> list[str]:
    task_ids = list(knowledge_space.tasks.task_ids)
    return [stable_state_id(state, task_ids) for state in ordered_declared_states(knowledge_space)]


def response_column_diagnostics(knowledge_space: KnowledgeSpace) -> dict[str, Any]:
    task_ids = list(knowledge_space.tasks.task_ids)
    states = ordered_declared_states(knowledge_space)
    columns: dict[str, list[int]] = {
        task_id: [1 if task_id in state else 0 for state in states]
        for task_id in task_ids
    }

    classes: list[dict[str, Any]] = []
    for task_id in task_ids:
        column = columns[task_id]
        for item in classes:
            if item["column"] == column:
                item["tasks"].append(task_id)
                break
        else:
            classes.append({"tasks": [task_id], "column": column})

    return {
        "task_order": task_ids,
        "state_order": [stable_state_id(state, task_ids) for state in states],
        "columns": columns,
        "equivalence_classes": classes,
        "equivalence_class_count": len(classes),
    }


def single_coordinate_witness_diagnostics(knowledge_space: KnowledgeSpace) -> dict[str, Any]:
    task_ids = list(knowledge_space.tasks.task_ids)
    states = ordered_declared_states(knowledge_space)
    state_ids = [stable_state_id(state, task_ids) for state in states]
    witnesses: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in task_ids}

    for left_index, left_state in enumerate(states):
        for right_index in range(left_index + 1, len(states)):
            right_state = states[right_index]
            diff = [
                task_id
                for task_id in task_ids
                if (task_id in left_state) != (task_id in right_state)
            ]
            if len(diff) == 1:
                witnesses[diff[0]].append(
                    {
                        "left_index": left_index,
                        "right_index": right_index,
                        "left_state_id": state_ids[left_index],
                        "right_state_id": state_ids[right_index],
                    }
                )

    indispensable = [task_id for task_id in task_ids if witnesses[task_id]]
    return {
        "task_order": task_ids,
        "state_order": state_ids,
        "witnesses": witnesses,
        "witness_count_by_task": {
            task_id: len(witnesses[task_id]) for task_id in task_ids
        },
        "indispensable_tasks": indispensable,
        "all_tasks_indispensable": indispensable == task_ids,
    }


def exhaustive_structured_population_closure(
    knowledge_space: KnowledgeSpace,
    *,
    max_task_count: int = 12,
) -> dict[str, Any]:
    task_ids = list(knowledge_space.tasks.task_ids)
    if len(task_ids) > max_task_count:
        raise ValueError(
            f"Exhaustive closure oracle is bounded to n <= {max_task_count}; got {len(task_ids)}."
        )

    declared = {
        stable_state_id(state, task_ids) for state in knowledge_space.valid_states
    }
    accepted: set[str] = set()
    for mask in range(1 << len(task_ids)):
        selected = [
            task_ids[task_index]
            for task_index in range(len(task_ids))
            if (mask >> task_index) & 1
        ]
        state = KnowledgeState(selected)
        if knowledge_space.is_valid_state(state):
            accepted.add(stable_state_id(state, task_ids))

    declared_not_accepted = sorted(declared - accepted)
    accepted_not_declared = sorted(accepted - declared)
    return {
        "task_count": len(task_ids),
        "enumerated_subset_count": 1 << len(task_ids),
        "declared_state_count": len(declared),
        "accepted_state_count": len(accepted),
        "matches": not declared_not_accepted and not accepted_not_declared,
        "declared_not_accepted": declared_not_accepted,
        "accepted_not_declared": accepted_not_declared,
    }


def fixed_metric_diagnostics(certificate_size: int, task_count: int, state_count: int) -> dict[str, Any]:
    information_lower_bound = math.ceil(math.log2(state_count)) if state_count > 1 else 0
    return {
        "fixed_task_ratio": certificate_size / task_count if task_count else 0.0,
        "fixed_task_savings": task_count - certificate_size,
        "information_lower_bound": information_lower_bound,
        "fixed_excess_over_information_bound": certificate_size - information_lower_bound,
    }


def validate_fixed_certificate_independent(
    knowledge_space: KnowledgeSpace,
    selected_tasks: Sequence[str],
) -> dict[str, Any]:
    task_ids = list(knowledge_space.tasks.task_ids)
    task_set = set(task_ids)
    selected = list(selected_tasks)
    unknown = [task_id for task_id in selected if task_id not in task_set]
    if unknown:
        raise ValueError(f"Certificate contains unknown tasks: {unknown}")
    if len(selected) != len(set(selected)):
        raise ValueError(f"Certificate contains duplicate tasks: {selected}")

    states = ordered_declared_states(knowledge_space)
    state_ids = [stable_state_id(state, task_ids) for state in states]
    restricted_signatures = [
        [1 if task_id in state else 0 for task_id in selected]
        for state in states
    ]
    separated_pairs = 0
    total_pairs = len(states) * (len(states) - 1) // 2
    for left_index, left_signature in enumerate(restricted_signatures):
        for right_index in range(left_index + 1, len(restricted_signatures)):
            if left_signature != restricted_signatures[right_index]:
                separated_pairs += 1

    return {
        "selected_tasks": selected,
        "valid": len({tuple(signature) for signature in restricted_signatures}) == len(states),
        "state_order": state_ids,
        "restricted_signatures": restricted_signatures,
        "separated_pairs": separated_pairs,
        "total_pairs": total_pairs,
    }


def block_coverage_diagnostics(
    blocks: Sequence[Sequence[str]],
    selected_tasks: Sequence[str],
) -> dict[str, Any]:
    selected = set(selected_tasks)
    known = {task_id for block in blocks for task_id in block}
    coverage = [
        {
            "block_index": block_index,
            "block_tasks": list(block),
            "selected_tasks": [task_id for task_id in block if task_id in selected],
        }
        for block_index, block in enumerate(blocks)
    ]
    for item in coverage:
        item["selected_count"] = len(item["selected_tasks"])

    return {
        "coverage": coverage,
        "unknown_selected_tasks": sorted(selected - known),
        "all_blocks_covered_once": all(item["selected_count"] == 1 for item in coverage)
        and not (selected - known),
    }


def reconstruct_adaptive_tree_metrics(
    tree_payload: Mapping[str, Any],
    knowledge_space: KnowledgeSpace,
) -> dict[str, Any]:
    task_ids = list(knowledge_space.tasks.task_ids)
    task_set = set(task_ids)
    states = ordered_declared_states(knowledge_space)
    state_ids = [stable_state_id(state, task_ids) for state in states]
    errors: list[str] = []
    per_state_depths: dict[str, int] = {}

    def walk(
        node: Mapping[str, Any] | None,
        candidates: Sequence[KnowledgeState],
        asked: set[str],
        depth: int,
    ) -> tuple[int, int]:
        if not isinstance(node, Mapping):
            errors.append(f"Expected tree node mapping at depth {depth}.")
            return 0, 0

        question = node.get("question")
        if question is None:
            candidate_state_ids = list(node.get("candidate_state_ids") or [])
            expected_state_ids = sorted(stable_state_id(state, task_ids) for state in candidates)
            if len(candidates) != 1:
                errors.append(
                    f"Leaf at depth {depth} covers {len(candidates)} candidate states."
                )
            if candidate_state_ids != expected_state_ids:
                errors.append(
                    f"Leaf state ids {candidate_state_ids} do not match expected {expected_state_ids}."
                )
            for state_id in candidate_state_ids:
                if state_id in per_state_depths:
                    errors.append(f"State {state_id} appears in multiple leaves.")
                per_state_depths[state_id] = depth
            return 1, 1

        if not isinstance(question, str):
            errors.append(f"Internal node question at depth {depth} is not a string.")
            return 1, 0
        if question not in task_set:
            errors.append(f"Unknown adaptive question {question!r}.")
            return 1, 0
        if question in asked:
            errors.append(f"Repeated adaptive question {question!r}.")
            return 1, 0

        yes_candidates = [state for state in candidates if question in state]
        no_candidates = [state for state in candidates if question not in state]
        if not yes_candidates or not no_candidates:
            errors.append(f"Question {question!r} does not split candidates at depth {depth}.")

        next_asked = set(asked)
        next_asked.add(question)
        yes_node_count, yes_leaf_count = walk(
            node.get("yes_child"),
            yes_candidates,
            next_asked,
            depth + 1,
        )
        no_node_count, no_leaf_count = walk(
            node.get("no_child"),
            no_candidates,
            next_asked,
            depth + 1,
        )
        return 1 + yes_node_count + no_node_count, yes_leaf_count + no_leaf_count

    node_count, leaf_count = walk(tree_payload, states, set(), 0)
    missing_state_ids = [state_id for state_id in state_ids if state_id not in per_state_depths]
    unexpected_state_ids = sorted(set(per_state_depths) - set(state_ids))
    if missing_state_ids:
        errors.append(f"States missing from leaves: {missing_state_ids}")
    if unexpected_state_ids:
        errors.append(f"Unknown leaf state ids: {unexpected_state_ids}")

    complete_depths = {
        state_id: per_state_depths[state_id]
        for state_id in state_ids
        if state_id in per_state_depths
    }
    average_depth = (
        sum(complete_depths.values()) / len(state_ids)
        if len(complete_depths) == len(state_ids)
        else 0.0
    )
    worst_case_depth = max(complete_depths.values()) if complete_depths else 0

    return {
        "valid": not errors,
        "node_count": node_count,
        "leaf_count": leaf_count,
        "per_state_depths": complete_depths,
        "average_depth": average_depth,
        "worst_case_depth": worst_case_depth,
        "errors": errors,
    }


def adaptive_diagnostics(knowledge_space: KnowledgeSpace, policy: str) -> dict[str, Any]:
    result = solve_adaptive_certificate(knowledge_space, policy=policy)
    payload = result.to_dict()
    reconstructed = reconstruct_adaptive_tree_metrics(payload["root"], knowledge_space)
    independently_valid = reconstructed["valid"]
    public_validator_valid = validate_adaptive_certificate(result.root, knowledge_space)
    metrics_agree = (
        reconstructed["node_count"] == result.node_count
        and reconstructed["worst_case_depth"] == result.worst_case_depth
        and abs(reconstructed["average_depth"] - result.average_depth) <= ABS_TOL
    )
    return {
        "solver": payload,
        "independent_validator_valid": independently_valid,
        "public_validator_valid": public_validator_valid,
        "reconstructed_metrics": reconstructed,
        "solver_metrics_agree_with_reconstruction": metrics_agree,
    }


def uniform_blocks(block_count: int, block_size: int) -> list[list[str]]:
    return [
        [f"B{block_index}_T{task_index}" for task_index in range(block_size)]
        for block_index in range(block_count)
    ]


def theoretical_state_count(family: str, block_count: int) -> int:
    if family == "independent_block":
        return 1 << block_count
    if family == "prefix_block":
        return block_count + 1
    raise ValueError(f"Unknown block family: {family}")


def _assert_structured_adaptive_oracles(
    *,
    family: str,
    block_count: int,
    exact_certificate_size: int,
    adaptive: Mapping[str, Any],
) -> None:
    solver = adaptive["solver"]
    reconstructed = adaptive["reconstructed_metrics"]
    _require(bool(solver["valid"]), f"{family} B={block_count} adaptive solver invalid.")
    _require(
        bool(adaptive["public_validator_valid"]),
        f"{family} B={block_count} public adaptive validator failed.",
    )
    _require(
        bool(adaptive["independent_validator_valid"]),
        f"{family} B={block_count} independent adaptive validator failed.",
    )
    _require(
        bool(adaptive["solver_metrics_agree_with_reconstruction"]),
        f"{family} B={block_count} adaptive metrics reconstruction mismatch.",
    )

    if family == "independent_block":
        _require(
            reconstructed["worst_case_depth"] == block_count,
            f"Independent block worst-case depth mismatch for B={block_count}.",
        )
        _require(
            abs(reconstructed["average_depth"] - block_count) <= ABS_TOL,
            f"Independent block average depth mismatch for B={block_count}.",
        )
        return

    expected_worst = math.ceil(math.log2(block_count + 1))
    expected_leaves = block_count + 1
    _require(
        reconstructed["worst_case_depth"] == expected_worst,
        f"Prefix block worst-case depth mismatch for B={block_count}.",
    )
    _require(
        reconstructed["leaf_count"] == expected_leaves,
        f"Prefix block leaf count mismatch for B={block_count}.",
    )
    _require(
        reconstructed["node_count"] == 2 * expected_leaves - 1,
        f"Prefix block node count mismatch for B={block_count}.",
    )
    _require(
        reconstructed["average_depth"] < exact_certificate_size,
        f"Prefix block average depth is not below fixed size for B={block_count}.",
    )
    if block_count == 2:
        _require(
            reconstructed["worst_case_depth"] == exact_certificate_size,
            "Prefix B=2 worst-case depth must equal fixed size.",
        )
    else:
        _require(
            reconstructed["worst_case_depth"] < exact_certificate_size,
            f"Prefix B={block_count} worst-case depth is not below fixed size.",
        )


def analyze_structured_cell(family: str, block_count: int, block_size: int) -> dict[str, Any]:
    blocks = uniform_blocks(block_count, block_size)
    if family == "independent_block":
        world = generate_independent_block_world(blocks)
    elif family == "prefix_block":
        world = generate_prefix_block_world(blocks)
    else:
        raise ValueError(f"Unknown structured family: {family}")

    task_count = len(world.tasks.task_ids)
    state_count = len(world.valid_states)
    expected_state_count = theoretical_state_count(family, block_count)
    identifiability = check_identifiability(world).to_dict()
    exact = solve_exact_certificate(world)
    fixed_validation = validate_fixed_certificate_independent(world, exact.selected_tasks)
    closure = exhaustive_structured_population_closure(world)
    columns = response_column_diagnostics(world)
    witnesses = single_coordinate_witness_diagnostics(world)
    block_coverage = block_coverage_diagnostics(blocks, exact.selected_tasks)
    fixed_metrics = fixed_metric_diagnostics(
        exact.certificate_size,
        task_count,
        state_count,
    )
    adaptive = {
        policy: adaptive_diagnostics(world, policy)
        for policy in ("entropy",)
    }

    _require(identifiability["identifiable"], f"{family} {block_count}x{block_size} not identifiable.")
    _require(closure["matches"], f"{family} {block_count}x{block_size} closure mismatch.")
    _require(
        state_count == expected_state_count,
        f"{family} {block_count}x{block_size} state count mismatch.",
    )
    _require(exact.valid, f"{family} {block_count}x{block_size} exact solver invalid.")
    _require(
        fixed_validation["valid"],
        f"{family} {block_count}x{block_size} independent fixed validator failed.",
    )
    _require(
        exact.certificate_size == block_count,
        f"{family} {block_count}x{block_size} fixed size mismatch.",
    )
    _require(
        block_coverage["all_blocks_covered_once"],
        f"{family} {block_count}x{block_size} certificate does not hit each block once.",
    )
    _require(
        columns["equivalence_class_count"] == block_count,
        f"{family} {block_count}x{block_size} response-column class mismatch.",
    )
    _require(
        [item["tasks"] for item in columns["equivalence_classes"]] == blocks,
        f"{family} {block_count}x{block_size} response-column classes do not match blocks.",
    )
    for policy, policy_result in adaptive.items():
        _assert_structured_adaptive_oracles(
            family=family,
            block_count=block_count,
            exact_certificate_size=exact.certificate_size,
            adaptive=policy_result,
        )

    return {
        "family": family,
        "block_count": block_count,
        "block_size": block_size,
        "blocks": blocks,
        "world": world.to_dict(),
        "canonical_state_signatures": canonical_state_signatures(world),
        "identifiability": identifiability,
        "exact": exact.to_dict(),
        "independent_fixed_validator": fixed_validation,
        "fixed_metrics": fixed_metrics,
        "closure": closure,
        "response_columns": columns,
        "single_coordinate_witnesses": witnesses,
        "block_coverage": block_coverage,
        "adaptive": adaptive,
        "structured_oracles_passed": True,
        "matched_controls": analyze_matched_controls(
            family=family,
            block_count=block_count,
            block_size=block_size,
            task_count=task_count,
            state_count=state_count,
        ),
    }


def _column_bits(column: int, state_count: int) -> list[int]:
    return [(column >> row_index) & 1 for row_index in range(state_count)]


def validate_matched_control_population(knowledge_space: KnowledgeSpace) -> dict[str, Any]:
    if knowledge_space.generator_rules.get("prerequisites", {}) not in ({}, None):
        raise ValueError("Matched controls must not define prerequisite metadata.")
    if knowledge_space.generator_rules.get("composition_constraints") is not None:
        raise ValueError("Matched controls must not define composition constraints.")

    task_ids = list(knowledge_space.tasks.task_ids)
    states = ordered_declared_states(knowledge_space)
    columns = response_column_diagnostics(knowledge_space)
    column_values = [tuple(columns["columns"][task_id]) for task_id in task_ids]
    state_count = len(states)

    constant_tasks = [
        task_ids[index]
        for index, column in enumerate(column_values)
        if sum(column) in (0, state_count)
    ]
    if constant_tasks:
        raise ValueError(f"Matched controls contain constant task columns: {constant_tasks}")
    if len(set(column_values)) != len(column_values):
        raise ValueError("Matched controls contain duplicate task columns.")

    identifiability = check_identifiability(knowledge_space)
    if not identifiability.identifiable:
        raise ValueError("Matched controls must have pairwise distinct state rows.")

    return {
        "task_count": len(task_ids),
        "state_count": state_count,
        "canonical_state_signatures": [stable_state_id(state, task_ids) for state in states],
        "non_constant_columns": True,
        "distinct_columns": True,
        "distinct_state_rows": True,
        "identifiability": identifiability.to_dict(),
    }


def generate_matched_control_world(
    task_count: int,
    state_count: int,
    seed: int,
    *,
    max_attempts: int = 10_000,
) -> tuple[KnowledgeSpace, int, list[int]]:
    if task_count > (1 << state_count) - 2:
        raise ValueError(
            "Cannot draw the requested number of distinct non-constant columns "
            f"for task_count={task_count}, state_count={state_count}."
        )

    rng = random.Random(seed)
    task_ids = [f"Q{task_index}" for task_index in range(task_count)]
    all_ones = (1 << state_count) - 1
    for attempt in range(1, max_attempts + 1):
        columns: list[int] = []
        seen_columns: set[int] = set()
        while len(columns) < task_count:
            column = rng.getrandbits(state_count)
            if column == 0 or column == all_ones or column in seen_columns:
                continue
            seen_columns.add(column)
            columns.append(column)

        rows = []
        for row_index in range(state_count):
            rows.append(
                KnowledgeState(
                    task_ids[column_index]
                    for column_index, column in enumerate(columns)
                    if (column >> row_index) & 1
                )
            )
        if len(set(rows)) != state_count:
            continue

        world = KnowledgeSpace(
            tasks=TaskUniverse(task_ids),
            valid_states=rows,
            generator_rules={},
            metadata={
                "family": "matched_random_control",
                "seed": seed,
                "successful_attempt": attempt,
            },
        )
        validate_matched_control_population(world)
        return world, attempt, columns

    raise RuntimeError(
        "Failed to generate matched random control after "
        f"{max_attempts} complete-population attempts for seed {seed}."
    )


def analyze_matched_controls(
    *,
    family: str,
    block_count: int,
    block_size: int,
    task_count: int,
    state_count: int,
) -> dict[str, Any]:
    runs = []
    for seed in MATCHED_CONTROL_SEEDS:
        world, successful_attempt, columns = generate_matched_control_world(
            task_count,
            state_count,
            seed,
        )
        control_validation = validate_matched_control_population(world)
        exact = solve_exact_certificate(world)
        fixed_validation = validate_fixed_certificate_independent(world, exact.selected_tasks)
        _require(exact.valid, f"Matched control exact solver invalid for seed {seed}.")
        _require(
            fixed_validation["valid"],
            f"Matched control fixed validator failed for seed {seed}.",
        )
        runs.append(
            {
                "seed": seed,
                "successful_attempt": successful_attempt,
                "sampled_columns": [
                    {
                        "task_id": f"Q{column_index}",
                        "integer": column,
                        "bits_by_generation_row": _column_bits(column, state_count),
                    }
                    for column_index, column in enumerate(columns)
                ],
                "canonical_state_signatures": canonical_state_signatures(world),
                "state_population": world.to_dict(),
                "construction_constraints": control_validation,
                "exact": exact.to_dict(),
                "independent_fixed_validator": fixed_validation,
            }
        )

    sizes = [int(run["exact"]["certificate_size"]) for run in runs]
    return {
        "matched_to": {
            "family": family,
            "block_count": block_count,
            "block_size": block_size,
            "task_count": task_count,
            "state_count": state_count,
        },
        "seed_count": len(MATCHED_CONTROL_SEEDS),
        "seeds": list(MATCHED_CONTROL_SEEDS),
        "aggregation": "equal-weighted over deterministic seeds; descriptive only, no p-values or confidence intervals",
        "exact_certificate_size_aggregate": {
            "min": min(sizes),
            "median": statistics.median(sizes),
            "mean": statistics.mean(sizes),
            "max": max(sizes),
        },
        "runs": runs,
    }


def canonical_no_compression_controls() -> list[dict[str, Any]]:
    specs: list[tuple[str, KnowledgeSpace]] = []
    for task_count in (4, 6, 8):
        specs.append(
            (
                f"chain_n{task_count}",
                generate_chain_world([f"Q{index}" for index in range(task_count)]),
            )
        )
    specs.append(("accepted_tree_A_BC_BD", generate_tree_world({"A": ["B", "C"], "B": ["D"]})))
    for task_count in (4, 6, 8):
        specs.append(
            (
                f"unstructured_n{task_count}",
                generate_unstructured_world([f"Q{index}" for index in range(task_count)]),
            )
        )

    controls = []
    for name, world in specs:
        task_count = len(world.tasks.task_ids)
        identifiability = check_identifiability(world).to_dict()
        exact = solve_exact_certificate(world)
        fixed_validation = validate_fixed_certificate_independent(world, exact.selected_tasks)
        witnesses = single_coordinate_witness_diagnostics(world)
        _require(identifiability["identifiable"], f"{name} control is not identifiable.")
        _require(witnesses["all_tasks_indispensable"], f"{name} lacks a witness for every task.")
        _require(exact.valid, f"{name} exact solver invalid.")
        _require(
            exact.certificate_size == task_count,
            f"{name} exact certificate size does not equal task count.",
        )
        _require(fixed_validation["valid"], f"{name} fixed validator failed.")
        controls.append(
            {
                "name": name,
                "world": world.to_dict(),
                "canonical_state_signatures": canonical_state_signatures(world),
                "identifiability": identifiability,
                "exact": exact.to_dict(),
                "independent_fixed_validator": fixed_validation,
                "single_coordinate_witnesses": witnesses,
                "no_compression_oracles_passed": True,
            }
        )
    return controls


def run_phase7_analysis() -> dict[str, Any]:
    structured_cells = [
        analyze_structured_cell(family, block_count, block_size)
        for block_count, block_size in GRID
        for family in ("independent_block", "prefix_block")
    ]
    matched_control_count = sum(
        len(cell["matched_controls"]["runs"]) for cell in structured_cells
    )
    return {
        "phase": "phase7_structural_compressibility",
        "grid": [
            {
                "block_count": block_count,
                "block_size": block_size,
                "task_count": block_count * block_size,
                "independent_state_count": 1 << block_count,
                "prefix_state_count": block_count + 1,
            }
            for block_count, block_size in GRID
        ],
        "matched_control_seeds": list(MATCHED_CONTROL_SEEDS),
        "statistical_semantics": {
            "adaptive_state_weighting": "uniform over declared states",
            "structured_and_canonical_controls": "single deterministic run per cell",
            "matched_controls": "equal-weighted deterministic seeds, descriptive only",
            "no_pooling": "families and task/state-count cells are not pooled",
        },
        "structured_cells": structured_cells,
        "canonical_no_compression_controls": canonical_no_compression_controls(),
        "matched_control_population_count": matched_control_count,
        "structured_oracle_status": "passed",
        "hypotheses": {
            "structured_fixed_compression": "gate: exact fixed size equals one task per block for both complete block families",
            "prefix_adaptive_average": "gate: strictly below fixed size for all frozen prefix cells",
            "prefix_adaptive_worst_case": "gate: equal to fixed size for B=2 and strictly below for B>=3",
            "matched_control_position": "descriptive scientific outcome, not an engineering gate",
        },
        "limitations": [
            "synchronized blocks are constructed observational redundancy, not learned latent capabilities",
            "matched random controls do not identify a causal effect of structure",
            "the grid is small and exact, not a scalability result",
            "uniform state weighting is not a claim about real deployment priors",
            "deterministic membership observations do not establish noisy or real-model validity",
        ],
    }


def _manifest_bytes_with_stable_self_size(manifest: dict[str, Any]) -> bytes:
    manifest_entry = next(
        item for item in manifest["file_inventory"] if item["path"].endswith(MANIFEST_FILENAME)
    )
    previous_size = -1
    payload = canonical_json_bytes(manifest)
    while len(payload) != previous_size:
        previous_size = len(payload)
        manifest_entry["size_bytes"] = previous_size
        payload = canonical_json_bytes(manifest)
    return payload


def publish_outputs_atomically(
    *,
    result: Mapping[str, Any],
    source: Mapping[str, Any],
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    if output_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing output root: {_relative(output_root)}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    if staging_root.exists():
        raise RuntimeError(f"Refusing to reuse existing staging root: {_relative(staging_root)}")

    result_relative = f"{_relative(output_root)}/{RESULT_FILENAME}"
    manifest_relative = f"{_relative(output_root)}/{MANIFEST_FILENAME}"
    result_bytes = canonical_json_bytes(result)
    result_hash = sha256_bytes(result_bytes)

    manifest: dict[str, Any] = {
        "phase": "phase7_structural_compressibility",
        "script": "scripts/phase7_structural_compressibility.py",
        "command": COMMAND,
        "python_version": platform.python_version(),
        "config": {
            "grid": [
                {"block_count": block_count, "block_size": block_size}
                for block_count, block_size in GRID
            ],
            "matched_control_seeds": list(MATCHED_CONTROL_SEEDS),
            "output_root": _relative(output_root),
        },
        "source": dict(source),
        "output_root": {
            "path": _relative(output_root),
            "publication": "staged in sibling directory and atomically renamed into place",
        },
        "file_inventory": [
            {
                "path": result_relative,
                "role": "scientific_result",
                "sha256": result_hash,
                "size_bytes": len(result_bytes),
            },
            {
                "path": manifest_relative,
                "role": "manifest",
                "sha256": None,
                "sha256_note": "Manifest self-hash is reported after write; embedding it would change the manifest bytes.",
                "size_bytes": 0,
            },
        ],
    }
    manifest_bytes = _manifest_bytes_with_stable_self_size(manifest)

    try:
        staging_root.mkdir()
        (staging_root / RESULT_FILENAME).write_bytes(result_bytes)
        (staging_root / MANIFEST_FILENAME).write_bytes(manifest_bytes)
        staging_root.rename(output_root)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    post_publish_status = verify_source_after_publish(source, output_root)
    manifest_hash = file_sha256(output_root / MANIFEST_FILENAME)
    return {
        "result_path": result_relative,
        "result_sha256": result_hash,
        "manifest_path": manifest_relative,
        "manifest_sha256": manifest_hash,
        "output_root": _relative(output_root),
        "post_publish_git_status_excluding_output_root": post_publish_status,
    }


def main() -> None:
    source = source_binding_before_run()
    result = run_phase7_analysis()
    artifacts = publish_outputs_atomically(result=result, source=source)
    print(json.dumps(artifacts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
