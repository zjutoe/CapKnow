from __future__ import annotations

import importlib
from math import log2
from pathlib import Path

from capability_certificate_lab.certificate import (
    AdaptiveCertificate,
    DecisionNode,
    solve_adaptive_certificate,
    solve_exact_certificate,
    tree_signature,
    validate_adaptive_certificate,
)
from capability_certificate_lab.certificate import policies
from capability_certificate_lab.knowledge_space.state import KnowledgeState
from capability_certificate_lab.knowledge_space import KnowledgeSpace, TaskUniverse
from capability_certificate_lab.certificate.policies import (
    resolve_policy,
    select_entropy_reduction_question,
)
from capability_certificate_lab.generators import (
    generate_chain_world,
    generate_tree_world,
    generate_unstructured_world,
)
from capability_certificate_lab.validation.identifiability.core import Signature
from collections.abc import Sequence
import pytest


def _is_internal(node: DecisionNode) -> bool:
    return node.question is not None


def test_chain_world_adaptive_improves_or_matches_fixed_depth():
    space = generate_chain_world(["A", "B", "C", "D"])
    fixed = solve_exact_certificate(space)
    adaptive = solve_adaptive_certificate(space, policy="entropy")

    assert isinstance(adaptive, AdaptiveCertificate)
    assert adaptive.valid
    assert adaptive.worst_case_depth <= fixed.certificate_size
    assert validate_adaptive_certificate(adaptive.root, space)


def test_tree_world_builds_branching_tree():
    space = generate_tree_world({"A": ["B", "C"], "B": ["D"]})
    adaptive = solve_adaptive_certificate(space, policy=select_entropy_reduction_question)

    assert adaptive.valid
    assert _is_internal(adaptive.root)
    assert adaptive.root.yes_child is not None
    assert adaptive.root.no_child is not None
    assert adaptive.node_count >= 3
    assert adaptive.average_depth > 0.0
    assert validate_adaptive_certificate(adaptive.root, space)


def test_unstructured_world_does_not_show_anomalous_advantage():
    space = generate_unstructured_world(["Q0", "Q1", "Q2"])
    fixed = solve_exact_certificate(space)
    adaptive = solve_adaptive_certificate(space, policy="entropy")

    assert adaptive.valid
    assert adaptive.worst_case_depth == fixed.certificate_size
    assert adaptive.worst_case_depth >= fixed.certificate_size
    assert validate_adaptive_certificate(adaptive.root, space)


def test_random_policy_is_reproducible_with_seed():
    space = generate_chain_world(["A", "B", "C"])
    run_a = solve_adaptive_certificate(space, policy="random", seed=42)
    run_b = solve_adaptive_certificate(space, policy="random", seed=42)

    assert tree_signature(run_a.root) == tree_signature(run_b.root)


def test_balanced_policy_is_rejected_by_resolver_and_solver():
    with pytest.raises(ValueError, match="Unsupported policy 'balanced'"):
        resolve_policy("balanced")

    with pytest.raises(ValueError, match="Unsupported policy 'balanced'"):
        solve_adaptive_certificate(generate_chain_world(["A"]), policy="balanced")


def test_entropy_equal_scores_preserve_task_id_order(monkeypatch):
    monkeypatch.setattr(
        policies,
        "_splitting_questions",
        lambda *args: [("first", 2, 2), ("second", 2, 2)],
    )

    selected = select_entropy_reduction_question([], ["first", "second"], set(), lambda *_: ())

    assert selected == "first"


def test_entropy_large_counts_use_exact_integer_order(monkeypatch):
    total = 268435456
    neighboring = (total // 2 - 1, total // 2 + 1)
    perfect = (total // 2, total // 2)

    def old_float_entropy(counts):
        no_count, yes_count = counts
        p_no = no_count / total
        p_yes = yes_count / total
        return -(p_no * log2(p_no) + p_yes * log2(p_yes))

    assert old_float_entropy(neighboring) == old_float_entropy(perfect)
    monkeypatch.setattr(
        policies,
        "_splitting_questions",
        lambda *args: [
            ("neighboring", *neighboring),
            ("perfect", *perfect),
        ],
    )
    candidate_states = range(total)

    selected = select_entropy_reduction_question(
        candidate_states,  # type: ignore[arg-type]
        ["neighboring", "perfect"],
        set(),
        lambda *_: (),
    )

    assert selected == "perfect"


def _collapsed_signature(_: KnowledgeState, task_ids: Sequence[str]) -> Signature:
    return (1,) if task_ids else ()


def _non_binary_signature(_: KnowledgeState, task_ids: Sequence[str]) -> Signature:
    return (2,) if task_ids else ()


def test_non_identifiable_signature_results_in_invalid_adaptive_certificate():
    space = generate_unstructured_world(["Q0", "Q1"])
    result = solve_adaptive_certificate(space, policy="entropy", response_signature_fn=_collapsed_signature)

    assert not result.valid
    assert result.worst_case_depth == 0
    assert result.average_depth == 0.0
    assert not validate_adaptive_certificate(
        result.root,
        space,
        response_signature_fn=_collapsed_signature,
    )


def test_non_binary_signature_is_rejected():
    space = generate_chain_world(["A", "B", "C"])

    with pytest.raises(ValueError, match="binary response signatures"):
        solve_adaptive_certificate(space, response_signature_fn=_non_binary_signature)


def test_random_policy_returns_valid_tree_for_many_seeds():
    space = generate_tree_world({"A": ["B", "C"], "B": ["D"]})

    for seed in range(100):
        result = solve_adaptive_certificate(space, policy="random", seed=seed)
        assert result.valid
        assert validate_adaptive_certificate(result.root, space)


def test_custom_policy_returning_non_splitting_task_raises():
    space = generate_unstructured_world(["A", "B"])

    def _bad_policy(*args):
        del args
        return "A"

    with pytest.raises(ValueError, match="non-splitting task"):
        solve_adaptive_certificate(
            space,
            policy=_bad_policy,
            response_signature_fn=_collapsed_signature,
        )


def test_custom_policy_cannot_stop_while_a_split_remains():
    space = generate_unstructured_world(["A", "B"])

    def _premature_stop(*args):
        del args
        return None

    with pytest.raises(ValueError, match="unasked splitting task remains"):
        solve_adaptive_certificate(space, policy=_premature_stop)


def test_adaptive_validator_rejects_incomplete_and_non_progressing_trees():
    space = generate_unstructured_world(["A", "B"])

    incomplete = DecisionNode(
        question="A",
        yes_child=DecisionNode(question=None, candidate_state_ids=['["A"]']),
        no_child=None,
    )
    assert not validate_adaptive_certificate(incomplete, space)

    leaf_with_two_states = DecisionNode(
        question=None,
        candidate_state_ids=["[]", '["A"]'],
    )
    assert not validate_adaptive_certificate(leaf_with_two_states, space)

    duplicated_leaf_id = DecisionNode(
        question=None,
        candidate_state_ids=["[]", "[]"],
    )
    assert not validate_adaptive_certificate(
        duplicated_leaf_id,
        generate_unstructured_world([]),
    )

    non_progressing = DecisionNode(
        question="A",
        yes_child=DecisionNode(question=None, candidate_state_ids=['["A"]']),
        no_child=DecisionNode(question=None, candidate_state_ids=["[]"]),
    )
    assert not validate_adaptive_certificate(
        non_progressing,
        space,
        response_signature_fn=_collapsed_signature,
    )


def test_adaptive_solver_rejects_empty_declared_state_population():
    space = KnowledgeSpace(tasks=TaskUniverse(("A",)), valid_states=())
    empty_leaf = DecisionNode(question=None, candidate_state_ids=[])

    with pytest.raises(ValueError, match="valid_states must not be empty"):
        solve_adaptive_certificate(space)

    assert not validate_adaptive_certificate(empty_leaf, space)


def test_adaptive_serialization_contains_concrete_tree_structure():
    space = generate_chain_world(["A"])
    result = solve_adaptive_certificate(space, policy="entropy")
    payload = result.to_dict()

    assert payload["root"]["question"] == "A"
    assert payload["root"]["yes_child"] == {
        "question": None,
        "candidate_state_ids": ['["A"]'],
    }
    assert payload["root"]["no_child"] == {
        "question": None,
        "candidate_state_ids": ["[]"],
    }


def test_phase4_world_results_have_only_current_policy_keys(monkeypatch):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    phase4 = importlib.import_module("revalidation_phase4_adaptive")
    monkeypatch.setattr(phase4, "RANDOM_SEEDS", [0])

    result = phase4._run_world("chain", generate_chain_world(["A", "B"]))

    assert set(result["policy_results"]) == {"entropy", "random"}
