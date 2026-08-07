from __future__ import annotations

from capability_certificate_lab.certificate import (
    AdaptiveCertificate,
    DecisionNode,
    solve_adaptive_certificate,
    solve_exact_certificate,
    tree_signature,
    validate_adaptive_certificate,
)
from capability_certificate_lab.certificate.policies import select_entropy_reduction_question
from capability_certificate_lab.generators import (
    generate_chain_world,
    generate_tree_world,
    generate_unstructured_world,
)


def _is_internal(node: DecisionNode) -> bool:
    return node.question is not None


def test_chain_world_adaptive_improves_or_matches_fixed_depth():
    space = generate_chain_world(["A", "B", "C", "D"])
    fixed = solve_exact_certificate(space)
    adaptive = solve_adaptive_certificate(space, policy="balanced")

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
