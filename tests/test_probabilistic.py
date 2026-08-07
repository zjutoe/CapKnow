from __future__ import annotations

from random import Random

import pytest
from capability_certificate_lab.certificate import solve_exact_certificate
from capability_certificate_lab.generators import generate_chain_world, generate_unstructured_world
from capability_certificate_lab.knowledge_space.state import KnowledgeState
from capability_certificate_lab.validation.identifiability.core import stable_state_id
from capability_certificate_lab.probabilistic import (
    ResponseNoiseModel,
    infer_state_posterior,
    simulate_probabilistic_response,
    solve_noisy_adaptive_certificate,
    solve_noisy_fixed_certificate,
)


def _collect_noisy_responses(
    state: KnowledgeState,
    task_ids: list[str],
    noise: ResponseNoiseModel,
    seed: int,
):
    rng = Random(seed)
    return {
        task: [simulate_probabilistic_response(
            task in state,
            noise=noise,
            attempts=1,
            rng=rng,
        )]
        for task in task_ids
    }


def test_noiseless_fixed_and_adaptive_match_deterministic_baseline():
    space = generate_chain_world(["A", "B", "C"])
    state = KnowledgeState(("A", "B", "C"))
    exact = solve_exact_certificate(space)
    task_ids = space.tasks.task_ids
    noise = ResponseNoiseModel(slip=0.0, guess=0.0)

    observations = _collect_noisy_responses(state, task_ids, noise, seed=7)
    fixed = solve_noisy_fixed_certificate(
        space,
        selected_tasks=task_ids,
        observations=observations,
        noise=noise,
        delta=1e-12,
    )
    adaptive = solve_noisy_adaptive_certificate(
        space,
        true_state=state,
        policy="entropy_reduction",
        noise=noise,
        delta=1e-12,
        seed=7,
    )

    assert fixed.valid
    assert fixed.map_state == stable_state_id(state, task_ids)
    assert fixed.confidence == 1.0
    assert fixed.certificate_size == exact.certificate_size
    assert adaptive.valid
    assert adaptive.map_state == stable_state_id(state, task_ids)
    assert adaptive.confidence == 1.0
    assert adaptive.query_count <= exact.certificate_size
    assert adaptive.reached_delta


def test_high_slip_reduces_identification_accuracy():
    space = generate_chain_world(["A", "B", "C"])
    state = KnowledgeState(("A", "B", "C"))
    task_ids = space.tasks.task_ids
    noise = ResponseNoiseModel(slip=0.45, guess=0.0)

    correct = 0
    for seed in range(20):
        observations = _collect_noisy_responses(state, task_ids, noise, seed)
        fixed = solve_noisy_fixed_certificate(
            space,
            selected_tasks=task_ids,
            observations=observations,
            noise=noise,
            delta=1.0,
        )
        if fixed.map_state == stable_state_id(state, task_ids):
            correct += 1

    assert 0 < correct < 20


def test_high_guess_prevents_overconfidence():
    space = generate_unstructured_world(["A", "B"])
    noise = ResponseNoiseModel(slip=0.01, guess=0.49)

    fixed = solve_noisy_fixed_certificate(
        space,
        selected_tasks=["B"],
        observations={"B": [1]},
        noise=noise,
        delta=0.05,
    )

    assert fixed.state_count == 4
    assert not fixed.valid
    assert fixed.confidence < 0.95


def test_posterior_mass_is_normalized_and_map_defined():
    space = generate_chain_world(["A", "B"])
    posterior = infer_state_posterior(
        space,
        observations=[],
        noise=ResponseNoiseModel(slip=0.2, guess=0.2),
    )

    assert abs(sum(posterior.state_posteriors.values()) - 1.0) < 1e-9
    assert posterior.map_state
    assert posterior.confidence == max(posterior.state_posteriors.values())


def test_repeated_attempts_reduce_uncertainty():
    space = generate_chain_world(["A", "B"])
    noise = ResponseNoiseModel(slip=0.3, guess=0.3)

    single = infer_state_posterior(space, [("A", [1])], noise=noise)
    repeated = infer_state_posterior(space, [("A", [1, 1, 1])], noise=noise)

    assert single.observation_count == 1
    assert repeated.observation_count == 3
    assert repeated.confidence >= single.confidence
    assert repeated.entropy <= single.entropy + 1e-12


def test_repeated_attempts_impact_policy_selection():
    space = generate_chain_world(["A", "B"])
    state = KnowledgeState(("A", "B"))
    result = solve_noisy_adaptive_certificate(
        space,
        true_state=state,
        policy="entropy_reduction",
        noise=ResponseNoiseModel(slip=0.2, guess=0.2),
        attempts_per_query=3,
        delta=0.0,
        seed=1,
    )

    assert result.query_count >= 1
    assert result.query_history
    assert all(len(entry["responses"]) == 3 for entry in result.query_history)


def test_non_binary_observation_is_rejected():
    space = generate_chain_world(["A", "B"])

    with pytest.raises(ValueError, match="Probabilistic observations must be 0/1"):
        solve_noisy_fixed_certificate(
            space,
            selected_tasks=["A"],
            observations={"A": [2]},
            noise=ResponseNoiseModel(slip=0.2, guess=0.2),
        )

    with pytest.raises(ValueError, match="Probabilistic observations must be 0/1"):
        solve_noisy_fixed_certificate(
            space,
            selected_tasks=["A"],
            observations={"A": [0.9]},
            noise=ResponseNoiseModel(slip=0.2, guess=0.2),
        )

    with pytest.raises(ValueError, match="must not be empty"):
        solve_noisy_fixed_certificate(
            space,
            selected_tasks=["A"],
            observations={"A": []},
            noise=ResponseNoiseModel(slip=0.2, guess=0.2),
        )


def test_contradictory_observations_mark_certificate_invalid():
    space = generate_chain_world(["A", "B"])

    result = solve_noisy_fixed_certificate(
        space,
        selected_tasks=["A", "B"],
        observations={"A": [0], "B": [1]},
        noise=ResponseNoiseModel(slip=0.0, guess=0.0),
    )

    assert not result.valid
    assert result.map_state == ""


def test_zero_query_entropy_matches_prior_uncertainty():
    space = generate_chain_world(["A", "B", "C"])
    result = solve_noisy_adaptive_certificate(
        space,
        true_state=KnowledgeState(("A", "B")),
        policy="entropy_reduction",
        noise=ResponseNoiseModel(slip=0.2, guess=0.2),
        max_queries=0,
    )

    assert result.query_count == 0
    assert result.entropy == pytest.approx(2.0)


def test_prior_must_cover_all_states():
    space = generate_chain_world(["A", "B"])
    task_ids = space.tasks.task_ids
    states = sorted(
        [state for state in space.valid_states if space.is_valid_state(state)],
        key=lambda state: state.as_tuple(task_ids),
    )
    state_ids = [stable_state_id(state, task_ids) for state in states]

    with pytest.raises(ValueError, match="Prior must provide a non-empty probability"):
        infer_state_posterior(space, observations=[], prior={state_ids[0]: 1.0})

    with pytest.raises(ValueError, match="Prior must provide a non-empty probability"):
        infer_state_posterior(
            space,
            observations=[],
            prior={**{sid: 0.5 for sid in state_ids}, "bad_state": 0.1},
        )


def test_unknown_task_in_observations_is_rejected():
    space = generate_chain_world(["A", "B"])

    with pytest.raises(ValueError, match="Unknown task id"):
        infer_state_posterior(space, [("Z", 1)])


def test_attempts_rejected_if_not_positive_integer():
    with pytest.raises(ValueError, match="attempts must be a positive integer"):
        simulate_probabilistic_response(
            state_has_task=True,
            noise=ResponseNoiseModel(),
            attempts=1.5,
        )

    space = generate_chain_world(["A", "B"])
    with pytest.raises(ValueError, match="attempts_per_query must be a positive integer"):
        solve_noisy_adaptive_certificate(
            space,
            true_state=KnowledgeState(("A", "B")),
            attempts_per_query=0,
        )


def test_delta_and_max_queries_validated():
    space = generate_chain_world(["A", "B"])

    with pytest.raises(ValueError, match="delta must be a finite number in \\[0.0, 1.0\\]"):
        solve_noisy_fixed_certificate(
            space,
            selected_tasks=["A"],
            observations={"A": [1]},
            noise=ResponseNoiseModel(),
            delta=-0.1,
        )

    with pytest.raises(ValueError, match="max_queries must be a non-negative integer"):
        solve_noisy_adaptive_certificate(
            space,
            true_state=KnowledgeState(("A", "B")),
            max_queries=-1,
        )
