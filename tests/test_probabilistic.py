from __future__ import annotations

from random import Random
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from capability_certificate_lab.certificate import solve_exact_certificate
from capability_certificate_lab.generators import generate_chain_world, generate_unstructured_world
from capability_certificate_lab.knowledge_space import KnowledgeSpace, TaskUniverse
from capability_certificate_lab.knowledge_space.state import KnowledgeState
from capability_certificate_lab.validation.identifiability.core import stable_state_id
from capability_certificate_lab.probabilistic import (
    ResponseNoiseModel,
    infer_state_posterior,
    select_entropy_reduction_question,
    select_expected_error_reduction_question,
    select_random_question,
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


class _ZeroRandom:
    def random(self):
        return 0.0


def test_zero_probability_bernoulli_boundary_returns_zero():
    assert (
        simulate_probabilistic_response(
            state_has_task=False,
            noise=ResponseNoiseModel(guess=0.0),
            rng=_ZeroRandom(),
        )
        == 0
    )


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
    for attempts in (1.5, True):
        with pytest.raises(ValueError, match="attempts must be a positive integer"):
            simulate_probabilistic_response(
                state_has_task=True,
                noise=ResponseNoiseModel(),
                attempts=attempts,
            )

    space = generate_chain_world(["A", "B"])
    states = list(space.valid_states)
    posterior = infer_state_posterior(space, observations=[]).state_posteriors
    for policy in (
        select_random_question,
        select_entropy_reduction_question,
        select_expected_error_reduction_question,
    ):
        with pytest.raises(ValueError, match="attempts must be a positive integer"):
            policy(
                states,
                space.tasks.task_ids,
                posterior,
                set(),
                ResponseNoiseModel(),
                attempts=True,
            )

    for attempts_per_query in (0, True):
        with pytest.raises(ValueError, match="attempts_per_query must be a positive integer"):
            solve_noisy_adaptive_certificate(
                space,
                true_state=KnowledgeState(("A", "B")),
                attempts_per_query=attempts_per_query,
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

    for max_queries in (-1, True):
        with pytest.raises(ValueError, match="max_queries must be a non-negative integer"):
            solve_noisy_adaptive_certificate(
                space,
                true_state=KnowledgeState(("A", "B")),
                max_queries=max_queries,
            )


def test_sequential_updates_match_batch_update():
    space = generate_unstructured_world(["A", "B"])
    noise = ResponseNoiseModel(slip=0.1, guess=0.1)

    batch = infer_state_posterior(space, [("A", 1), ("B", 0)], noise=noise)
    step_one = infer_state_posterior(space, [("A", 1)], noise=noise)
    sequential = infer_state_posterior(
        space,
        [("B", 0)],
        noise=noise,
        prior=step_one.state_posteriors,
    )

    assert sequential.map_state == batch.map_state
    assert sequential.confidence == pytest.approx(batch.confidence)
    for state_id, probability in batch.state_posteriors.items():
        assert sequential.state_posteriors[state_id] == pytest.approx(probability)


def test_fixed_certificate_exposes_solver_posterior():
    space = generate_unstructured_world(["A", "B"])
    noise = ResponseNoiseModel(slip=0.1, guess=0.1)

    fixed = solve_noisy_fixed_certificate(
        space,
        selected_tasks=["A", "B"],
        observations={"A": [1], "B": [0]},
        noise=noise,
    )
    batch = infer_state_posterior(space, [("A", 1), ("B", 0)], noise=noise)

    assert fixed.map_state == batch.map_state
    assert fixed.confidence == pytest.approx(batch.confidence)
    assert fixed.state_posteriors == pytest.approx(batch.state_posteriors)
    assert fixed.to_dict()["state_posteriors"] == pytest.approx(batch.state_posteriors)


def test_phase5_consistency_gate_rejects_near_tie_set_drift(monkeypatch):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import revalidation_phase5_robustness as phase5_robustness

    batch = SimpleNamespace(
        state_posteriors={"a": 0.5, "b": 0.5 - 0.9e-12},
        map_state="a",
        confidence=0.5,
    )
    sequential = SimpleNamespace(
        state_posteriors={"a": 0.5, "b": 0.5 - 1.1e-12},
        map_state="a",
        confidence=0.5,
    )
    posterior_results = iter((batch, batch, sequential))
    monkeypatch.setattr(
        phase5_robustness,
        "infer_state_posterior",
        lambda *args, **kwargs: next(posterior_results),
    )

    consistency = phase5_robustness._posterior_consistency(
        object(),
        [("A", 1)],
        ResponseNoiseModel(),
        observed_posteriors=batch.state_posteriors,
        observed_map_state=batch.map_state,
        observed_confidence=batch.confidence,
    )

    assert not consistency["ok"]
    assert not consistency["sequential_tie_set_matches_batch"]


def test_two_task_exact_posterior_values():
    space = generate_unstructured_world(["A", "B"])
    posterior = infer_state_posterior(
        space,
        [("A", 1), ("B", 0)],
        noise=ResponseNoiseModel(slip=0.1, guess=0.1),
    )

    assert posterior.state_posteriors == pytest.approx(
        {
            "[]": 0.09,
            '["A"]': 0.81,
            '["A","B"]': 0.09,
            '["B"]': 0.01,
        }
    )
    assert posterior.map_state == '["A"]'
    assert posterior.confidence == pytest.approx(0.81)


def test_long_possible_observation_sequence_does_not_underflow():
    space = generate_unstructured_world(["A"])
    posterior = infer_state_posterior(
        space,
        [("A", [1] * 10000)],
        noise=ResponseNoiseModel(slip=0.1, guess=0.1),
    )

    assert posterior.map_state == '["A"]'
    assert posterior.confidence == pytest.approx(1.0)
    assert sum(posterior.state_posteriors.values()) == pytest.approx(1.0)


def test_genuinely_impossible_observations_remain_explicit():
    space = generate_unstructured_world(["A"])
    posterior = infer_state_posterior(
        space,
        [("A", 1), ("A", 0)],
        noise=ResponseNoiseModel(slip=0.0, guess=0.0),
    )

    assert posterior.map_state == ""
    assert posterior.confidence == 0.0
    assert posterior.state_posteriors == {"[]": 0.0, '["A"]': 0.0}


def test_adaptive_stopping_threshold_does_not_double_count_old_observations():
    space = generate_unstructured_world(["A", "B"])
    questions = iter(["A", "B"])

    def _fixed_policy(*args):
        del args
        return next(questions, None)

    result = solve_noisy_adaptive_certificate(
        space,
        true_state=KnowledgeState(("A",)),
        policy=_fixed_policy,
        noise=ResponseNoiseModel(slip=0.1, guess=0.1),
        delta=0.15,
        seed=1,
    )

    assert result.query_count == 2
    assert result.confidence == pytest.approx(0.81)
    assert not result.reached_delta
    assert not result.valid


def test_probabilistic_inference_rejects_empty_invalid_and_duplicate_states():
    tasks = TaskUniverse(["A", "B"])

    with pytest.raises(ValueError, match="must not be empty"):
        infer_state_posterior(KnowledgeSpace(tasks=tasks, valid_states=[]), [])

    invalid_world = KnowledgeSpace(
        tasks=tasks,
        valid_states=[KnowledgeState(("Z",))],
    )
    with pytest.raises(ValueError, match="violate world rules"):
        infer_state_posterior(invalid_world, [])

    duplicate_world = KnowledgeSpace(
        tasks=tasks,
        valid_states=[KnowledgeState(("A",)), KnowledgeState(("A",))],
    )
    with pytest.raises(ValueError, match="Duplicate declared states"):
        solve_noisy_adaptive_certificate(
            duplicate_world,
            true_state=KnowledgeState(("A",)),
        )


def test_noisy_adaptive_solver_rejects_true_state_outside_declared_population():
    space = generate_chain_world(["A", "B"])

    with pytest.raises(ValueError, match="true_state must belong to declared valid_states"):
        solve_noisy_adaptive_certificate(
            space,
            true_state=KnowledgeState(("B",)),
        )
