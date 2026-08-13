from __future__ import annotations

from collections import defaultdict
from random import Random
from statistics import mean

from capability_certificate_lab.certificate import solve_exact_certificate
from capability_certificate_lab.generators import (
    generate_chain_world,
    generate_tree_world,
    generate_unstructured_world,
)
from capability_certificate_lab.knowledge_space.state import KnowledgeState
from capability_certificate_lab.probabilistic import (
    ResponseNoiseModel,
    infer_state_posterior,
    simulate_probabilistic_response,
    solve_noisy_adaptive_certificate,
    solve_noisy_fixed_certificate,
)
from capability_certificate_lab.probabilistic.posterior import MAP_TIE_TOLERANCE
from capability_certificate_lab.validation.identifiability.core import stable_state_id

from revalidation_common import source_binding_before_run, write_result_and_manifest


COMMAND = "PYTHONPATH=. python scripts/revalidation_phase5_robustness.py"
SEEDS = list(range(100))
ATTEMPTS = [1, 3, 5]
NOISE_CONDITIONS = [
    (0.00, 0.00),
    (0.05, 0.05),
    (0.10, 0.10),
    (0.20, 0.20),
    (0.30, 0.30),
    (0.30, 0.05),
    (0.05, 0.30),
]
POSTERIOR_TOLERANCE = 1e-12
ZERO_NOISE_ORACLE_POLICY = "entropy"
ZERO_NOISE_EXPECTED_DEPTHS = {
    "chain": {
        '["A","B","C"]': 2,
        '["A","B"]': 2,
        '["A"]': 2,
        "[]": 2,
    },
    "tree": {
        '["A","B","C"]': 2,
        '["A","B"]': 2,
        '["A","C"]': 3,
        '["A"]': 3,
        "[]": 2,
    },
    "unstructured": {
        '["A","B"]': 2,
        '["A"]': 2,
        '["B"]': 2,
        "[]": 2,
    },
}
WORLD_DEFINITIONS = [
    {
        "name": "chain",
        "generator": "generate_chain_world",
        "task_ids": ["A", "B", "C"],
    },
    {
        "name": "tree",
        "generator": "generate_tree_world",
        "edges": {"A": ["B", "C"]},
    },
    {
        "name": "unstructured",
        "generator": "generate_unstructured_world",
        "task_ids": ["A", "B"],
    },
]


def _build_world(definition):
    if definition["generator"] == "generate_chain_world":
        return generate_chain_world(definition["task_ids"])
    if definition["generator"] == "generate_tree_world":
        return generate_tree_world(definition["edges"])
    if definition["generator"] == "generate_unstructured_world":
        return generate_unstructured_world(definition["task_ids"])
    raise ValueError(f"Unsupported world generator {definition['generator']}.")


def _ordered_states(world):
    task_ids = world.tasks.task_ids
    return sorted(world.valid_states, key=lambda state: state.as_tuple(task_ids))


def _simulate_fixed_observations(world, true_state: KnowledgeState, selected_tasks, noise, attempts, seed):
    rng = Random(seed)
    observations = {}
    flat = []
    for task_id in selected_tasks:
        responses = simulate_probabilistic_response(
            task_id in true_state,
            noise=noise,
            attempts=attempts,
            rng=rng,
        )
        response_list = responses if isinstance(responses, list) else [responses]
        observations[task_id] = response_list
        flat.extend((task_id, value) for value in response_list)
    return observations, flat


def _posterior_consistency(
    world,
    observations,
    noise,
    *,
    observed_posteriors,
    observed_map_state,
    observed_confidence,
):
    batch = infer_state_posterior(world, observations, noise=noise)
    sequential = infer_state_posterior(world, [], noise=noise)
    for task_id, value in observations:
        sequential = infer_state_posterior(
            world,
            [(task_id, value)],
            noise=noise,
            prior=sequential.state_posteriors,
        )
    compared_posteriors = observed_posteriors
    compared_map_state = observed_map_state
    compared_confidence = observed_confidence
    if set(compared_posteriors) != set(batch.state_posteriors):
        max_diff = float("inf")
    else:
        max_diff = max(
            abs(batch.state_posteriors[state_id] - compared_posteriors[state_id])
            for state_id in batch.state_posteriors
        )
    sequential_batch_diff = max(
        abs(batch.state_posteriors[state_id] - sequential.state_posteriors[state_id])
        for state_id in batch.state_posteriors
    )
    confidence_diff = abs(batch.confidence - compared_confidence)
    sequential_confidence_diff = abs(batch.confidence - sequential.confidence)
    batch_ties = [
        state_id
        for state_id, probability in batch.state_posteriors.items()
        if batch.confidence - probability <= MAP_TIE_TOLERANCE
    ]
    sequential_ties = [
        state_id
        for state_id, probability in sequential.state_posteriors.items()
        if sequential.confidence - probability <= MAP_TIE_TOLERANCE
    ]
    compared_ties = [
        state_id
        for state_id, probability in compared_posteriors.items()
        if compared_confidence - probability <= MAP_TIE_TOLERANCE
    ]
    observed_tie_set_matches_batch = set(batch_ties) == set(compared_ties)
    sequential_tie_set_matches_batch = set(batch_ties) == set(sequential_ties)
    return {
        "ok": (
            max_diff <= POSTERIOR_TOLERANCE
            and sequential_batch_diff <= POSTERIOR_TOLERANCE
            and confidence_diff <= POSTERIOR_TOLERANCE
            and sequential_confidence_diff <= POSTERIOR_TOLERANCE
            and batch.map_state == compared_map_state
            and batch.map_state == sequential.map_state
            and observed_tie_set_matches_batch
            and sequential_tie_set_matches_batch
        ),
        "max_posterior_diff": max_diff,
        "max_sequential_batch_posterior_diff": sequential_batch_diff,
        "confidence_diff": confidence_diff,
        "sequential_confidence_diff": sequential_confidence_diff,
        "batch_map_state": batch.map_state,
        "sequential_map_state": sequential.map_state,
        "observed_map_state": compared_map_state,
        "batch_confidence": batch.confidence,
        "sequential_confidence": sequential.confidence,
        "observed_confidence": compared_confidence,
        "batch_map_ties": batch_ties,
        "sequential_map_ties": sequential_ties,
        "observed_map_ties": compared_ties,
        "observed_tie_set_matches_batch": observed_tie_set_matches_batch,
        "sequential_tie_set_matches_batch": sequential_tie_set_matches_batch,
    }


def _flatten_adaptive_observations(query_history):
    if query_history is None:
        return []
    flat = []
    for index, entry in enumerate(query_history):
        question = entry.get("question")
        responses = entry.get("responses")
        if not isinstance(question, str):
            raise RuntimeError(f"Adaptive history entry {index} has no string question.")
        if not isinstance(responses, list):
            raise RuntimeError(f"Adaptive history entry {index} has no response list.")
        flat.extend((question, value) for value in responses)
    return flat


def _empty_metrics():
    return {
        "map_accuracy": [],
        "confidence": [],
        "threshold_reached": [],
        "query_count": [],
        "total_response_count": [],
        "consistency_ok": [],
        "posterior_diff": [],
        "tie_cases": 0,
        "tie_set_mismatches": 0,
        "map_mismatches": 0,
    }


def _record(metrics, *, correct, confidence, threshold, query_count, response_count, consistency=None):
    metrics["map_accuracy"].append(1.0 if correct else 0.0)
    metrics["confidence"].append(confidence)
    metrics["threshold_reached"].append(1.0 if threshold else 0.0)
    metrics["query_count"].append(query_count)
    metrics["total_response_count"].append(response_count)
    if consistency is not None:
        metrics["consistency_ok"].append(1.0 if consistency["ok"] else 0.0)
        metrics["posterior_diff"].append(
            max(
                consistency["max_posterior_diff"],
                consistency["max_sequential_batch_posterior_diff"],
            )
        )
        if (
            len(consistency["batch_map_ties"]) > 1
            or len(consistency["sequential_map_ties"]) > 1
            or len(consistency["observed_map_ties"]) > 1
        ):
            metrics["tie_cases"] += 1
        if (
            not consistency["observed_tie_set_matches_batch"]
            or not consistency["sequential_tie_set_matches_batch"]
        ):
            metrics["tie_set_mismatches"] += 1
        if (
            consistency["batch_map_state"] != consistency["observed_map_state"]
            or consistency["batch_map_state"] != consistency["sequential_map_state"]
        ):
            metrics["map_mismatches"] += 1


def _summarize(metrics):
    return {
        "run_count": len(metrics["map_accuracy"]),
        "map_accuracy": mean(metrics["map_accuracy"]),
        "mean_posterior_confidence": mean(metrics["confidence"]),
        "threshold_reach_rate": mean(metrics["threshold_reached"]),
        "average_query_count": mean(metrics["query_count"]),
        "worst_case_query_count": max(metrics["query_count"]),
        "average_total_response_count": mean(metrics["total_response_count"]),
        "sequential_batch_consistency_rate": (
            mean(metrics["consistency_ok"]) if metrics["consistency_ok"] else None
        ),
        "max_sequential_batch_posterior_diff": (
            max(metrics["posterior_diff"]) if metrics["posterior_diff"] else None
        ),
        "map_tie_case_count": metrics["tie_cases"],
        "tie_set_mismatch_count": metrics["tie_set_mismatches"],
        "map_mismatch_count": metrics["map_mismatches"],
    }


def _zero_noise_adaptive_depth_oracle(name, world, states, task_ids):
    expected_depths = ZERO_NOISE_EXPECTED_DEPTHS[name]
    state_ids = [stable_state_id(state, task_ids) for state in states]
    if set(state_ids) != set(expected_depths):
        raise RuntimeError(
            f"{name} zero-noise depth oracle state IDs do not match the generated world."
        )

    failures = []
    attempt_checks = []
    zero_noise = ResponseNoiseModel(slip=0.0, guess=0.0)
    for attempts in ATTEMPTS:
        observed_depths = {}
        observed_response_counts = {}
        threshold_failures = 0
        for state in states:
            state_id = stable_state_id(state, task_ids)
            expected_depth = expected_depths[state_id]
            adaptive = solve_noisy_adaptive_certificate(
                world,
                true_state=state,
                policy="entropy_reduction",
                noise=zero_noise,
                delta=0.05,
                attempts_per_query=attempts,
                seed=attempts,
            )
            response_count = len(_flatten_adaptive_observations(adaptive.query_history))
            observed_depths[state_id] = adaptive.query_count
            observed_response_counts[state_id] = response_count
            reached = adaptive.reached_delta and adaptive.valid
            expected_responses = attempts * expected_depth
            if not reached:
                threshold_failures += 1
            if (
                adaptive.map_state != state_id
                or adaptive.query_count != expected_depth
                or not reached
                or adaptive.confidence != 1.0
                or response_count != expected_responses
            ):
                failures.append(
                    {
                        "attempts": attempts,
                        "state_id": state_id,
                        "expected_depth": expected_depth,
                        "adaptive_query_count": adaptive.query_count,
                        "adaptive_map_state": adaptive.map_state,
                        "adaptive_confidence": adaptive.confidence,
                        "adaptive_valid": adaptive.valid,
                        "adaptive_reached_delta": adaptive.reached_delta,
                        "expected_response_count": expected_responses,
                        "adaptive_response_count": response_count,
                    }
                )

        expected_values = list(expected_depths.values())
        observed_values = list(observed_depths.values())
        observed_response_values = list(observed_response_counts.values())
        attempt_checks.append(
            {
                "attempts": attempts,
                "average_expected_depth": mean(expected_values),
                "average_adaptive_query_count": mean(observed_values),
                "worst_expected_depth": max(expected_values),
                "worst_adaptive_query_count": max(observed_values),
                "average_expected_response_count": mean(
                    attempts * depth for depth in expected_values
                ),
                "average_adaptive_response_count": mean(observed_response_values),
                "threshold_semantic_failures": threshold_failures,
                "state_query_depth_mismatches": sum(
                    1
                    for state_id, observed in observed_depths.items()
                    if observed != expected_depths[state_id]
                ),
            }
        )

    if failures:
        raise RuntimeError(f"Zero-noise adaptive depth oracle failed for {name}: {failures[:3]}")
    return {
        "expected_depth_source": "literal_zero_noise_expected_depths",
        "deterministic_policy": ZERO_NOISE_ORACLE_POLICY,
        "adaptive_policy": "entropy_reduction",
        "state_expected_depths": expected_depths,
        "attempt_checks": attempt_checks,
        "failure_count": len(failures),
    }


def _run_world(name, world):
    task_ids = world.tasks.task_ids
    states = _ordered_states(world)
    exact = solve_exact_certificate(world)
    zero_noise_depth_oracle = _zero_noise_adaptive_depth_oracle(name, world, states, task_ids)
    true_zero_noise_expected = True
    buckets = defaultdict(_empty_metrics)
    fixed_consistency_failures = []
    adaptive_consistency_failures = []

    for slip, guess in NOISE_CONDITIONS:
        noise = ResponseNoiseModel(slip=slip, guess=guess)
        for attempts in ATTEMPTS:
            for seed in SEEDS:
                true_state = states[seed % len(states)]
                true_id = stable_state_id(true_state, task_ids)
                fixed_observations, flat = _simulate_fixed_observations(
                    world,
                    true_state,
                    exact.selected_tasks,
                    noise,
                    attempts,
                    seed=seed + attempts * 1000,
                )
                fixed = solve_noisy_fixed_certificate(
                    world,
                    selected_tasks=exact.selected_tasks,
                    observations=fixed_observations,
                    noise=noise,
                    delta=0.05,
                )
                if fixed.state_posteriors is None:
                    raise RuntimeError("Fixed result did not expose final state posteriors.")
                consistency = _posterior_consistency(
                    world,
                    flat,
                    noise,
                    observed_posteriors=fixed.state_posteriors,
                    observed_map_state=fixed.map_state,
                    observed_confidence=fixed.confidence,
                )
                if not consistency["ok"]:
                    fixed_consistency_failures.append(
                        {
                            "method": "fixed",
                            "world": name,
                            "slip": slip,
                            "guess": guess,
                            "attempts": attempts,
                            "seed": seed,
                            **consistency,
                        }
                    )
                _record(
                    buckets[("fixed", slip, guess, attempts)],
                    correct=fixed.map_state == true_id,
                    confidence=fixed.confidence,
                    threshold=fixed.valid,
                    query_count=fixed.certificate_size,
                    response_count=fixed.observations,
                    consistency=consistency,
                )

                adaptive = solve_noisy_adaptive_certificate(
                    world,
                    true_state=true_state,
                    policy="entropy_reduction",
                    noise=noise,
                    delta=0.05,
                    attempts_per_query=attempts,
                    seed=seed + attempts * 1000,
                )
                adaptive_flat = _flatten_adaptive_observations(adaptive.query_history)
                response_count = len(adaptive_flat)
                if adaptive.state_posteriors is None:
                    raise RuntimeError("Adaptive result did not expose final state posteriors.")
                adaptive_consistency = _posterior_consistency(
                    world,
                    adaptive_flat,
                    noise,
                    observed_posteriors=adaptive.state_posteriors,
                    observed_map_state=adaptive.map_state,
                    observed_confidence=adaptive.confidence,
                )
                if not adaptive_consistency["ok"]:
                    adaptive_consistency_failures.append(
                        {
                            "method": "adaptive",
                            "world": name,
                            "slip": slip,
                            "guess": guess,
                            "attempts": attempts,
                            "seed": seed,
                            **adaptive_consistency,
                        }
                    )
                _record(
                    buckets[("adaptive", slip, guess, attempts)],
                    correct=adaptive.map_state == true_id,
                    confidence=adaptive.confidence,
                    threshold=adaptive.reached_delta,
                    query_count=adaptive.query_count,
                    response_count=response_count,
                    consistency=adaptive_consistency,
                )

                if slip == 0.0 and guess == 0.0:
                    true_zero_noise_expected = (
                        true_zero_noise_expected
                        and fixed.map_state == true_id
                        and adaptive.map_state == true_id
                    )

    consistency_failures = fixed_consistency_failures + adaptive_consistency_failures
    if consistency_failures:
        raise RuntimeError(f"Sequential/batch posterior consistency failed: {consistency_failures[:3]}")
    if not true_zero_noise_expected:
        raise RuntimeError(f"Zero-noise sanity check failed for {name}.")

    summaries = []
    for (method, slip, guess, attempts), metrics in sorted(buckets.items()):
        summaries.append(
            {
                "method": method,
                "slip": slip,
                "guess": guess,
                "attempts": attempts,
                **_summarize(metrics),
            }
        )
    return {
        "name": name,
        "task_ids": task_ids,
        "state_count": len(states),
        "exact_certificate_size": exact.certificate_size,
        "summaries": summaries,
        "sanity_checks": {
            "zero_noise_matches_deterministic": true_zero_noise_expected,
            "zero_noise_adaptive_depth_oracle": zero_noise_depth_oracle,
            "fixed_sequential_batch_consistency_failures": len(fixed_consistency_failures),
            "adaptive_sequential_batch_consistency_failures": len(adaptive_consistency_failures),
            "sequential_batch_consistency_failures": len(consistency_failures),
            "symmetric_noise_note": "Symmetric-noise aggregates are reported descriptively; any non-monotone sampled accuracy is not interpreted as improved evidence.",
        },
    }


def main() -> None:
    source = source_binding_before_run()
    worlds = [
        (definition["name"], _build_world(definition))
        for definition in WORLD_DEFINITIONS
    ]
    result = {
        "phase": "phase5_robustness",
        "seeds": SEEDS,
        "attempts": ATTEMPTS,
        "noise_conditions": [{"slip": slip, "guess": guess} for slip, guess in NOISE_CONDITIONS],
        "worlds": [_run_world(name, world) for name, world in worlds],
    }
    artifacts = write_result_and_manifest(
        phase="phase5_robustness",
        script_name="scripts/revalidation_phase5_robustness.py",
        command=COMMAND,
        config={
            "seeds": SEEDS,
            "attempts": ATTEMPTS,
            "noise_conditions": NOISE_CONDITIONS,
            "posterior_consistency_tolerance": POSTERIOR_TOLERANCE,
            "zero_noise_oracle_policy": ZERO_NOISE_ORACLE_POLICY,
            "zero_noise_expected_depths": ZERO_NOISE_EXPECTED_DEPTHS,
            "world_definitions": WORLD_DEFINITIONS,
        },
        result=result,
        source=source,
    )
    print(artifacts)


if __name__ == "__main__":
    main()
