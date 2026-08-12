from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import log2
from random import Random

from ..knowledge_space.state import KnowledgeState
from ..validation.identifiability.core import stable_state_id
from .response_model import ResponseNoiseModel, response_probability


AdaptiveQuestionPolicy = Callable[
    [
        Sequence[KnowledgeState],
        Sequence[str],
        Mapping[str, float],
        set[str],
        ResponseNoiseModel,
        int,
        Random | None,
    ],
    str | None,
]


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _available_questions(task_ids: Sequence[str], asked: set[str]) -> list[str]:
    return [task_id for task_id in task_ids if task_id not in asked]


def _entropy_from_distribution(values: Sequence[float]) -> float:
    total = 0.0
    for value in values:
        if value <= 0.0:
            continue
        total -= value * log2(value)
    return total


def _binomial_pmf(success_prob: float, trials: int) -> list[float]:
    if type(trials) is not int or trials < 0:
        raise ValueError("trials must be a non-negative integer.")
    if trials == 0:
        return [1.0]
    if success_prob <= 0.0:
        probs = [0.0] * (trials + 1)
        probs[0] = 1.0
        return probs
    if success_prob >= 1.0:
        probs = [0.0] * (trials + 1)
        probs[trials] = 1.0
        return probs

    probs = [0.0] * (trials + 1)
    no_prob = 1.0 - success_prob
    probs[0] = no_prob**trials
    for successes in range(1, trials + 1):
        probs[successes] = (
            probs[successes - 1]
            * (trials - successes + 1)
            / successes
            * success_prob
            / no_prob
        )
    return probs


def _branch_statistics(
    states: Sequence[KnowledgeState],
    state_ids: Sequence[str],
    posterior: Mapping[str, float],
    question: str,
    noise: ResponseNoiseModel,
    attempts: int,
) -> list[tuple[float, dict[str, float]]]:
    attempts = _require_positive_int(attempts, "attempts")
    branch_masses = [(0.0, {}) for _ in range(attempts + 1)]
    for state, sid in zip(states, state_ids):
        prior = posterior.get(sid, 0.0)
        if prior == 0.0:
            continue
        p_yes = response_probability(question in state, noise)
        for successes, p_count in enumerate(_binomial_pmf(p_yes, attempts)):
            count_mass = prior * p_count
            branch_mass, branch_states = branch_masses[successes]
            branch_mass += count_mass
            branch_states[sid] = branch_states.get(sid, 0.0) + count_mass
            branch_masses[successes] = (branch_mass, branch_states)
    return branch_masses


def _safe_entropy_from_mass(mass: dict[str, float], branch_mass: float) -> float:
    if branch_mass == 0.0:
        return 0.0
    probs = [weight / branch_mass for weight in mass.values()]
    return _entropy_from_distribution(probs)


def _safe_error_from_mass(mass: dict[str, float], branch_mass: float) -> float:
    if branch_mass == 0.0:
        return 0.0
    return 1.0 - max(weight / branch_mass for weight in mass.values())


def select_random_question(
    states: Sequence[KnowledgeState],
    task_ids: Sequence[str],
    posterior: Mapping[str, float],
    asked: set[str],
    noise: ResponseNoiseModel,
    attempts: int,
    rng: Random | None = None,
) -> str | None:
    del states
    del posterior
    del noise
    _ = _require_positive_int(attempts, "attempts")
    candidates = _available_questions(task_ids, asked)
    if not candidates:
        return None
    chooser = rng if rng is not None else Random()
    return chooser.choice(candidates)


def select_entropy_reduction_question(
    states: Sequence[KnowledgeState],
    task_ids: Sequence[str],
    posterior: Mapping[str, float],
    asked: set[str],
    noise: ResponseNoiseModel,
    attempts: int = 1,
    rng: Random | None = None,
) -> str | None:
    del rng
    attempts = _require_positive_int(attempts, "attempts")

    state_ids = [stable_state_id(state, task_ids) for state in states]
    current_entropy = _entropy_from_distribution(list(posterior.values()))

    best_gain = -1.0
    best_question: str | None = None
    for question in _available_questions(task_ids, asked):
        branches = _branch_statistics(
            states,
            state_ids,
            posterior,
            question,
            noise,
            attempts,
        )
        expected_entropy = 0.0
        for branch_mass, branch_states in branches:
            expected_entropy += branch_mass * _safe_entropy_from_mass(
                branch_states,
                branch_mass,
            )
        gain = current_entropy - expected_entropy
        if gain > best_gain:
            best_gain = gain
            best_question = question

    return best_question


def select_expected_error_reduction_question(
    states: Sequence[KnowledgeState],
    task_ids: Sequence[str],
    posterior: Mapping[str, float],
    asked: set[str],
    noise: ResponseNoiseModel,
    attempts: int = 1,
    rng: Random | None = None,
) -> str | None:
    del rng
    attempts = _require_positive_int(attempts, "attempts")

    state_ids = [stable_state_id(state, task_ids) for state in states]
    current_error = 1.0 - max(posterior.values(), default=0.0)

    best_reduction = -1.0
    best_question: str | None = None
    for question in _available_questions(task_ids, asked):
        branches = _branch_statistics(
            states,
            state_ids,
            posterior,
            question,
            noise,
            attempts,
        )
        expected_error = 0.0
        for branch_mass, branch_states in branches:
            expected_error += branch_mass * _safe_error_from_mass(
                branch_states,
                branch_mass,
            )
        reduction = current_error - expected_error
        if reduction > best_reduction:
            best_reduction = reduction
            best_question = question

    return best_question


POLICIES = {
    "random": select_random_question,
    "entropy": select_entropy_reduction_question,
    "entropy_reduction": select_entropy_reduction_question,
    "expected_error": select_expected_error_reduction_question,
    "expected_error_reduction": select_expected_error_reduction_question,
}


def resolve_policy(policy: str | AdaptiveQuestionPolicy) -> tuple[str, AdaptiveQuestionPolicy]:
    if callable(policy):
        return "custom", policy
    if policy not in POLICIES:
        raise ValueError(f"Unsupported adaptive policy '{policy}'")
    return policy, POLICIES[policy]
