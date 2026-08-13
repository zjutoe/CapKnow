from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log, log2

from ..knowledge_space.space import KnowledgeSpace
from ..knowledge_space.state import KnowledgeState
from ..certificate.validator import _ordered_valid_states
from ..validation.identifiability.core import stable_state_id
from .response_model import ResponseNoiseModel, response_probability


Observation = tuple[str, int | Sequence[int]]
MAP_TIE_TOLERANCE = 1e-12


def _as_binary(value: object) -> int:
    if not isinstance(value, int):
        raise ValueError("Probabilistic observations must be 0/1 integers.")
    if value not in (0, 1):
        raise ValueError("Probabilistic observations must be 0/1.")
    return value


@dataclass(frozen=True)
class PosteriorResult:
    state_count: int
    state_posteriors: dict[str, float]
    map_state: str
    entropy: float
    confidence: float
    observation_count: int


def _flatten_observations(
    observations: Sequence[Observation],
    valid_task_ids: set[str],
) -> list[tuple[str, int]]:
    expanded: list[tuple[str, int]] = []
    for task_id, values in observations:
        if task_id not in valid_task_ids:
            raise ValueError(f"Unknown task id '{task_id}' in probabilistic observations.")
        if isinstance(values, int):
            expanded.append((task_id, _as_binary(values)))
        else:
            for raw_value in values:
                expanded.append((task_id, _as_binary(raw_value)))
    return expanded


def _entropy(probabilities: Iterable[float]) -> float:
    total = 0.0
    for probability in probabilities:
        if probability <= 0.0:
            continue
        total -= probability * log2(probability)
    return total


def _observation_likelihood(
    state: KnowledgeState,
    response: tuple[str, int],
    noise: ResponseNoiseModel,
) -> float:
    task_id, value = response
    if value not in (0, 1):
        raise ValueError("Probabilistic observations must be 0/1.")
    p_yes = response_probability(task_id in state, noise)
    return p_yes if value == 1 else (1.0 - p_yes)


def _normalize_prior(
    state_ids: list[str],
    prior: Mapping[str, float] | None,
) -> dict[str, float]:
    if prior is None:
        prior_mass = 1.0 / len(state_ids)
        return {state_id: prior_mass for state_id in state_ids}

    prior_map = dict(prior)
    expected = set(state_ids)
    provided = set(prior_map.keys())
    if provided != expected:
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        raise ValueError(
            "Prior must provide a non-empty probability for every valid state id and no extras. "
            f"Missing: {missing}, extra: {extra}"
        )

    normalized: dict[str, float] = {}
    total = 0.0
    for state_id in state_ids:
        value = float(prior_map[state_id])
        if not isfinite(value) or value < 0.0:
            raise ValueError("Prior probabilities must be finite and non-negative.")
        normalized[state_id] = value
        total += value

    if total <= 0.0:
        raise ValueError("Prior probabilities must sum to a positive mass.")

    return {state_id: value / total for state_id, value in normalized.items()}


def infer_state_posterior(
    knowledge_space: KnowledgeSpace,
    observations: Sequence[Observation],
    noise: ResponseNoiseModel = ResponseNoiseModel(),
    prior: Mapping[str, float] | None = None,
) -> PosteriorResult:
    """Compute posterior over valid states under an i.i.d. noisy response model."""

    ordered_states, task_ids = _ordered_valid_states(knowledge_space)

    if not ordered_states:
        raise ValueError("Declared valid_states must not be empty for probabilistic inference.")

    expanded = _flatten_observations(observations, set(task_ids))
    state_ids = [stable_state_id(state, task_ids) for state in ordered_states]
    state_prior = _normalize_prior(state_ids, prior)

    log_weights: dict[str, float] = {}
    for state_id, state in zip(state_ids, ordered_states):
        prior_mass = state_prior[state_id]
        log_weight = float("-inf") if prior_mass == 0.0 else log(prior_mass)
        for response in expanded:
            if log_weight == float("-inf"):
                break
            likelihood = _observation_likelihood(state, response, noise)
            if likelihood == 0.0:
                log_weight = float("-inf")
                break
            log_weight += log(likelihood)
        log_weights[state_id] = log_weight

    finite_weights = [weight for weight in log_weights.values() if weight != float("-inf")]
    if not finite_weights:
        return PosteriorResult(
            state_count=len(ordered_states),
            state_posteriors={state_id: 0.0 for state_id in state_ids},
            map_state="",
            entropy=0.0,
            confidence=0.0,
            observation_count=len(expanded),
        )

    max_log_weight = max(finite_weights)
    scaled = {
        sid: 0.0 if weight == float("-inf") else exp(weight - max_log_weight)
        for sid, weight in log_weights.items()
    }
    total_mass = sum(scaled.values())
    normalized = {sid: weight / total_mass for sid, weight in scaled.items()}
    confidence = max(normalized.values())
    map_state = min(
        sid
        for sid, probability in normalized.items()
        if confidence - probability <= MAP_TIE_TOLERANCE
    )
    return PosteriorResult(
        state_count=len(ordered_states),
        state_posteriors=normalized,
        map_state=map_state,
        entropy=_entropy(normalized.values()),
        confidence=confidence,
        observation_count=len(expanded),
    )
