from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random
from time import perf_counter

from ..knowledge_space.space import KnowledgeSpace
from ..knowledge_space.state import KnowledgeState
from ..validation.identifiability.core import stable_state_id
from .adaptive_policy import AdaptiveQuestionPolicy, resolve_policy
from .posterior import _as_binary, infer_state_posterior
from .response_model import ResponseNoiseModel, simulate_probabilistic_response
from math import isfinite


ObservationMap = Mapping[str, int | Sequence[int]]


@dataclass(frozen=True)
class NoisyFixedCertificate:
    task_count: int
    state_count: int
    certificate_size: int
    selected_tasks: list[str]
    valid: bool
    map_state: str
    confidence: float
    entropy: float
    delta: float
    method: str = "noisy_fixed"
    observations: int = 0
    map_probability: float | None = None
    runtime_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "task_count": self.task_count,
            "state_count": self.state_count,
            "certificate_size": self.certificate_size,
            "selected_tasks": self.selected_tasks,
            "valid": self.valid,
            "map_state": self.map_state,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "delta": self.delta,
            "observations": self.observations,
            "map_probability": self.map_probability,
            "runtime_ms": self.runtime_ms,
        }


@dataclass(frozen=True)
class NoisyAdaptiveCertificate:
    task_count: int
    state_count: int
    query_count: int
    map_state: str
    confidence: float
    entropy: float
    valid: bool
    policy: str | None = None
    seed: int | None = None
    delta: float = 0.05
    reached_delta: bool = False
    runtime_ms: float = 0.0
    method: str = "noisy_adaptive"
    asked_tasks: list[str] | None = None
    map_probability: float | None = None
    query_history: list[dict[str, object]] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "task_count": self.task_count,
            "state_count": self.state_count,
            "query_count": self.query_count,
            "map_state": self.map_state,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "valid": self.valid,
            "policy": self.policy,
            "seed": self.seed,
            "delta": self.delta,
            "reached_delta": self.reached_delta,
            "runtime_ms": self.runtime_ms,
            "asked_tasks": self.asked_tasks,
            "map_probability": self.map_probability,
            "query_history": self.query_history,
        }


def _normalize_observations(
    selected_tasks: Sequence[str],
    observations: ObservationMap,
) -> list[tuple[str, int]]:
    expanded: list[tuple[str, int]] = []
    for task in selected_tasks:
        if task not in observations:
            raise ValueError(f"Missing observation for selected task '{task}'")
        values = observations[task]
        if isinstance(values, int):
            expanded.append((task, _as_binary(values)))
        else:
            values_list = list(values)
            if len(values_list) == 0:
                raise ValueError(f"Observation list for task '{task}' must not be empty.")
            expanded.extend((task, _as_binary(value)) for value in values_list)
    return expanded


def _state_ids(states: Sequence[KnowledgeState], task_ids: Sequence[str]) -> list[str]:
    return [stable_state_id(state, task_ids) for state in states]


def _flatten_query_records(records: list[tuple[str, Sequence[int]]]) -> list[tuple[str, int]]:
    expanded: list[tuple[str, int]] = []
    for task_id, values in records:
        expanded.extend((task_id, _as_binary(value)) for value in values)
    return expanded


def _aggregate_observations(records: list[tuple[str, Sequence[int]]]) -> list[tuple[str, int]]:
    return _flatten_query_records(records)


def _validate_delta(delta: float) -> None:
    if not isfinite(delta) or not (0.0 <= delta <= 1.0):
        raise ValueError("delta must be a finite number in [0.0, 1.0].")


def _validate_non_negative_int(value: int | None, name: str) -> int | None:
    if value is not None and (not isinstance(value, int) or value < 0):
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def solve_noisy_fixed_certificate(
    knowledge_space: KnowledgeSpace,
    selected_tasks: Sequence[str],
    observations: ObservationMap,
    noise: ResponseNoiseModel,
    delta: float = 0.05,
    prior: Mapping[str, float] | None = None,
) -> NoisyFixedCertificate:
    start = perf_counter()
    _validate_delta(delta)

    selected = list(selected_tasks)
    task_set = set(knowledge_space.tasks.task_ids)
    for task_id in selected:
        if task_id not in task_set:
            raise ValueError(f"Task '{task_id}' not in knowledge space")

    expanded = _normalize_observations(selected, observations)
    posterior = infer_state_posterior(
        knowledge_space,
        observations=[(task_id, value) for task_id, value in expanded],
        noise=noise,
        prior=prior,
    )
    valid = posterior.confidence >= 1.0 - delta and posterior.map_state != ""
    return NoisyFixedCertificate(
        task_count=len(knowledge_space.tasks.task_ids),
        state_count=posterior.state_count,
        certificate_size=len(selected),
        selected_tasks=selected,
        valid=valid,
        map_state=posterior.map_state,
        confidence=posterior.confidence,
        map_probability=posterior.confidence,
        entropy=posterior.entropy,
        delta=delta,
        observations=len(expanded),
        runtime_ms=(perf_counter() - start) * 1000,
    )


def solve_noisy_adaptive_certificate(
    knowledge_space: KnowledgeSpace,
    true_state: KnowledgeState,
    policy: str | AdaptiveQuestionPolicy = "entropy_reduction",
    noise: ResponseNoiseModel = ResponseNoiseModel(),
    delta: float = 0.05,
    max_queries: int | None = None,
    attempts_per_query: int = 1,
    seed: int | None = None,
) -> NoisyAdaptiveCertificate:
    start = perf_counter()
    _validate_delta(delta)
    _validate_positive_int(attempts_per_query, "attempts_per_query")
    _validate_non_negative_int(max_queries, "max_queries")

    task_ids = list(knowledge_space.tasks.task_ids)
    ordered_states = sorted(
        [state for state in knowledge_space.valid_states if knowledge_space.is_valid_state(state)],
        key=lambda state: state.as_tuple(task_ids),
    )
    state_count = len(ordered_states)
    state_ids = _state_ids(ordered_states, task_ids)

    if state_count == 0:
        return NoisyAdaptiveCertificate(
            task_count=len(task_ids),
            state_count=0,
            query_count=0,
            map_state="",
            confidence=0.0,
            entropy=0.0,
            valid=False,
            policy="custom" if callable(policy) else str(policy),
            seed=seed,
            delta=delta,
            reached_delta=False,
            runtime_ms=(perf_counter() - start) * 1000,
            method="noisy_adaptive",
            asked_tasks=[],
            map_probability=None,
            query_history=[],
        )

    prior = {sid: 1.0 / state_count for sid in state_ids}
    posterior = infer_state_posterior(
        knowledge_space,
        observations=[],
        noise=noise,
        prior=prior,
    )

    policy_name, policy_fn = resolve_policy(policy)
    rng = Random(seed)

    asked: set[str] = set()
    query_records: list[tuple[str, Sequence[int]]] = []
    history: list[dict[str, object]] = []

    asked_limit = max_queries if max_queries is not None else len(task_ids)
    while (
        len(asked) < asked_limit
        and len(query_records) < asked_limit
        and posterior.confidence < 1.0 - delta
        and posterior.map_state != ""
    ):
        question = policy_fn(
            ordered_states,
            task_ids,
            posterior.state_posteriors,
            asked,
            noise,
            attempts_per_query,
            rng,
        )
        if question is None:
            break

        responses = simulate_probabilistic_response(
            question in true_state,
            noise=noise,
            attempts=attempts_per_query,
            rng=rng,
        )
        responses_list = responses if isinstance(responses, list) else [responses]
        responses_norm = [int(v) for v in responses_list]
        query_records.append((question, responses_norm))
        asked.add(question)

        expanded = _aggregate_observations(query_records)
        posterior = infer_state_posterior(
            knowledge_space,
            expanded,
            noise=noise,
            prior=posterior.state_posteriors,
        )
        history.append(
            {
                "question": question,
                "responses": responses_norm,
                "confidence": posterior.confidence,
                "entropy": posterior.entropy,
            }
        )

    reached_delta = posterior.confidence >= 1.0 - delta and posterior.map_state != ""
    return NoisyAdaptiveCertificate(
        task_count=len(task_ids),
        state_count=state_count,
        query_count=len(query_records),
        map_state=posterior.map_state,
        confidence=posterior.confidence,
        entropy=posterior.entropy,
        valid=reached_delta,
        policy=policy_name,
        seed=seed,
        delta=delta,
        reached_delta=reached_delta,
        runtime_ms=(perf_counter() - start) * 1000,
        method="noisy_adaptive",
        asked_tasks=sorted(asked),
        map_probability=posterior.confidence,
        query_history=history,
    )
