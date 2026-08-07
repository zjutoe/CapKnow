from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence


@dataclass(frozen=True)
class ResponseNoiseModel:
    """Slip and guess rates for noisy binary responses."""

    slip: float = 0.0
    guess: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.slip <= 1.0:
            raise ValueError("slip must be between 0.0 and 1.0.")
        if not 0.0 <= self.guess <= 1.0:
            raise ValueError("guess must be between 0.0 and 1.0.")


def response_probability(
    state_has_task: bool,
    noise: ResponseNoiseModel,
) -> float:
    """Return P(Y=1) under one task-specific noisy response."""

    return noise.guess if not state_has_task else 1.0 - noise.slip


def _binary_response_value(value: int) -> int:
    return 1 if value else 0


def simulate_probabilistic_response(
    state_has_task: bool,
    noise: ResponseNoiseModel,
    attempts: int = 1,
    rng: Random | None = None,
) -> int | list[int]:
    """Simulate one or more Bernoulli draws for the same task."""

    if not isinstance(attempts, int) or attempts <= 0:
        raise ValueError("attempts must be a positive integer.")

    generator = rng if rng is not None else Random()
    p_yes = response_probability(state_has_task, noise)
    responses = [_binary_response_value(generator.random() <= p_yes) for _ in range(attempts)]

    return responses[0] if attempts == 1 else responses
