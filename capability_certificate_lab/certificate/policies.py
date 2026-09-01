from __future__ import annotations

from collections.abc import Callable, Sequence
from random import Random
from typing import Mapping

from ..knowledge_space.state import KnowledgeState
from ..validation.identifiability.core import Signature


QuestionPolicy = Callable[
    [
        Sequence[KnowledgeState],
        Sequence[str],
        set[str],
        Callable[[KnowledgeState, Sequence[str]], Signature],
        Random | None,
    ],
    str | None,
]


def _answer_bit(
    state: KnowledgeState,
    question: str,
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature],
) -> int:
    signature = response_signature_fn(state, [question])
    if len(signature) != 1:
        raise ValueError(
            "Response signature length must be one for adaptive policy evaluation."
        )
    value = signature[0]
    if value not in (0, 1):
        raise ValueError(
            "Adaptive policies require binary response signatures: each response must be 0 or 1."
        )
    return int(value)


def _available_questions(task_ids: Sequence[str], asked: set[str]) -> list[str]:
    return [task_id for task_id in task_ids if task_id not in asked]


def _splitting_questions(
    candidate_states: Sequence[KnowledgeState],
    task_ids: Sequence[str],
    asked: set[str],
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature],
) -> list[tuple[str, int, int]]:
    questions: list[tuple[str, int, int]] = []
    for question in _available_questions(task_ids, asked):
        no_count, yes_count = _split_counts(
            candidate_states,
            question,
            response_signature_fn,
        )
        if no_count and yes_count:
            questions.append((question, no_count, yes_count))
    return questions


def _split_counts(
    candidate_states: Sequence[KnowledgeState],
    question: str,
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature],
) -> tuple[int, int]:
    yes_count = 0
    for state in candidate_states:
        response = _answer_bit(state, question, response_signature_fn)
        if response:
            yes_count += 1
    no_count = len(candidate_states) - yes_count
    return no_count, yes_count


def select_random_question(
    candidate_states: Sequence[KnowledgeState],
    task_ids: Sequence[str],
    asked: set[str],
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature],
    rng: Random | None = None,
) -> str | None:
    candidates = [
        question
        for question, _, _ in _splitting_questions(
            candidate_states,
            task_ids,
            asked,
            response_signature_fn,
        )
    ]
    if not candidates:
        return None

    chooser = rng if rng is not None else Random()
    return chooser.choice(candidates)


def select_entropy_reduction_question(
    candidate_states: Sequence[KnowledgeState],
    task_ids: Sequence[str],
    asked: set[str],
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature],
    rng: Random | None = None,
) -> str | None:
    del rng

    questions = _splitting_questions(
        candidate_states,
        task_ids,
        asked,
        response_signature_fn,
    )
    if not questions:
        return None

    best_question: str | None = None
    best_score = -1

    for question, no_count, yes_count in questions:
        score = min(no_count, yes_count)
        if score > best_score:
            best_score = score
            best_question = question

    return best_question


POLICIES: Mapping[str, QuestionPolicy] = {
    "random": select_random_question,
    "entropy": select_entropy_reduction_question,
}


def resolve_policy(
    policy: str | QuestionPolicy,
) -> tuple[str, QuestionPolicy]:
    if callable(policy):
        return "custom", policy
    if policy not in POLICIES:
        raise ValueError(f"Unsupported policy '{policy}'")
    return policy, POLICIES[policy]
