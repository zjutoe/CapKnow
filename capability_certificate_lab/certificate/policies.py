from __future__ import annotations

from collections.abc import Callable, Sequence
from math import log2
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
    return int(signature[0])


def _available_questions(task_ids: Sequence[str], asked: set[str]) -> list[str]:
    return [task_id for task_id in task_ids if task_id not in asked]


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
    del response_signature_fn
    del candidate_states

    candidates = _available_questions(task_ids, asked)
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

    questions = _available_questions(task_ids, asked)
    if not questions:
        return None

    best_question: str | None = None
    best_gain = -1.0

    total = len(candidate_states)
    for question in questions:
        no_count, yes_count = _split_counts(candidate_states, question, response_signature_fn)
        if yes_count == 0 or no_count == 0:
            gain = 0.0
        else:
            p_yes = yes_count / total
            p_no = no_count / total
            gain = -(p_yes * log2(p_yes) + p_no * log2(p_no))
        if gain > best_gain:
            best_gain = gain
            best_question = question

    return best_question


def select_balanced_split_question(
    candidate_states: Sequence[KnowledgeState],
    task_ids: Sequence[str],
    asked: set[str],
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature],
    rng: Random | None = None,
) -> str | None:
    del rng

    questions = _available_questions(task_ids, asked)
    if not questions:
        return None

    best_question: str | None = None
    best_balance = len(candidate_states) + 1

    for question in questions:
        no_count, yes_count = _split_counts(candidate_states, question, response_signature_fn)
        balance = max(no_count, yes_count)
        if balance < best_balance:
            best_balance = balance
            best_question = question

    return best_question


POLICIES: Mapping[str, QuestionPolicy] = {
    "random": select_random_question,
    "entropy": select_entropy_reduction_question,
    "balanced": select_balanced_split_question,
}


def resolve_policy(
    policy: str | QuestionPolicy,
) -> tuple[str, QuestionPolicy]:
    if callable(policy):
        return "custom", policy
    if policy not in POLICIES:
        raise ValueError(f"Unsupported policy '{policy}'")
    return policy, POLICIES[policy]
