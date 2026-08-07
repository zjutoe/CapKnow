from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
from random import Random
from time import perf_counter
from typing import FrozenSet

from ..knowledge_space.space import KnowledgeSpace
from ..knowledge_space.state import KnowledgeState
from ..validation.identifiability.core import Signature, response_signature
from .result import CertificateResult
from .validator import validate_certificate, _ordered_valid_states


def _pair_difference_task_sets(
    states: Sequence[KnowledgeState],
    task_ids: Sequence[str],
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature],
) -> tuple[list[FrozenSet[str]], bool]:
    num_states = len(states)
    signatures = [response_signature_fn(state, task_ids) for state in states]
    pair_differences: list[FrozenSet[str]] = []
    for i in range(num_states):
        for j in range(i + 1, num_states):
            sig_i = signatures[i]
            sig_j = signatures[j]
            if sig_i == sig_j:
                return [], False

            diff = frozenset(
                task_ids[task_idx]
                for task_idx, bits in enumerate(zip(sig_i, sig_j))
                if bits[0] != bits[1]
            )
            pair_differences.append(diff)
    return pair_differences, True


def _is_hitting_set(candidate: Sequence[str], pair_differences: Sequence[FrozenSet[str]]) -> bool:
    selected = set(candidate)
    return all(selected & diff for diff in pair_differences)


def solve_exact_certificate(
    knowledge_space: KnowledgeSpace,
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature] = response_signature,
) -> CertificateResult:
    start = perf_counter()

    states, task_ids = _ordered_valid_states(knowledge_space)
    state_count = len(states)
    total_pairs = state_count * (state_count - 1) // 2

    pair_differences, identifiable = _pair_difference_task_sets(
        states,
        task_ids,
        response_signature_fn,
    )
    if not identifiable:
        return CertificateResult(
            task_count=len(task_ids),
            state_count=state_count,
            certificate_size=0,
            selected_tasks=[],
            valid=False,
            separated_pairs=0,
            total_pairs=total_pairs,
            runtime_ms=(perf_counter() - start) * 1000,
            method="exact",
        )

    if total_pairs == 0:
        return CertificateResult(
            task_count=len(task_ids),
            state_count=state_count,
            certificate_size=0,
            selected_tasks=[],
            valid=True,
            separated_pairs=0,
            total_pairs=0,
            runtime_ms=(perf_counter() - start) * 1000,
            method="exact",
        )

    for size in range(1, len(task_ids) + 1):
        for selected in combinations(task_ids, size):
            if _is_hitting_set(selected, pair_differences):
                selected_tasks = list(selected)
                return CertificateResult(
                    task_count=len(task_ids),
                    state_count=state_count,
                    certificate_size=size,
                    selected_tasks=selected_tasks,
                    valid=validate_certificate(
                        knowledge_space,
                        selected_tasks,
                        response_signature_fn=response_signature_fn,
                    ),
                    separated_pairs=total_pairs,
                    total_pairs=total_pairs,
                    runtime_ms=(perf_counter() - start) * 1000,
                    method="exact",
                )

    # Should not happen when identifiable under current response model.
    return CertificateResult(
        task_count=len(task_ids),
        state_count=state_count,
        certificate_size=len(task_ids),
        selected_tasks=list(task_ids),
        valid=False,
        separated_pairs=0,
        total_pairs=total_pairs,
        runtime_ms=(perf_counter() - start) * 1000,
        method="exact",
    )


def solve_greedy_certificate(
    knowledge_space: KnowledgeSpace,
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature] = response_signature,
) -> CertificateResult:
    start = perf_counter()

    states, task_ids = _ordered_valid_states(knowledge_space)
    state_count = len(states)
    total_pairs = state_count * (state_count - 1) // 2
    pair_differences, identifiable = _pair_difference_task_sets(
        states,
        task_ids,
        response_signature_fn,
    )
    if not identifiable:
        return CertificateResult(
            task_count=len(task_ids),
            state_count=state_count,
            certificate_size=0,
            selected_tasks=[],
            valid=False,
            separated_pairs=0,
            total_pairs=total_pairs,
            runtime_ms=(perf_counter() - start) * 1000,
            method="greedy",
        )
    if total_pairs == 0:
        return CertificateResult(
            task_count=len(task_ids),
            state_count=state_count,
            certificate_size=0,
            selected_tasks=[],
            valid=True,
            separated_pairs=0,
            total_pairs=0,
            runtime_ms=(perf_counter() - start) * 1000,
            method="greedy",
        )

    unresolved = set(range(len(pair_differences)))
    pair_list = list(pair_differences)
    candidate = []
    selected = set[str]()
    while unresolved:
        best_task = ""
        best_gain = -1
        for task_id in task_ids:
            if task_id in selected:
                continue
            gain = 0
            for pair_idx in unresolved:
                if task_id in pair_list[pair_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_task = task_id
        if best_gain <= 0:
            break

        selected.add(best_task)
        candidate.append(best_task)
        unresolved = {
            pair_idx for pair_idx in unresolved if best_task not in pair_list[pair_idx]
        }

    return CertificateResult(
        task_count=len(task_ids),
        state_count=state_count,
        certificate_size=len(candidate),
        selected_tasks=candidate,
        valid=validate_certificate(
            knowledge_space,
            candidate,
            response_signature_fn=response_signature_fn,
        )
        and len(unresolved) == 0,
        separated_pairs=total_pairs - len(unresolved),
        total_pairs=total_pairs,
        runtime_ms=(perf_counter() - start) * 1000,
        method="greedy",
    )


def solve_random_certificate(
    knowledge_space: KnowledgeSpace,
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature] = response_signature,
    seed: int | None = None,
) -> CertificateResult:
    start = perf_counter()

    states, task_ids = _ordered_valid_states(knowledge_space)
    state_count = len(states)
    total_pairs = state_count * (state_count - 1) // 2
    pair_differences, identifiable = _pair_difference_task_sets(
        states,
        task_ids,
        response_signature_fn,
    )
    if not identifiable:
        return CertificateResult(
            task_count=len(task_ids),
            state_count=state_count,
            certificate_size=0,
            selected_tasks=[],
            valid=False,
            separated_pairs=0,
            total_pairs=total_pairs,
            runtime_ms=(perf_counter() - start) * 1000,
            method="random",
        )
    if total_pairs == 0:
        return CertificateResult(
            task_count=len(task_ids),
            state_count=state_count,
            certificate_size=0,
            selected_tasks=[],
            valid=True,
            separated_pairs=0,
            total_pairs=0,
            runtime_ms=(perf_counter() - start) * 1000,
            method="random",
        )

    pair_list = list(pair_differences)
    unresolved = set(range(len(pair_list)))

    rng = Random(seed)
    order = list(task_ids)
    rng.shuffle(order)

    selected = []
    chosen = set[str]()
    for task_id in order:
        if not unresolved:
            break
        chosen.add(task_id)
        selected.append(task_id)
        unresolved = {
            pair_idx for pair_idx in unresolved if task_id not in pair_list[pair_idx]
        }

    return CertificateResult(
        task_count=len(task_ids),
        state_count=state_count,
        certificate_size=len(selected),
        selected_tasks=selected,
        valid=validate_certificate(
            knowledge_space,
            selected,
            response_signature_fn=response_signature_fn,
        )
        and len(unresolved) == 0,
        separated_pairs=total_pairs - len(unresolved),
        total_pairs=total_pairs,
        runtime_ms=(perf_counter() - start) * 1000,
        method="random",
    )
