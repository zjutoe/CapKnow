from __future__ import annotations

from collections.abc import Callable, Sequence
from random import Random
from time import perf_counter

from ..knowledge_space.space import KnowledgeSpace
from ..knowledge_space.state import KnowledgeState
from ..validation.identifiability.core import Signature, response_signature, stable_state_id
from .decision_tree import DecisionNode
from .policies import QuestionPolicy, resolve_policy
from .result import AdaptiveCertificate
from .validator import _ordered_valid_states


def _answer_bit(
    state: KnowledgeState,
    question: str,
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature],
) -> int:
    signature = response_signature_fn(state, [question])
    if len(signature) != 1:
        raise ValueError(
            "Response signature length must be one for adaptive certificate evaluation."
        )
    value = signature[0]
    if value not in (0, 1):
        raise ValueError(
            "Adaptive policies require binary response signatures: each response must be 0 or 1."
        )
    return int(value)


def _split_states(
    candidate_states: Sequence[KnowledgeState],
    question: str,
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature],
) -> tuple[list[KnowledgeState], list[KnowledgeState]]:
    yes_states: list[KnowledgeState] = []
    no_states: list[KnowledgeState] = []
    for state in candidate_states:
        if _answer_bit(state, question, response_signature_fn):
            yes_states.append(state)
        else:
            no_states.append(state)
    return no_states, yes_states


def _candidate_state_ids(
    states: Sequence[KnowledgeState],
    task_ids: Sequence[str],
) -> list[str]:
    return [stable_state_id(state, task_ids) for state in states]


def _build_adaptive_tree(
    candidate_states: Sequence[KnowledgeState],
    task_ids: Sequence[str],
    asked: set[str],
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature],
    policy: QuestionPolicy,
    rng: Random | None,
) -> DecisionNode:
    if len(candidate_states) <= 1:
        return DecisionNode(
            question=None,
            candidate_state_ids=_candidate_state_ids(candidate_states, task_ids),
        )

    question = policy(
        candidate_states,
        task_ids,
        asked,
        response_signature_fn,
        rng,
    )
    if question is None:
        for task_id in task_ids:
            if task_id in asked:
                continue
            no_states, yes_states = _split_states(
                candidate_states,
                task_id,
                response_signature_fn,
            )
            if no_states and yes_states:
                raise ValueError(
                    "Adaptive policy returned None while an unasked splitting task remains."
                )
        return DecisionNode(
            question=None,
            candidate_state_ids=_candidate_state_ids(candidate_states, task_ids),
        )
    if question not in task_ids:
        raise ValueError(f"Adaptive policy returned unknown task '{question}'.")
    if question in asked:
        raise ValueError(f"Adaptive policy returned repeated task '{question}'.")

    no_states, yes_states = _split_states(candidate_states, question, response_signature_fn)
    if not no_states or not yes_states:
        raise ValueError(
            f"Adaptive policy returned non-splitting task '{question}' for current candidates."
        )

    next_asked = set(asked)
    next_asked.add(question)

    return DecisionNode(
        question=question,
        yes_child=_build_adaptive_tree(
            yes_states,
            task_ids,
            next_asked,
            response_signature_fn,
            policy,
            rng,
        ),
        no_child=_build_adaptive_tree(
            no_states,
            task_ids,
            next_asked,
            response_signature_fn,
            policy,
            rng,
        ),
    )


def _tree_metrics(
    node: DecisionNode,
    depth: int = 0,
) -> tuple[int, int, float, int]:
    if node.is_leaf():
        state_count = max(len(node.candidate_state_ids or ()), 0)
        depth_sum = float(depth * state_count)
        return 1, state_count, depth_sum, depth

    node_count = 1
    child_states = 0
    depth_sum = 0.0
    max_depth = depth

    if node.yes_child is not None:
        child_count, leaf_states, child_depth_sum, child_max_depth = _tree_metrics(
            node.yes_child,
            depth + 1,
        )
        node_count += child_count
        child_states += leaf_states
        depth_sum += child_depth_sum
        max_depth = max(max_depth, child_max_depth)

    if node.no_child is not None:
        child_count, leaf_states, child_depth_sum, child_max_depth = _tree_metrics(
            node.no_child,
            depth + 1,
        )
        node_count += child_count
        child_states += leaf_states
        depth_sum += child_depth_sum
        max_depth = max(max_depth, child_max_depth)

    return node_count, child_states, depth_sum, max_depth


def solve_adaptive_certificate(
    knowledge_space: KnowledgeSpace,
    policy: str | QuestionPolicy = "entropy",
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature] = response_signature,
    seed: int | None = None,
) -> AdaptiveCertificate:
    start = perf_counter()
    states, task_ids = _ordered_valid_states(knowledge_space)
    if not states:
        raise ValueError("Declared valid_states must not be empty for adaptive certificates.")

    policy_name, policy_fn = resolve_policy(policy)
    rng = Random(seed) if seed is not None else None
    root = _build_adaptive_tree(
        states,
        task_ids,
        set(),
        response_signature_fn,
        policy_fn,
        rng,
    )

    node_count, state_cover_count, depth_sum, worst_case_depth = _tree_metrics(root, depth=0)
    average_depth = depth_sum / state_cover_count if state_cover_count else 0.0
    valid = validate_adaptive_certificate(
        root,
        knowledge_space,
        response_signature_fn=response_signature_fn,
    )

    if not valid:
        worst_case_depth = 0
        average_depth = 0.0

    return AdaptiveCertificate(
        root=root,
        worst_case_depth=worst_case_depth,
        average_depth=average_depth,
        node_count=node_count,
        valid=valid,
        method="adaptive",
        policy=policy_name,
        seed=seed,
        runtime_ms=(perf_counter() - start) * 1000,
    )


def validate_adaptive_certificate(
    certificate: DecisionNode | None,
    knowledge_space: KnowledgeSpace,
    response_signature_fn: Callable[[KnowledgeState, Sequence[str]], Signature] = response_signature,
) -> bool:
    states, task_ids = _ordered_valid_states(knowledge_space)
    if not states:
        return False

    if certificate is None:
        return False

    task_set = set(knowledge_space.tasks.task_ids)

    def _validate_node(
        node: DecisionNode | None,
        candidate_states: Sequence[KnowledgeState],
        asked: set[str],
    ) -> bool:
        if node is None:
            return False
        if node.question is None:
            if len(candidate_states) != 1:
                return False
            candidate_state_ids = list(node.candidate_state_ids or ())
            if len(candidate_state_ids) != 1:
                return False
            expected_id = stable_state_id(candidate_states[0], task_ids)
            return candidate_state_ids == [expected_id]

        if node.question not in task_set or node.question in asked:
            return False
        if node.yes_child is None or node.no_child is None:
            return False

        no_states, yes_states = _split_states(
            candidate_states,
            node.question,
            response_signature_fn,
        )
        if not no_states or not yes_states:
            return False

        next_asked = set(asked)
        next_asked.add(node.question)
        return _validate_node(node.yes_child, yes_states, next_asked) and _validate_node(
            node.no_child,
            no_states,
            next_asked,
        )

    return _validate_node(certificate, states, set())
