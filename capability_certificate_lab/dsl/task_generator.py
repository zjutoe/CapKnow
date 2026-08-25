from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from ..knowledge_space.space import KnowledgeSpace
from ..knowledge_space.state import KnowledgeState
from ..knowledge_space.tasks import TaskUniverse
from ..validation.validator import validate_state
from .composition import DEFAULT_COMPOSITION_RULES, CompositionRule, build_composite_program
from .primitives import PRIMITIVE_TASKS, get_primitive
from .program import Program, PrimitiveNode


def generate_dsl_primitive_task_map() -> dict[str, Program]:
    return {
        op_id: PrimitiveNode(op=get_primitive(op_id))
        for op_id in PRIMITIVE_TASKS
    }


def _build_composite_rules(
    base_programs: dict[str, Program],
    composition_rules: Sequence[CompositionRule],
) -> dict[str, Program]:
    _validate_composition_rules(base_programs, composition_rules)
    program_map = dict(base_programs)
    pending = list(composition_rules)
    changed = True

    while pending and changed:
        changed = False
        next_pending: list[CompositionRule] = []
        for rule in pending:
            left_program = program_map.get(rule.left)
            right_program = program_map.get(rule.right)
            if left_program is None or right_program is None:
                next_pending.append(rule)
                continue
            if rule.result in program_map:
                raise ValueError(f"Composition result id already exists: {rule.result}")
            program_map[rule.result] = build_composite_program(
                rule,
                left_program,
                right_program,
            )
            changed = True

        pending = next_pending

    if pending:
        missing_left = sorted({rule.left for rule in pending if rule.left not in program_map})
        missing_right = sorted({rule.right for rule in pending if rule.right not in program_map})
        raise ValueError(
            "Cannot resolve DSL composition rules for current task set. "
            f"Missing left={missing_left}, right={missing_right}."
        )

    return program_map


def _validate_composition_rules(
    base_programs: Mapping[str, Program],
    composition_rules: Sequence[CompositionRule],
) -> None:
    primitive_ids = set(base_programs)
    result_ids: set[str] = set()
    pair_results: dict[tuple[str, str], str] = {}
    for rule in composition_rules:
        if rule.result in primitive_ids:
            raise ValueError(
                f"Composition result id '{rule.result}' collides with a primitive id."
            )
        if rule.result in result_ids:
            raise ValueError(f"Duplicate composition result id '{rule.result}'.")
        result_ids.add(rule.result)

        pair = (rule.left, rule.right)
        previous = pair_results.get(pair)
        if previous is not None and previous != rule.result:
            raise ValueError(
                f"Conflicting composition results for {rule.left}+{rule.right}: "
                f"{previous} vs {rule.result}."
            )
        pair_results[pair] = rule.result


def _build_composition_constraints_checks(
    composition_rules: Sequence[CompositionRule],
) -> tuple[Callable[[KnowledgeState], bool], ...]:
    checks: list[Callable[[KnowledgeState], bool]] = []

    for rule in composition_rules:
        def _check(state: KnowledgeState, rule=rule) -> bool:
            return not (
                rule.left in state
                and rule.right in state
                and rule.result not in state
            )

        checks.append(_check)

    return tuple(checks)


def _to_dicts(rules: Sequence[CompositionRule]) -> tuple[dict[str, str], ...]:
    return tuple(rule.to_dict() for rule in rules)


def generate_dsl_world(
    include_composite: bool = True,
    composition_rules: Sequence[CompositionRule] = DEFAULT_COMPOSITION_RULES,
) -> tuple[KnowledgeSpace, dict[str, Program]]:
    primitive_map = generate_dsl_primitive_task_map()

    program_map = dict(primitive_map)
    active_composition_rules: tuple[CompositionRule, ...] = ()
    composition_constraints: tuple[dict[str, str], ...] = ()
    composition_constraint_checks: tuple[Callable[[KnowledgeState], bool], ...] = ()
    if include_composite:
        program_map = _build_composite_rules(program_map, composition_rules)
        active_composition_rules = tuple(composition_rules)
        composition_constraints = _to_dicts(active_composition_rules)
        composition_constraint_checks = _build_composition_constraints_checks(active_composition_rules)

    all_task_ids = list(program_map.keys())

    prerequisites: dict[str, tuple[str, ...]] = {}
    for rule in active_composition_rules:
        prerequisites[rule.result] = (rule.left, rule.right)

    universe = TaskUniverse(all_task_ids)
    universe_task_ids = universe.task_ids
    valid_states: list[KnowledgeState] = []
    for mask in range(1 << len(universe_task_ids)):
        selected = [
            universe_task_ids[idx]
            for idx in range(len(universe_task_ids))
            if (mask >> idx) & 1
        ]
        state = KnowledgeState(selected)
        if validate_state(
            state,
            task_universe=universe,
            prerequisites=prerequisites,
            composition_constraints=composition_constraint_checks,
        ):
            valid_states.append(state)

    world = KnowledgeSpace(
        tasks=universe,
        valid_states=sorted(
            valid_states,
            key=lambda state: (len(state.tasks), state.as_tuple(universe_task_ids)),
        ),
        generator_rules={
            "type": "dsl",
            "composition_rules": _to_dicts(active_composition_rules),
            "prerequisites": {k: tuple(v) for k, v in prerequisites.items()},
            "composition_constraints": composition_constraints,
        },
        metadata={"composition_rules": _to_dicts(active_composition_rules)},
    )

    return world, program_map
