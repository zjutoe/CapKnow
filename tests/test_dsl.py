from __future__ import annotations

import pytest

from capability_certificate_lab.certificate import solve_exact_certificate
from capability_certificate_lab.dsl import (
    CompositionRule,
    execute,
    ConditionNode,
    ExecutionResult,
    LoopNode,
    InvalidProgramError,
    MissingCapabilityError,
    PrimitiveNode,
    SequenceNode,
    default_input_context,
    generate_dsl_world,
    get_primitive,
    make_dsl_response_signature,
)
import capability_certificate_lab.dsl.executor as executor_module
import capability_certificate_lab.dsl.primitives as primitives_module
from capability_certificate_lab.dsl.primitives import PrimitiveOperation
from capability_certificate_lab.knowledge_space import KnowledgeSpace, TaskUniverse
from capability_certificate_lab.knowledge_space.state import KnowledgeState
from capability_certificate_lab.validation.identifiability.core import stable_state_id


def test_dsl_primitive_execution_matches_membership():
    program = PrimitiveNode(op=get_primitive("MEMORY"))
    input_context = {"memory": {"answer": 42}, "key": "answer", "items": [42]}

    assert program.to_task_id() == "MEMORY"
    assert program.required_primitive_ids() == {"MEMORY"}
    result = execute(program, {"MEMORY": True}, input_context)
    assert isinstance(result, ExecutionResult)
    assert result.output_type == "memory_context"
    assert result.value["value"] == 42
    with pytest.raises(MissingCapabilityError, match="MEMORY"):
        execute(program, {"MEMORY": False}, input_context)

    add = PrimitiveNode(op=get_primitive("ADD"))
    assert execute(add, {"ADD": True}, (2, 3)).value == 5
    for invalid_pair in ((True, 3), (2, False)):
        with pytest.raises(InvalidProgramError, match="ADD requires numeric inputs"):
            execute(add, {"ADD": True}, invalid_pair)
    compare = PrimitiveNode(op=get_primitive("COMPARE"))
    assert execute(compare, {"COMPARE": True}, (5, 5)).value is True
    with pytest.raises(InvalidProgramError, match="requires a two-value input"):
        execute(compare, {"COMPARE": True}, True)


def test_dsl_capability_values_must_be_boolean():
    add = PrimitiveNode(op=get_primitive("ADD"))

    with pytest.raises(InvalidProgramError, match="must be a boolean"):
        execute(add, {"ADD": "false"}, (2, 3))
    with pytest.raises(InvalidProgramError, match="must be a boolean"):
        execute(add, {"ADD": 1}, (2, 3))
    with pytest.raises(MissingCapabilityError, match="ADD"):
        execute(add, {"ADD": False}, (2, 3))
    assert execute(add, {"ADD": True}, (2, 3)).value == 5


def test_dsl_composition_execution_uses_semantics():
    memory = PrimitiveNode(op=get_primitive("MEMORY"))
    search = PrimitiveNode(op=get_primitive("SEARCH"))
    retrieval = SequenceNode((memory, search))
    input_context = {
        "memory": {"needle": "A"},
        "key": "needle",
        "items": ["A", "B"],
    }

    state_with_both = {"MEMORY": True, "SEARCH": True}
    state_with_one = {"MEMORY": True, "SEARCH": False}

    assert retrieval.to_task_id() == "SEQ[MEMORY,SEARCH]"
    assert retrieval.required_primitive_ids() == {"MEMORY", "SEARCH"}
    assert execute(retrieval, state_with_both, input_context).value is True
    with pytest.raises(MissingCapabilityError, match="SEARCH"):
        execute(retrieval, state_with_one, input_context)

    cond = ConditionNode(
        condition=PrimitiveNode(op=get_primitive("COMPARE")),
        then=search,
        otherwise=PrimitiveNode(op=get_primitive("COMPARE")),
    )
    condition_input = {
        "left": 1,
        "right": 1,
        "items": ["needle"],
        "target": "needle",
    }
    assert execute(
        cond,
        {"CONDITION": True, "COMPARE": True, "SEARCH": True},
        condition_input,
    ).value is True
    inactive_search_input = {**condition_input, "right": 2}
    with pytest.raises(MissingCapabilityError, match="SEARCH"):
        execute(
            cond,
            {"CONDITION": True, "COMPARE": True},
            inactive_search_input,
        )

    condition = PrimitiveNode(op=get_primitive("CONDITION"))
    assert execute(condition, {"CONDITION": True}, True).value is True
    assert execute(condition, {"CONDITION": True}, {"condition": False}).value is False
    assert execute(condition, {"CONDITION": True}, ["match"]).value is True
    assert execute(condition, {"CONDITION": True}, []).value is False
    for invalid_input in ("false", {"condition": "false"}, {"other": True}, 1):
        with pytest.raises(InvalidProgramError, match="CONDITION"):
            execute(condition, {"CONDITION": True}, invalid_input)


def test_dsl_invalid_program_rejected():
    retrieval = SequenceNode((PrimitiveNode(op=get_primitive("MEMORY")), PrimitiveNode(op=get_primitive("SEARCH"))))
    looping = LoopNode(body=retrieval, max_iterations=0)

    with pytest.raises(ValueError, match="Loop max_iterations must be a positive integer"):
        looping.to_dict()

    with pytest.raises(InvalidProgramError, match="Unsupported program node type"):
        execute("not_a_program", {"MEMORY": True})  # type: ignore[arg-type]

    with pytest.raises(InvalidProgramError, match="ADD requires"):
        execute(PrimitiveNode(op=get_primitive("ADD")), {"ADD": True}, "bad")

    spoofed = PrimitiveNode(
        op=PrimitiveOperation(
            op_id="ADD",
            input_type="numeric_pair",
            output_type="bad_type",
        )
    )
    with pytest.raises(InvalidProgramError, match="does not match the registry"):
        execute(spoofed, {"ADD": True}, (1, 2))


def test_loop_requires_loop_capability_and_returns_concrete_value():
    loop = LoopNode(
        body=PrimitiveNode(op=get_primitive("CONDITION")),
        max_iterations=2,
    )

    with pytest.raises(MissingCapabilityError, match="LOOP"):
        execute(loop, {"CONDITION": True}, True)

    result = execute(loop, {"LOOP": True, "CONDITION": True}, True)
    assert result.value is True
    assert result.output_type == "bool"


@pytest.mark.parametrize("max_iterations", [1.5, True, False, 0, -1])
def test_loop_rejects_positive_integer_contract_violations(max_iterations):
    loop = LoopNode(
        body=PrimitiveNode(op=get_primitive("CONDITION")),
        max_iterations=max_iterations,
    )

    with pytest.raises(InvalidProgramError, match="Loop max_iterations must be a positive integer"):
        execute(loop, {"LOOP": True, "CONDITION": True}, True)


def test_loop_positive_integer_executes_exact_iteration_count(monkeypatch):
    increment = PrimitiveOperation(
        op_id="INC",
        input_type="number",
        output_type="number",
    )
    monkeypatch.setitem(primitives_module.PRIMITIVES, "INC", increment)
    original_execute_primitive = executor_module._execute_primitive

    def execute_test_primitive(op_id, input_context):
        if op_id == "INC":
            if type(input_context) is not int:
                raise InvalidProgramError("INC requires integer input.")
            return input_context + 1
        return original_execute_primitive(op_id, input_context)

    monkeypatch.setattr(executor_module, "_execute_primitive", execute_test_primitive)

    one_iteration = LoopNode(
        body=PrimitiveNode(op=get_primitive("INC")),
        max_iterations=1,
    )
    two_iterations = LoopNode(
        body=PrimitiveNode(op=get_primitive("INC")),
        max_iterations=2,
    )

    assert execute(one_iteration, {"LOOP": True, "INC": True}, 0).value == 1
    assert execute(two_iterations, {"LOOP": True, "INC": True}, 0).value == 2


def _state_id_list(world):
    return [stable_state_id(state, world.tasks.task_ids) for state in world.valid_states]


def test_same_capability_state_reproducible():
    world_a, programs_a = generate_dsl_world()
    world_b, programs_b = generate_dsl_world()

    assert world_a.tasks.task_ids == world_b.tasks.task_ids
    assert _state_id_list(world_a) == _state_id_list(world_b)
    assert programs_a.keys() == programs_b.keys()


def test_composition_task_does_not_leak_primitive_label():
    world, task_programs = generate_dsl_world()

    signature_fn = make_dsl_response_signature(task_programs)
    signature = signature_fn(
        KnowledgeState(("MEMORY", "SEARCH", "RETRIEVAL")),
        tuple(world.tasks.task_ids),
    )
    retrieval_index = world.tasks.task_ids.index("RETRIEVAL")
    memory_index = world.tasks.task_ids.index("MEMORY")

    assert signature[retrieval_index] == 1
    assert signature[memory_index] == 1

    no_search = signature_fn(
        KnowledgeState(("CONDITION", "MEMORY")),
        tuple(world.tasks.task_ids),
    )
    assert no_search[retrieval_index] == 0
    assert no_search[memory_index] == 1


def test_dsl_signature_is_deterministic_and_invalid_programs_propagate():
    world, task_programs = generate_dsl_world()
    signature_fn = make_dsl_response_signature(task_programs)
    state = KnowledgeState(("MEMORY", "SEARCH", "RETRIEVAL"))

    assert signature_fn(state, tuple(world.tasks.task_ids)) == signature_fn(
        state,
        tuple(world.tasks.task_ids),
    )

    bad_signature = make_dsl_response_signature({"BAD": SequenceNode(())})
    with pytest.raises(InvalidProgramError, match="SequenceNode requires"):
        bad_signature(KnowledgeState(()), ("BAD",))

    hidden_bad = make_dsl_response_signature(
        {
            "BAD": SequenceNode(
                (
                    PrimitiveNode(op=get_primitive("MEMORY")),
                    SequenceNode(()),
                )
            )
        }
    )
    with pytest.raises(InvalidProgramError, match="SequenceNode requires"):
        hidden_bad(KnowledgeState(()), ("BAD",))


def test_dsl_world_can_run_certificate_solvers():
    world, task_programs = generate_dsl_world()
    signature_fn = make_dsl_response_signature(task_programs)
    result = solve_exact_certificate(world, response_signature_fn=signature_fn)

    assert result.task_count == len(world.tasks.task_ids)
    assert result.state_count == len(world.valid_states)
    assert result.task_count >= 1
    assert result.certificate_size <= result.task_count
    assert isinstance(result.valid, bool)
    assert result.valid

    primitive_world, primitive_programs = generate_dsl_world(include_composite=False)
    primitive_signature_fn = make_dsl_response_signature(primitive_programs)
    primitive_result = solve_exact_certificate(
        primitive_world,
        response_signature_fn=primitive_signature_fn,
    )
    assert primitive_result.valid


def test_dsl_world_without_composite_tasks_has_empty_composition_meta():
    primitive_world, _ = generate_dsl_world(include_composite=False)

    assert primitive_world.generator_rules["composition_rules"] == ()
    assert primitive_world.generator_rules["composition_constraints"] == ()
    assert primitive_world.metadata["composition_rules"] == ()


def test_dsl_composition_result_conflicts_raise_and_chains_resolve():
    with pytest.raises(ValueError, match="collides with a primitive"):
        generate_dsl_world(
            composition_rules=(CompositionRule("MEMORY", "SEARCH", "ADD"),),
        )

    with pytest.raises(ValueError, match="Duplicate composition result"):
        generate_dsl_world(
            composition_rules=(
                CompositionRule("MEMORY", "SEARCH", "CUSTOM"),
                CompositionRule("FILTER", "CONDITION", "CUSTOM"),
            ),
        )

    chained_world, programs = generate_dsl_world(
        composition_rules=(
            CompositionRule("FILTER", "CONDITION", "FILTER_CHECK"),
            CompositionRule("FILTER_CHECK", "LOOP", "FILTER_CHECK_LOOP"),
        ),
    )
    assert "FILTER_CHECK_LOOP" in programs
    assert chained_world.metadata["composition_rules"] == (
        {"left": "FILTER", "right": "CONDITION", "result": "FILTER_CHECK"},
        {"left": "FILTER_CHECK", "right": "LOOP", "result": "FILTER_CHECK_LOOP"},
    )


def test_knowledge_space_composition_constraints_accept_mapping_and_callable_contracts():
    tasks = TaskUniverse(("A", "B", "C"))

    mapping_world = KnowledgeSpace(
        tasks=tasks,
        valid_states=(),
        generator_rules={
            "composition_constraints": [
                {"left": "A", "right": "B", "result": "C"},
            ],
        },
    )

    callable_world = KnowledgeSpace(
        tasks=tasks,
        valid_states=(),
        generator_rules={
            "composition_constraints": [
                lambda state: not (
                    "A" in state and "B" in state and "C" not in state
                ),
            ],
        },
    )

    assert mapping_world.is_valid_state(KnowledgeState(("A", "B", "C")))
    assert not mapping_world.is_valid_state(KnowledgeState(("A", "B")))
    assert callable_world.is_valid_state(KnowledgeState(("C",)))
    assert not callable_world.is_valid_state(KnowledgeState(("A", "B")))
