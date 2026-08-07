from __future__ import annotations

import pytest

from capability_certificate_lab.certificate import solve_exact_certificate
from capability_certificate_lab.dsl import (
    execute,
    ConditionNode,
    LoopNode,
    InvalidProgramError,
    PrimitiveNode,
    SequenceNode,
    generate_dsl_world,
    get_primitive,
    make_dsl_response_signature,
)
from capability_certificate_lab.knowledge_space import KnowledgeSpace, TaskUniverse
from capability_certificate_lab.knowledge_space.state import KnowledgeState
from capability_certificate_lab.validation.identifiability.core import stable_state_id


def test_dsl_primitive_execution_matches_membership():
    program = PrimitiveNode(op=get_primitive("MEMORY"))

    assert program.to_task_id() == "MEMORY"
    assert program.required_primitive_ids() == {"MEMORY"}
    assert execute(program, {"MEMORY": True}) == 1
    assert execute(program, {"MEMORY": False}) == 0


def test_dsl_composition_execution_uses_semantics():
    memory = PrimitiveNode(op=get_primitive("MEMORY"))
    search = PrimitiveNode(op=get_primitive("SEARCH"))
    retrieval = SequenceNode((memory, search))

    state_with_both = {"MEMORY": True, "SEARCH": True}
    state_with_one = {"MEMORY": True, "SEARCH": False}

    assert retrieval.to_task_id() == "SEQ[MEMORY,SEARCH]"
    assert retrieval.required_primitive_ids() == {"MEMORY", "SEARCH"}
    assert execute(retrieval, state_with_both) == 1
    assert execute(retrieval, state_with_one) == 0

    cond = ConditionNode(
        condition=memory,
        then=search,
        otherwise=PrimitiveNode(op=get_primitive("COMPARE")),
    )
    assert execute(cond, state_with_both) == 1


def test_dsl_invalid_program_rejected():
    retrieval = SequenceNode((PrimitiveNode(op=get_primitive("MEMORY")), PrimitiveNode(op=get_primitive("SEARCH"))))
    looping = LoopNode(body=retrieval, max_iterations=0)

    with pytest.raises(ValueError, match="Loop max_iterations must be a positive integer"):
        looping.to_dict()

    with pytest.raises(InvalidProgramError, match="Unsupported program node type"):
        execute("not_a_program", {"MEMORY": True})  # type: ignore[arg-type]


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
