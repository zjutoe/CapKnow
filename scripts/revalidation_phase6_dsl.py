from __future__ import annotations

from capability_certificate_lab.certificate import solve_exact_certificate
from capability_certificate_lab.dsl import (
    CompositionRule,
    DEFAULT_COMPOSITION_RULES,
    PrimitiveNode,
    SequenceNode,
    default_input_context,
    execute,
    generate_dsl_world,
    get_primitive,
    make_dsl_response_signature,
)
from capability_certificate_lab.generators import generate_unstructured_world
from capability_certificate_lab.knowledge_space.state import KnowledgeState

from revalidation_common import (
    canonical_json_bytes,
    sha256_bytes,
    source_binding_before_run,
    write_result_and_manifest,
)


COMMAND = "PYTHONPATH=. python scripts/revalidation_phase6_dsl.py"
HELD_OUT_RULE = CompositionRule("FILTER", "CONDITION", "FILTER_CHECK")
HELD_OUT_INPUT_CONTEXT = {"items": ["keep", "drop", "keep"], "target": "keep"}
EXPECTED_MEMORY_RESULT = {
    "items": [42],
    "key": "answer",
    "memory": {"answer": 42},
    "target": 42,
    "value": 42,
}
EXPECTED_COMPOSITE_MEMORY_RESULT = {
    "items": ["A", "B"],
    "key": "needle",
    "memory": {"needle": "A"},
    "target": "A",
    "value": "A",
}
EXPECTED_PRIMITIVE_EXECUTION = {
    "ADD": {"value": 5, "output_type": "number"},
    "COMPARE": {"value": True, "output_type": "bool"},
    "MEMORY": {"value": EXPECTED_MEMORY_RESULT, "output_type": "memory_context"},
    "SEARCH": {"value": True, "output_type": "bool"},
    "FILTER": {"value": ["keep", "keep"], "output_type": "list"},
    "LOOP": {"value": "unchanged", "output_type": "loop_value"},
    "CONDITION": {"value": False, "output_type": "bool"},
}
EXPECTED_COMPOSITE_EXECUTION = {
    "program": "SEQ[MEMORY,SEARCH]",
    "required_primitives": ["MEMORY", "SEARCH"],
    "step_results_without_composite_label": [
        {
            "program": "MEMORY",
            "value": EXPECTED_COMPOSITE_MEMORY_RESULT,
            "output_type": "memory_context",
        },
        {"program": "SEARCH", "value": True, "output_type": "bool"},
    ],
    "result_without_composite_label": {"value": True, "output_type": "bool"},
}
EXPECTED_HELD_OUT_COMPOSITION = {
    "rule": HELD_OUT_RULE.to_dict(),
    "input_context": HELD_OUT_INPUT_CONTEXT,
    "present_in_default_rules": False,
    "program": "SEQ[FILTER,CONDITION]",
    "required_primitives": ["CONDITION", "FILTER"],
    "step_results_without_composite_label": [
        {"program": "FILTER", "value": ["keep", "keep"], "output_type": "list"},
        {"program": "CONDITION", "value": True, "output_type": "bool"},
    ],
    "result_without_composite_label": {"value": True, "output_type": "bool"},
}
EXPECTED_CERTIFICATE_TRANSFER_INVARIANTS = {
    "direct_state_count": 128,
    "dsl_state_count": 128,
    "direct_certificate_size": 7,
    "dsl_certificate_size": 7,
    "direct_valid": True,
    "dsl_valid": True,
    "same_state_count": True,
    "same_certificate_cost": True,
    "direct_selected_tasks": [
        "ADD",
        "COMPARE",
        "MEMORY",
        "SEARCH",
        "FILTER",
        "LOOP",
        "CONDITION",
    ],
    "dsl_selected_tasks": [
        "ADD",
        "COMPARE",
        "MEMORY",
        "SEARCH",
        "FILTER",
        "LOOP",
        "CONDITION",
    ],
}
EXPECTED_REPRODUCIBILITY = {
    "first": (0, 0, 1, 1, 0, 0, 0, 1, 0),
    "second": (0, 0, 1, 1, 0, 0, 0, 1, 0),
    "equal": True,
    "canonical_sha256": "6ad9598c205cc3fe9a3b7e864a2c998e9f402c8da0c67a7eff0c9b67504e11d0",
}
ACCEPTANCE_ORACLES = {
    "primitive_execution": EXPECTED_PRIMITIVE_EXECUTION,
    "composite_execution": EXPECTED_COMPOSITE_EXECUTION,
    "held_out_composition": EXPECTED_HELD_OUT_COMPOSITION,
    "certificate_transfer_invariants": EXPECTED_CERTIFICATE_TRANSFER_INVARIANTS,
    "reproducibility": EXPECTED_REPRODUCIBILITY,
}


def _assert_equal(label, actual, expected):
    if actual != expected:
        raise RuntimeError(f"{label} mismatch. expected={expected!r}, actual={actual!r}")


def _default_composition_rules():
    return [rule.to_dict() for rule in DEFAULT_COMPOSITION_RULES]


def _result_to_dict(result):
    return {"value": result.value, "output_type": result.output_type}


def _primitive_checks():
    return {
        "ADD": _result_to_dict(
            execute(PrimitiveNode(get_primitive("ADD")), {"ADD": True}, (2, 3))
        ),
        "COMPARE": _result_to_dict(execute(
            PrimitiveNode(get_primitive("COMPARE")),
            {"COMPARE": True},
            (5, 5),
        )),
        "MEMORY": _result_to_dict(execute(
            PrimitiveNode(get_primitive("MEMORY")),
            {"MEMORY": True},
            {"memory": {"answer": 42}, "key": "answer", "items": [42]},
        )),
        "SEARCH": _result_to_dict(execute(
            PrimitiveNode(get_primitive("SEARCH")),
            {"SEARCH": True},
            {"items": ["needle"], "target": "needle"},
        )),
        "FILTER": _result_to_dict(execute(
            PrimitiveNode(get_primitive("FILTER")),
            {"FILTER": True},
            {"items": ["keep", "drop", "keep"], "target": "keep"},
        )),
        "LOOP": _result_to_dict(execute(
            PrimitiveNode(get_primitive("LOOP")),
            {"LOOP": True},
            "unchanged",
        )),
        "CONDITION": _result_to_dict(execute(
            PrimitiveNode(get_primitive("CONDITION")),
            {"CONDITION": True},
            {"condition": False},
        )),
    }


def _composite_checks():
    retrieval = SequenceNode(
        (
            PrimitiveNode(get_primitive("MEMORY")),
            PrimitiveNode(get_primitive("SEARCH")),
        )
    )
    context = {"memory": {"needle": "A"}, "key": "needle", "items": ["A", "B"]}
    memory_result = execute(
        PrimitiveNode(get_primitive("MEMORY")),
        {"MEMORY": True, "SEARCH": True},
        context,
    )
    search_result = execute(
        PrimitiveNode(get_primitive("SEARCH")),
        {"MEMORY": True, "SEARCH": True},
        memory_result.value,
    )
    return {
        "program": retrieval.to_task_id(),
        "required_primitives": sorted(retrieval.required_primitive_ids()),
        "step_results_without_composite_label": [
            {"program": "MEMORY", **_result_to_dict(memory_result)},
            {"program": "SEARCH", **_result_to_dict(search_result)},
        ],
        "result_without_composite_label": _result_to_dict(execute(
            retrieval,
            {"MEMORY": True, "SEARCH": True},
            context,
        )),
    }


def _held_out_composition():
    held_out = SequenceNode(
        (
            PrimitiveNode(get_primitive("FILTER")),
            PrimitiveNode(get_primitive("CONDITION")),
        )
    )
    filter_result = execute(
        PrimitiveNode(get_primitive("FILTER")),
        {"FILTER": True, "CONDITION": True},
        HELD_OUT_INPUT_CONTEXT,
    )
    condition_result = execute(
        PrimitiveNode(get_primitive("CONDITION")),
        {"FILTER": True, "CONDITION": True},
        filter_result.value,
    )
    return {
        "rule": HELD_OUT_RULE.to_dict(),
        "input_context": HELD_OUT_INPUT_CONTEXT,
        "present_in_default_rules": HELD_OUT_RULE in DEFAULT_COMPOSITION_RULES,
        "program": held_out.to_task_id(),
        "required_primitives": sorted(held_out.required_primitive_ids()),
        "step_results_without_composite_label": [
            {"program": "FILTER", **_result_to_dict(filter_result)},
            {"program": "CONDITION", **_result_to_dict(condition_result)},
        ],
        "result_without_composite_label": _result_to_dict(execute(
            held_out,
            {"FILTER": True, "CONDITION": True},
            HELD_OUT_INPUT_CONTEXT,
        )),
    }


def _certificate_transfer():
    primitive_world, primitive_programs = generate_dsl_world(include_composite=False)
    direct_world = generate_unstructured_world(primitive_world.tasks.task_ids)
    direct = solve_exact_certificate(direct_world)
    dsl = solve_exact_certificate(
        primitive_world,
        response_signature_fn=make_dsl_response_signature(primitive_programs),
    )
    return {
        "direct": direct.to_dict(),
        "dsl": dsl.to_dict(),
        "same_state_count": len(direct_world.valid_states) == len(primitive_world.valid_states),
        "same_certificate_cost": direct.certificate_size == dsl.certificate_size,
        "note": "Pure primitive DSL world matches the direct-task world; composite worlds add derived labels and are compared separately.",
    }


def _certificate_transfer_invariants(result):
    return {
        "direct_state_count": result["direct"]["state_count"],
        "dsl_state_count": result["dsl"]["state_count"],
        "direct_certificate_size": result["direct"]["certificate_size"],
        "dsl_certificate_size": result["dsl"]["certificate_size"],
        "direct_valid": result["direct"]["valid"],
        "dsl_valid": result["dsl"]["valid"],
        "same_state_count": result["same_state_count"],
        "same_certificate_cost": result["same_certificate_cost"],
        "direct_selected_tasks": result["direct"]["selected_tasks"],
        "dsl_selected_tasks": result["dsl"]["selected_tasks"],
    }


def _reproducibility():
    world, programs = generate_dsl_world()
    signature_fn = make_dsl_response_signature(programs, input_context=default_input_context())
    state = KnowledgeState(("MEMORY", "SEARCH", "RETRIEVAL"))
    first = signature_fn(state, tuple(world.tasks.task_ids))
    second = signature_fn(state, tuple(world.tasks.task_ids))
    return {
        "first": first,
        "second": second,
        "equal": first == second,
        "canonical_sha256": sha256_bytes(canonical_json_bytes({"first": first, "second": second})),
    }


def _assert_phase6_oracles(result):
    _assert_equal(
        "primitive_execution",
        result["primitive_execution"],
        EXPECTED_PRIMITIVE_EXECUTION,
    )
    _assert_equal(
        "composite_execution",
        result["composite_execution"],
        EXPECTED_COMPOSITE_EXECUTION,
    )
    _assert_equal(
        "held_out_composition",
        result["held_out_composition"],
        EXPECTED_HELD_OUT_COMPOSITION,
    )
    _assert_equal(
        "certificate_transfer_invariants",
        _certificate_transfer_invariants(result["certificate_transfer"]),
        EXPECTED_CERTIFICATE_TRANSFER_INVARIANTS,
    )
    _assert_equal(
        "reproducibility",
        result["reproducibility"],
        EXPECTED_REPRODUCIBILITY,
    )


def main() -> None:
    source = source_binding_before_run()
    result = {
        "phase": "phase6_dsl",
        "primitive_execution": _primitive_checks(),
        "composite_execution": _composite_checks(),
        "held_out_composition": _held_out_composition(),
        "certificate_transfer": _certificate_transfer(),
        "reproducibility": _reproducibility(),
        "acceptance_oracles": ACCEPTANCE_ORACLES,
        "default_composition_rules": _default_composition_rules(),
    }
    _assert_phase6_oracles(result)
    artifacts = write_result_and_manifest(
        phase="phase6_dsl",
        script_name="scripts/revalidation_phase6_dsl.py",
        command=COMMAND,
        config={
            "held_out_rule": "FILTER+CONDITION->FILTER_CHECK",
            "held_out_input_context": HELD_OUT_INPUT_CONTEXT,
            "default_composition_rules": _default_composition_rules(),
            "acceptance_oracles": ACCEPTANCE_ORACLES,
        },
        result=result,
        source=source,
    )
    print(artifacts)


if __name__ == "__main__":
    main()
