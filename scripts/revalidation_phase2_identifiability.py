from __future__ import annotations

from collections.abc import Sequence

from capability_certificate_lab.generators import (
    generate_chain_world,
    generate_tree_world,
    generate_unstructured_world,
)
from capability_certificate_lab.knowledge_space.space import KnowledgeSpace
from capability_certificate_lab.knowledge_space.state import KnowledgeState
from capability_certificate_lab.knowledge_space.tasks import TaskUniverse
from capability_certificate_lab.validation.identifiability import check_identifiability, response_signature

from revalidation_common import source_binding_before_run, write_result_and_manifest


COMMAND = "PYTHONPATH=. python scripts/revalidation_phase2_identifiability.py"


def _collapsed_signature(state: KnowledgeState, task_ids: Sequence[str]) -> tuple[int, ...]:
    return (1,) * len(task_ids)


def _run_case(name: str, world: KnowledgeSpace, signature_fn=None) -> dict[str, object]:
    report = check_identifiability(
        world,
        response_signature_fn=signature_fn or response_signature,
    )
    return {
        "name": name,
        "task_ids": world.tasks.task_ids,
        **report.to_dict(),
    }


def main() -> None:
    source = source_binding_before_run()
    cases = [
        (
            "chain",
            generate_chain_world(["A", "B", "C", "D"]),
            None,
        ),
        (
            "tree",
            generate_tree_world({"A": ["B", "C"], "B": ["D"]}),
            None,
        ),
        (
            "unstructured",
            generate_unstructured_world(["Q0", "Q1", "Q2"]),
            None,
        ),
        (
            "artificial_full_vector_collision",
            KnowledgeSpace(
                tasks=TaskUniverse(["A", "B"]),
                valid_states=[KnowledgeState(("A",)), KnowledgeState(("B",))],
                metadata={"type": "artificial_collision"},
            ),
            _collapsed_signature,
        ),
    ]
    result = {
        "phase": "phase2_identifiability",
        "cases": [_run_case(name, world, signature_fn) for name, world, signature_fn in cases],
    }
    artifacts = write_result_and_manifest(
        phase="phase2_identifiability",
        script_name="scripts/revalidation_phase2_identifiability.py",
        command=COMMAND,
        config={"cases": [name for name, _, _ in cases]},
        result=result,
        source=source,
    )
    print(artifacts)


if __name__ == "__main__":
    main()
