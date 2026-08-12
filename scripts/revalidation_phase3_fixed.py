from __future__ import annotations

from capability_certificate_lab.certificate import (
    solve_exact_certificate,
    solve_greedy_certificate,
    solve_random_certificate,
    validate_certificate,
)
from capability_certificate_lab.generators import (
    generate_chain_world,
    generate_tree_world,
    generate_unstructured_world,
)

from revalidation_common import source_binding_before_run, write_result_and_manifest


COMMAND = "PYTHONPATH=. python scripts/revalidation_phase3_fixed.py"
RANDOM_SEED = 123


def _run_world(name, world):
    results = {}
    for method, solver in (
        ("exact", solve_exact_certificate),
        ("greedy", solve_greedy_certificate),
    ):
        outcome = solver(world)
        results[method] = {
            **outcome.to_dict(),
            "validated_independently": validate_certificate(world, outcome.selected_tasks),
        }
    random_outcome = solve_random_certificate(world, seed=RANDOM_SEED)
    results["random"] = {
        **random_outcome.to_dict(),
        "seed": RANDOM_SEED,
        "validated_independently": validate_certificate(world, random_outcome.selected_tasks),
    }
    return {
        "name": name,
        "task_ids": world.tasks.task_ids,
        "state_count": len(world.valid_states),
        "results": results,
    }


def main() -> None:
    source = source_binding_before_run()
    worlds = [
        ("chain", generate_chain_world(["A", "B", "C", "D"])),
        ("tree", generate_tree_world({"A": ["B", "C"], "B": ["D"]})),
        ("unstructured", generate_unstructured_world(["Q0", "Q1", "Q2"])),
    ]
    result = {
        "phase": "phase3_fixed_regression",
        "exact_solver_note": "Exhaustive subset search remains the accepted temporary exact method.",
        "worlds": [_run_world(name, world) for name, world in worlds],
    }
    artifacts = write_result_and_manifest(
        phase="phase3_fixed_regression",
        script_name="scripts/revalidation_phase3_fixed.py",
        command=COMMAND,
        config={"worlds": [name for name, _ in worlds], "random_seed": RANDOM_SEED},
        result=result,
        source=source,
    )
    print(artifacts)


if __name__ == "__main__":
    main()
