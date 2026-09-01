from __future__ import annotations

from statistics import mean

from capability_certificate_lab.certificate import (
    solve_adaptive_certificate,
    solve_exact_certificate,
    validate_adaptive_certificate,
)
from capability_certificate_lab.generators import (
    generate_chain_world,
    generate_tree_world,
    generate_unstructured_world,
)

from revalidation_common import source_binding_before_run, write_result_and_manifest


COMMAND = "PYTHONPATH=. python scripts/revalidation_phase4_adaptive.py"
RANDOM_SEEDS = list(range(100))


def _summarize_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "run_count": len(runs),
        "valid_run_rate": mean(1.0 if run["valid"] else 0.0 for run in runs),
        "average_depth_mean": mean(float(run["average_depth"]) for run in runs),
        "worst_case_depth_mean": mean(float(run["worst_case_depth"]) for run in runs),
        "worst_case_depth_max": max(int(run["worst_case_depth"]) for run in runs),
        "node_count_mean": mean(float(run["node_count"]) for run in runs),
        "runs": runs,
    }


def _run_world(name, world):
    fixed = solve_exact_certificate(world)
    policy_results = {}
    for policy in ("entropy",):
        outcome = solve_adaptive_certificate(world, policy=policy)
        independently_valid = validate_adaptive_certificate(outcome.root, world)
        if not outcome.valid or not independently_valid:
            raise RuntimeError(f"{name}/{policy} produced invalid adaptive tree.")
        policy_results[policy] = _summarize_runs(
            [
                {
                    **outcome.to_dict(),
                    "independently_valid": independently_valid,
                }
            ]
        )

    random_runs = []
    for seed in RANDOM_SEEDS:
        outcome = solve_adaptive_certificate(world, policy="random", seed=seed)
        independently_valid = validate_adaptive_certificate(outcome.root, world)
        if not outcome.valid or not independently_valid:
            raise RuntimeError(f"{name}/random seed {seed} produced invalid adaptive tree.")
        random_runs.append(
            {
                **outcome.to_dict(),
                "independently_valid": independently_valid,
            }
        )
    policy_results["random"] = _summarize_runs(random_runs)

    return {
        "name": name,
        "fixed_exact_certificate_size": fixed.certificate_size,
        "fixed_valid": fixed.valid,
        "state_count": len(world.valid_states),
        "task_count": len(world.tasks.task_ids),
        "policy_results": policy_results,
        "aggregation": "average_depth is equal-weighted over states within each tree; policy summaries are equal-weighted over runs/seeds.",
    }


def main() -> None:
    source = source_binding_before_run()
    worlds = [
        ("chain", generate_chain_world(["A", "B", "C", "D"])),
        ("tree", generate_tree_world({"A": ["B", "C"], "B": ["D"]})),
        ("unstructured", generate_unstructured_world(["Q0", "Q1", "Q2"])),
    ]
    result = {
        "phase": "phase4_adaptive_regression",
        "random_seeds": RANDOM_SEEDS,
        "worlds": [_run_world(name, world) for name, world in worlds],
    }
    artifacts = write_result_and_manifest(
        phase="phase4_adaptive_regression",
        script_name="scripts/revalidation_phase4_adaptive.py",
        command=COMMAND,
        config={"worlds": [name for name, _ in worlds], "random_seeds": RANDOM_SEEDS},
        result=result,
        source=source,
    )
    print(artifacts)


if __name__ == "__main__":
    main()
