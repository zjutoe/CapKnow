from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly from repo root or examples dir.
ROOT_DIR = Path(__file__).resolve().parent
repo_root = str(ROOT_DIR.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from capability_certificate_lab.generators import generate_chain_world
from capability_certificate_lab.validation.identifiability import check_identifiability


def main() -> None:
    world = generate_chain_world(["A", "B", "C", "D"])
    report = check_identifiability(world)

    print("Knowledge Space:")
    print(f"type: {world.metadata.get('type', 'unknown')}")
    print(f"tasks: {report.num_tasks}")
    print(f"states: {report.num_states}")
    print("\nUnique signatures:")
    print(report.num_unique_signatures)
    print("\nIdentifiable:")
    print(report.identifiable)
    print("\nCollisions:")
    print(report.collision_count)


if __name__ == "__main__":
    main()
