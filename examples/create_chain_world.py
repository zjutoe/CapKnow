from capability_certificate_lab.generators import generate_chain_world


def format_state(world, state):
    if not state.tasks:
        return "{}"
    ordered = [task_id for task_id in world.tasks.task_ids if task_id in state.tasks]
    return "{" + ",".join(ordered) + "}"


def main():
    world = generate_chain_world(["A", "B", "C", "D"])
    print("Tasks:")
    print(" ".join(world.tasks.task_ids))
    print("\nStates:")
    for state in world.valid_states:
        print(format_state(world, state))


if __name__ == "__main__":
    main()
