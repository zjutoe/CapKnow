from capability_certificate_lab.knowledge_space.state import KnowledgeState
from capability_certificate_lab.generators import generate_chain_world
from capability_certificate_lab.simulator.response import simulate_response, simulate_response_matrix


def test_response_matches_membership():
    state = KnowledgeState(("A", "C"))
    assert simulate_response(state, "A") == 1
    assert simulate_response(state, "B") == 0
    assert simulate_response(state, "C") == 1


def test_response_matrix_shape_and_values():
    space = generate_chain_world(["A", "B", "C"])
    tasks = space.tasks
    matrix = simulate_response_matrix([KnowledgeState(()), KnowledgeState(("A", "C"))], tasks)
    assert matrix == [[0, 0, 0], [1, 0, 1]]
