from capability_certificate_lab.generators import generate_chain_world
from capability_certificate_lab.knowledge_space.state import KnowledgeState


def test_chain_valid_states():
    space = generate_chain_world(["A", "B", "C", "D"])
    assert len(space.valid_states) == 5
    expected = [
        KnowledgeState(()),
        KnowledgeState(("A",)),
        KnowledgeState(("A", "B")),
        KnowledgeState(("A", "B", "C")),
        KnowledgeState(("A", "B", "C", "D")),
    ]
    assert space.valid_states == expected


def test_chain_invalid_states_are_rejected():
    space = generate_chain_world(["A", "B", "C", "D"])
    assert not space.is_valid_state(KnowledgeState(("B",)))
    assert not space.is_valid_state(KnowledgeState(("A", "C")))
