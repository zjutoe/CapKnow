from capability_certificate_lab.generators import generate_tree_world
from capability_certificate_lab.knowledge_space.state import KnowledgeState


def test_tree_child_requires_parent():
    space = generate_tree_world({"A": ["B", "C"]})
    assert space.is_valid_state(KnowledgeState(()))
    assert space.is_valid_state(KnowledgeState(("A",)))
    assert space.is_valid_state(KnowledgeState(("A", "B")))
    assert space.is_valid_state(KnowledgeState(("A", "C")))
    assert space.is_valid_state(KnowledgeState(("A", "B", "C")))
    assert not space.is_valid_state(KnowledgeState(("B",)))
    assert not space.is_valid_state(KnowledgeState(("C",)))
