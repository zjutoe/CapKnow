from capability_certificate_lab.generators import generate_unstructured_world


def test_unstructured_has_all_states():
    space = generate_unstructured_world(["Q0", "Q1", "Q2", "Q3", "Q4"])
    assert len(space.valid_states) == 32
