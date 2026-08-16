from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch

from capability_certificate_lab.lm_bridge import corpus_generator as cg
from capability_certificate_lab.dsl.executor import MissingCapabilityError
from capability_certificate_lab.lm_bridge.model import ToyCausalTransformer, TransformerConfig, build_model, transformer_config
from capability_certificate_lab.lm_bridge.tokenizer import BOS_ID, EOS_ID, PAD_ID, SEP_ID, ByteTokenizer
from capability_certificate_lab.lm_bridge.train import (
    TextRecord,
    encode_record_batch,
    load_model_from_checkpoint,
    make_optimizer,
    response_only_labels,
    response_only_loss,
    run_byte_copy_overfit_control,
    save_checkpoint,
    set_deterministic_backend,
)


def test_frozen_primitive_state_and_task_order() -> None:
    assert cg.PRIMITIVE_ORDER == ("MEMORY", "SEARCH", "FILTER", "CONDITION")
    assert cg.STATE_MASKS == tuple(range(16))
    assert cg.TASK_ORDER == (
        "MEMORY",
        "SEARCH",
        "FILTER",
        "CONDITION",
        "MEMORY_FILTER",
        "FILTER_CONDITION",
        "SEARCH_CONDITION",
        "MEMORY_SEARCH",
    )
    assert cg.state_from_mask(5) == {
        "MEMORY": True,
        "SEARCH": False,
        "FILTER": True,
        "CONDITION": False,
    }


@pytest.mark.parametrize("task_id", cg.TASK_ORDER)
def test_program_contracts_context_schemas_and_outputs(task_id: str) -> None:
    context = cg.context_for_payload(task_id, "evaluation", 0)
    cg.validate_context_schema(task_id, context)
    assert cg.required_primitives(task_id) == tuple(sorted(cg.program_for_task(task_id).required_primitive_ids(), key=cg.PRIMITIVE_ORDER.index))
    answer = cg.canonical_answer(task_id, context)
    assert answer in {"true", "false"} or answer.startswith('"') or answer.startswith("[")

    missing = dict(cg.state_from_mask(15))
    missing[cg.required_primitives(task_id)[0]] = False
    with pytest.raises(MissingCapabilityError):
        cg.execute_task(task_id, context, missing)


def test_exact_context_schema_rejects_extra_target_or_value_for_memory_sequences() -> None:
    context = dict(cg.context_for_payload("MEMORY_SEARCH", "evaluation", 0))
    context["target"] = next(iter(context["memory"].values()))
    with pytest.raises(ValueError, match="exactly memory, key, and items"):
        cg.validate_context_schema("MEMORY_SEARCH", context)
    context.pop("target")
    context["value"] = next(iter(context["memory"].values()))
    with pytest.raises(ValueError, match="exactly memory, key, and items"):
        cg.validate_context_schema("MEMORY_SEARCH", context)


@pytest.mark.parametrize("task_id", ("MEMORY_FILTER", "MEMORY_SEARCH"))
def test_memory_sequence_dependency_perturbation_covers_all_evaluation_contexts(task_id: str) -> None:
    original_false_count = 0
    for record_index in range(64):
        context = cg.context_for_payload(task_id, "evaluation", record_index)
        items = list(context["items"])
        present_context = {
            "memory": {context["key"]: items[0]},
            "key": context["key"],
            "items": items,
        }

        if task_id == "MEMORY_SEARCH" and cg.canonical_answer(task_id, context) == "false":
            original_false_count += 1

        present_answer, absent_answer = cg.validate_memory_dependency(task_id, context)

        assert present_answer == cg.canonical_answer(task_id, present_context)
        assert present_answer != absent_answer
        if task_id == "MEMORY_SEARCH":
            assert present_answer == "true"
            assert absent_answer == "false"
        else:
            assert absent_answer == "[]"

    if task_id == "MEMORY_SEARCH":
        assert original_false_count == 32


def test_ground_truth_matrix_identifiability_and_unique_certificate() -> None:
    audit = cg.audit_ground_truth_oracle()
    assert audit["identifiable"] is True
    assert audit["accepted_identifiability"]["identifiable"] is True
    assert audit["accepted_solver_certificate_size"] == 4
    assert audit["accepted_solver_selected_certificate"] == cg.PRIMITIVE_ORDER
    assert audit["certificate_size"] == 4
    assert audit["minimum_certificates"] == (cg.PRIMITIVE_ORDER,)
    matrix = audit["matrix"]
    assert len(matrix) == 16
    assert all(len(row) == 8 for row in matrix)
    assert matrix[0] == (0, 0, 0, 0, 0, 0, 0, 0)
    assert matrix[15] == (1, 1, 1, 1, 1, 1, 1, 1)


def test_leakage_oracle_rejects_forbidden_model_facing_text() -> None:
    for text in (
        "Use MEMORY now.",
        "This primitive is available.",
        "state id 3 prompt.",
        "A+B->C rule.",
        "seed 300000 appears.",
    ):
        with pytest.raises(ValueError):
            cg.leak_check_model_text(text)


def test_split_disjointness_and_condition_exception() -> None:
    train = cg.build_split_records("training", "CONDITION", 4)
    evaluation = cg.build_split_records("evaluation", "CONDITION", 4)
    cg.validate_split_disjointness(train, evaluation)
    assert {record.normalized_payload for record in train} == {(False,), (True,)}
    assert {record.normalized_payload for record in evaluation} == {(False,), (True,)}


def test_memory_search_is_held_out_from_training() -> None:
    with pytest.raises(ValueError, match="held-out composition"):
        cg.build_split_records("training", "MEMORY_SEARCH", 1)
    evaluation = cg.build_split_records("evaluation", "MEMORY_SEARCH", 1)
    assert len(evaluation) == 1


def test_split_disjointness_rejects_tampered_training_memory_search_record() -> None:
    train = list(cg.build_split_records("training", "MEMORY", 2))
    train[0] = replace(train[0], task_id="MEMORY_SEARCH")
    evaluation = cg.build_split_records("evaluation", "MEMORY", 2)

    with pytest.raises(ValueError, match="held-out composition"):
        cg.validate_split_disjointness(tuple(train), evaluation)


def test_split_disjointness_rejects_tampered_training_memory_search_program_dict() -> None:
    train = list(cg.build_split_records("training", "MEMORY", 2))
    train[0] = replace(train[0], program_dict=cg.program_dict("MEMORY_SEARCH"))
    evaluation = cg.build_split_records("evaluation", "MEMORY", 2)

    with pytest.raises(ValueError, match="held-out composition"):
        cg.validate_split_disjointness(tuple(train), evaluation)


def test_split_disjointness_rejects_tampered_training_memory_search_template_id() -> None:
    train = list(cg.build_split_records("training", "MEMORY", 2))
    train[0] = replace(train[0], template_id="memory_search__train_a")
    evaluation = cg.build_split_records("evaluation", "MEMORY", 2)

    with pytest.raises(ValueError, match="held-out composition"):
        cg.validate_split_disjointness(tuple(train), evaluation)


@pytest.mark.parametrize("task_id", ("MEMORY", "SEARCH", "FILTER", "CONDITION"))
def test_primitive_train_and_evaluation_templates_use_distinct_phrasings(task_id: str) -> None:
    context = cg.context_for_payload(task_id, "evaluation", 0)
    train_prompt = cg.render_prompt(
        task_id,
        context,
        f"{task_id.lower()}__train_a",
        "train",
    )
    eval_prompt = cg.render_prompt(
        task_id,
        context,
        "neutral_eval__neutral_a",
        "neutral",
    )
    assert train_prompt != eval_prompt


def test_split_disjointness_rejects_duplicate_canonical_context() -> None:
    train = cg.build_split_records("training", "MEMORY", 2)
    evaluation = list(cg.build_split_records("evaluation", "MEMORY", 2))
    evaluation[0] = replace(evaluation[0], canonical_context=train[0].canonical_context)
    with pytest.raises(ValueError, match="canonical_context"):
        cg.validate_split_disjointness(train, tuple(evaluation))


def test_split_disjointness_rejects_duplicate_normalized_payload() -> None:
    train = cg.build_split_records("training", "SEARCH", 2)
    evaluation = list(cg.build_split_records("evaluation", "SEARCH", 2))
    evaluation[0] = replace(evaluation[0], normalized_payload=train[0].normalized_payload)
    with pytest.raises(ValueError, match="normalized_payload"):
        cg.validate_split_disjointness(train, tuple(evaluation))


def test_evaluation_pack_identity_balance_styles_and_checksum() -> None:
    pack = cg.build_evaluation_probe_pack()
    cg.validate_evaluation_pack(pack)
    assert len(pack) == 512
    assert cg.evaluation_pack_checksum(pack) == cg.EVALUATION_PACK_CHECKSUM
    assert tuple((probe.task_id, probe.record_index) for probe in pack[:65]) == (
        *[("MEMORY", i) for i in range(64)],
        ("SEARCH", 0),
    )
    by_task = {task_id: [probe for probe in pack if probe.task_id == task_id] for task_id in cg.TASK_ORDER}
    for task_id in ("FILTER_CONDITION", "SEARCH_CONDITION", "MEMORY_SEARCH"):
        probes = by_task[task_id]
        assert [probe.style for probe in probes[:32]] == ["explicit"] * 32
        assert [probe.style for probe in probes[32:]] == ["indirect"] * 32
        for block in (probes[:32], probes[32:]):
            assert [probe.answer for probe in block].count("true") == 16
            assert [probe.answer for probe in block].count("false") == 16
    memory_filter = by_task["MEMORY_FILTER"]
    assert len({probe.answer for probe in memory_filter[:32]}) == 32
    assert len({probe.answer for probe in memory_filter[32:]}) == 32
    for task_id in ("MEMORY", "FILTER", "MEMORY_FILTER"):
        assert len({probe.answer for probe in by_task[task_id]}) == 64


def test_evaluation_pack_prefix_budget_is_enforced() -> None:
    pack = cg.build_evaluation_probe_pack()

    assert max(cg.evaluation_prefix_token_count(probe.prompt) + 64 for probe in pack) <= 256

    tampered = list(pack)
    tampered[0] = replace(tampered[0], prompt="x" * 191)
    with pytest.raises(ValueError, match="prefix length"):
        cg.validate_evaluation_pack(tuple(tampered))


def test_evaluation_pack_rejects_answer_tampering_preserving_boolean_balance() -> None:
    pack = list(cg.build_evaluation_probe_pack())
    search_indices = [idx for idx, probe in enumerate(pack) if probe.task_id == "SEARCH"]
    true_index = next(idx for idx in search_indices if pack[idx].answer == "true")
    false_index = next(idx for idx in search_indices if pack[idx].answer == "false")
    pack[true_index] = replace(pack[true_index], answer="false")
    pack[false_index] = replace(pack[false_index], answer="true")

    assert [probe.answer for probe in pack if probe.task_id == "SEARCH"].count("true") == 32
    assert [probe.answer for probe in pack if probe.task_id == "SEARCH"].count("false") == 32
    with pytest.raises(ValueError, match="canonical full-capability execution"):
        cg.validate_evaluation_pack(tuple(pack))


def test_evaluation_pack_rejects_wrong_or_transparent_probe_key() -> None:
    pack = list(cg.build_evaluation_probe_pack())
    pack[0] = replace(pack[0], probe_key="phase8-probe|0|0")

    with pytest.raises(ValueError, match="probe_key"):
        cg.validate_evaluation_pack(tuple(pack))


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("template_id", "neutral_eval__tampered_a"),
        ("payload_id", "evaluation-payload-00-999"),
        ("program_dict", {"type": "primitive", "op": {"op_id": "SEARCH"}, "args": ()}),
        ("canonical_context", '{"key":"tampered","memory":{"tampered":"0000000000000000"}}'),
        ("normalized_payload", ("tampered", "0000000000000000")),
        ("prompt", "In the table, 0000000000000000 maps to 1111111111111111. Return compact JSON."),
    ),
)
def test_evaluation_pack_rejects_checksum_identity_mismatch(
    field_name: str,
    field_value: object,
) -> None:
    pack = list(cg.build_evaluation_probe_pack())
    pack[0] = replace(pack[0], **{field_name: field_value})
    tampered_pack = tuple(pack)

    assert cg.evaluation_pack_checksum(tampered_pack) != cg.EVALUATION_PACK_CHECKSUM
    with pytest.raises(ValueError):
        cg.validate_evaluation_pack(tampered_pack)


def test_evaluation_pack_program_dict_is_deeply_immutable() -> None:
    probe = cg.build_evaluation_probe_pack()[4 * 64]

    with pytest.raises(TypeError):
        probe.program_dict["type"] = "primitive"
    with pytest.raises(TypeError):
        probe.program_dict["steps"][0]["type"] = "tampered"


def test_no_raw_seed_ids_or_metadata_leak_in_generated_prompts() -> None:
    for probe in cg.build_evaluation_probe_pack():
        assert "300000" not in probe.prompt
        assert probe.template_id not in probe.prompt
        assert probe.task_id not in probe.prompt
        assert probe.task_id.lower() not in probe.prompt
        cg.leak_check_model_text(probe.prompt)


def test_condition_prompts_keep_metadata_out_while_allowing_semantic_overlap() -> None:
    train = cg.build_split_records("training", "CONDITION", 64)
    evaluation = cg.build_split_records("evaluation", "CONDITION", 64)
    cg.validate_split_disjointness(train, evaluation)
    assert {record.normalized_payload for record in train} == {(False,), (True,)}
    assert {record.normalized_payload for record in evaluation} == {(False,), (True,)}
    for record in (*train, *evaluation):
        assert record.template_id not in record.prompt
        assert record.task_id not in record.prompt
        cg.leak_check_model_text(record.prompt)


@pytest.mark.parametrize("condition", cg.CONDITIONS)
@pytest.mark.parametrize("corpus_size, records_per_family", (("base", 128), ("large", 512)))
def test_training_corpus_counts_and_held_out_exclusion(
    condition: str,
    corpus_size: str,
    records_per_family: int,
) -> None:
    corpus = cg.build_training_corpus(condition, seed=0, state_mask=15, corpus_size=corpus_size)

    assert len(corpus) == 7 * records_per_family
    assert [record.task_id for record in corpus[:records_per_family]] == ["MEMORY"] * records_per_family
    assert {record.task_id for record in corpus} == set(cg.TRAINING_TASK_ORDER)
    assert "MEMORY_SEARCH" not in {record.task_id for record in corpus}
    assert not any(record.program_dict == cg.program_dict("MEMORY_SEARCH") for record in corpus)
    assert not any(record.template_id.startswith("memory_search__") for record in corpus)
    cg.validate_training_corpus(corpus, condition, seed=0, state_mask=15, corpus_size=corpus_size)


def test_training_corpus_base_payload_prefix_of_large_for_structured_conditions() -> None:
    base_a = cg.build_training_corpus("A", seed=1, state_mask=7, corpus_size="base")
    large_a = cg.build_training_corpus("A", seed=1, state_mask=7, corpus_size="large")
    base_b = cg.build_training_corpus("B", seed=1, state_mask=7, corpus_size="base")
    large_b = cg.build_training_corpus("B", seed=1, state_mask=7, corpus_size="large")

    for task_id in cg.TRAINING_TASK_ORDER:
        base_task_a = [record for record in base_a if record.task_id == task_id]
        large_task_a = [record for record in large_a if record.task_id == task_id]
        base_task_b = [record for record in base_b if record.task_id == task_id]
        large_task_b = [record for record in large_b if record.task_id == task_id]
        assert base_task_a == large_task_a[:128]
        assert base_task_b == large_task_b[:128]


def test_record_order_indices_are_deterministic_and_condition_independent() -> None:
    corpus_a = cg.build_training_corpus("A", seed=2, state_mask=3)
    corpus_b = cg.build_training_corpus("B", seed=2, state_mask=3)
    order = cg.record_order_indices(seed=2, state_mask=3, record_count=len(corpus_a))

    assert order == cg.record_order_indices(seed=2, state_mask=3, record_count=len(corpus_a))
    assert set(order) == set(range(len(corpus_a)))
    assert order != tuple(range(len(corpus_a)))
    assert [corpus_a[index].record_index for index in order] == [corpus_a[index].record_index for index in order]
    assert [corpus_a[index].task_id for index in order] == [corpus_b[index].task_id for index in order]


def test_a_b_outcome_agreement_and_primitive_record_identity() -> None:
    cg.validate_a_b_agreement(seed=0, state_mask=11, corpus_size="base")
    corpus_a = cg.build_training_corpus("A", seed=0, state_mask=11)
    corpus_b = cg.build_training_corpus("B", seed=0, state_mask=11)

    for record_a, record_b in zip(corpus_a, corpus_b):
        assert record_a.answer == record_b.answer
        if record_a.task_id in cg.PRIMITIVE_ORDER:
            assert record_a == record_b
        else:
            assert record_a.prompt != record_b.prompt
            assert record_a.template_id != record_b.template_id


def test_c_randomized_control_gates_degrees_histograms_changes_and_determinism() -> None:
    audit = cg.validate_randomized_control(seed=0, corpus_size="base")
    corpora_a = cg.build_training_corpora("A", seed=0, corpus_size="base")
    corpora_c = cg.build_training_corpora("C", seed=0, corpus_size="base")

    assert audit["changed_fraction"] >= 0.15
    assert all(audit["changed_by_family"][task_id] >= 1 for task_id in cg.SEEN_COMPOSITION_TASKS)
    assert cg.aggregate_utf8_byte_histogram(corpora_a) == cg.aggregate_utf8_byte_histogram(corpora_c)
    assert cg.aggregate_token_histogram(corpora_a) == cg.aggregate_token_histogram(corpora_c)
    for state_mask in cg.STATE_MASKS:
        records_a = corpora_a[state_mask]
        records_c = corpora_c[state_mask]
        for record_a, record_c in zip(records_a, records_c):
            assert record_a.prompt == record_c.prompt
            if record_a.task_id in cg.PRIMITIVE_ORDER:
                assert record_a == record_c
        a_positive = sum(record.answer != cg.UNABLE_RESPONSE for record in records_a if record.task_id in cg.SEEN_COMPOSITION_TASKS)
        c_positive = sum(record.answer != cg.UNABLE_RESPONSE for record in records_c if record.task_id in cg.SEEN_COMPOSITION_TASKS)
        assert a_positive == c_positive


def test_randomized_control_rejects_malformed_label_and_histogram_changes() -> None:
    corpora_a = cg.build_training_corpora("A", seed=0, corpus_size="base")
    corpora_c = dict(cg.build_training_corpora("C", seed=0, corpus_size="base"))
    state_records = list(corpora_c[0])
    target_index = next(index for index, record in enumerate(state_records) if record.task_id in cg.SEEN_COMPOSITION_TASKS)
    state_records[target_index] = replace(state_records[target_index], answer='"tampered"')
    corpora_c[0] = tuple(state_records)

    with pytest.raises(
        ValueError,
        match="response target|deterministic switch randomization|UTF-8 byte histogram|tokenizer-token histogram",
    ):
        cg.validate_randomized_control(
            seed=0,
            corpus_size="base",
            corpus_a_by_state=corpora_a,
            corpus_c_by_state=corpora_c,
        )


def test_condition_c_single_state_validation_rejects_seen_composition_label_tamper() -> None:
    corpus = list(cg.build_training_corpus("C", seed=0, state_mask=3, corpus_size="base"))
    target_index = next(index for index, record in enumerate(corpus) if record.task_id in cg.SEEN_COMPOSITION_TASKS)
    target = corpus[target_index]
    context = json.loads(target.canonical_context)
    tampered_answer = (
        cg.UNABLE_RESPONSE
        if target.answer != cg.UNABLE_RESPONSE
        else cg.canonical_answer(target.task_id, context)
    )
    corpus[target_index] = replace(target, answer=tampered_answer)

    with pytest.raises(ValueError, match="deterministic switch randomization"):
        cg.validate_training_corpus(tuple(corpus), "C", seed=0, state_mask=3, corpus_size="base")


def test_randomized_control_rejects_insufficient_randomization() -> None:
    corpora_a = cg.build_training_corpora("A", seed=0, corpus_size="base")

    with pytest.raises(ValueError, match="deterministic switch randomization|changed-label fraction|at least one label"):
        cg.validate_randomized_control(
            seed=0,
            corpus_size="base",
            corpus_a_by_state=corpora_a,
            corpus_c_by_state=corpora_a,
        )


def test_training_split_rejects_prompt_id_context_and_payload_collisions() -> None:
    train = list(cg.build_training_corpus("A", seed=0, state_mask=15))
    evaluation = cg.build_evaluation_probe_pack()

    for field_name, match in (
        ("prompt", "prompt"),
        ("template_id", "template_id"),
        ("payload_id", "payload_id"),
        ("canonical_context", "canonical_context"),
        ("normalized_payload", "normalized_payload"),
    ):
        tampered = list(train)
        source_probe = next(probe for probe in evaluation if probe.task_id == "MEMORY")
        target_index = next(index for index, record in enumerate(tampered) if record.task_id == "MEMORY")
        tampered[target_index] = replace(tampered[target_index], **{field_name: getattr(source_probe, field_name)})
        with pytest.raises(ValueError, match=match):
            cg.validate_split_disjointness(tuple(tampered), evaluation)


def test_response_target_length_diagnostics_and_paired_differences_are_deterministic() -> None:
    corpora_a = cg.build_training_corpora("A", seed=0, corpus_size="base")
    corpora_c = cg.build_training_corpora("C", seed=0, corpus_size="base")
    diagnostics_a = cg.response_target_length_diagnostics(corpora_a)
    diagnostics_c = cg.response_target_length_diagnostics(corpora_c)
    differences = cg.paired_target_length_diagnostic_differences(corpora_a, corpora_c)

    assert diagnostics_a == cg.response_target_length_diagnostics(corpora_a)
    assert set(diagnostics_a) == set(diagnostics_c) == set(differences)
    assert diagnostics_a[(0, "MEMORY")]["literal_unable_target_ratio"] == 1.0
    assert diagnostics_a[(15, "MEMORY")]["literal_unable_target_ratio"] == 0.0
    assert any(
        differences[(state_mask, task_id)]["literal_unable_target_ratio"] != 0
        for state_mask in cg.STATE_MASKS
        for task_id in cg.SEEN_COMPOSITION_TASKS
    )


def test_byte_tokenizer_round_trip_specials_padding_length_gates_and_response_mask() -> None:
    tokenizer = ByteTokenizer()
    text = "utf-8 bytes: café ☃"
    assert tokenizer.decode_text(tokenizer.encode_text(text)) == text

    record = tokenizer.encode_training_record("xy", "ab")
    assert record.input_ids == (BOS_ID, ord("x"), ord("y"), SEP_ID, ord("a"), ord("b"), EOS_ID)
    padded = tokenizer.pad(record.input_ids, 10)
    assert padded[-3:] == (PAD_ID, PAD_ID, PAD_ID)
    with pytest.raises(ValueError, match="BOS"):
        tokenizer.validate_special_token_placement((ord("x"), BOS_ID, SEP_ID, EOS_ID), mode="training")
    with pytest.raises(ValueError, match="PAD"):
        tokenizer.validate_special_token_placement((BOS_ID, ord("x"), PAD_ID, SEP_ID, EOS_ID), mode="training")
    with pytest.raises(ValueError, match="EOS"):
        tokenizer.validate_special_token_placement((*tokenizer.encode_evaluation_prefix("xy"), EOS_ID), mode="evaluation_prefix")
    with pytest.raises(ValueError, match="exceeds maximum sequence length|truncation"):
        tokenizer.encode_training_record("p" * 254, "r")
    with pytest.raises(ValueError, match="prefix length|truncation"):
        tokenizer.encode_evaluation_prefix("p" * 191)

    input_ids = torch.tensor([padded], dtype=torch.long)
    labels = response_only_labels(input_ids)
    assert labels[0, :3].tolist() == [-100, -100, -100]
    assert labels[0, 3].item() == ord("a")
    assert labels[0, 4].item() == ord("b")
    assert labels[0, 5].item() == EOS_ID
    assert labels[0, 6:].tolist() == [-100, -100, -100, -100]
    with pytest.raises(ValueError, match="BOS"):
        response_only_labels(torch.tensor([[BOS_ID, ord("x"), SEP_ID, BOS_ID, EOS_ID]], dtype=torch.long))


def test_frozen_small_medium_model_constants_and_recorded_parameter_counts() -> None:
    small_config = transformer_config("small")
    medium_config = transformer_config("medium")
    assert (small_config.d_model, small_config.n_heads, small_config.n_layers, small_config.d_ff) == (64, 4, 2, 256)
    assert (medium_config.d_model, medium_config.n_heads, medium_config.n_layers, medium_config.d_ff) == (128, 4, 4, 512)
    assert small_config.max_seq_len == medium_config.max_seq_len == 256
    assert small_config.dropout == medium_config.dropout == 0.0

    small = build_model("small")
    medium = build_model("medium")
    assert small.parameter_count == sum(parameter.numel() for parameter in small.parameters())
    assert medium.parameter_count == sum(parameter.numel() for parameter in medium.parameters())
    assert medium.parameter_count > small.parameter_count


def test_causal_mask_prevents_future_token_access() -> None:
    set_deterministic_backend(0)
    model = ToyCausalTransformer(TransformerConfig(name="test-tiny", d_model=16, n_heads=2, n_layers=1, d_ff=32))
    model.eval()
    left = torch.tensor([[BOS_ID, ord("a"), ord("b"), ord("c"), ord("d"), EOS_ID]])
    right = left.clone()
    right[0, 4] = ord("z")

    with torch.no_grad():
        left_logits = model(left)
        right_logits = model(right)

    torch.testing.assert_close(left_logits[:, :4, :], right_logits[:, :4, :], atol=0.0, rtol=0.0)
    assert not torch.equal(left_logits[:, 4, :], right_logits[:, 4, :])


def test_deterministic_initialization_optimizer_step_and_greedy_decode() -> None:
    config = TransformerConfig(name="test-tiny", d_model=16, n_heads=2, n_layers=1, d_ff=32)
    set_deterministic_backend(3)
    model_a = ToyCausalTransformer(config)
    set_deterministic_backend(3)
    model_b = ToyCausalTransformer(config)
    for state_a, state_b in zip(model_a.state_dict().values(), model_b.state_dict().values()):
        torch.testing.assert_close(state_a, state_b, atol=0.0, rtol=0.0)

    tokenizer = ByteTokenizer()
    batch = encode_record_batch((TextRecord("copy a", "a"), TextRecord("copy b", "b")), tokenizer)
    for model in (model_a, model_b):
        optimizer = make_optimizer(model)
        optimizer.zero_grad(set_to_none=True)
        loss = response_only_loss(model(batch), batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for state_a, state_b in zip(model_a.state_dict().values(), model_b.state_dict().values()):
        torch.testing.assert_close(state_a, state_b, atol=0.0, rtol=0.0)
    prefix = torch.tensor([tokenizer.encode_evaluation_prefix("copy a")], dtype=torch.long)
    torch.testing.assert_close(model_a.greedy_decode(prefix), model_b.greedy_decode(prefix), atol=0, rtol=0)
    with pytest.raises(ValueError, match="64"):
        model_a.greedy_decode(prefix, max_new_tokens=65)


def test_model_save_load_round_trip_preserves_state_dict_and_logits(tmp_path: Path) -> None:
    set_deterministic_backend(4)
    model = ToyCausalTransformer(TransformerConfig(name="test-tiny", d_model=16, n_heads=2, n_layers=1, d_ff=32))
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(str(path), model, metadata={"purpose": "unit-test"})

    loaded = load_model_from_checkpoint(str(path))
    assert loaded.parameter_count == model.parameter_count == sum(parameter.numel() for parameter in model.parameters())
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[key], atol=0.0, rtol=0.0)

    sample = torch.tensor([[BOS_ID, ord("x"), SEP_ID, ord("y"), EOS_ID]], dtype=torch.long)
    with torch.no_grad():
        torch.testing.assert_close(model(sample), loaded(sample), atol=0.0, rtol=0.0)


def test_four_record_cpu_byte_copy_overfit_control() -> None:
    result = run_byte_copy_overfit_control(
        seed=0,
        max_steps=500,
        config=TransformerConfig(name="test-tiny", d_model=32, n_heads=4, n_layers=1, d_ff=128),
    )

    assert result.steps <= 500
    assert result.training_accuracy == 1.0


def _passing_feasibility_cells() -> list[dict[str, object]]:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    return [
        {
            "family": family,
            "model_size": model_size,
            "seed": seed,
            "eval_count": sf.EVAL_RECORDS_PER_FAMILY,
            "exact_matches": sf.PASS_THRESHOLD,
            "passed": True,
            "generations_path": f"{family}__{model_size}__seed{seed}/generations.jsonl",
            "checkpoint_path": f"{family}__{model_size}__seed{seed}/checkpoint_step1500.pt",
            "parameter_count": 123,
        }
        for family in sf.FAMILIES
        for model_size in sf.MODEL_SIZES
        for seed in sf.SEEDS
    ]


def test_feasibility_families_disjoint_cell_gate_raw_retention_and_marker_rejection(tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")

    groups = sf.grouped_records()
    assert set(groups) == set(sf.FAMILIES)
    for family, splits in groups.items():
        assert len(splits["train"]) == 512
        assert len(splits["eval"]) == 64
        train_operands = {record.operand_id for record in splits["train"]}
        eval_operands = {record.operand_id for record in splits["eval"]}
        train_templates = {record.template_id for record in splits["train"]}
        eval_templates = {record.template_id for record in splits["eval"]}
        train_prompts = {record.prompt for record in splits["train"]}
        eval_prompts = {record.prompt for record in splits["eval"]}
        train_semantic = {value for record in splits["train"] for value in sf.semantic_values_for_record(record)}
        eval_semantic = {value for record in splits["eval"] for value in sf.semantic_values_for_record(record)}
        assert train_operands.isdisjoint(eval_operands)
        assert train_templates.isdisjoint(eval_templates)
        assert train_prompts.isdisjoint(eval_prompts)
        assert train_semantic.isdisjoint(eval_semantic)
        assert set(sf.PROMPT_SURFACES[family]["train"]).isdisjoint(sf.PROMPT_SURFACES[family]["eval"])
        for record in (*splits["train"], *splits["eval"]):
            sf.reject_scientific_markers(record.prompt)
            sf.reject_scientific_markers(record.answer)
            assert "template" not in record.prompt.lower()
        decoy_record = sf.FeasibilityRecord(
            **{**splits["train"][0].__dict__, "semantic_values": ("metadata-decoy",)}
        )
        assert "metadata-decoy" not in sf.semantic_values_for_record(decoy_record)

    cells = _passing_feasibility_cells()
    sf.validate_cell_counts(cells)
    tampered = [dict(cell) for cell in cells]
    tampered[0]["passed"] = False
    with pytest.raises(ValueError, match="52/64"):
        sf.validate_cell_counts(tampered)
    tampered = [dict(cell) for cell in cells]
    tampered[0]["exact_matches"] = 51
    with pytest.raises(ValueError, match="52/64"):
        sf.validate_cell_counts(tampered)
    tampered = [dict(cell) for cell in cells]
    tampered[0]["exact_matches"] = 51
    tampered[0]["passed"] = False
    with pytest.raises(ValueError, match="52/64"):
        sf.validate_cell_counts(tampered)
    tampered = [dict(cell) for cell in cells]
    tampered[0]["exact_matches"] = 65
    with pytest.raises(ValueError, match="exact_matches"):
        sf.validate_cell_counts(tampered)
    for field_name, bad_value in (
        ("exact_matches", 52.9),
        ("eval_count", "64"),
        ("seed", "0"),
        ("passed", 1),
    ):
        tampered = [dict(cell) for cell in cells]
        tampered[0][field_name] = bad_value
        with pytest.raises(ValueError, match=field_name):
            sf.validate_cell_counts(tampered)

    cell_dir = tmp_path / "hex_copy__small__seed0"
    cell_dir.mkdir()
    generations = cell_dir / "generations.jsonl"
    checkpoint = cell_dir / "checkpoint_step1500.pt"
    generations.write_text('{"generated":"abc","exact_match":true}\n')
    checkpoint.write_bytes(b"checkpoint")
    inventory = sf.inventory(tmp_path)
    assert any(row["path"] == "hex_copy__small__seed0/generations.jsonl" for row in inventory)
    assert any(row["path"] == "hex_copy__small__seed0/checkpoint_step1500.pt" for row in inventory)

    bad = list(groups["hex_copy"]["train"])
    bad[0] = sf.FeasibilityRecord(**{**bad[0].__dict__, "prompt": "MEMORY marker"})
    with pytest.raises(ValueError, match="forbidden Phase 8"):
        sf.validate_feasibility_records(tuple(bad))


def test_feasibility_semantic_train_eval_overlap_is_rejected_from_raw_prompt() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    records = list(sf.build_family_records("named_value_json"))
    train_value = next(
        value
        for value in sf.semantic_values_for_record(next(record for record in records if record.split == "train"))
        if value.startswith(("red-", "blue-", "green-", "silver-"))
    )
    eval_index = next(index for index, record in enumerate(records) if record.split == "eval")
    records[eval_index] = sf.FeasibilityRecord(
        **{
            **records[eval_index].__dict__,
            "prompt": f"{records[eval_index].prompt} stray overlap token {train_value}",
            "semantic_values": (),
        }
    )

    with pytest.raises(ValueError, match="semantic values"):
        sf.validate_feasibility_records(tuple(records))


def test_feasibility_hex_semantic_overlap_is_case_insensitive() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    lower_hex = "abcdef0123456789"
    upper_hex = lower_hex.upper()
    train = sf.FeasibilityRecord(
        family="hex_copy",
        split="train",
        index=0,
        template_id="train-surface",
        operand_id="train-value",
        prompt=f"copy {lower_hex}",
        answer=lower_hex,
    )
    eval_record = sf.FeasibilityRecord(
        family="hex_copy",
        split="eval",
        index=0,
        template_id="eval-surface",
        operand_id="eval-value",
        prompt=f"copy {upper_hex}",
        answer=upper_hex,
    )

    assert sf.semantic_values_for_record(eval_record) == frozenset({lower_hex})
    with pytest.raises(ValueError, match="semantic values"):
        sf.validate_feasibility_records((train, eval_record))


def test_feasibility_named_and_array_semantic_overlap_is_case_insensitive() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")

    named_train = sf.FeasibilityRecord(
        family="named_value_json",
        split="train",
        index=0,
        template_id="named-train-surface",
        operand_id="named-train-value",
        prompt="choose red-tdeadbeef from pairs red-tdeadbeef=value",
        answer=json.dumps("red-tdeadbeef"),
    )
    named_eval = sf.FeasibilityRecord(
        family="named_value_json",
        split="eval",
        index=0,
        template_id="named-eval-surface",
        operand_id="named-eval-value",
        prompt="choose RED-TDEADBEEF from pairs RED-TDEADBEEF=value",
        answer=json.dumps("RED-TDEADBEEF"),
    )
    assert "red-tdeadbeef" in sf.semantic_values_for_record(named_eval)
    with pytest.raises(ValueError, match="semantic values"):
        sf.validate_feasibility_records((named_train, named_eval))

    array_train = sf.FeasibilityRecord(
        family="array_json",
        split="train",
        index=0,
        template_id="array-train-surface",
        operand_id="array-train-value",
        prompt="chunks ste00ff11",
        answer=json.dumps(["ste00ff11"]),
    )
    array_eval = sf.FeasibilityRecord(
        family="array_json",
        split="eval",
        index=0,
        template_id="array-eval-surface",
        operand_id="array-eval-value",
        prompt="pieces STE00FF11",
        answer=json.dumps(["STE00FF11"]),
    )
    assert "ste00ff11" in sf.semantic_values_for_record(array_eval)
    with pytest.raises(ValueError, match="semantic values"):
        sf.validate_feasibility_records((array_train, array_eval))


def test_feasibility_template_and_operand_overlap_are_rejected_independently() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")

    base_train = sf.FeasibilityRecord(
        family="hex_copy",
        split="train",
        index=0,
        template_id="shared-form",
        operand_id="train-operand",
        prompt="copy 1111111111111111",
        answer="1111111111111111",
    )
    base_eval = sf.FeasibilityRecord(
        family="hex_copy",
        split="eval",
        index=0,
        template_id="SHARED-FORM",
        operand_id="eval-operand",
        prompt="copy 2222222222222222",
        answer="2222222222222222",
    )
    with pytest.raises(ValueError, match="template_id"):
        sf.validate_feasibility_records((base_train, base_eval))

    base_eval = sf.FeasibilityRecord(
        **{
            **base_eval.__dict__,
            "template_id": "eval-form",
            "operand_id": "TRAIN-OPERAND",
        }
    )
    with pytest.raises(ValueError, match="operand_id"):
        sf.validate_feasibility_records((base_train, base_eval))


def test_feasibility_prompt_surface_overlap_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    surfaces = {
        family: {split: tuple(values) for split, values in by_split.items()}
        for family, by_split in sf.PROMPT_SURFACES.items()
    }
    surfaces["hex_copy"]["eval"] = (surfaces["hex_copy"]["train"][0], *surfaces["hex_copy"]["eval"][1:])
    monkeypatch.setattr(sf, "PROMPT_SURFACES", surfaces)

    with pytest.raises(ValueError, match="surface overlap"):
        sf.validate_prompt_surface_contract("hex_copy")
    with pytest.raises(ValueError, match="surface overlap"):
        sf.grouped_records()


def test_feasibility_script_direct_cli_inspect_records() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/phase8_sequence_feasibility.py",
            "inspect-records",
            "--family",
            "hex_copy",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        text=True,
    )
    data = json.loads(completed.stdout)
    assert len(data["hex_copy"]) == 576


def test_feasibility_root_numbering_refuses_overwrite_and_skips(tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    with pytest.raises(ValueError, match="next numbered root"):
        sf.validate_new_root(tmp_path / "feasibility_002")
    with pytest.raises(ValueError, match="canonical path spelling"):
        sf.validate_new_root(tmp_path / "alias" / ".." / "feasibility_001")
    with pytest.raises(ValueError, match="canonical path spelling"):
        sf.main(["run", "--root", f"{tmp_path}//feasibility_001"])

    predecessor = tmp_path / "feasibility_001"
    predecessor.mkdir()
    predecessor_manifest = predecessor / "manifest.json"
    predecessor_manifest.write_text('{"old":true,"terminal_status":"FAILED"}\n')
    predecessor_failed = predecessor / "FAILED.json"
    predecessor_failed.write_text(
        json.dumps({"status": "FAILED", "manifest_sha256": sf.file_sha256(predecessor_manifest)}) + "\n"
    )

    sf.validate_new_root(tmp_path / "feasibility_002", (predecessor,), ())
    with pytest.raises(ValueError, match="next numbered root"):
        sf.validate_new_root(tmp_path / "feasibility_003", (predecessor,), ())

    existing = tmp_path / "feasibility_002"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        sf.validate_new_root(existing, (predecessor,), ())


def test_feasibility_failed_terminal_binds_manifest(tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    sf.write_terminal(tmp_path, "FAILED", [], (), (), failure="synthetic failure")

    manifest = tmp_path / "manifest.json"
    summary = tmp_path / "summary.json"
    failed = tmp_path / "FAILED.json"
    assert manifest.exists()
    assert summary.exists()
    assert failed.exists()
    failed_data = json.loads(failed.read_text())
    manifest_data = json.loads(manifest.read_text())
    summary_data = json.loads(summary.read_text())
    assert failed_data["status"] == "FAILED"
    assert failed_data["manifest_sha256"] == sf.file_sha256(manifest)
    assert manifest_data["terminal_status"] == "FAILED"
    assert manifest_data["failure"] == "synthetic failure"
    assert manifest_data["environment"]["gpu_driver"] == sf.gpu_driver_version()
    assert summary_data["protocol"] == "phase8_sequence_feasibility"
    assert summary_data["source"]["script"] == "scripts/phase8_sequence_feasibility.py"
    assert summary_data["configuration"]["pass_threshold"] == sf.PASS_THRESHOLD
    assert summary_data["cells"] == []
    summary_inventory = [row for row in manifest_data["file_inventory"] if row["path"] == "summary.json"]
    assert summary_inventory == [
        {"path": "summary.json", "sha256": sf.file_sha256(summary), "bytes": summary.stat().st_size}
    ]


def test_feasibility_root_rejects_dual_terminal_and_manifest_status_mismatch(tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    root = tmp_path / "feasibility_001"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text('{"terminal_status":"DONE"}\n')
    done = root / "DONE.json"
    done.write_text(json.dumps({"status": "DONE", "manifest_sha256": sf.file_sha256(manifest)}) + "\n")
    failed = root / "FAILED.json"
    failed.write_text(json.dumps({"status": "FAILED", "manifest_sha256": sf.file_sha256(manifest)}) + "\n")

    with pytest.raises(ValueError, match="both DONE.json and FAILED.json"):
        sf.load_terminal_binding(root)

    failed.unlink()
    manifest.write_text('{"terminal_status":"FAILED"}\n')
    done.write_text(json.dumps({"status": "DONE", "manifest_sha256": sf.file_sha256(manifest)}) + "\n")
    with pytest.raises(ValueError, match="terminal_status"):
        sf.load_terminal_binding(root)


def test_source_clean_only_allows_inventory_bound_predecessor_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    source_commit = "source-sha"
    configuration = {"training_steps": 1500}
    cells = _passing_feasibility_cells()

    def write_inventory_root(root: Path, status: str) -> tuple[Path, Path, list[Path]]:
        root.mkdir(parents=True)
        summary = root / "summary.json"
        summary.write_text("{}\n")
        cell_dir = root / "hex_copy__small__seed0"
        cell_dir.mkdir()
        generations = cell_dir / "generations.jsonl"
        generations.write_text('{"generated":"abc","exact_match":true}\n')
        file_inventory = [
            {"path": "hex_copy__small__seed0/generations.jsonl", "sha256": sf.file_sha256(generations), "bytes": generations.stat().st_size},
            {"path": "summary.json", "sha256": sf.file_sha256(summary), "bytes": summary.stat().st_size},
        ]
        root_cells = cells if status == "DONE" else []
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "terminal_status": status,
                    "source_commit": source_commit,
                    "configuration": configuration,
                    "cells": root_cells,
                    "predecessor_roots": [],
                    "predecessor_selections": [],
                    "file_inventory": file_inventory,
                },
                sort_keys=True,
            )
            + "\n"
        )
        terminal = root / f"{status}.json"
        terminal.write_text(
            json.dumps({"status": status, "manifest_sha256": sf.file_sha256(manifest), "cells": root_cells})
            + "\n"
        )
        return manifest, terminal, [generations, summary]

    selected_root = tmp_path / "feasibility_001"
    manifest, terminal, inventory_files = write_inventory_root(selected_root, "DONE")
    selection = tmp_path / "feasibility_selection_001.json"
    selection.write_text(
        json.dumps(
            {
                "selected_root": str(selected_root),
                "selected_manifest_sha256": sf.file_sha256(manifest),
                "source_commit": source_commit,
                "configuration": configuration,
                "per_cell_counts": cells,
                "pass_decision": True,
                "independent_review_verdict": "ACCEPT",
                "predecessor_roots": [],
                "predecessor_selections": [],
            },
            sort_keys=True,
        )
        + "\n"
    )

    def set_git_status(paths: list[Path]) -> None:
        def fake_run(args: list[str], check: bool, text: bool, stdout: object) -> subprocess.CompletedProcess[str]:
            assert args == ["git", "status", "--porcelain=v1", "--untracked-files=all"]
            assert check is True
            assert text is True
            return subprocess.CompletedProcess(args, 0, stdout="".join(f"?? {path}\n" for path in paths))

        monkeypatch.setattr(sf.subprocess, "run", fake_run)

    allowed_paths = [selection, manifest, terminal, *inventory_files]
    set_git_status(allowed_paths)
    sf.validate_source_clean(tmp_path / "feasibility_002", (), (selection,))

    extra = selected_root / "unbound_extra.txt"
    extra.write_text("not bound\n")
    set_git_status([*allowed_paths, extra])
    with pytest.raises(RuntimeError, match="Untracked file"):
        sf.validate_source_clean(tmp_path / "feasibility_002", (), (selection,))

    inventory_files[-1].write_text("tampered after manifest\n")
    set_git_status(allowed_paths)
    with pytest.raises(ValueError, match="file_inventory"):
        sf.validate_source_clean(tmp_path / "feasibility_002", (), (selection,))


def test_selection_record_validation_binds_root_predecessors_and_checksums(tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    source_commit = "source-sha"
    configuration = {"training_steps": 1500}
    cells = _passing_feasibility_cells()

    def write_root(
        root: Path,
        status: str,
        root_cells: list[dict[str, object]],
        predecessor_roots: list[dict[str, object]],
        predecessor_selections: list[dict[str, object]],
    ) -> tuple[Path, Path]:
        root.mkdir()
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "terminal_status": status,
                    "source_commit": source_commit,
                    "configuration": configuration,
                    "cells": root_cells,
                    "predecessor_roots": predecessor_roots,
                    "predecessor_selections": predecessor_selections,
                },
                sort_keys=True,
            )
            + "\n"
        )
        terminal = root / f"{status}.json"
        terminal.write_text(
            json.dumps({"status": status, "manifest_sha256": sf.file_sha256(manifest), "cells": root_cells})
            + "\n"
        )
        return manifest, terminal

    def write_selection(
        path: Path,
        root: Path,
        manifest: Path,
        predecessor_roots: list[dict[str, object]],
        predecessor_selections: list[dict[str, object]],
        *,
        selected_cells: list[dict[str, object]] | None = None,
        overrides: dict[str, object] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "selected_root": str(root),
            "selected_manifest_sha256": sf.file_sha256(manifest),
            "source_commit": source_commit,
            "configuration": configuration,
            "per_cell_counts": cells if selected_cells is None else selected_cells,
            "pass_decision": True,
            "independent_review_verdict": "ACCEPT",
            "predecessor_roots": predecessor_roots,
            "predecessor_selections": predecessor_selections,
        }
        if overrides:
            data.update(overrides)
        path.write_text(json.dumps(data, sort_keys=True) + "\n")

    first_bad_root = tmp_path / "feasibility_999"
    first_bad_manifest, _first_bad_done = write_root(first_bad_root, "DONE", cells, [], [])
    first_bad_selection = tmp_path / "bad_first_selected_root" / "feasibility_selection_001.json"
    write_selection(first_bad_selection, first_bad_root, first_bad_manifest, [], [])
    with pytest.raises(ValueError, match="next numbered root"):
        sf.validate_selection_record(first_bad_selection)

    predecessor = tmp_path / "feasibility_001"
    predecessor.mkdir()
    predecessor_manifest = predecessor / "manifest.json"
    predecessor_manifest.write_text('{"old":true,"terminal_status":"FAILED"}\n')
    predecessor_done = predecessor / "FAILED.json"
    predecessor_done.write_text(
        json.dumps({"status": "FAILED", "manifest_sha256": sf.file_sha256(predecessor_manifest)}) + "\n"
    )
    predecessor_roots = [
        {
            "path": str(predecessor),
            "terminal_state": "FAILED",
            "terminal_sha256": sf.file_sha256(predecessor_done),
            "manifest_sha256": sf.file_sha256(predecessor_manifest),
        }
    ]

    gapped_dir = tmp_path / "gapped_selected_root"
    gapped_dir.mkdir()
    gapped_root = gapped_dir / "feasibility_003"
    gapped_manifest, _gapped_done = write_root(gapped_root, "DONE", cells, predecessor_roots, [])
    gapped_selection = gapped_dir / "feasibility_selection_001.json"
    write_selection(gapped_selection, gapped_root, gapped_manifest, predecessor_roots, [])
    with pytest.raises(ValueError, match="next numbered root"):
        sf.validate_selection_record(gapped_selection)

    lineage_root = tmp_path / "feasibility_002"
    lineage_manifest, _lineage_done = write_root(lineage_root, "DONE", cells, predecessor_roots, [])
    predecessor_selection = tmp_path / "feasibility_selection_001.json"
    write_selection(predecessor_selection, lineage_root, lineage_manifest, predecessor_roots, [])
    predecessor_selections = [
        {"path": str(predecessor_selection), "sha256": sf.file_sha256(predecessor_selection)}
    ]
    full_predecessor_roots = [*predecessor_roots, sf.terminal_binding(lineage_root)]

    root = tmp_path / "feasibility_003"
    manifest, done = write_root(root, "DONE", cells, full_predecessor_roots, predecessor_selections)

    selection = tmp_path / "feasibility_selection_002.json"
    write_selection(selection, root, manifest, full_predecessor_roots, predecessor_selections)

    assert sf.validate_selection_record(selection)["pass_decision"] is True

    duplicate_predecessor_selections = [dict(predecessor_selections[0]), dict(predecessor_selections[0])]
    duplicate_selection_root = tmp_path / "bad_duplicate_predecessor_selection_root" / "feasibility_003"
    duplicate_selection_root.parent.mkdir()
    duplicate_selection_manifest, _duplicate_selection_done = write_root(
        duplicate_selection_root,
        "DONE",
        cells,
        full_predecessor_roots,
        duplicate_predecessor_selections,
    )
    duplicate_predecessor_selection = (
        tmp_path / "bad_duplicate_predecessor_selection" / "feasibility_selection_002.json"
    )
    write_selection(
        duplicate_predecessor_selection,
        duplicate_selection_root,
        duplicate_selection_manifest,
        full_predecessor_roots,
        duplicate_predecessor_selections,
    )
    with pytest.raises(ValueError, match="duplicate canonical paths"):
        sf.validate_selection_record(duplicate_predecessor_selection)

    descending_predecessor_selections = [
        {"path": str(selection), "sha256": sf.file_sha256(selection)},
        dict(predecessor_selections[0]),
    ]
    descending_predecessor_roots = [*full_predecessor_roots, sf.terminal_binding(root)]
    descending_selection_root = tmp_path / "bad_descending_predecessor_selection_root" / "feasibility_004"
    descending_selection_root.parent.mkdir()
    descending_selection_manifest, _descending_selection_done = write_root(
        descending_selection_root,
        "DONE",
        cells,
        descending_predecessor_roots,
        descending_predecessor_selections,
    )
    descending_predecessor_selection = (
        tmp_path / "bad_descending_predecessor_selection" / "feasibility_selection_003.json"
    )
    write_selection(
        descending_predecessor_selection,
        descending_selection_root,
        descending_selection_manifest,
        descending_predecessor_roots,
        descending_predecessor_selections,
    )
    with pytest.raises(ValueError, match="strictly ascending"):
        sf.validate_selection_record(descending_predecessor_selection)

    auto_manifest_root = tmp_path / "auto_manifest" / "feasibility_004"
    auto_manifest_root.mkdir(parents=True)
    assert sf.build_manifest(auto_manifest_root, cells, (predecessor,), (predecessor_selection,), "DONE")[
        "predecessor_roots"
    ] == full_predecessor_roots

    omitted_transitive_root = tmp_path / "omitted_transitive_root" / "feasibility_003"
    omitted_transitive_root.parent.mkdir()
    omitted_transitive_roots = [sf.terminal_binding(lineage_root)]
    omitted_transitive_manifest, _omitted_transitive_done = write_root(
        omitted_transitive_root,
        "DONE",
        cells,
        omitted_transitive_roots,
        predecessor_selections,
    )
    omitted_transitive_selection = (
        tmp_path / "bad_omitted_transitive_root" / "feasibility_selection_002.json"
    )
    write_selection(
        omitted_transitive_selection,
        omitted_transitive_root,
        omitted_transitive_manifest,
        omitted_transitive_roots,
        predecessor_selections,
    )
    with pytest.raises(ValueError, match="complete root lineage closure"):
        sf.validate_selection_record(omitted_transitive_selection)

    for index, bad_value in enumerate(("REJECT", "accepted", "", 1)):
        bad_selection = tmp_path / f"bad_verdict_{index}" / "feasibility_selection_002.json"
        write_selection(
            bad_selection,
            root,
            manifest,
            full_predecessor_roots,
            predecessor_selections,
            overrides={"independent_review_verdict": bad_value},
        )
        with pytest.raises(ValueError, match="independent_review_verdict"):
            sf.validate_selection_record(bad_selection)

    duplicate_root = tmp_path / "duplicate_root_binding" / "feasibility_003"
    duplicate_root.parent.mkdir()
    duplicate_roots = [*full_predecessor_roots, dict(full_predecessor_roots[0])]
    duplicate_manifest, _duplicate_done = write_root(
        duplicate_root,
        "DONE",
        cells,
        duplicate_roots,
        predecessor_selections,
    )
    duplicate_selection = tmp_path / "bad_duplicate_root_binding" / "feasibility_selection_002.json"
    write_selection(duplicate_selection, duplicate_root, duplicate_manifest, duplicate_roots, predecessor_selections)
    with pytest.raises(ValueError, match="duplicate root bindings"):
        sf.validate_selection_record(duplicate_selection)

    bad_selection = tmp_path / "bad_selected_root" / "feasibility_selection_002.json"
    write_selection(
        bad_selection,
        root,
        manifest,
        full_predecessor_roots,
        predecessor_selections,
        overrides={"selected_root": str(tmp_path / "selected_root")},
    )
    with pytest.raises(ValueError, match="feasibility_NNN"):
        sf.validate_selection_record(bad_selection)

    bad_selection = tmp_path / "bad_selected_root_type" / "feasibility_selection_002.json"
    write_selection(
        bad_selection,
        root,
        manifest,
        full_predecessor_roots,
        predecessor_selections,
        overrides={"selected_root": 12},
    )
    with pytest.raises(ValueError, match="selected_root"):
        sf.validate_selection_record(bad_selection)

    bad_selection = tmp_path / "bad_selected_root_alias" / "feasibility_selection_002.json"
    write_selection(
        bad_selection,
        root,
        manifest,
        full_predecessor_roots,
        predecessor_selections,
        overrides={"selected_root": str(root.parent / "alias" / ".." / root.name)},
    )
    with pytest.raises(ValueError, match="canonical path spelling"):
        sf.validate_selection_record(bad_selection)

    for spelling in (
        f"./{root.name}",
        f"{root}/",
        f"{root.parent}//{root.name}",
    ):
        bad_selection = tmp_path / f"bad_selected_root_spelling_{len(spelling)}" / "feasibility_selection_002.json"
        write_selection(
            bad_selection,
            root,
            manifest,
            full_predecessor_roots,
            predecessor_selections,
            overrides={"selected_root": spelling},
        )
        with pytest.raises(ValueError, match="canonical path spelling"):
            sf.validate_selection_record(bad_selection)

    bad_name_selection = tmp_path / "feasibility_selection_bad_name.json"
    bad_name_selection.write_text(selection.read_text())
    with pytest.raises(ValueError, match="feasibility_selection_NNN"):
        sf.validate_selection_record(bad_name_selection)

    skip_selection = tmp_path / "feasibility_selection_999.json"
    skip_selection.write_text(selection.read_text())
    with pytest.raises(ValueError, match="next numbered selection"):
        sf.validate_selection_record(skip_selection)

    bad_selection = tmp_path / "bad_source" / "feasibility_selection_002.json"
    write_selection(
        bad_selection,
        root,
        manifest,
        full_predecessor_roots,
        predecessor_selections,
        overrides={"source_commit": "wrong-source"},
    )
    with pytest.raises(ValueError, match="source_commit"):
        sf.validate_selection_record(bad_selection)

    bad_counts = [dict(cell) for cell in cells]
    bad_counts[0]["exact_matches"] = 53
    bad_selection = tmp_path / "bad_counts" / "feasibility_selection_002.json"
    write_selection(
        bad_selection,
        root,
        manifest,
        full_predecessor_roots,
        predecessor_selections,
        selected_cells=bad_counts,
    )
    with pytest.raises(ValueError, match="per_cell_counts"):
        sf.validate_selection_record(bad_selection)

    bad_selection = tmp_path / "bad_predecessor_roots" / "feasibility_selection_002.json"
    write_selection(bad_selection, root, manifest, [], predecessor_selections)
    with pytest.raises(ValueError, match="predecessor_roots"):
        sf.validate_selection_record(bad_selection)

    bad_selection = tmp_path / "bad_predecessor_selections" / "feasibility_selection_001.json"
    write_selection(bad_selection, root, manifest, full_predecessor_roots, [])
    with pytest.raises(ValueError, match="predecessor_selections"):
        sf.validate_selection_record(bad_selection)

    for bad_value in ("true", 1, False):
        bad_selection = tmp_path / f"bad_pass_{bad_value!r}" / "feasibility_selection_002.json"
        write_selection(
            bad_selection,
            root,
            manifest,
            full_predecessor_roots,
            predecessor_selections,
            overrides={"pass_decision": bad_value},
        )
        with pytest.raises(ValueError, match="pass_decision"):
            sf.validate_selection_record(bad_selection)

    bad_cells = [dict(cell) for cell in cells]
    bad_cells[0]["exact_matches"] = 51
    bad_cells[0]["passed"] = False
    bad_root = tmp_path / "feasibility_004"
    bad_manifest, _bad_done = write_root(bad_root, "DONE", bad_cells, full_predecessor_roots, predecessor_selections)
    bad_selection = tmp_path / "bad_52_64" / "feasibility_selection_002.json"
    write_selection(
        bad_selection,
        bad_root,
        bad_manifest,
        full_predecessor_roots,
        predecessor_selections,
        selected_cells=bad_cells,
    )
    with pytest.raises(ValueError, match="52/64"):
        sf.validate_selection_record(bad_selection)

    aliased_predecessor_roots = [dict(full_predecessor_roots[0]), *full_predecessor_roots[1:]]
    aliased_predecessor_roots[0]["path"] = str(predecessor.parent / "alias" / ".." / predecessor.name)
    alias_root = tmp_path / "feasibility_005"
    alias_manifest, _alias_done = write_root(alias_root, "DONE", cells, aliased_predecessor_roots, predecessor_selections)
    bad_selection = tmp_path / "bad_predecessor_root_alias" / "feasibility_selection_002.json"
    write_selection(bad_selection, alias_root, alias_manifest, aliased_predecessor_roots, predecessor_selections)
    with pytest.raises(ValueError, match="canonical path spelling"):
        sf.validate_selection_record(bad_selection)

    aliased_predecessor_selections = [dict(predecessor_selections[0])]
    aliased_predecessor_selections[0]["path"] = str(predecessor_selection.parent / "alias" / ".." / predecessor_selection.name)
    alias_root = tmp_path / "feasibility_006"
    alias_manifest, _alias_done = write_root(alias_root, "DONE", cells, full_predecessor_roots, aliased_predecessor_selections)
    bad_selection = tmp_path / "bad_predecessor_selection_alias" / "feasibility_selection_002.json"
    write_selection(bad_selection, alias_root, alias_manifest, full_predecessor_roots, aliased_predecessor_selections)
    with pytest.raises(ValueError, match="canonical path spelling"):
        sf.validate_selection_record(bad_selection)

    malformed_predecessor = tmp_path / "feasibility_007"
    malformed_predecessor.mkdir()
    malformed_manifest = malformed_predecessor / "manifest.json"
    malformed_manifest.write_text('{"old":true,"terminal_status":"FAILED"}\n')
    malformed_done = malformed_predecessor / "FAILED.json"
    malformed_done.write_text('{"status":"FAILED"}\n')
    with pytest.raises(ValueError, match="does not bind"):
        sf.terminal_binding(malformed_predecessor)

    predecessor_done.write_text('{"status":"DONE"}\n')
    with pytest.raises(ValueError, match="status does not match"):
        sf.validate_selection_record(selection)
