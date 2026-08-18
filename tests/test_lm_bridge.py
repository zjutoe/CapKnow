from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path
import re
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
    training_accuracy,
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
    with pytest.raises(ValueError, match="256-token"):
        tokenizer.encode_training_record("xy", "ab", max_length=257)
    with pytest.raises(ValueError, match="256-token"):
        tokenizer.pad(record.input_ids, 257)
    with pytest.raises(ValueError, match="256-token"):
        tokenizer.batch_pad((record.input_ids,), length=257)
    with pytest.raises(ValueError, match="exact integer"):
        tokenizer.batch_pad((record.input_ids,), length=256.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        tokenizer.encode_evaluation_prefix("xy", max_generated_tokens=-1)
    with pytest.raises(ValueError, match="64-token"):
        tokenizer.decode_generated_response((BOS_ID, ord("p"), SEP_ID, *([ord("a")] * 65)))
    with pytest.raises(ValueError, match="256-token"):
        tokenizer.decode_generated_response((BOS_ID, *([ord("p")] * 300), SEP_ID, EOS_ID))
    with pytest.raises(ValueError, match="256-token"):
        tokenizer.validate_special_token_placement((BOS_ID, *([ord("p")] * 300), SEP_ID, EOS_ID), mode="generated")

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
    with pytest.raises(ValueError, match="non-negative"):
        model_a.greedy_decode(prefix, max_new_tokens=-1)
    long_prefix = torch.tensor([[BOS_ID, *([ord("p")] * 191), SEP_ID]], dtype=torch.long)
    with pytest.raises(ValueError, match="256-token"):
        model_a.greedy_decode(long_prefix, max_new_tokens=64)


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
            "parameter_count": sf.expected_parameter_count(model_size),
        }
        for family in sf.FAMILIES
        for model_size in sf.MODEL_SIZES
        for seed in sf.SEEDS
    ]


def _source_snapshot(source_commit: str | None = None) -> object:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    return sf.SourceSnapshot(commit=_test_source_commit() if source_commit is None else source_commit, status_lines=(), ignored_inputs=())


def _test_source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        cwd=REPO_ROOT,
    ).stdout.strip()


def _historical_source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD^"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        cwd=REPO_ROOT,
    ).stdout.strip()


def _write_feasibility_root(
    root: Path,
    status: str,
    cells: list[dict[str, object]],
    predecessor_roots: tuple[Path, ...] = (),
    predecessor_selections: tuple[Path, ...] = (),
    *,
    lightweight: bool = False,
    source_commit: str | None = None,
) -> tuple[Path, Path]:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    root.mkdir(parents=True)
    record_groups = None if lightweight else sf.grouped_records()
    for cell in cells:
        generations = root / str(cell["generations_path"])
        checkpoint = root / str(cell["checkpoint_path"])
        generations.parent.mkdir(parents=True, exist_ok=True)
        if lightweight:
            generations.write_text("{}\n")
            checkpoint.write_bytes(b"checkpoint")
        else:
            assert record_groups is not None
            records = record_groups[str(cell["family"])]["eval"]
            tokenizer = ByteTokenizer()
            generations.write_text(
                "\n".join(
                    json.dumps(_generation_row(record, tokenizer, exact_match=index < int(cell["exact_matches"])))
                    for index, record in enumerate(records[: int(cell["eval_count"])])
                )
                + "\n"
            )
            model = build_model(str(cell["model_size"]))
            save_checkpoint(
                str(checkpoint),
                model,
                metadata={
                    "family": cell["family"],
                    "model_size": cell["model_size"],
                    "seed": cell["seed"],
                    "training_steps": sf.TRAINING_STEPS,
                },
            )
    sf.write_terminal(
        root,
        status,
        cells,
        predecessor_roots,
        predecessor_selections,
        failure=None if status == "DONE" else "synthetic failure",
        source_snapshot=_source_snapshot(source_commit),
    )
    return root / "manifest.json", root / f"{status}.json"


def _generation_row(record: object, tokenizer: ByteTokenizer, *, exact_match: bool) -> dict[str, object]:
    generated = record.answer if exact_match else "mismatch"
    raw_token_ids = [*tokenizer.encode_evaluation_prefix(record.prompt), *tokenizer.encode_text(generated), EOS_ID]
    return {
        "family": record.family,
        "index": record.index,
        "template_id": record.template_id,
        "operand_id": record.operand_id,
        "prompt": record.prompt,
        "expected": record.answer,
        "generated": generated,
        "raw_token_ids": raw_token_ids,
        "invalid_generation": False,
        "generation_error": None,
        "exact_match": exact_match,
    }


def _historical_failed_cell() -> dict[str, object]:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    return {
        "family": "hex_copy",
        "model_size": "small",
        "seed": 0,
        "eval_count": 2,
        "exact_matches": 1,
        "passed": False,
        "generations_path": "hex_copy__small__seed0/generations.jsonl",
        "checkpoint_path": "hex_copy__small__seed0/checkpoint_step1500.pt",
        "parameter_count": sf.expected_parameter_count("small"),
    }


def _legacy_generation_rows() -> list[dict[str, object]]:
    tokenizer = ByteTokenizer()
    rows = []
    for index, expected, generated in (
        (0, "legacy-expected-0000", "legacy-expected-0000"),
        (1, "legacy-expected-0001", "legacy-generated-0001"),
    ):
        prompt = f"Legacy source prompt {index}: abcdef01234567{index:02d}"
        raw_token_ids = [*tokenizer.encode_evaluation_prefix(prompt), *tokenizer.encode_text(generated), EOS_ID]
        rows.append(
            {
                "family": "hex_copy",
                "index": index,
                "template_id": f"legacy-template-{index}",
                "operand_id": f"legacy-operand-{index}",
                "prompt": prompt,
                "expected": expected,
                "generated": generated,
                "raw_token_ids": raw_token_ids,
                "invalid_generation": False,
                "generation_error": None,
                "exact_match": generated == expected,
            }
        )
    return rows


def _write_historical_generation_root(
    root: Path,
    *,
    source_commit: str,
    rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    cell = _historical_failed_cell()
    root.mkdir(parents=True)
    generations = root / str(cell["generations_path"])
    checkpoint = root / str(cell["checkpoint_path"])
    generations.parent.mkdir(parents=True, exist_ok=True)
    generations.write_text("\n".join(json.dumps(row) for row in (rows or _legacy_generation_rows())) + "\n")
    save_checkpoint(
        str(checkpoint),
        build_model("small"),
        metadata={
            "family": cell["family"],
            "model_size": cell["model_size"],
            "seed": cell["seed"],
            "training_steps": sf.TRAINING_STEPS,
        },
    )
    sf.write_terminal(
        root,
        "FAILED",
        [cell],
        (),
        (),
        failure="historical synthetic failure",
        source_snapshot=_source_snapshot(source_commit),
    )
    return cell


def _write_selection(
    path: Path,
    root: Path,
    manifest: Path,
    predecessor_roots: list[dict[str, object]],
    predecessor_selections: list[dict[str, object]],
    *,
    cells: list[dict[str, object]] | None = None,
    overrides: dict[str, object] | None = None,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    data = {
        "selected_root": str(root),
        "selected_manifest_sha256": sf.file_sha256(manifest),
        "source_commit": _test_source_commit(),
        "configuration": sf.frozen_configuration(),
        "per_cell_counts": _passing_feasibility_cells() if cells is None else cells,
        "pass_decision": True,
        "independent_review_verdict": "ACCEPT",
        "predecessor_roots": predecessor_roots,
        "predecessor_selections": predecessor_selections,
    }
    if overrides:
        data.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n")


def _unchecked_terminal_binding(root: Path) -> dict[str, object]:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    terminal, _terminal_data, _manifest, manifest_sha = sf.load_terminal_binding(root)
    return {
        "path": str(root),
        "terminal_state": terminal.stem,
        "terminal_sha256": sf.file_sha256(terminal),
        "manifest_sha256": manifest_sha,
    }


def _rewrite_terminal_manifest_sha(root: Path, status: str) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    terminal = root / f"{status}.json"
    terminal_data = json.loads(terminal.read_text())
    terminal_data["manifest_sha256"] = sf.file_sha256(root / "manifest.json")
    sf.write_json(terminal, terminal_data)


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
        train_records = {(record.prompt, record.answer) for record in splits["train"]}
        eval_records = {(record.prompt, record.answer) for record in splits["eval"]}
        train_semantic = {value for record in splits["train"] for value in sf.semantic_values_for_record(record)}
        eval_semantic = {value for record in splits["eval"] for value in sf.semantic_values_for_record(record)}
        assert len(train_operands) == 512
        assert len(eval_operands) == 64
        assert len(train_templates) == 4
        assert len(eval_templates) == 4
        assert train_operands.isdisjoint(eval_operands)
        assert train_templates.isdisjoint(eval_templates)
        assert train_prompts.isdisjoint(eval_prompts)
        assert train_records.isdisjoint(eval_records)
        assert train_semantic.isdisjoint(eval_semantic)
        assert [len(record.prompt.encode("utf-8")) for record in splits["train"][:64]] == [
            len(record.prompt.encode("utf-8")) for record in splits["eval"][:64]
        ]
        assert [len(record.answer.encode("utf-8")) for record in splits["train"][:64]] == [
            len(record.answer.encode("utf-8")) for record in splits["eval"][:64]
        ]
        assert set(sf.prompt_placeholder_sequence(surface) for surface in sf.PROMPT_SURFACES[family]["train"]) == set(
            sf.prompt_placeholder_sequence(surface) for surface in sf.PROMPT_SURFACES[family]["eval"]
        )
        for train_surface, eval_surface in zip(
            sf.PROMPT_SURFACES[family]["train"],
            sf.PROMPT_SURFACES[family]["eval"],
            strict=True,
        ):
            assert train_surface != eval_surface
            assert len(train_surface.encode("utf-8")) == len(eval_surface.encode("utf-8"))
            assert sf.prompt_placeholder_sequence(train_surface) == sf.prompt_placeholder_sequence(eval_surface)
        if family == "hex_copy":
            train_hex = {
                value
                for record in splits["train"]
                for value in sf.HEX_OPERAND_RE.findall(f"{record.prompt}\n{record.answer}")
            }
            eval_hex = {
                value
                for record in splits["eval"]
                for value in sf.HEX_OPERAND_RE.findall(f"{record.prompt}\n{record.answer}")
            }
            assert all(re.fullmatch(r"[0-9a-f]{16}", value) for value in train_hex | eval_hex)
            assert {int(value[0], 16) >= 8 for value in train_hex} == {False, True}
            assert {int(value[0], 16) >= 8 for value in eval_hex} == {False, True}
        if family == "named_value_json":
            train_keys = {key.casefold() for record in splits["train"] for key, _value in sf.NAMED_FIELD_RE.findall(record.prompt)}
            eval_keys = {key.casefold() for record in splits["eval"] for key, _value in sf.NAMED_FIELD_RE.findall(record.prompt)}
            assert train_keys == eval_keys == set(sf.NAMED_VALUE_KEYS)
            generated_values = [
                value
                for record in (*splits["train"], *splits["eval"])
                for value in sf.NAMED_VALUE_RE.findall(f"{record.prompt}\n{record.answer}")
            ]
            assert generated_values
            assert all(
                re.fullmatch(r"(?:red|blue|green|silver)-[0-9a-f]{4}", value, re.IGNORECASE)
                for value in generated_values
            )
            assert sf.NAMED_VALUE_RE.fullmatch("red-deadbeef") is None
            suffixes_by_split = {
                split_name: [
                    value.rsplit("-", 1)[1].casefold()
                    for record in splits[split_name]
                    for _key, value in sf.NAMED_FIELD_RE.findall(record.prompt)
                ]
                for split_name in ("train", "eval")
            }
            assert len(set(suffixes_by_split["train"] + suffixes_by_split["eval"])) == sum(
                len(suffixes) for suffixes in suffixes_by_split.values()
            )
            for suffixes in suffixes_by_split.values():
                assert all({suffix[position] for suffix in suffixes} == set("0123456789abcdef") for position in range(4))
        if family == "boolean_json":
            operand_sets = {}
            suffix_sets = {}
            for split_name in ("train", "eval"):
                split_operands = [
                    match.group(0).casefold()
                    for record in splits[split_name]
                    for match in sf.BOOLEAN_OPERAND_RE.finditer(record.prompt)
                ]
                assert len(split_operands) == len(splits[split_name])
                assert all(re.fullmatch(r"(?:affirm|reject)-[0-9a-f]{16}", operand) for operand in split_operands)
                assert {operand.split("-", 1)[0] for operand in split_operands} == set(sf.BOOLEAN_LABELS)
                assert len({len(operand.encode("utf-8")) for operand in split_operands}) == 1
                assert not any(
                    phrase in record.prompt
                    for record in splits[split_name]
                    for phrase in ("is at most", "is greater than", "comparison")
                )
                operand_sets[split_name] = set(split_operands)
                suffix_sets[split_name] = {operand.rsplit("-", 1)[1] for operand in split_operands}
                assert len(suffix_sets[split_name]) == len(split_operands)
                suffix_tails_by_label = {
                    label: sorted(
                        suffix[-1]
                        for suffix in suffix_sets[split_name]
                        if any(operand == f"{label}-{suffix}" for operand in split_operands)
                    )
                    for label in sf.BOOLEAN_LABELS
                }
                assert suffix_tails_by_label["affirm"] == suffix_tails_by_label["reject"]
                for template_id in {record.template_id for record in splits[split_name]}:
                    template_answers = [record.answer for record in splits[split_name] if record.template_id == template_id]
                    template_labels = [
                        sf.parsed_boolean_operands(record.prompt)[0][0]
                        for record in splits[split_name]
                        if record.template_id == template_id
                    ]
                    assert template_answers.count("true") == template_answers.count("false")
                    assert template_labels.count("affirm") == template_labels.count("reject")
            assert operand_sets["train"].isdisjoint(operand_sets["eval"])
            assert suffix_sets["train"].isdisjoint(suffix_sets["eval"])
        if family == "array_json":
            items = [
                value
                for record in (*splits["train"], *splits["eval"])
                for value in sf.ARRAY_ITEM_RE.findall(f"{record.prompt}\n{record.answer}")
            ]
            assert items
            assert all(re.fullmatch(r"q[0-9a-f]{4}", value) for value in items)
            assert sf.ARRAY_ITEM_RE.fullmatch("q0badcafe") is None
            suffixes_by_split = {
                split_name: [
                    value[1:].casefold()
                    for record in splits[split_name]
                    for value in sf.ARRAY_ITEM_RE.findall(record.prompt)
                ]
                for split_name in ("train", "eval")
            }
            assert len(set(suffixes_by_split["train"] + suffixes_by_split["eval"])) == sum(
                len(suffixes) for suffixes in suffixes_by_split.values()
            )
            for suffixes in suffixes_by_split.values():
                assert all({suffix[position] for suffix in suffixes} == set("0123456789abcdef") for position in range(4))
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

    for forbidden in (
        "A recorded claim has value true. Write that Boolean in compact JSON.",
        "state 0101",
        "graph A->B",
        "seed 123",
        "model id 7",
        "prerequisite rule table",
    ):
        with pytest.raises(ValueError, match="forbidden Phase 8"):
            sf.reject_scientific_markers(forbidden)

    condition_prompts = {
        record.prompt
        for split in ("training", "evaluation")
        for record in cg.build_split_records(split, "CONDITION", 8)
    }
    assert len(condition_prompts) >= 8
    for prompt in condition_prompts:
        with pytest.raises(ValueError):
            sf.reject_scientific_markers(prompt)
        with pytest.raises(ValueError):
            sf.reject_scientific_markers(prompt.lower())
    with pytest.raises(ValueError):
        sf.reject_scientific_markers(
            "A table entry says newkey has stored value newvalue. Write the value for newkey in compact JSON."
        )
    composition_context = {
        "memory": {"novel-key": "novel-value"},
        "key": "novel-key",
        "items": ["novel-value", "other-value"],
    }
    composition_prompt = cg.render_prompt(
        "MEMORY_FILTER",
        composition_context,
        "memory_filter__train_explicit_a",
        "train_explicit",
    )
    with pytest.raises(ValueError):
        sf.reject_scientific_markers(composition_prompt)
    with pytest.raises(ValueError):
        sf.reject_scientific_markers(composition_prompt.swapcase())
    for contaminated in (
        f"Feasibility check: {composition_prompt}",
        f"{composition_prompt} This is only a feasibility check.",
    ):
        with pytest.raises(ValueError):
            sf.reject_scientific_markers(contaminated)
    contaminated_records = list(groups["hex_copy"]["train"])
    contaminated_records[0] = sf.FeasibilityRecord(
        **{**contaminated_records[0].__dict__, "prompt": f"Feasibility check: {composition_prompt}"}
    )
    with pytest.raises(ValueError):
        sf.validate_feasibility_records(tuple(contaminated_records))
    multiline_context = {
        "memory": {"novel-key": "novel\nvalue"},
        "key": "novel-key",
        "items": ["novel\nvalue", "other-value"],
    }
    multiline_prompt = cg.render_prompt(
        "MEMORY_FILTER",
        multiline_context,
        "memory_filter__train_explicit_a",
        "train_explicit",
    )
    for contaminated in (
        multiline_prompt,
        f"Feasibility check: {multiline_prompt}",
        f"{multiline_prompt}\nThis is only a feasibility check.",
    ):
        with pytest.raises(ValueError):
            sf.reject_scientific_markers(contaminated)
    contaminated_records[0] = sf.FeasibilityRecord(
        **{**contaminated_records[0].__dict__, "prompt": f"Feasibility check: {multiline_prompt}"}
    )
    with pytest.raises(ValueError):
        sf.validate_feasibility_records(tuple(contaminated_records))
    empty_operand_cases = (
        (
            "SEARCH_CONDITION",
            {"items": ["a", "b", "c"], "target": ""},
            "search_condition__probe_a",
            "explicit",
        ),
        (
            "FILTER_CONDITION",
            {"items": ["a", "b", "c"], "target": ""},
            "filter_condition__probe_a",
            "explicit",
        ),
        (
            "MEMORY_FILTER",
            {"memory": {"": ""}, "key": "", "items": ["", "other-value"]},
            "memory_filter__train_explicit_a",
            "train_explicit",
        ),
    )
    empty_operand_prompts = [
        cg.render_prompt(task_id, context, template_id, style)
        for task_id, context, template_id, style in empty_operand_cases
    ]
    for prompt in empty_operand_prompts:
        for contaminated in (
            prompt,
            f"Feasibility check: {prompt}",
            f"{prompt} This is only a feasibility check.",
        ):
            with pytest.raises(ValueError):
                sf.reject_scientific_markers(contaminated)
    contaminated_records[0] = sf.FeasibilityRecord(
        **{**contaminated_records[0].__dict__, "prompt": f"Feasibility check: {empty_operand_prompts[0]}"}
    )
    with pytest.raises(ValueError):
        sf.validate_feasibility_records(tuple(contaminated_records))
    condition_counterfactual_prompts = (
        cg.render_prompt("CONDITION", {"condition": True}, "condition__train_b", "train"),
        cg.render_prompt("CONDITION", {"condition": False}, "condition__train_c", "train"),
        cg.render_prompt("CONDITION", {"condition": False}, "neutral_eval__neutral_a", "neutral"),
    )
    generic_memory_search_prompts = (
        cg.render_prompt(
            "MEMORY_SEARCH",
            {"memory": {"alpha": "bravo"}, "key": "alpha", "items": ["bravo", "charlie"]},
            "memory_search__generic_a",
            "train",
        ),
        cg.render_prompt(
            "MEMORY_SEARCH",
            {"memory": {"": ""}, "key": "", "items": [""]},
            "memory_search__generic_a",
            "train",
        ),
        cg.render_prompt(
            "MEMORY_SEARCH",
            {"memory": {"novel-key": "novel\nvalue"}, "key": "novel-key", "items": ["novel\nvalue"]},
            "memory_search__generic_a",
            "train",
        ),
    )
    for prompt in (*condition_counterfactual_prompts, *generic_memory_search_prompts):
        for contaminated in (prompt, f"Feasibility check: {prompt}", f"{prompt} This is only a feasibility check."):
            with pytest.raises(ValueError):
                sf.reject_scientific_markers(contaminated)
    condition_contaminated_records = list(groups["hex_copy"]["train"])
    condition_contaminated_records[0] = sf.FeasibilityRecord(
        **{
            **condition_contaminated_records[0].__dict__,
            "prompt": f"Feasibility check: {condition_counterfactual_prompts[0]}",
        }
    )
    with pytest.raises(ValueError):
        sf.validate_feasibility_records(tuple(condition_contaminated_records))
    memory_search_contaminated_records = list(groups["hex_copy"]["train"])
    memory_search_contaminated_records[0] = sf.FeasibilityRecord(
        **{
            **memory_search_contaminated_records[0].__dict__,
            "prompt": f"Feasibility check: {generic_memory_search_prompts[0]}",
        }
    )
    with pytest.raises(ValueError):
        sf.validate_feasibility_records(tuple(memory_search_contaminated_records))


def test_feasibility_named_value_template_target_balance_and_coupled_tamper_rejection() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    records = list(sf.build_family_records("named_value_json"))

    for split_name, expected_count in (("train", sf.TRAIN_RECORDS_PER_FAMILY), ("eval", sf.EVAL_RECORDS_PER_FAMILY)):
        split_records = [record for record in records if record.split == split_name]
        counts: dict[tuple[str, str], int] = {}
        for record in split_records:
            key = (record.template_id, sf.named_value_target_key(record))
            counts[key] = counts.get(key, 0) + 1
        assert len(counts) == 16
        assert set(counts.values()) == {expected_count // 16}

    surface_tampered = []
    for record in records:
        target = sf.named_value_target_key(record)
        prompt = sf.PROMPT_SURFACES["named_value_json"][record.split][0].format(
            a=target,
            b=sf.named_value_field_text(record),
        )
        surface_tampered.append(sf.FeasibilityRecord(**{**record.__dict__, "prompt": prompt}))

    with pytest.raises(ValueError, match="template_id must identify the actual prompt surface"):
        sf.validate_feasibility_records(tuple(surface_tampered))

    coupled_records = []
    for record in records:
        fields = {key.casefold(): value.casefold() for key, value in sf.NAMED_FIELD_RE.findall(record.prompt)}
        target = sf.NAMED_VALUE_KEYS[record.index % len(sf.NAMED_VALUE_KEYS)]
        prompt = re.sub(
            r"\b(key|name)\s+(red|blue|green|silver)\b",
            lambda match: f"{match.group(1)} {target}",
            record.prompt,
            count=1,
            flags=re.IGNORECASE,
        )
        coupled_records.append(
            sf.FeasibilityRecord(
                **{
                    **record.__dict__,
                    "prompt": prompt,
                    "answer": sf.compact_json(fields[target]),
                }
            )
        )

    with pytest.raises(ValueError, match="template×target-key coverage"):
        sf.validate_feasibility_records(tuple(coupled_records))


def test_feasibility_array_template_count_balance_and_coupled_tamper_rejection() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    records = list(sf.build_family_records("array_json"))

    for split_name, expected_count in (("train", sf.TRAIN_RECORDS_PER_FAMILY), ("eval", sf.EVAL_RECORDS_PER_FAMILY)):
        split_records = [record for record in records if record.split == split_name]
        counts: dict[tuple[str, int], int] = {}
        for record in split_records:
            key = (record.template_id, sf.array_item_count(record))
            counts[key] = counts.get(key, 0) + 1
        assert len(counts) == 16
        assert set(counts.values()) == {expected_count // 16}

    surface_tampered = []
    for record in records:
        items = json.loads(record.answer)
        prompt = sf.PROMPT_SURFACES["array_json"][record.split][0].format(a=" | ".join(items))
        surface_tampered.append(sf.FeasibilityRecord(**{**record.__dict__, "prompt": prompt}))

    with pytest.raises(ValueError, match="template_id must identify the actual prompt surface"):
        sf.validate_feasibility_records(tuple(surface_tampered))

    coupled_records = []
    for record in records:
        target_count = 1 + record.index % 4
        items = [value.casefold() for value in sf.ARRAY_ITEM_RE.findall(record.prompt)[:target_count]]
        offset = 0x4000 if record.split == "train" else 0xC000
        while len(items) < target_count:
            items.append(f"q{offset + record.index * 4 + len(items):04x}")
        surface = sf.PROMPT_SURFACES["array_json"][record.split][record.index % 4]
        coupled_records.append(
            sf.FeasibilityRecord(
                **{
                    **record.__dict__,
                    "prompt": surface.format(a=" | ".join(items)),
                    "answer": sf.compact_json(items),
                    "semantic_values": (*items, sf.compact_json(items)),
                }
            )
        )

    with pytest.raises(ValueError, match="template×item-count coverage"):
        sf.validate_feasibility_records(tuple(coupled_records))


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


def test_feasibility_boolean_operand_overlap_is_rejected_from_raw_prompt() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    records = list(sf.build_family_records("boolean_json"))
    train_operand = sf.BOOLEAN_OPERAND_RE.search(
        next(record.prompt for record in records if record.split == "train")
    )
    assert train_operand is not None
    eval_index = next(index for index, record in enumerate(records) if record.split == "eval")
    records[eval_index] = sf.FeasibilityRecord(
        **{
            **records[eval_index].__dict__,
            "prompt": f"{records[eval_index].prompt} repeated mark {train_operand.group(0)}",
            "semantic_values": (),
        }
    )

    with pytest.raises(ValueError, match="semantic values"):
        sf.validate_feasibility_records(tuple(records))


def test_feasibility_boolean_suffix_reuse_under_opposite_label_is_rejected() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    records = list(sf.build_family_records("boolean_json"))
    train_record = next(
        record
        for record in records
        if record.split == "train" and sf.parsed_boolean_operands(record.prompt)[0][0] == "affirm"
    )
    train_suffix = sf.parsed_boolean_operands(train_record.prompt)[0][1]
    eval_index = next(
        index
        for index, record in enumerate(records)
        if record.split == "eval" and sf.parsed_boolean_operands(record.prompt)[0][0] == "reject"
    )
    eval_record = records[eval_index]
    eval_operand = sf.BOOLEAN_OPERAND_RE.search(eval_record.prompt)
    assert eval_operand is not None
    records[eval_index] = sf.FeasibilityRecord(
        **{
            **eval_record.__dict__,
            "prompt": eval_record.prompt.replace(eval_operand.group(0), f"reject-{train_suffix}"),
        }
    )

    with pytest.raises(ValueError, match="suffixes must be disjoint independent of label"):
        sf.validate_feasibility_records(tuple(records))


def test_feasibility_boolean_label_coded_suffix_tail_is_rejected() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    records = sf.build_family_records("boolean_json")
    tampered = []
    for record in records:
        operand = sf.BOOLEAN_OPERAND_RE.search(record.prompt)
        assert operand is not None
        label, suffix = sf.parsed_boolean_operands(record.prompt)[0]
        replacement = f"{label}-{suffix[:-1]}{'0' if label == 'affirm' else 'f'}"
        tampered.append(
            sf.FeasibilityRecord(
                **{
                    **record.__dict__,
                    "prompt": record.prompt.replace(operand.group(0), replacement),
                }
            )
        )

    with pytest.raises(ValueError, match="suffix-tail distributions must match exactly across labels"):
        sf.validate_feasibility_records(tuple(tampered))


@pytest.mark.parametrize("family", ("hex_copy", "boolean_json"))
def test_feasibility_hex_and_boolean_prompt_surface_collapse_is_rejected(family: str) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    records = list(sf.build_family_records(family))
    tampered = []
    for record in records:
        surface = sf.PROMPT_SURFACES[family][record.split][0]
        if family == "hex_copy":
            prompt = surface.format(a=record.index % 17, b=record.answer)
        else:
            operand = sf.BOOLEAN_OPERAND_RE.search(record.prompt)
            assert operand is not None
            prompt = surface.format(a=operand.group(0))
        tampered.append(sf.FeasibilityRecord(**{**record.__dict__, "prompt": prompt}))

    with pytest.raises(ValueError, match="template_id must identify the actual prompt surface"):
        sf.validate_feasibility_records(tuple(tampered))


def test_evaluate_model_retains_malformed_generations_as_invalid_mismatches() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    tokenizer = ByteTokenizer()
    record = sf.FeasibilityRecord(
        family="hex_copy",
        split="eval",
        index=0,
        template_id="eval-form",
        operand_id="eval-operand",
        prompt="copy abcdef0123456789",
        answer="abcdef0123456789",
    )

    class BadModel:
        def greedy_decode(self, prefix_tensor: torch.Tensor) -> torch.Tensor:
            bad_utf8_byte = 255
            return torch.tensor([[*prefix_tensor[0].tolist(), bad_utf8_byte, EOS_ID]], dtype=torch.long)

    correct, rows = sf.evaluate_model(BadModel(), (record,), tokenizer, torch.device("cpu"))

    assert correct == 0
    assert rows[0]["exact_match"] is False
    assert rows[0]["invalid_generation"] is True
    assert rows[0]["generated"] is None
    assert rows[0]["raw_token_ids"][-2:] == [255, EOS_ID]
    assert "UnicodeDecodeError" in rows[0]["generation_error"]


def test_training_accuracy_counts_malformed_generations_as_incorrect() -> None:
    tokenizer = ByteTokenizer()
    records = (TextRecord(prompt="copy abcdef0123456789", answer="abcdef0123456789"),)

    class BadTrainingModel:
        training = True

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

        def greedy_decode(self, prefix_tensor: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[*prefix_tensor[0].tolist(), 255, EOS_ID]], dtype=torch.long)

    model = BadTrainingModel()
    assert training_accuracy(model, records, tokenizer, device=torch.device("cpu")) == 0.0
    assert model.training is True


def test_checkpoint_replay_rejects_generation_rows_not_produced_by_checkpoint(tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    cell = _passing_feasibility_cells()[0]
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        str(checkpoint),
        build_model(str(cell["model_size"])),
        metadata={
            "family": cell["family"],
            "model_size": cell["model_size"],
            "seed": cell["seed"],
            "training_steps": sf.TRAINING_STEPS,
        },
    )
    tokenizer = ByteTokenizer()
    rows = [
        _generation_row(record, tokenizer, exact_match=index < int(cell["exact_matches"]))
        for index, record in enumerate(sf.grouped_records()[str(cell["family"])]["eval"])
    ]
    with pytest.raises(ValueError, match="Checkpoint replay"):
        sf.validate_checkpoint_replays_generations(checkpoint, cell, rows)


def test_checkpoint_artifact_rejects_unloadable_checkpoint(tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-a-checkpoint")
    with pytest.raises(ValueError, match="Checkpoint artifact"):
        sf.validate_checkpoint_artifact(checkpoint, _passing_feasibility_cells()[0])


def test_generation_artifact_rejects_fabricated_exact_match_rows(tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    cell = _passing_feasibility_cells()[0]
    tokenizer = ByteTokenizer()
    rows = [
        _generation_row(record, tokenizer, exact_match=index < int(cell["exact_matches"]))
        for index, record in enumerate(sf.grouped_records()[str(cell["family"])]["eval"])
    ]
    rows[0]["generated"] = "wrong"
    generations = tmp_path / "generations.jsonl"
    generations.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="generated text|exact_match"):
        sf.validate_generation_artifact(generations, cell)


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

    legacy_named = sf.FeasibilityRecord(
        family="named_value_json",
        split="train",
        index=0,
        template_id="legacy-named-surface",
        operand_id="legacy-named-value",
        prompt=(
            "choose key red; fields "
            "red=red-deadbeef; blue=blue-00000000; green=green-00000000; silver=silver-00000000"
        ),
        answer=json.dumps("red-deadbeef"),
    )
    with pytest.raises(ValueError, match="fixed held-out value grammar"):
        sf.semantic_values_for_record(legacy_named)

    legacy_array = sf.FeasibilityRecord(
        family="array_json",
        split="train",
        index=0,
        template_id="legacy-array-surface",
        operand_id="legacy-array-value",
        prompt="chunks q0badcafe",
        answer=json.dumps(["q0badcafe"]),
    )
    with pytest.raises(ValueError, match="fixed held-out item grammar"):
        sf.semantic_values_for_record(legacy_array)

    named_train = sf.FeasibilityRecord(
        family="named_value_json",
        split="train",
        index=0,
        template_id="named-train-surface",
        operand_id="named-train-value",
        prompt=(
            "choose key red; fields "
            "red=red-beef; blue=blue-2222; green=green-2a2a; silver=silver-3333"
        ),
        answer=json.dumps("red-beef"),
    )
    named_eval = sf.FeasibilityRecord(
        family="named_value_json",
        split="eval",
        index=0,
        template_id="named-eval-surface",
        operand_id="named-eval-value",
        prompt=(
            "select key RED; fields "
            "RED=RED-BEEF; BLUE=BLUE-2222; GREEN=GREEN-2A2A; SILVER=SILVER-3333"
        ),
        answer=json.dumps("RED-BEEF"),
    )
    assert "red-beef" in sf.semantic_values_for_record(named_eval)
    with pytest.raises(ValueError, match="semantic values"):
        sf.validate_feasibility_records((named_train, named_eval))

    array_train = sf.FeasibilityRecord(
        family="array_json",
        split="train",
        index=0,
        template_id="array-train-surface",
        operand_id="array-train-value",
        prompt="chunks q0bad",
        answer=json.dumps(["q0bad"]),
    )
    array_eval = sf.FeasibilityRecord(
        family="array_json",
        split="eval",
        index=0,
        template_id="array-eval-surface",
        operand_id="array-eval-value",
        prompt="pieces Q0BAD",
        answer=json.dumps(["Q0BAD"]),
    )
    assert "q0bad" in sf.semantic_values_for_record(array_eval)
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


def test_feasibility_root_numbering_refuses_overwrite_and_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    artifact_parent = tmp_path / "artifacts" / "phase8_toy_lm_bridge"
    artifact_parent.mkdir(parents=True)
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", artifact_parent)

    with pytest.raises(ValueError, match="feasibility_001"):
        sf.validate_new_root(artifact_parent / "feasibility_000")
    with pytest.raises(ValueError, match="complete and continuous"):
        sf.validate_new_root(artifact_parent / "feasibility_002")
    with pytest.raises(ValueError, match="canonical path spelling"):
        sf.validate_new_root(artifact_parent / "alias" / ".." / "feasibility_001")
    with pytest.raises(ValueError, match="canonical path spelling"):
        sf.main(["run", "--root", f"{artifact_parent}//feasibility_001"])
    with pytest.raises(ValueError, match="located directly"):
        sf.validate_new_root(tmp_path / "feasibility_001")

    predecessor = artifact_parent / "feasibility_001"
    predecessor.mkdir()
    sf.write_terminal(
        predecessor,
        "FAILED",
        [],
        (),
        (),
        failure="synthetic predecessor failure",
        source_snapshot=_source_snapshot(),
    )

    sf.validate_new_root(artifact_parent / "feasibility_002", (predecessor,), ())
    with pytest.raises(ValueError, match="complete and continuous"):
        sf.validate_new_root(artifact_parent / "feasibility_003", (predecessor,), ())
    second_predecessor = artifact_parent / "feasibility_002"
    _write_feasibility_root(second_predecessor, "FAILED", [], predecessor_roots=(predecessor,))
    with pytest.raises(ValueError, match="strictly ascending"):
        sf.validate_new_root(artifact_parent / "feasibility_003", (second_predecessor, predecessor), ())
    selection = artifact_parent / "feasibility_selection_001.json"
    selection.write_text("{}\n")
    monkeypatch.setattr(sf, "validate_selection_record", lambda path, **kwargs: {"selected_root": str(predecessor)})
    sf.validate_new_root(artifact_parent / "feasibility_003", (second_predecessor,), (selection,))

    with pytest.raises(FileExistsError, match="overwrite"):
        sf.validate_new_root(second_predecessor, (predecessor,), ())

    symlink_parent = tmp_path / "symlink_artifacts" / "phase8_toy_lm_bridge"
    symlink_parent.mkdir(parents=True)
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", symlink_parent)
    dangling = symlink_parent / "feasibility_001"
    dangling.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with pytest.raises((FileExistsError, ValueError), match="symlink|overwrite"):
        sf.validate_new_root(dangling)


def test_feasibility_failed_terminal_binds_manifest(tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    sf.write_terminal(
        tmp_path,
        "FAILED",
        [],
        (),
        (),
        failure="synthetic failure",
        source_snapshot=_source_snapshot(),
    )

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


def test_historical_predecessor_generation_rows_are_not_reinterpreted_as_current_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    artifact_parent = tmp_path / "artifacts" / "phase8_toy_lm_bridge"
    artifact_parent.mkdir(parents=True)
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", artifact_parent)

    historical_commit = _historical_source_commit()
    predecessor = artifact_parent / "feasibility_001"
    _write_historical_generation_root(predecessor, source_commit=historical_commit)
    current_first = sf.grouped_records()["hex_copy"]["eval"][0]
    historical_first = json.loads(
        (predecessor / "hex_copy__small__seed0" / "generations.jsonl").read_text().splitlines()[0]
    )
    assert historical_first["prompt"] != current_first.prompt
    assert historical_first["expected"] != current_first.answer

    sf.validate_new_root(artifact_parent / "feasibility_002", (predecessor,), ())


def test_historical_predecessor_generation_internal_inconsistency_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    artifact_parent = tmp_path / "artifacts" / "phase8_toy_lm_bridge"
    artifact_parent.mkdir(parents=True)
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", artifact_parent)

    historical_commit = _historical_source_commit()
    predecessor = artifact_parent / "feasibility_001"
    cell = _write_historical_generation_root(predecessor, source_commit=historical_commit)
    rows = _legacy_generation_rows()
    rows[0] = {**rows[0], "prompt": "tampered retained prompt with unchanged raw prefix"}
    (predecessor / str(cell["generations_path"])).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    sf.write_terminal(
        predecessor,
        "FAILED",
        [cell],
        (),
        (),
        failure="historical synthetic failure",
        source_snapshot=_source_snapshot(historical_commit),
    )

    with pytest.raises(ValueError, match="raw_token_ids prefix"):
        sf.validate_new_root(artifact_parent / "feasibility_002", (predecessor,), ())


def test_current_source_generation_prompt_mismatch_is_still_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    artifact_parent = tmp_path / "artifacts" / "phase8_toy_lm_bridge"
    artifact_parent.mkdir(parents=True)
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", artifact_parent)

    current_source_root = artifact_parent / "feasibility_001"
    tokenizer = ByteTokenizer()
    current_records = sf.grouped_records()["hex_copy"]["eval"][:2]
    rows = [
        _generation_row(record, tokenizer, exact_match=index == 0)
        for index, record in enumerate(current_records)
    ]
    rows[0]["prompt"] = "Current-source retained prompt tamper"
    rows[0]["raw_token_ids"] = [
        *tokenizer.encode_evaluation_prefix(str(rows[0]["prompt"])),
        *tokenizer.encode_text(str(rows[0]["generated"])),
        EOS_ID,
    ]
    _write_historical_generation_root(current_source_root, source_commit=_test_source_commit(), rows=rows)

    with pytest.raises(ValueError, match="row prompt"):
        sf.validate_new_root(artifact_parent / "feasibility_002", (current_source_root,), ())


def test_source_clean_only_allows_inventory_bound_predecessor_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    cells = _passing_feasibility_cells()
    monkeypatch.setattr(sf, "validate_generation_artifact", lambda path, cell: [])
    monkeypatch.setattr(sf, "validate_historical_generation_artifact", lambda path, cell: [])
    monkeypatch.setattr(sf, "validate_checkpoint_artifact", lambda path, cell: None)
    monkeypatch.setattr(sf, "validate_checkpoint_replays_generations", lambda path, cell, rows: None)
    artifact_parent = tmp_path / "artifacts" / "phase8_toy_lm_bridge"
    artifact_parent.mkdir(parents=True)
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", artifact_parent)

    selected_root = artifact_parent / "feasibility_001"
    manifest, terminal = _write_feasibility_root(selected_root, "DONE", cells, lightweight=True)
    selection = artifact_parent / "feasibility_selection_001.json"
    _write_selection(selection, selected_root, manifest, [], [], cells=cells)
    inventory_files = [selected_root / row["path"] for row in json.loads(manifest.read_text())["file_inventory"]]
    inventory_bytes = {path: path.read_bytes() for path in inventory_files}

    def set_git_status(paths: list[Path], *, ignored_lines: tuple[str, ...] = ()) -> None:
        def fake_run(
            args: list[str],
            check: bool,
            text: bool,
            stdout: object,
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            if args[:4] == ["git", "status", "--porcelain=v1", "--untracked-files=all"]:
                return subprocess.CompletedProcess(args, 0, stdout="".join(f"?? {path}\n" for path in paths))
            if args[:5] == ["git", "status", "--porcelain=v1", "--ignored", "--untracked-files=all"]:
                return subprocess.CompletedProcess(args, 0, stdout="".join(f"!! {line}\n" for line in ignored_lines))
            if args == ["git", "rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(args, 0, stdout="a" * 40 + "\n")
            if args[:3] == ["git", "cat-file", "-e"]:
                return subprocess.CompletedProcess(args, 0, stdout="")
            raise AssertionError(args)

        monkeypatch.setattr(sf.subprocess, "run", fake_run)

    allowed_paths = [selection, manifest, terminal, *inventory_files]
    set_git_status(allowed_paths)
    snapshot = sf.validate_source_clean(artifact_parent / "feasibility_002", (), (selection,))
    assert snapshot.commit == "a" * 40

    extra = selected_root / "unbound_extra.txt"
    extra.write_text("not bound\n")
    set_git_status([*allowed_paths, extra])
    with pytest.raises(ValueError, match="file_inventory|Expecting value"):
        sf.validate_source_clean(artifact_parent / "feasibility_002", (), (selection,))
    extra.unlink()

    inventory_files[-1].write_text("tampered after manifest\n")
    set_git_status(allowed_paths)
    with pytest.raises(ValueError, match="file_inventory|Expecting value"):
        sf.validate_source_clean(artifact_parent / "feasibility_002", (), (selection,))

    inventory_files[-1].write_bytes(inventory_bytes[inventory_files[-1]])
    set_git_status(allowed_paths, ignored_lines=("scripts/phase8_sequence_feasibility.py",))
    with pytest.raises(RuntimeError, match="Ignored executable source input"):
        sf.validate_source_clean(artifact_parent / "feasibility_002", (), (selection,))


def test_source_snapshot_rejects_head_or_status_change_before_terminal_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    snapshot = sf.SourceSnapshot(commit="a" * 40, status_lines=(), ignored_inputs=())

    def changed_head(
        args: list[str],
        check: bool,
        text: bool,
        stdout: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if args == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="b" * 40 + "\n")
        raise AssertionError(args)

    monkeypatch.setattr(sf.subprocess, "run", changed_head)
    with pytest.raises(sf.SourceChangedError, match="HEAD changed"):
        sf.verify_source_unchanged(snapshot)

    def changed_status(
        args: list[str],
        check: bool,
        text: bool,
        stdout: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if args == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="a" * 40 + "\n")
        if args == ["git", "status", "--porcelain=v1", "--untracked-files=all"]:
            return subprocess.CompletedProcess(args, 0, stdout=" M scripts/phase8_sequence_feasibility.py\n")
        raise AssertionError(args)

    monkeypatch.setattr(sf.subprocess, "run", changed_status)
    with pytest.raises(sf.SourceChangedError, match="worktree status changed"):
        sf.verify_source_unchanged(snapshot)

    def changed_ignored(
        args: list[str],
        check: bool,
        text: bool,
        stdout: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if args == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="a" * 40 + "\n")
        if args == ["git", "status", "--porcelain=v1", "--untracked-files=all"]:
            return subprocess.CompletedProcess(args, 0, stdout="")
        if args[:5] == ["git", "status", "--porcelain=v1", "--ignored", "--untracked-files=all"]:
            return subprocess.CompletedProcess(args, 0, stdout="!! scripts/phase8_sequence_feasibility.py\n")
        raise AssertionError(args)

    monkeypatch.setattr(sf.subprocess, "run", changed_ignored)
    with pytest.raises(sf.SourceChangedError, match="Ignored executable"):
        sf.verify_source_unchanged(snapshot)

    active_root = sf.ARTIFACT_PARENT / "feasibility_001.tmp"

    def active_output_only(
        args: list[str],
        check: bool,
        text: bool,
        stdout: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if args == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="a" * 40 + "\n")
        if args == ["git", "status", "--porcelain=v1", "--untracked-files=all"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="?? artifacts/phase8_toy_lm_bridge/feasibility_001.tmp/in_progress.txt\n",
            )
        if args[:5] == ["git", "status", "--porcelain=v1", "--ignored", "--untracked-files=all"]:
            return subprocess.CompletedProcess(args, 0, stdout="")
        raise AssertionError(args)

    monkeypatch.setattr(sf.subprocess, "run", active_output_only)
    sf.verify_source_unchanged(snapshot, active_output_root=active_root)


def test_strict_selection_validation_rejects_fabricated_roots_and_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    cells = _passing_feasibility_cells()
    original_write_feasibility_root = _write_feasibility_root

    def lightweight_write_feasibility_root(
        root: Path,
        status: str,
        cells: list[dict[str, object]],
        predecessor_roots: tuple[Path, ...] = (),
        predecessor_selections: tuple[Path, ...] = (),
    ) -> tuple[Path, Path]:
        return original_write_feasibility_root(
            root,
            status,
            cells,
            predecessor_roots,
            predecessor_selections,
            lightweight=True,
        )

    monkeypatch.setattr(sys.modules[__name__], "_write_feasibility_root", lightweight_write_feasibility_root)
    monkeypatch.setattr(
        sys.modules[__name__],
        "save_checkpoint",
        lambda path, model, metadata: Path(path).write_bytes(b"checkpoint"),
    )
    monkeypatch.setattr(sf, "validate_generation_artifact", lambda path, cell: [])
    monkeypatch.setattr(sf, "validate_checkpoint_artifact", lambda path, cell: None)
    monkeypatch.setattr(sf, "validate_checkpoint_replays_generations", lambda path, cell, rows: None)

    def set_parent(name: str) -> Path:
        parent = tmp_path / name / "artifacts" / "phase8_toy_lm_bridge"
        parent.mkdir(parents=True)
        monkeypatch.setattr(sf, "ARTIFACT_PARENT", parent)
        return parent

    parent = set_parent("valid")
    predecessor = parent / "feasibility_001"
    _write_feasibility_root(predecessor, "FAILED", [])
    predecessor_roots = [sf.terminal_binding(predecessor)]
    selected_root = parent / "feasibility_002"
    selected_manifest, _selected_done = _write_feasibility_root(
        selected_root,
        "DONE",
        cells,
        predecessor_roots=(predecessor,),
    )
    selection = parent / "feasibility_selection_001.json"
    _write_selection(selection, selected_root, selected_manifest, predecessor_roots, [], cells=cells)
    assert sf.validate_selection_record(selection)["selected_root"] == str(selected_root)

    parent = set_parent("valid_selection_lineage")
    predecessor = parent / "feasibility_001"
    _write_feasibility_root(predecessor, "FAILED", [])
    predecessor_roots = [sf.terminal_binding(predecessor)]
    first_root = parent / "feasibility_002"
    first_manifest, _first_done = _write_feasibility_root(
        first_root,
        "DONE",
        cells,
        predecessor_roots=(predecessor,),
    )
    first_selection = parent / "feasibility_selection_001.json"
    _write_selection(first_selection, first_root, first_manifest, predecessor_roots, [], cells=cells)
    assert sf.validate_selection_record(first_selection)["selected_root"] == str(first_root)
    second_root = parent / "feasibility_003"
    second_manifest, _second_done = _write_feasibility_root(
        second_root,
        "DONE",
        cells,
        predecessor_roots=(predecessor,),
        predecessor_selections=(first_selection,),
    )
    second_manifest_data = json.loads(second_manifest.read_text())
    second_selection = parent / "feasibility_selection_002.json"
    _write_selection(
        second_selection,
        second_root,
        second_manifest,
        second_manifest_data["predecessor_roots"],
        second_manifest_data["predecessor_selections"],
        cells=cells,
    )
    assert sf.validate_selection_record(second_selection)["selected_root"] == str(second_root)

    parent = set_parent("valid_retry_after_selected_root")
    first_root = parent / "feasibility_001"
    first_manifest, _first_done = _write_feasibility_root(first_root, "DONE", cells)
    first_selection = parent / "feasibility_selection_001.json"
    _write_selection(first_selection, first_root, first_manifest, [], [], cells=cells)
    failed_retry = parent / "feasibility_002"
    _write_feasibility_root(failed_retry, "FAILED", [], predecessor_selections=(first_selection,))
    retry_root = parent / "feasibility_003"
    retry_manifest, _retry_done = _write_feasibility_root(
        retry_root,
        "DONE",
        cells,
        predecessor_roots=(failed_retry,),
        predecessor_selections=(first_selection,),
    )
    retry_manifest_data = json.loads(retry_manifest.read_text())
    assert [Path(binding["path"]).name for binding in retry_manifest_data["predecessor_roots"]] == [
        "feasibility_001",
        "feasibility_002",
    ]
    retry_selection = parent / "feasibility_selection_002.json"
    _write_selection(
        retry_selection,
        retry_root,
        retry_manifest,
        retry_manifest_data["predecessor_roots"],
        retry_manifest_data["predecessor_selections"],
        cells=cells,
    )
    assert sf.validate_selection_record(retry_selection)["selected_root"] == str(retry_root)

    parent = set_parent("nested_fabricated_intermediate_lineage")
    first_root = parent / "feasibility_001"
    _write_feasibility_root(first_root, "FAILED", [])
    fabricated_second_root = parent / "feasibility_002"
    _write_feasibility_root(fabricated_second_root, "FAILED", [])
    selected_root = parent / "feasibility_003"
    selected_manifest, _selected_done = _write_feasibility_root(
        selected_root,
        "DONE",
        cells,
        predecessor_roots=(first_root,),
    )
    selected_manifest_data = json.loads(selected_manifest.read_text())
    selected_manifest_data["predecessor_roots"] = [
        _unchecked_terminal_binding(first_root),
        _unchecked_terminal_binding(fabricated_second_root),
    ]
    sf.write_json(selected_manifest, selected_manifest_data)
    _rewrite_terminal_manifest_sha(selected_root, "DONE")
    selected_manifest_data = json.loads(selected_manifest.read_text())
    fabricated_selection = parent / "feasibility_selection_001.json"
    _write_selection(
        fabricated_selection,
        selected_root,
        selected_manifest,
        selected_manifest_data["predecessor_roots"],
        [],
        cells=cells,
    )
    with pytest.raises(ValueError, match="complete and continuous"):
        sf.validate_selection_record(fabricated_selection)

    parent = set_parent("nested_forged_selection_sha")
    first_root = parent / "feasibility_001"
    first_manifest, _first_done = _write_feasibility_root(first_root, "DONE", cells)
    first_selection = parent / "feasibility_selection_001.json"
    _write_selection(first_selection, first_root, first_manifest, [], [], cells=cells)
    forged_second_root = parent / "feasibility_002"
    second_manifest, _second_failed = _write_feasibility_root(
        forged_second_root,
        "FAILED",
        [],
        predecessor_selections=(first_selection,),
    )
    second_manifest_data = json.loads(second_manifest.read_text())
    second_manifest_data["predecessor_selections"][0]["sha256"] = "0" * 64
    sf.write_json(second_manifest, second_manifest_data)
    _rewrite_terminal_manifest_sha(forged_second_root, "FAILED")
    final_root = parent / "feasibility_003"
    final_manifest, _final_done = _write_feasibility_root(
        final_root,
        "DONE",
        cells,
        predecessor_roots=(first_root,),
    )
    final_manifest_data = json.loads(final_manifest.read_text())
    final_manifest_data["predecessor_roots"] = [
        _unchecked_terminal_binding(first_root),
        _unchecked_terminal_binding(forged_second_root),
    ]
    final_manifest_data["predecessor_selections"] = [sf.selection_binding(first_selection)]
    sf.write_json(final_manifest, final_manifest_data)
    _rewrite_terminal_manifest_sha(final_root, "DONE")
    final_manifest_data = json.loads(final_manifest.read_text())
    final_selection = parent / "feasibility_selection_002.json"
    _write_selection(
        final_selection,
        final_root,
        final_manifest,
        final_manifest_data["predecessor_roots"],
        final_manifest_data["predecessor_selections"],
        cells=cells,
    )
    with pytest.raises(ValueError, match="selection checksum"):
        sf.validate_selection_record(final_selection)

    parent = set_parent("root_validation_cache")
    roots: list[Path] = []
    for number in range(1, 9):
        root = parent / f"feasibility_{number:03d}"
        _write_feasibility_root(root, "FAILED", [], predecessor_roots=tuple(roots))
        roots.append(root)
    root_validation_calls: list[Path] = []
    original_validate_root_artifacts = sf._validate_feasibility_root_artifacts

    def counted_validate_root_artifacts(
        root: Path,
        *,
        require_passing: bool,
        context: object,
        current_commit: str,
    ) -> None:
        root_validation_calls.append(root.resolve())
        original_validate_root_artifacts(
            root,
            require_passing=require_passing,
            context=context,
            current_commit=current_commit,
        )

    monkeypatch.setattr(sf, "_validate_feasibility_root_artifacts", counted_validate_root_artifacts)
    sf.validate_feasibility_root_artifacts(roots[-1], require_passing=False)
    assert len(root_validation_calls) <= len(roots)
    assert len(set(root_validation_calls)) == len(root_validation_calls)

    parent = set_parent("root_validation_cycle")
    cyclic_root = parent / "feasibility_001"
    _write_feasibility_root(cyclic_root, "FAILED", [])
    manifest = cyclic_root / "manifest.json"
    failed = cyclic_root / "FAILED.json"
    manifest_data = json.loads(manifest.read_text())
    manifest_data["predecessor_roots"] = [
        {
            "path": str(cyclic_root),
            "terminal_state": "FAILED",
            "terminal_sha256": "0" * 64,
            "manifest_sha256": "0" * 64,
        }
    ]
    sf.write_json(manifest, manifest_data)
    failed_data = json.loads(failed.read_text())
    failed_data["manifest_sha256"] = sf.file_sha256(manifest)
    sf.write_json(failed, failed_data)
    with pytest.raises(ValueError, match="checksum mismatch|lineage contains a cycle"):
        sf.validate_feasibility_root_artifacts(cyclic_root, require_passing=False)

    outside_selection = tmp_path / "feasibility_selection_001.json"
    outside_selection.write_text(selection.read_text())
    with pytest.raises(ValueError, match="located directly"):
        sf.validate_selection_record(outside_selection)

    parent = set_parent("missing_checkpoint")
    bad_root = parent / "feasibility_001"
    bad_manifest, _bad_done = _write_feasibility_root(bad_root, "DONE", cells)
    (bad_root / str(cells[0]["checkpoint_path"])).unlink()
    bad_selection = parent / "feasibility_selection_001.json"
    _write_selection(bad_selection, bad_root, bad_manifest, [], [], cells=cells)
    with pytest.raises(ValueError, match="file_inventory|checkpoint"):
        sf.validate_selection_record(bad_selection)

    parent = set_parent("symlink_inventory")
    bad_root = parent / "feasibility_001"
    bad_manifest, _bad_done = _write_feasibility_root(bad_root, "DONE", cells)
    generation = bad_root / str(cells[0]["generations_path"])
    external_generation = tmp_path / "external_generations.jsonl"
    external_generation.write_bytes(generation.read_bytes())
    generation.unlink()
    generation.symlink_to(external_generation)
    bad_selection = parent / "feasibility_selection_001.json"
    _write_selection(bad_selection, bad_root, bad_manifest, [], [], cells=cells)
    with pytest.raises(ValueError, match="symlink"):
        sf.validate_selection_record(bad_selection)

    parent = set_parent("tampered_summary")
    bad_root = parent / "feasibility_001"
    bad_manifest, _bad_done = _write_feasibility_root(bad_root, "DONE", cells)
    (bad_root / "summary.json").write_text('{"tampered":true}\n')
    bad_selection = parent / "feasibility_selection_001.json"
    _write_selection(bad_selection, bad_root, bad_manifest, [], [], cells=cells)
    with pytest.raises(ValueError, match="Summary|file_inventory|checksum"):
        sf.validate_selection_record(bad_selection)

    parent = set_parent("bad_config")
    bad_root = parent / "feasibility_001"
    bad_manifest, _bad_done = _write_feasibility_root(bad_root, "DONE", cells)
    manifest_data = json.loads(bad_manifest.read_text())
    manifest_data["configuration"] = {"training_steps": 1500}
    bad_manifest.write_text(json.dumps(manifest_data, sort_keys=True) + "\n")
    (bad_root / "DONE.json").write_text(
        json.dumps(
            {
                "status": "DONE",
                "manifest_path": "manifest.json",
                "manifest_sha256": sf.file_sha256(bad_manifest),
                "pass_threshold": sf.PASS_THRESHOLD,
                "cells": cells,
            }
        )
        + "\n"
    )
    bad_selection = parent / "feasibility_selection_001.json"
    _write_selection(bad_selection, bad_root, bad_manifest, [], [], cells=cells)
    with pytest.raises(ValueError, match="frozen feasibility schema"):
        sf.validate_selection_record(bad_selection)

    parent = set_parent("bad_source_provenance")
    bad_root = parent / "feasibility_001"
    bad_manifest, _bad_done = _write_feasibility_root(bad_root, "DONE", cells)
    manifest_data = json.loads(bad_manifest.read_text())
    manifest_data["source_provenance"]["commit"] = "b" * 40
    bad_manifest.write_text(json.dumps(manifest_data, sort_keys=True) + "\n")
    (bad_root / "DONE.json").write_text(
        json.dumps(
            {
                "status": "DONE",
                "manifest_path": "manifest.json",
                "manifest_sha256": sf.file_sha256(bad_manifest),
                "pass_threshold": sf.PASS_THRESHOLD,
                "cells": cells,
            }
        )
        + "\n"
    )
    bad_selection = parent / "feasibility_selection_001.json"
    _write_selection(bad_selection, bad_root, bad_manifest, [], [], cells=cells)
    with pytest.raises(ValueError, match="source_provenance"):
        sf.validate_selection_record(bad_selection)

    parent = set_parent("dirty_source_provenance")
    bad_root = parent / "feasibility_001"
    _write_feasibility_root(bad_root, "DONE", cells)
    sf.write_terminal(
        bad_root,
        "DONE",
        cells,
        (),
        (),
        source_snapshot=sf.SourceSnapshot(
            commit=_test_source_commit(),
            status_lines=(" M scripts/phase8_sequence_feasibility.py",),
            ignored_inputs=(),
        ),
    )
    bad_manifest = bad_root / "manifest.json"
    bad_selection = parent / "feasibility_selection_001.json"
    _write_selection(bad_selection, bad_root, bad_manifest, [], [], cells=cells)
    with pytest.raises(ValueError, match="tracked or staged"):
        sf.validate_selection_record(bad_selection)

    parent = set_parent("unbound_source_provenance")
    failed_root = parent / "feasibility_001"
    failed_root.mkdir()
    sf.write_terminal(
        failed_root,
        "FAILED",
        [],
        (),
        (),
        failure="synthetic failure",
        source_snapshot=sf.SourceSnapshot(
            commit=_test_source_commit(),
            status_lines=("?? arbitrary_unbound_input.py",),
            ignored_inputs=(),
        ),
    )
    with pytest.raises(ValueError, match="unbound untracked"):
        sf.terminal_binding(failed_root)

    parent = set_parent("ignored_source_provenance")
    failed_root = parent / "feasibility_001"
    failed_root.mkdir()
    sf.write_terminal(
        failed_root,
        "FAILED",
        [],
        (),
        (),
        failure="synthetic failure",
        source_snapshot=sf.SourceSnapshot(
            commit=_test_source_commit(),
            status_lines=(),
            ignored_inputs=({"path": "scripts/untracked_exec.py", "sha256": "0" * 64, "bytes": 1},),
        ),
    )
    with pytest.raises(ValueError, match="ignored executable"):
        sf.terminal_binding(failed_root)

    parent = set_parent("missing_source_commit")
    bad_root = parent / "feasibility_001"
    _write_feasibility_root(bad_root, "DONE", cells)
    missing_commit = "a" * 40
    sf.write_terminal(
        bad_root,
        "DONE",
        cells,
        (),
        (),
        source_snapshot=sf.SourceSnapshot(commit=missing_commit, status_lines=(), ignored_inputs=()),
    )
    bad_manifest = bad_root / "manifest.json"
    bad_selection = parent / "feasibility_selection_001.json"
    _write_selection(
        bad_selection,
        bad_root,
        bad_manifest,
        [],
        [],
        cells=cells,
        overrides={"source_commit": missing_commit},
    )
    with pytest.raises(ValueError, match="source_commit"):
        sf.validate_selection_record(bad_selection)

    parent = set_parent("bad_terminal_threshold")
    bad_root = parent / "feasibility_001"
    bad_manifest, bad_done = _write_feasibility_root(bad_root, "DONE", cells)
    terminal_data = json.loads(bad_done.read_text())
    terminal_data["pass_threshold"] = 999
    bad_done.write_text(json.dumps(terminal_data, sort_keys=True) + "\n")
    bad_selection = parent / "feasibility_selection_001.json"
    _write_selection(bad_selection, bad_root, bad_manifest, [], [], cells=cells)
    with pytest.raises(ValueError, match="pass_threshold"):
        sf.validate_selection_record(bad_selection)

    parent = set_parent("failed_terminal_error_mismatch")
    failed_root = parent / "feasibility_001"
    _write_feasibility_root(failed_root, "FAILED", [])
    failed_terminal = failed_root / "FAILED.json"
    failed_data = json.loads(failed_terminal.read_text())
    failed_data["error"] = "contradictory failure"
    failed_terminal.write_text(json.dumps(failed_data, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="FAILED terminal error"):
        sf.terminal_binding(failed_root)

    parent = set_parent("bad_parameter_count")
    bad_cells = [dict(cell) for cell in cells]
    bad_cells[0]["parameter_count"] += 1
    bad_root = parent / "feasibility_001"
    bad_manifest, _bad_done = _write_feasibility_root(bad_root, "DONE", bad_cells)
    bad_selection = parent / "feasibility_selection_001.json"
    _write_selection(bad_selection, bad_root, bad_manifest, [], [], cells=bad_cells)
    with pytest.raises(ValueError, match="parameter_count"):
        sf.validate_selection_record(bad_selection)

    parent = set_parent("gapped_lineage")
    predecessor = parent / "feasibility_001"
    _write_feasibility_root(predecessor, "FAILED", [])
    bad_root = parent / "feasibility_003"
    bad_manifest, _bad_done = _write_feasibility_root(
        bad_root,
        "DONE",
        cells,
        predecessor_roots=(predecessor,),
    )
    bad_selection = parent / "feasibility_selection_001.json"
    _write_selection(bad_selection, bad_root, bad_manifest, [sf.terminal_binding(predecessor)], [], cells=cells)
    with pytest.raises(ValueError, match="complete and continuous"):
        sf.validate_selection_record(bad_selection)

    parent = set_parent("symlink_root")
    target = tmp_path / "outside_target"
    target.mkdir()
    symlink_root = parent / "feasibility_001"
    symlink_root.symlink_to(target, target_is_directory=True)
    symlink_selection = parent / "feasibility_selection_001.json"
    symlink_selection.write_text(
        json.dumps(
            {
                "selected_root": str(symlink_root),
                "selected_manifest_sha256": "0" * 64,
                "source_commit": "a" * 40,
                "configuration": sf.frozen_configuration(),
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
    with pytest.raises(ValueError, match="symlink|located directly"):
        sf.validate_selection_record(symlink_selection)
