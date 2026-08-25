from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from types import MappingProxyType

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch

from capability_certificate_lab.lm_bridge import corpus_generator as cg
from capability_certificate_lab.dsl.executor import MissingCapabilityError
from capability_certificate_lab.lm_bridge.model import (
    TIED_MODEL_PROTOCOL_REVISION,
    ToyCausalTransformer,
    TransformerConfig,
    build_historical_model,
    build_model,
    transformer_config,
)
from capability_certificate_lab.lm_bridge.tokenizer import BOS_ID, EOS_ID, PAD_ID, SEP_ID, ByteTokenizer
from capability_certificate_lab.lm_bridge.train import (
    TextRecord,
    contiguous_uint8_bytes,
    encode_record_batch,
    load_model_from_checkpoint,
    make_optimizer,
    response_only_labels,
    response_only_loss,
    run_byte_copy_overfit_control,
    save_checkpoint,
    set_deterministic_backend,
    training_accuracy,
    validate_tied_checkpoint_payload,
)


def _tied_config(name: str = "test-tiny", d_model: int = 16, n_heads: int = 2, n_layers: int = 1, d_ff: int = 32) -> TransformerConfig:
    return TransformerConfig(
        name=name,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        embedding_weight_tying=True,
        model_protocol_revision=TIED_MODEL_PROTOCOL_REVISION,
    )


def _snapshot_diagnostic_process_state(sf: object) -> dict[str, object]:
    return {
        "rng_states": sf.snapshot_rng_states(),
        "num_threads": torch.get_num_threads(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "env": {key: os.environ.get(key) for key in sf.DIAGNOSTIC_REQUIRED_ENV},
    }


def _restore_diagnostic_process_state(sf: object, state: dict[str, object]) -> None:
    env = state["env"]
    assert isinstance(env, dict)
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    sf.restore_rng_states(state["rng_states"])
    torch.set_num_threads(state["num_threads"])
    torch.backends.mkldnn.enabled = state["mkldnn_enabled"]
    torch.use_deterministic_algorithms(
        state["deterministic_algorithms"],
        warn_only=state["deterministic_warn_only"],
    )
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = state["cuda_tf32"]
    torch.backends.cudnn.allow_tf32 = state["cudnn_tf32"]


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
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    small_config = transformer_config("small")
    medium_config = transformer_config("medium")
    assert (small_config.d_model, small_config.n_heads, small_config.n_layers, small_config.d_ff) == (64, 4, 2, 256)
    assert (medium_config.d_model, medium_config.n_heads, medium_config.n_layers, medium_config.d_ff) == (128, 4, 4, 512)
    assert small_config.max_seq_len == medium_config.max_seq_len == 256
    assert small_config.dropout == medium_config.dropout == 0.0
    assert small_config.embedding_weight_tying is True
    assert medium_config.embedding_weight_tying is True
    assert small_config.model_protocol_revision == TIED_MODEL_PROTOCOL_REVISION
    assert medium_config.model_protocol_revision == TIED_MODEL_PROTOCOL_REVISION

    small = build_model("small")
    medium = build_model("medium")
    assert small.lm_head.weight is small.token_embedding.weight
    assert medium.lm_head.weight is medium.token_embedding.weight
    assert small.parameter_count == sum(parameter.numel() for parameter in small.parameters())
    assert medium.parameter_count == sum(parameter.numel() for parameter in medium.parameters())
    assert small.parameter_count == sf.FROZEN_PARAMETER_COUNTS["small"] == 133120
    assert medium.parameter_count == sf.FROZEN_PARAMETER_COUNTS["medium"] == 859392
    assert medium.parameter_count > small.parameter_count


def test_d2_tied_constructor_matches_historical_token_embedding_rng_and_optimizer(
    request: pytest.FixtureRequest,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    process_state = _snapshot_diagnostic_process_state(sf)
    request.addfinalizer(lambda: _restore_diagnostic_process_state(sf, process_state))

    for model_size in ("small", "medium"):
        for seed in (0, 1, 2):
            set_deterministic_backend(seed)
            historical = build_historical_model(model_size)
            historical_token_embedding = historical.token_embedding.weight.detach().clone()
            historical_rng = sf.snapshot_rng_states()

            set_deterministic_backend(seed)
            tied = build_model(model_size)
            tied_rng = sf.snapshot_rng_states()

            assert tied.config.embedding_weight_tying is True
            assert tied.config.model_protocol_revision == TIED_MODEL_PROTOCOL_REVISION
            assert tied.lm_head.weight is tied.token_embedding.weight
            assert tied.parameter_count == sf.FROZEN_PARAMETER_COUNTS[model_size]
            torch.testing.assert_close(tied.token_embedding.weight, historical_token_embedding, atol=0.0, rtol=0.0)
            assert contiguous_uint8_bytes(tied.token_embedding.weight) == contiguous_uint8_bytes(historical_token_embedding)
            assert repr(tied_rng[0]) == repr(historical_rng[0])
            torch.testing.assert_close(tied_rng[1], historical_rng[1], atol=0, rtol=0)
            if historical_rng[2] is None:
                assert tied_rng[2] is None
            else:
                assert tied_rng[2] is not None
                assert len(tied_rng[2]) == len(historical_rng[2])
                for tied_cuda_state, historical_cuda_state in zip(tied_rng[2], historical_rng[2], strict=True):
                    torch.testing.assert_close(tied_cuda_state, historical_cuda_state, atol=0, rtol=0)

            optimizer_params = list(make_optimizer(tied).param_groups[0]["params"])
            assert len({id(parameter) for parameter in optimizer_params}) == len(optimizer_params)
            assert sum(parameter is tied.token_embedding.weight for parameter in optimizer_params) == 1

            set_deterministic_backend(seed)
            replay_a = build_model(model_size)
            set_deterministic_backend(seed)
            replay_b = build_model(model_size)
            sample = torch.tensor([[BOS_ID, ord("x"), SEP_ID, ord("y"), EOS_ID]], dtype=torch.long)
            with torch.no_grad():
                torch.testing.assert_close(replay_a(sample), replay_b(sample), atol=0.0, rtol=0.0)
            prefix = torch.tensor([[BOS_ID, ord("x"), SEP_ID]], dtype=torch.long)
            torch.testing.assert_close(replay_a.greedy_decode(prefix), replay_b.greedy_decode(prefix), atol=0, rtol=0)


def test_causal_mask_prevents_future_token_access() -> None:
    set_deterministic_backend(0)
    model = ToyCausalTransformer(_tied_config())
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
    config = _tied_config()
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
    model = ToyCausalTransformer(_tied_config())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(str(path), model, metadata={"purpose": "unit-test"})

    loaded = load_model_from_checkpoint(str(path))
    assert loaded.parameter_count == model.parameter_count == sum(parameter.numel() for parameter in model.parameters())
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[key], atol=0.0, rtol=0.0)

    sample = torch.tensor([[BOS_ID, ord("x"), SEP_ID, ord("y"), EOS_ID]], dtype=torch.long)
    with torch.no_grad():
        torch.testing.assert_close(model(sample), loaded(sample), atol=0.0, rtol=0.0)


def test_d2_tied_checkpoint_schema_rejects_ambiguous_or_divergent_payloads(tmp_path: Path) -> None:
    set_deterministic_backend(0)
    model = build_model("small")
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(str(path), model, metadata={"purpose": "unit-test"})
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    assert validate_tied_checkpoint_payload(checkpoint).model_protocol_revision == TIED_MODEL_PROTOCOL_REVISION
    assert set(checkpoint["model_state_dict"]) == set(model.state_dict())
    assert contiguous_uint8_bytes(checkpoint["model_state_dict"]["token_embedding.weight"]) == contiguous_uint8_bytes(
        checkpoint["model_state_dict"]["lm_head.weight"]
    )
    loaded = load_model_from_checkpoint(str(path))
    assert loaded.lm_head.weight is loaded.token_embedding.weight

    def reject(mutator: object, match: str) -> None:
        mutated = {
            "model_state_dict": dict(checkpoint["model_state_dict"]),
            "config": dict(checkpoint["config"]),
            "parameter_count": checkpoint["parameter_count"],
            "metadata": dict(checkpoint["metadata"]),
        }
        assert callable(mutator)
        mutator(mutated)
        mutated_path = tmp_path / f"mutated_{len(list(tmp_path.iterdir()))}.pt"
        torch.save(mutated, mutated_path)
        with pytest.raises(ValueError, match=match):
            load_model_from_checkpoint(str(mutated_path))

    reject(lambda payload: payload["config"].pop("model_protocol_revision"), "schema|revision")
    reject(lambda payload: payload["config"].pop("embedding_weight_tying"), "schema|tying")
    reject(lambda payload: payload["config"].__setitem__("embedding_weight_tying", False), "tying=true")
    reject(lambda payload: payload["config"].__setitem__("model_protocol_revision", "phase8_untied_legacy_v1"), "revision")
    reject(lambda payload: payload["model_state_dict"].pop("lm_head.weight"), "keys")
    reject(lambda payload: payload["model_state_dict"].__setitem__("lm_head.weight", [0]), "tensor")
    reject(
        lambda payload: payload["model_state_dict"].__setitem__(
            "lm_head.weight",
            payload["model_state_dict"]["lm_head.weight"].to(torch.float64),
        ),
        "dtype|shape",
    )
    reject(
        lambda payload: payload["model_state_dict"].__setitem__(
            "lm_head.weight",
            payload["model_state_dict"]["lm_head.weight"][:1],
        ),
        "shape",
    )
    reject(
        lambda payload: payload["model_state_dict"].__setitem__(
            "lm_head.weight",
            payload["model_state_dict"]["lm_head.weight"].to_sparse(),
        ),
        "layout",
    )
    reject(
        lambda payload: payload["model_state_dict"].__setitem__(
            "lm_head.weight",
            payload["model_state_dict"]["lm_head.weight"].clone().add_(1.0),
        ),
        "byte-equal",
    )
    reject(
        lambda payload: (
            payload["model_state_dict"].__setitem__(
                "token_embedding.weight",
                torch.zeros_like(payload["model_state_dict"]["token_embedding.weight"]),
            ),
            payload["model_state_dict"].__setitem__(
                "lm_head.weight",
                -torch.zeros_like(payload["model_state_dict"]["lm_head.weight"]),
            ),
        ),
        "byte-equal",
    )

    nan_payload = {
        "model_state_dict": dict(checkpoint["model_state_dict"]),
        "config": dict(checkpoint["config"]),
        "parameter_count": checkpoint["parameter_count"],
        "metadata": dict(checkpoint["metadata"]),
    }
    nan_weight = nan_payload["model_state_dict"]["token_embedding.weight"].clone()
    nan_weight[0, 0] = float("nan")
    nan_payload["model_state_dict"]["token_embedding.weight"] = nan_weight
    nan_payload["model_state_dict"]["lm_head.weight"] = nan_weight.clone()
    nan_path = tmp_path / "matching_nan_payload.pt"
    torch.save(nan_payload, nan_path)
    assert validate_tied_checkpoint_payload(torch.load(nan_path, map_location="cpu", weights_only=True)).model_protocol_revision == TIED_MODEL_PROTOCOL_REVISION
    reject(lambda payload: payload.__setitem__("unexpected", True), "schema")


def test_four_record_cpu_byte_copy_overfit_control() -> None:
    result = run_byte_copy_overfit_control(
        seed=0,
        max_steps=500,
        config=_tied_config(d_model=32, n_heads=4, n_layers=1, d_ff=128),
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
            "embedding_weight_tying": True,
            "model_protocol_revision": TIED_MODEL_PROTOCOL_REVISION,
        }
        for family in sf.FAMILIES
        for model_size in sf.MODEL_SIZES
        for seed in sf.SEEDS
    ]


def _source_snapshot(source_commit: str | None = None) -> object:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    return sf.SourceSnapshot(commit=_test_source_commit() if source_commit is None else source_commit, status_lines=(), ignored_inputs=())


def _current_deterministic_flags() -> dict[str, object]:
    return {
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": ".",
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_tf32": False,
        "cudnn_tf32": False,
    }


def _allow_synthetic_historical_checkpoints(monkeypatch: pytest.MonkeyPatch, sf: object) -> None:
    monkeypatch.setattr(sf, "validate_historical_checkpoint_allowlist", lambda path: None)


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
    command_roots = tuple(
        Path(str(binding["path"]))
        for binding in sf.complete_predecessor_root_bindings(predecessor_roots, predecessor_selections)
    )
    sf.write_terminal(
        root,
        status,
        cells,
        predecessor_roots,
        predecessor_selections,
        failure=None if status == "DONE" else "synthetic failure",
        source_snapshot=_source_snapshot(source_commit),
        decision_diagnostic=sf.DECISION_DIAGNOSTIC_ROOT_BINDING,
        exact_command=sf.feasibility_exact_command(root, command_roots, Path(sf.FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT)),
        deterministic_flags=_current_deterministic_flags(),
        record_hashes=sf.FEASIBILITY_RECORD_HASHES,
    )
    manifest = root / "manifest.json"
    manifest_data = json.loads(manifest.read_text())
    manifest_data["environment"] = dict(sf.FEASIBILITY_REQUIRED_RUNTIME_ENV)
    sf.write_json(manifest, manifest_data)
    _rewrite_terminal_manifest_sha(root, status)
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
        "parameter_count": sf.HISTORICAL_PARAMETER_COUNTS["small"],
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
    sf.save_historical_checkpoint(
        str(checkpoint),
        build_historical_model("small"),
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


def _rewrite_current_publication_command(
    root: Path,
    status: str,
    output_root: Path,
    predecessor_roots: tuple[Path, ...] = (),
    predecessor_selections: tuple[Path, ...] = (),
    decision_diagnostic_root: Path | None = None,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    decision_root = decision_diagnostic_root or Path(sf.FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT)
    exact_command = sf.feasibility_exact_command(output_root, predecessor_roots, decision_root)
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.json"
    terminal_path = root / f"{status}.json"

    summary = json.loads(summary_path.read_text())
    summary["exact_command"] = exact_command
    sf.write_json(summary_path, summary)

    manifest = json.loads(manifest_path.read_text())
    manifest["exact_command"] = exact_command
    manifest["file_inventory"] = sf.inventory(root)
    sf.write_json(manifest_path, manifest)

    terminal = json.loads(terminal_path.read_text())
    terminal["exact_command"] = exact_command
    terminal["manifest_sha256"] = sf.file_sha256(manifest_path)
    sf.write_json(terminal_path, terminal)


def _refresh_current_publication_hashes(root: Path, status: str) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_inventory"] = sf.inventory(root)
    sf.write_json(manifest_path, manifest)
    _rewrite_terminal_manifest_sha(root, status)


def test_d2_current_configuration_and_record_hashes_are_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    expected_hashes = {
        "hex_train512": "8798d57c3bdde6693d8046e3a7525687f06976074dc0cceffaf1c674a0f8c390",
        "hex_eval64": "be97a3a2877fa5f8e6c2e85a4213f975ccc89de6b3c61338dff100d1d1f21868",
        "named_train512": "9b55084d281a9420e12a1b6a35c3eb199abd74f4b766a14a833e6241c59564b8",
        "named_eval64": "6cc73c98616430a12a357de99702e8cb4db95258225adab80fd11212aa203d20",
        "boolean_train512": "52b5d74bec65bc903938b8b1993c1bf6df489d9c13659ad62aca76336c2d77f1",
        "boolean_eval64": "3c2ce21e86dd88ce85c6d2927c8ff09eb8f87cecb223bdeb1009debff68dff3b",
        "array_train512": "6bbcae6203dfcf443f9871f30b48c29580fa721317387ff26b757b4066a98e94",
        "array_eval64": "8d504dd63ad2538aedc8195a4f3c0729ed38ec95a078b9c9895fbf93cf8c173a",
    }
    expected_configuration = {
        "model_protocol_revision": "phase8_tied_io_v1",
        "embedding_weight_tying": True,
        "device": "cuda:0",
        "families": ["hex_copy", "named_value_json", "boolean_json", "array_json"],
        "model_sizes": ["small", "medium"],
        "seeds": [0, 1, 2],
        "train_records_per_family": 512,
        "eval_records_per_family": 64,
        "training_steps": 1500,
        "batch_size": 64,
        "pass_threshold": 52,
        "context_window_tokens": 256,
        "generation_window_tokens": 64,
        "record_hashes": expected_hashes,
        "model_parameter_counts": {"small": 133120, "medium": 859392},
    }

    assert sf.FEASIBILITY_RECORD_HASHES == expected_hashes
    assert sf.validate_feasibility_record_hashes() == expected_hashes
    assert sf.frozen_configuration() == expected_configuration

    records = sf.feasibility_record_sets()
    tampered = dict(records)
    tampered["hex_train512"] = (replace(records["hex_train512"][0], prompt="tampered prompt"), *records["hex_train512"][1:])
    monkeypatch.setattr(sf, "feasibility_record_sets", lambda: tampered)
    with pytest.raises(ValueError, match="record hash mismatch"):
        sf.validate_feasibility_record_hashes()


def test_d2_current_manifest_summary_terminal_and_cell_schemas_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    artifact_parent = tmp_path / "artifacts" / "phase8_toy_lm_bridge"
    artifact_parent.mkdir(parents=True)
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", artifact_parent)
    monkeypatch.setattr(sf, "source_provenance_allowed_paths", lambda manifest_data, context=None: set())
    monkeypatch.setattr(sf, "validate_generation_artifact", lambda path, cell: [])
    monkeypatch.setattr(sf, "validate_checkpoint_artifact", lambda path, cell: None)
    monkeypatch.setattr(sf, "validate_checkpoint_replays_generations", lambda path, cell, rows: None)

    root = artifact_parent / "feasibility_001"
    manifest, terminal = _write_feasibility_root(root, "DONE", _passing_feasibility_cells(), lightweight=True)
    manifest_data = json.loads(manifest.read_text())
    manifest_data["environment"] = dict(sf.FEASIBILITY_REQUIRED_RUNTIME_ENV)
    sf.write_json(manifest, manifest_data)
    _rewrite_terminal_manifest_sha(root, "DONE")

    sf.validate_feasibility_root_artifacts(root, require_passing=True)
    summary_data = json.loads((root / "summary.json").read_text())
    terminal_data = json.loads(terminal.read_text())
    assert set(manifest_data) == set(sf.CURRENT_MANIFEST_KEYS)
    assert set(summary_data) == set(sf.CURRENT_SUMMARY_KEYS)
    assert set(terminal_data) == set(sf.CURRENT_TERMINAL_KEYS)
    assert all(set(cell) == set(sf.CURRENT_CELL_KEYS) for cell in manifest_data["cells"])
    assert "protocol_revision" not in manifest_data["configuration"]
    assert manifest_data["configuration"]["model_protocol_revision"] == TIED_MODEL_PROTOCOL_REVISION

    bad_cell = dict(_passing_feasibility_cells()[0])
    bad_cell.pop("embedding_weight_tying")
    with pytest.raises(ValueError, match="exact current tied schema"):
        sf.validate_cell_counts([bad_cell])

    extra_cell = {**_passing_feasibility_cells()[0], "extra": True}
    with pytest.raises(ValueError, match="exact current tied schema"):
        sf.validate_cell_counts([extra_cell])

    historical_cell = _historical_failed_cell()
    sf.validate_cell_artifact_schema([historical_cell], require_pass=False, historical=True)
    with pytest.raises(ValueError, match="exact protocol schema"):
        sf.validate_cell_artifact_schema([{**historical_cell, "embedding_weight_tying": False}], require_pass=False, historical=True)

    manifest_data = json.loads(manifest.read_text())
    legacy_alias_configuration = dict(sf.frozen_configuration())
    legacy_alias_configuration["protocol_revision"] = legacy_alias_configuration.pop("model_protocol_revision")
    manifest_data["configuration"] = legacy_alias_configuration
    sf.write_json(manifest, manifest_data)
    _rewrite_terminal_manifest_sha(root, "DONE")
    with pytest.raises(ValueError, match="configuration"):
        sf.validate_feasibility_root_artifacts(root, require_passing=True)

    manifest_data["configuration"] = sf.frozen_configuration()
    sf.write_json(manifest, manifest_data)
    _rewrite_terminal_manifest_sha(root, "DONE")

    manifest_data = json.loads(manifest.read_text())
    manifest_data["unexpected"] = True
    sf.write_json(manifest, manifest_data)
    _rewrite_terminal_manifest_sha(root, "DONE")
    with pytest.raises(ValueError, match="Current manifest"):
        sf.validate_feasibility_root_artifacts(root, require_passing=True)


def test_d2_legacy_and_d1_allowlists_accept_only_exact_frozen_artifact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    original_checkpoint_validator = sf.validate_checkpoint_artifact
    monkeypatch.setattr(sf, "validate_historical_generation_artifact", lambda path, cell: [])
    monkeypatch.setattr(sf, "validate_checkpoint_artifact", lambda path, cell: None)
    monkeypatch.setattr(sf, "validate_checkpoint_replays_generations", lambda path, cell, rows: None)
    for root_name, expected in sf.HISTORICAL_FEASIBILITY_ROOTS.items():
        root = sf.ARTIFACT_PARENT / root_name
        terminal, _terminal_data, manifest, manifest_sha = sf.load_terminal_binding(root)
        assert manifest_sha == expected["manifest_sha256"]
        assert sf.file_sha256(terminal) == expected["terminal_sha256"]
        assert sf.validate_historical_feasibility_identity(
            root,
            source_commit=expected["source_commit"],
            manifest_sha=manifest_sha,
            terminal_path=terminal,
        ) is True
        sf.validate_feasibility_root_artifacts(root, require_passing=False)

    root_004 = sf.ARTIFACT_PARENT / "feasibility_004"
    root_004_manifest = json.loads((root_004 / "manifest.json").read_text())
    original_checkpoint_validator(root_004 / root_004_manifest["cells"][0]["checkpoint_path"], root_004_manifest["cells"][0])

    expected_decision = {
        "path": "artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001",
        "source_commit": "cb49ebdf577df78e97b7748aadc48f8547a70f6a",
        "artifact_class": "non_evidence_feasibility_diagnostic",
        "feasibility_selection_eligible": False,
        "task_010d_authorized": False,
        "manifest_sha256": "5589ac3a1215ea4cf451bd14f75ab64ab53b183548d1d42134e3f52b6ad12241",
        "summary_sha256": "91582eda4629f29436289896c9f3802fe243c2b86a8f622c98c6c97e579a34ac",
        "terminal_sha256": "39f3893627d24162d94f541217f4f754c2e193fe4a61361dacc6156b84826f92",
        "accepted_proposal_commit": "9a767c6708c7c69f5ba98848250afcf50c8c5a6f",
        "independent_review_verdict": "ACCEPT",
    }
    decision_root = sf.ARTIFACT_PARENT / "feasibility_diagnostic_001"
    deep_calls: list[Path] = []

    def deep_loader(root: Path) -> tuple[Path, dict[str, object], Path, str]:
        deep_calls.append(root)
        return root / "DONE.json", {}, root / "manifest.json", expected_decision["manifest_sha256"]

    monkeypatch.setattr(sf, "load_diagnostic_terminal_binding", deep_loader)
    assert sf.validate_decision_diagnostic_binding(decision_root, deep=True) == expected_decision
    assert deep_calls == [decision_root]
    assert sf.validate_decision_diagnostic_binding(decision_root.resolve(), deep=False) == expected_decision

    d1_checkpoint = decision_root / "array_json__small__3000__seed0" / "checkpoint_step3000.pt"
    sf.validate_historical_checkpoint_allowlist(d1_checkpoint)
    copied_checkpoint = tmp_path / "checkpoint_step3000.pt"
    copied_checkpoint.write_bytes(d1_checkpoint.read_bytes())
    with pytest.raises(ValueError, match="allowlist"):
        sf.validate_historical_checkpoint_allowlist(copied_checkpoint)

    original_file_sha256 = sf.file_sha256

    def tampered_sha(path: Path) -> str:
        if path == d1_checkpoint:
            return "0" * 64
        return original_file_sha256(path)

    monkeypatch.setattr(sf, "file_sha256", tampered_sha)
    with pytest.raises(ValueError, match="checksum|inventory"):
        sf.validate_historical_checkpoint_allowlist(d1_checkpoint)


def test_d2_shallow_inventory_schema_modes_bind_exact_keys_paths_and_hashes(tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")

    def make_root(name: str) -> Path:
        root = tmp_path / name
        root.mkdir()
        (root / "summary.json").write_text("{}\n")
        (root / "rows.jsonl").write_text("{}\n")
        (root / "checkpoint.pt").write_bytes(b"checkpoint\n")
        return root

    for schema_mode, inventory_fn in (
        ("historical", sf.inventory),
        ("diagnostic", sf.diagnostic_inventory),
    ):
        root = make_root(schema_mode)
        rows = inventory_fn(root)
        expected_allowed = {(root / str(row["path"])).resolve() for row in rows}
        assert sf.shallow_inventory_bound_paths_from_rows(root, rows, schema_mode=schema_mode) == expected_allowed

        missing_path_rows = [dict(row) for row in rows[:-1]]
        with pytest.raises(ValueError, match="exactly match root files"):
            sf.shallow_inventory_bound_paths_from_rows(root, missing_path_rows, schema_mode=schema_mode)

        bad_size_rows = [dict(row) for row in rows]
        bad_size_rows[0]["bytes"] = int(bad_size_rows[0]["bytes"]) + 1
        with pytest.raises(ValueError, match="byte count mismatch"):
            sf.shallow_inventory_bound_paths_from_rows(root, bad_size_rows, schema_mode=schema_mode)

        bad_hash_rows = [dict(row) for row in rows]
        bad_hash_rows[0]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="checksum mismatch"):
            sf.shallow_inventory_bound_paths_from_rows(root, bad_hash_rows, schema_mode=schema_mode)

        stale_rows = [dict(row) for row in rows]
        (root / "extra.txt").write_text("extra\n")
        with pytest.raises(ValueError, match="exactly match root files"):
            sf.shallow_inventory_bound_paths_from_rows(root, stale_rows, schema_mode=schema_mode)


def test_d2_shallow_inventory_schema_modes_reject_role_schema_mismatches(tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    root = tmp_path / "role_schema"
    root.mkdir()
    (root / "summary.json").write_text("{}\n")
    (root / "rows.jsonl").write_text("{}\n")
    (root / "checkpoint.pt").write_bytes(b"checkpoint\n")
    historical_rows = sf.inventory(root)
    diagnostic_rows = sf.diagnostic_inventory(root)

    with pytest.raises(TypeError):
        sf.shallow_inventory_bound_paths_from_rows(root, historical_rows)

    historical_with_role = [{**row, "role": "diagnostic_artifact"} for row in historical_rows]
    with pytest.raises(ValueError, match="exact schema"):
        sf.shallow_inventory_bound_paths_from_rows(root, historical_with_role, schema_mode="historical")

    with pytest.raises(ValueError, match="exact schema"):
        sf.shallow_inventory_bound_paths_from_rows(root, historical_rows, schema_mode="diagnostic")

    diagnostic_missing_role = [dict(row) for row in diagnostic_rows]
    diagnostic_missing_role[0].pop("role")
    with pytest.raises(ValueError, match="exact schema"):
        sf.shallow_inventory_bound_paths_from_rows(root, diagnostic_missing_role, schema_mode="diagnostic")

    diagnostic_extra_key = [dict(row) for row in diagnostic_rows]
    diagnostic_extra_key[0]["unexpected"] = True
    with pytest.raises(ValueError, match="exact schema"):
        sf.shallow_inventory_bound_paths_from_rows(root, diagnostic_extra_key, schema_mode="diagnostic")

    diagnostic_wrong_role = [dict(row) for row in diagnostic_rows]
    diagnostic_wrong_role[0]["role"] = "wrong_role"
    with pytest.raises(ValueError, match="role mismatch"):
        sf.shallow_inventory_bound_paths_from_rows(root, diagnostic_wrong_role, schema_mode="diagnostic")

    with pytest.raises(ValueError, match="exact schema"):
        sf.shallow_inventory_bound_paths_from_rows(root, diagnostic_rows, schema_mode="historical")

    with pytest.raises(ValueError, match="schema_mode"):
        sf.shallow_inventory_bound_paths_from_rows(root, historical_rows, schema_mode="legacy")


def test_d2_shallow_decision_diagnostic_paths_bind_real_frozen_d1_inventory() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    root = REPO_ROOT / sf.FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT
    manifest = root / "manifest.json"
    summary = root / "summary.json"
    terminal = root / "DONE.json"
    manifest_data = json.loads(manifest.read_text())
    file_inventory = manifest_data["file_inventory"]

    assert file_inventory == sf.diagnostic_inventory(root)
    assert all(set(row) == {"path", "sha256", "bytes", "role"} for row in file_inventory)

    allowed = sf.shallow_decision_diagnostic_paths(root)
    inventory_paths = {(root / str(row["path"])).resolve() for row in file_inventory}
    assert allowed == {manifest.resolve(), summary.resolve(), terminal.resolve(), *inventory_paths}
    assert (root / "array_json__small__3000__seed0" / "checkpoint_step3000.pt").resolve() in allowed


def test_d2_decision_diagnostic_cannot_be_selected_or_used_as_current_cell() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    diagnostic_root = sf.ARTIFACT_PARENT / "feasibility_diagnostic_001"
    with pytest.raises(ValueError, match="Feasibility root basename"):
        sf.validate_feasibility_root_artifacts(diagnostic_root, require_passing=True)

    checkpoint = diagnostic_root / "array_json__small__3000__seed0" / "checkpoint_step3000.pt"
    with pytest.raises(ValueError, match="parameter_count|Current checkpoint"):
        sf.validate_checkpoint_artifact(checkpoint, _passing_feasibility_cells()[0])


def test_d2_run_cli_requires_exact_real_argv_environment_and_no_programmatic_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    root = Path("artifacts/phase8_toy_lm_bridge/feasibility_005")
    predecessors = tuple(Path(path) for path in sf.FEASIBILITY_REQUIRED_PREDECESSOR_ROOTS)
    diagnostic_root = Path(sf.FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT)
    for key, value in sf.FEASIBILITY_REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sf, "current_environment_dict", lambda: dict(sf.FEASIBILITY_REQUIRED_RUNTIME_ENV))
    monkeypatch.setattr(sf, "diagnostic_kernel_argv", lambda: sf.feasibility_exact_argv(root, predecessors, diagnostic_root))

    sf.validate_feasibility_cli_contract(
        device="cuda:0",
        root=root,
        predecessor_roots=predecessors,
        predecessor_selections=(),
        decision_diagnostic_root=diagnostic_root,
    )

    for bad_device in ("cpu", "cuda"):
        with pytest.raises(ValueError, match="cuda:0"):
            sf.validate_feasibility_cli_contract(
                device=bad_device,
                root=root,
                predecessor_roots=predecessors,
                predecessor_selections=(),
                decision_diagnostic_root=diagnostic_root,
            )
    with pytest.raises(ValueError, match="predecessor selections"):
        sf.validate_feasibility_cli_contract(
            device="cuda:0",
            root=root,
            predecessor_roots=predecessors,
            predecessor_selections=(Path("artifacts/phase8_toy_lm_bridge/feasibility_selection_001.json"),),
            decision_diagnostic_root=diagnostic_root,
        )
    with pytest.raises(ValueError, match="root"):
        sf.validate_feasibility_cli_contract(
            device="cuda:0",
            root=Path("artifacts/phase8_toy_lm_bridge/feasibility_006"),
            predecessor_roots=predecessors,
            predecessor_selections=(),
            decision_diagnostic_root=diagnostic_root,
        )
    with pytest.raises(ValueError, match="predecessor roots"):
        sf.validate_feasibility_cli_contract(
            device="cuda:0",
            root=root,
            predecessor_roots=tuple(reversed(predecessors)),
            predecessor_selections=(),
            decision_diagnostic_root=diagnostic_root,
        )
    with pytest.raises(ValueError, match="decision diagnostic"):
        sf.validate_feasibility_cli_contract(
            device="cuda:0",
            root=root,
            predecessor_roots=predecessors,
            predecessor_selections=(),
            decision_diagnostic_root=Path("artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_002"),
        )
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    with pytest.raises(ValueError, match="PYTHONPATH"):
        sf.validate_feasibility_cli_contract(
            device="cuda:0",
            root=root,
            predecessor_roots=predecessors,
            predecessor_selections=(),
            decision_diagnostic_root=diagnostic_root,
        )
    monkeypatch.setenv("PYTHONPATH", ".")
    monkeypatch.setattr(sf, "current_environment_dict", lambda: {**sf.FEASIBILITY_REQUIRED_RUNTIME_ENV, "gpu": "different"})
    with pytest.raises(ValueError, match="environment dictionary"):
        sf.validate_feasibility_cli_contract(
            device="cuda:0",
            root=root,
            predecessor_roots=predecessors,
            predecessor_selections=(),
            decision_diagnostic_root=diagnostic_root,
        )

    monkeypatch.setattr(sf, "current_environment_dict", lambda: dict(sf.FEASIBILITY_REQUIRED_RUNTIME_ENV))
    monkeypatch.setattr(sf, "diagnostic_kernel_argv", lambda: ["python", "-O", *sf.feasibility_exact_argv(root, predecessors, diagnostic_root)[1:]])
    with pytest.raises(ValueError, match="process argv"):
        sf.validate_feasibility_cli_contract(
            device="cuda:0",
            root=root,
            predecessor_roots=predecessors,
            predecessor_selections=(),
            decision_diagnostic_root=diagnostic_root,
        )
    with pytest.raises(ValueError, match="main\\(argv"):
        sf.main([
            "run",
            "--device",
            "cuda:0",
            "--root",
            str(root),
            *[item for predecessor in predecessors for item in ("--predecessor-root", str(predecessor))],
            "--decision-diagnostic-root",
            str(diagnostic_root),
        ])


def test_d2_run_current_suite_direct_import_rejects_before_tmp_or_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    root = tmp_path / "feasibility_005"
    predecessors = tuple(tmp_path / f"feasibility_{index:03d}" for index in range(1, 5))
    diagnostic_root = tmp_path / "feasibility_diagnostic_001"

    for name in (
        "validate_current_run_root_contract",
        "shallow_current_run_allowed_paths",
        "capture_source_provenance",
        "validate_feasibility_cli_contract",
        "build_model",
    ):
        monkeypatch.setattr(sf, name, lambda *args, _name=name, **kwargs: (_ for _ in ()).throw(AssertionError(f"{_name} ran before real main gate")))

    with pytest.raises(ValueError, match="real __main__"):
        sf.run_current_suite(
            root,
            predecessors,
            (),
            device="cuda:0",
            decision_diagnostic_root=diagnostic_root,
        )
    assert not root.exists()
    assert not root.with_name(root.name + ".tmp").exists()


def test_d2_record_hash_mismatch_stops_before_model_construction_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    events: list[str] = []
    root = tmp_path / "feasibility_005"
    predecessors = tuple(tmp_path / f"feasibility_{index:03d}" for index in range(1, 5))
    diagnostic_root = tmp_path / "feasibility_diagnostic_001"

    monkeypatch.setattr(sf, "require_feasibility_real_main_context", lambda: events.append("main_context"))
    monkeypatch.setattr(sf, "validate_feasibility_cli_contract", lambda **kwargs: events.append("cli"))
    monkeypatch.setattr(sf, "validate_current_run_root_contract", lambda *args: events.append("root_contract"))
    monkeypatch.setattr(sf, "shallow_current_run_allowed_paths", lambda *args: events.append("shallow") or set())
    monkeypatch.setattr(
        sf,
        "capture_source_provenance",
        lambda *args, **kwargs: events.append("clean") or sf.SourceSnapshot(commit="a" * 40, status_lines=(), ignored_inputs=()),
    )
    monkeypatch.setattr(sf, "configure_feasibility_deterministic_backend", lambda: events.append("backend"))

    def hash_mismatch() -> dict[str, str]:
        events.append("record_hashes")
        raise ValueError("record hash mismatch")

    monkeypatch.setattr(sf, "validate_feasibility_record_hashes", hash_mismatch)
    monkeypatch.setattr(sf, "build_model", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model constructed")))
    with pytest.raises(ValueError, match="record hash mismatch"):
        sf.run_current_suite(
            root,
            predecessors,
            (),
            device="cuda:0",
            decision_diagnostic_root=diagnostic_root,
        )
    assert events == ["main_context", "cli", "root_contract", "shallow", "clean", "backend", "record_hashes"]
    assert not root.exists()
    assert not root.with_name(root.name + ".tmp").exists()


def test_d2_shallow_inventory_then_source_cleanliness_precedes_deep_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    artifact_parent = Path(".pytest_d2_preflight") / tmp_path.name / "phase8_toy_lm_bridge"
    root = artifact_parent / "feasibility_005"
    predecessors = tuple(artifact_parent / f"feasibility_{index:03d}" for index in range(1, 5))
    diagnostic_root = artifact_parent / "feasibility_diagnostic_001"
    events: list[str] = []

    monkeypatch.setattr(sf, "ARTIFACT_PARENT", artifact_parent)
    monkeypatch.setattr(sf, "FEASIBILITY_REQUIRED_ROOT", str(root))
    monkeypatch.setattr(sf, "FEASIBILITY_REQUIRED_PREDECESSOR_ROOTS", tuple(str(path) for path in predecessors))
    monkeypatch.setattr(sf, "FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT", str(diagnostic_root))
    monkeypatch.setattr(sf, "require_feasibility_real_main_context", lambda: events.append("main_context"))
    monkeypatch.setattr(sf, "validate_feasibility_cli_contract", lambda **kwargs: events.append("cli"))
    monkeypatch.setattr(sf, "shallow_current_run_allowed_paths", lambda *args: events.append("shallow") or set())

    def dirty_source(*args: object, **kwargs: object) -> object:
        events.append("clean")
        raise RuntimeError("dirty source")

    monkeypatch.setattr(sf, "capture_source_provenance", dirty_source)
    for name in (
        "configure_feasibility_deterministic_backend",
        "validate_feasibility_record_hashes",
        "validate_feasibility_root_artifacts",
        "validate_decision_diagnostic_binding",
        "build_model",
        "deterministic_batch_indices",
        "load_model_from_checkpoint",
    ):
        monkeypatch.setattr(sf, name, lambda *args, _name=name, **kwargs: (_ for _ in ()).throw(AssertionError(f"{_name} ran before source cleanliness failed")))

    with pytest.raises(RuntimeError, match="dirty source"):
        sf.run_current_suite(
            root,
            predecessors,
            (),
            device="cuda:0",
            decision_diagnostic_root=diagnostic_root,
        )
    assert events == ["main_context", "cli", "shallow", "clean"]
    assert not (REPO_ROOT / root).exists()
    assert not (REPO_ROOT / root.with_name(root.name + ".tmp")).exists()

    events.clear()
    monkeypatch.setattr(
        sf,
        "capture_source_provenance",
        lambda *args, **kwargs: events.append("clean") or sf.SourceSnapshot(commit="a" * 40, status_lines=(), ignored_inputs=()),
    )
    monkeypatch.setattr(sf, "configure_feasibility_deterministic_backend", lambda: events.append("backend"))
    monkeypatch.setattr(sf, "validate_feasibility_record_hashes", lambda: events.append("record_hashes") or dict(sf.FEASIBILITY_RECORD_HASHES))
    monkeypatch.setattr(sf, "validate_feasibility_root_artifacts", lambda predecessor_root, **kwargs: events.append(f"deep:{predecessor_root.name}"))

    def d1_probe(root: Path, *, deep: bool) -> dict[str, object]:
        events.append(f"d1_deep:{deep}")
        raise RuntimeError("deep validation probe")

    monkeypatch.setattr(sf, "validate_decision_diagnostic_binding", d1_probe)
    with pytest.raises(RuntimeError, match="deep validation probe"):
        sf.run_current_suite(
            root,
            predecessors,
            (),
            device="cuda:0",
            decision_diagnostic_root=diagnostic_root,
        )
    assert events == [
        "main_context",
        "cli",
        "shallow",
        "clean",
        "backend",
        "record_hashes",
        "deep:feasibility_001",
        "deep:feasibility_002",
        "deep:feasibility_003",
        "deep:feasibility_004",
        "d1_deep:True",
    ]
    assert not (REPO_ROOT / root).exists()
    assert not (REPO_ROOT / root.with_name(root.name + ".tmp")).exists()


def test_d2_run_current_suite_uses_cpu_initialized_tied_models_and_all_24_fake_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")

    class FakeModel(torch.nn.Module):
        def __init__(self, model_size: str) -> None:
            super().__init__()
            self.model_size = model_size
            self.token_embedding = torch.nn.Embedding(1, 1)
            self.lm_head = torch.nn.Linear(1, 1, bias=False)
            self.lm_head.weight = self.token_embedding.weight
            self.parameter_count = sf.FROZEN_PARAMETER_COUNTS[model_size]
            self.transfers: list[str] = []

        def to(self, device: object) -> "FakeModel":
            self.transfers.append(str(device))
            return self

    def fake_records() -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            family: {
                "train": (sf.FeasibilityRecord(family, "train", 0, f"{family}_train", f"{family}_train", "prompt", "answer"),),
                "eval": (sf.FeasibilityRecord(family, "eval", 0, f"{family}_eval", f"{family}_eval", "prompt", "answer"),),
            }
            for family in sf.FAMILIES
        }

    def configure_common(match_overrides: dict[tuple[str, str, int], int]) -> tuple[Path, tuple[Path, ...], Path, list[tuple[str, str, int]], list[FakeModel]]:
        root = tmp_path / f"feasibility_005_{len(match_overrides)}"
        predecessors = tuple(tmp_path / f"feasibility_{index:03d}" for index in range(1, 5))
        diagnostic_root = tmp_path / "feasibility_diagnostic_001"
        executed: list[tuple[str, str, int]] = []
        models: list[FakeModel] = []
        original_build_manifest = sf.build_manifest
        for key, value in sf.FEASIBILITY_REQUIRED_ENV.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(sf, "require_feasibility_real_main_context", lambda: None)
        monkeypatch.setattr(sf, "validate_feasibility_cli_contract", lambda **kwargs: None)
        monkeypatch.setattr(sf, "validate_current_run_root_contract", lambda *args: None)
        monkeypatch.setattr(sf, "shallow_current_run_allowed_paths", lambda *args: set())
        monkeypatch.setattr(
            sf,
            "capture_source_provenance",
            lambda *args, **kwargs: sf.SourceSnapshot(commit="a" * 40, status_lines=(), ignored_inputs=()),
        )
        monkeypatch.setattr(sf, "configure_feasibility_deterministic_backend", lambda: None)
        monkeypatch.setattr(sf, "validate_feasibility_record_hashes", lambda: dict(sf.FEASIBILITY_RECORD_HASHES))
        monkeypatch.setattr(sf, "validate_feasibility_root_artifacts", lambda *args, **kwargs: None)
        monkeypatch.setattr(sf, "validate_decision_diagnostic_binding", lambda root, *, deep: dict(sf.DECISION_DIAGNOSTIC_ROOT_BINDING))
        monkeypatch.setattr(sf, "grouped_records", fake_records)
        monkeypatch.setattr(sf, "verify_source_unchanged", lambda *args, **kwargs: None)
        monkeypatch.setattr(sf, "validate_generation_artifact", lambda path, cell: [])
        monkeypatch.setattr(sf, "validate_checkpoint_artifact", lambda path, cell: None)
        monkeypatch.setattr(sf, "validate_checkpoint_replays_generations", lambda path, cell, rows: None)
        monkeypatch.setattr(sf, "expected_parameter_count", lambda model_size: sf.FROZEN_PARAMETER_COUNTS[model_size])

        def build_manifest_with_required_environment(*args: object, **kwargs: object) -> dict[str, object]:
            manifest = original_build_manifest(*args, **kwargs)
            manifest["environment"] = dict(sf.FEASIBILITY_REQUIRED_RUNTIME_ENV)
            return manifest

        monkeypatch.setattr(sf, "build_manifest", build_manifest_with_required_environment)
        monkeypatch.setattr(
            sf,
            "complete_predecessor_root_bindings",
            lambda roots, selections, context=None: [
                {
                    "path": str(path),
                    "terminal_state": "FAILED",
                    "terminal_sha256": f"{index}" * 64,
                    "manifest_sha256": f"{index + 4}" * 64,
                }
                for index, path in enumerate(roots, start=1)
            ],
        )

        def build_fake_model(model_size: str) -> FakeModel:
            model = FakeModel(model_size)
            models.append(model)
            assert all(parameter.device.type == "cpu" for parameter in model.parameters())
            return model

        def fake_train(model: FakeModel, records: object, *, seed: int, **kwargs: object) -> object:
            assert model.transfers == ["cuda:0"]
            return type("FakeTrainResult", (), {"final_loss": 0.0, "training_accuracy": 1.0})()

        def fake_evaluate(model: FakeModel, records: tuple[object, ...], tokenizer: object, device: torch.device) -> tuple[int, list[dict[str, object]]]:
            assert model.transfers == ["cuda:0"]
            assert str(device) == "cuda:0"
            family = records[0].family
            key = (family, model.model_size, len([item for item in executed if item[0] == family and item[1] == model.model_size]))
            seed = key[2]
            executed.append((family, model.model_size, seed))
            exact_matches = match_overrides.get((family, model.model_size, seed), sf.PASS_THRESHOLD)
            return exact_matches, [{"family": family, "index": 0, "exact_match": exact_matches >= sf.PASS_THRESHOLD}]

        monkeypatch.setattr(sf, "build_model", build_fake_model)
        monkeypatch.setattr(sf, "train_text_records", fake_train)
        monkeypatch.setattr(sf, "evaluate_model", fake_evaluate)
        monkeypatch.setattr(sf, "save_checkpoint", lambda path, model, metadata: Path(path).write_bytes(b"checkpoint"))
        return root, predecessors, diagnostic_root, executed, models

    root, predecessors, diagnostic_root, executed, models = configure_common({})
    sf.run_current_suite(
        root,
        predecessors,
        (),
        device="cuda:0",
        decision_diagnostic_root=diagnostic_root,
    )
    assert len(executed) == len(sf.FAMILIES) * len(sf.MODEL_SIZES) * len(sf.SEEDS) == 24
    assert all(model.transfers == ["cuda:0"] for model in models)
    assert (root / "DONE.json").exists()
    assert json.loads((root / "summary.json").read_text())["summary"]["all_cells_passed"] is True

    fail_key = (sf.FAMILIES[-1], sf.MODEL_SIZES[-1], sf.SEEDS[-1])
    root, predecessors, diagnostic_root, executed, _models = configure_common({fail_key: sf.PASS_THRESHOLD - 1})
    sf.run_current_suite(
        root,
        predecessors,
        (),
        device="cuda:0",
        decision_diagnostic_root=diagnostic_root,
    )
    assert len(executed) == 24
    assert (root / "FAILED.json").exists()
    failed = json.loads((root / "FAILED.json").read_text())
    assert len(failed["cells"]) == 24
    assert any(cell["passed"] is False for cell in failed["cells"])
    assert failed["error"] == "complete feasibility matrix did not satisfy all 24 pass thresholds."
    assert json.loads((root / "summary.json").read_text())["summary"]["all_cells_passed"] is False


@pytest.mark.parametrize(
    ("mutation", "cause_match"),
    (
        ("generation", "Generation artifact raw_token_ids prefix|Generation artifact row prompt"),
        ("checkpoint", "Checkpoint artifact is not a loadable"),
        ("cell_schema", "exact current tied schema"),
        ("lineage", "predecessor_roots"),
        ("replay", "Checkpoint replay mismatch probe"),
    ),
)
def test_d2_current_publication_gate_rejects_mutated_artifacts_schema_lineage_or_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    cause_match: str,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    status = "FAILED"
    root_dir = tmp_path / mutation
    temp_root = root_dir / "feasibility_005.tmp"
    output_root = root_dir / "feasibility_005"
    cell = dict(_passing_feasibility_cells()[0])
    use_real_generation = mutation == "generation"
    _write_feasibility_root(temp_root, status, [cell], lightweight=not use_real_generation)
    _rewrite_current_publication_command(temp_root, status, output_root)

    if mutation == "generation":
        generation_path = temp_root / str(cell["generations_path"])
        rows = [json.loads(line) for line in generation_path.read_text().splitlines()]
        rows[0]["prompt"] = f"{rows[0]['prompt']} tampered"
        generation_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        _refresh_current_publication_hashes(temp_root, status)
        monkeypatch.setattr(sf, "validate_checkpoint_artifact", lambda path, cell: None)
        monkeypatch.setattr(sf, "validate_checkpoint_replays_generations", lambda path, cell, rows: None)
    elif mutation == "checkpoint":
        checkpoint_path = temp_root / str(cell["checkpoint_path"])
        checkpoint_path.write_bytes(b"tampered checkpoint")
        _refresh_current_publication_hashes(temp_root, status)
        monkeypatch.setattr(sf, "validate_generation_artifact", lambda path, cell: [])
        monkeypatch.setattr(sf, "validate_checkpoint_replays_generations", lambda path, cell, rows: None)
    elif mutation == "cell_schema":
        for filename in ("summary.json", "manifest.json", f"{status}.json"):
            path = temp_root / filename
            data = json.loads(path.read_text())
            for row in data["cells"]:
                row.pop("model_protocol_revision")
            if filename == "manifest.json":
                data["file_inventory"] = sf.inventory(temp_root)
            sf.write_json(path, data)
        _refresh_current_publication_hashes(temp_root, status)
        monkeypatch.setattr(sf, "validate_generation_artifact", lambda path, cell: [])
        monkeypatch.setattr(sf, "validate_checkpoint_artifact", lambda path, cell: None)
        monkeypatch.setattr(sf, "validate_checkpoint_replays_generations", lambda path, cell, rows: None)
    elif mutation == "lineage":
        manifest_path = temp_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["predecessor_roots"] = [
            {
                "path": "artifacts/phase8_toy_lm_bridge/feasibility_001",
                "terminal_state": "FAILED",
                "terminal_sha256": "0" * 64,
                "manifest_sha256": "1" * 64,
            }
        ]
        sf.write_json(manifest_path, manifest)
        _rewrite_terminal_manifest_sha(temp_root, status)
        monkeypatch.setattr(sf, "validate_generation_artifact", lambda path, cell: [])
        monkeypatch.setattr(sf, "validate_checkpoint_artifact", lambda path, cell: None)
        monkeypatch.setattr(sf, "validate_checkpoint_replays_generations", lambda path, cell, rows: None)
    elif mutation == "replay":
        monkeypatch.setattr(sf, "validate_generation_artifact", lambda path, cell: [])
        monkeypatch.setattr(sf, "validate_checkpoint_artifact", lambda path, cell: None)

        def replay_mismatch(path: Path, cell: dict[str, object], rows: list[dict[str, object]]) -> None:
            raise ValueError("Checkpoint replay mismatch probe")

        monkeypatch.setattr(sf, "validate_checkpoint_replays_generations", replay_mismatch)
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(mutation)

    with pytest.raises(sf.FeasibilityPublicationError) as excinfo:
        sf.publish_current_feasibility_root_or_leave_incomplete(
            temp_root,
            output_root,
            terminal_status=status,
            predecessor_roots=(),
            predecessor_selections=(),
            decision_diagnostic_root=Path(sf.FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT),
            source_snapshot=_source_snapshot(),
            allowed_source_paths=set(),
        )
    assert excinfo.value.__cause__ is not None
    assert re.search(cause_match, str(excinfo.value.__cause__))
    assert not output_root.exists()
    assert temp_root.exists()
    assert not (temp_root / "DONE.json").exists()
    assert not (temp_root / "FAILED.json").exists()
    assert (temp_root / "manifest.json").exists()


def test_d2_current_publication_success_validates_before_rename_and_replays_on_cuda0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    status = "FAILED"
    temp_root = tmp_path / "feasibility_005.tmp"
    output_root = tmp_path / "feasibility_005"
    _write_feasibility_root(temp_root, status, [dict(_passing_feasibility_cells()[0])], lightweight=True)
    _rewrite_current_publication_command(temp_root, status, output_root)
    events: list[str] = []
    replay_devices: list[str] = []

    monkeypatch.setattr(sf, "validate_generation_artifact", lambda path, cell: events.append("generation") or [])
    monkeypatch.setattr(sf, "validate_checkpoint_artifact", lambda path, cell: events.append("checkpoint"))
    monkeypatch.setattr(sf.torch.cuda, "is_available", lambda: True)

    def replay_probe(path: Path, cell: dict[str, object], rows: list[dict[str, object]]) -> None:
        events.append("replay")
        replay_devices.append(str(sf.feasibility_replay_device()))

    monkeypatch.setattr(sf, "validate_checkpoint_replays_generations", replay_probe)

    def source_callback() -> None:
        events.append("callback")

    def rename_probe(source: Path, destination: Path) -> None:
        events.append("rename")
        source.rename(destination)

    monkeypatch.setattr(sf, "atomic_rename_noreplace", rename_probe)
    sf.publish_current_feasibility_root_or_leave_incomplete(
        temp_root,
        output_root,
        terminal_status=status,
        predecessor_roots=(),
        predecessor_selections=(),
        decision_diagnostic_root=Path(sf.FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT),
        source_snapshot=_source_snapshot(),
        allowed_source_paths=set(),
        final_callback=source_callback,
    )
    assert output_root.exists()
    assert not temp_root.exists()
    assert events == ["generation", "checkpoint", "replay", "callback", "callback", "rename"]
    assert replay_devices == ["cuda:0"]


def test_d2_current_publication_callback_failure_blocks_rename_and_leaves_tmp_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    status = "FAILED"
    temp_root = tmp_path / "feasibility_005.tmp"
    output_root = tmp_path / "feasibility_005"
    _write_feasibility_root(temp_root, status, [dict(_passing_feasibility_cells()[0])], lightweight=True)
    _rewrite_current_publication_command(temp_root, status, output_root)

    monkeypatch.setattr(sf, "validate_generation_artifact", lambda path, cell: [])
    monkeypatch.setattr(sf, "validate_checkpoint_artifact", lambda path, cell: None)
    monkeypatch.setattr(sf, "validate_checkpoint_replays_generations", lambda path, cell, rows: None)
    monkeypatch.setattr(sf, "atomic_rename_noreplace", lambda source, destination: (_ for _ in ()).throw(AssertionError("rename after failed callback")))

    with pytest.raises(sf.FeasibilityPublicationError) as excinfo:
        sf.publish_current_feasibility_root_or_leave_incomplete(
            temp_root,
            output_root,
            terminal_status=status,
            predecessor_roots=(),
            predecessor_selections=(),
            decision_diagnostic_root=Path(sf.FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT),
            source_snapshot=_source_snapshot(),
            allowed_source_paths=set(),
            final_callback=lambda: (_ for _ in ()).throw(sf.SourceChangedError("source changed probe")),
        )
    assert isinstance(excinfo.value.__cause__, sf.SourceChangedError)
    assert not output_root.exists()
    assert temp_root.exists()
    assert not (temp_root / "DONE.json").exists()
    assert not (temp_root / "FAILED.json").exists()


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


def test_checkpoint_replay_rejects_generation_rows_not_produced_by_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    monkeypatch.setattr(sf, "feasibility_replay_device", lambda: torch.device("cpu"))
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
        sf.require_canonical_path_string(f"{artifact_parent}//feasibility_001", "root", sf.ROOT_RE)
    with pytest.raises(ValueError, match="real process command"):
        sf.main([
            "run",
            "--device",
            "cuda:0",
            "--root",
            sf.FEASIBILITY_REQUIRED_ROOT,
            "--predecessor-root",
            sf.FEASIBILITY_REQUIRED_PREDECESSOR_ROOTS[0],
            "--predecessor-root",
            sf.FEASIBILITY_REQUIRED_PREDECESSOR_ROOTS[1],
            "--predecessor-root",
            sf.FEASIBILITY_REQUIRED_PREDECESSOR_ROOTS[2],
            "--predecessor-root",
            sf.FEASIBILITY_REQUIRED_PREDECESSOR_ROOTS[3],
            "--decision-diagnostic-root",
            sf.FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT,
        ])
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

    monkeypatch.setattr(sf, "terminal_binding", lambda root, context=None: _unchecked_terminal_binding(root))
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

    with pytest.raises(ValueError, match="decision_diagnostic|configuration|allowlist"):
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

    with pytest.raises(ValueError, match="decision_diagnostic|configuration|allowlist"):
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

    with pytest.raises(ValueError, match="decision_diagnostic|configuration|allowlist"):
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
    monkeypatch.setattr(sf, "validate_feasibility_root_artifacts", lambda *args, **kwargs: None)
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
            if args == ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"]:
                return subprocess.CompletedProcess(args, 0, stdout="".join(f"{line}\0" for line in ignored_lines))
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
    with pytest.raises((RuntimeError, ValueError), match="exact supplied|file_inventory|Expecting value"):
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


def test_repo_wide_ignored_source_scan_detects_root_sourceless_import_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".gitignore").write_text("*.pyc\n*.pyo\n*.so\n*.pth\nignored_link\n__pycache__/\n", encoding="utf-8")
    sitecustomize = repo / "sitecustomize.pyc"
    shadow_module = repo / "shadow_module.pyc"
    nested_cache = repo / "pkg" / "__pycache__" / "nested.cpython-313.pyc"
    nested_cache.parent.mkdir(parents=True)
    sitecustomize.write_bytes(b"root ignored sourceless sitecustomize")
    shadow_module.write_bytes(b"root ignored sourceless shadow module")
    nested_cache.write_bytes(b"nested ignored cache")
    symlink_path = repo / "ignored_link"
    symlink_created = False
    try:
        symlink_path.symlink_to("shadow_module.pyc")
        symlink_created = True
    except OSError:
        pass

    monkeypatch.setattr(sf, "REPO_ROOT", repo)
    rows = sf.ignored_source_inputs()
    by_path = {row["path"]: row for row in rows}
    assert "sitecustomize.pyc" in by_path
    assert "shadow_module.pyc" in by_path
    assert "pkg/__pycache__/nested.cpython-313.pyc" in by_path
    assert by_path["sitecustomize.pyc"] == {
        "path": "sitecustomize.pyc",
        "sha256": sf.file_sha256(sitecustomize),
        "bytes": sitecustomize.stat().st_size,
    }
    assert by_path["shadow_module.pyc"] == {
        "path": "shadow_module.pyc",
        "sha256": sf.file_sha256(shadow_module),
        "bytes": shadow_module.stat().st_size,
    }
    if symlink_created:
        assert by_path["ignored_link"]["sha256"] == sha256(b"shadow_module.pyc").hexdigest()
        assert by_path["ignored_link"]["bytes"] == len(b"shadow_module.pyc")


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
        if args == ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"]:
            return subprocess.CompletedProcess(args, 0, stdout="scripts/phase8_sequence_feasibility.py\0")
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
        if args == ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"]:
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
    monkeypatch.setattr(sf, "shallow_decision_diagnostic_paths", lambda root: set())

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
    sf.write_json(bad_manifest, manifest_data)
    _rewrite_terminal_manifest_sha(bad_root, "DONE")
    bad_selection = parent / "feasibility_selection_001.json"
    _write_selection(bad_selection, bad_root, bad_manifest, [], [], cells=cells)
    with pytest.raises(ValueError, match="frozen feasibility schema"):
        sf.validate_selection_record(bad_selection)

    parent = set_parent("bad_source_provenance")
    bad_root = parent / "feasibility_001"
    bad_manifest, _bad_done = _write_feasibility_root(bad_root, "DONE", cells)
    manifest_data = json.loads(bad_manifest.read_text())
    manifest_data["source_provenance"]["commit"] = "b" * 40
    sf.write_json(bad_manifest, manifest_data)
    _rewrite_terminal_manifest_sha(bad_root, "DONE")
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


def test_diagnostic_named_matrix_has_exact_four_cells_and_no_relabeling() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    matrix = sf.build_named_diagnostic_matrix()

    assert tuple(matrix) == sf.NAMED_DIAGNOSTIC_CELLS
    for cell, rows in matrix.items():
        assert len(rows) == 64
        counts: dict[tuple[str, str], int] = {}
        for row in rows:
            assert row["diagnostic_cell"] == cell
            assert row["operand_source_split"] == ("train" if cell.endswith("seen_operand") else "eval")
            assert row["surface_source_split"] == ("train" if cell.startswith("seen_surface") else "eval")
            assert row["source_template_id"].startswith(f"seq_named_value_json_{row['operand_source_split']}_")
            assert row["template_id"].startswith(f"seq_named_value_json_{row['surface_source_split']}_")
            counts[(row["template_id"], row["target_key"])] = counts.get((row["template_id"], row["target_key"]), 0) + 1
        assert len(counts) == 16
        assert set(counts.values()) == {4}

    seen = matrix["seen_surface_seen_operand"][0]
    held_surface = matrix["held_surface_seen_operand"][0]
    assert seen["operand_id"] == held_surface["operand_id"]
    assert seen["operand_source_split"] == held_surface["operand_source_split"] == "train"
    assert seen["source_template_id"] == held_surface["source_template_id"]
    assert seen["prompt"] != held_surface["prompt"]
    assert len(seen["prompt"].encode("utf-8")) == len(held_surface["prompt"].encode("utf-8"))

    identity_fields = (
        ("operand_source_split", "eval"),
        ("operand_source_index", 999),
        ("operand_id", "seq_operand_named_value_json_99999"),
        ("source_template_id", "seq_named_value_json_eval_0"),
        ("target_key", "blue"),
        ("expected", '"red-dead"'),
        ("roster", [{"key": key, "value": f"{key}-dead"} for key in sf.NAMED_VALUE_KEYS]),
        ("prompt", held_surface["prompt"]),
        ("template_id", "seq_named_value_json_eval_0"),
    )
    for field_name, replacement in identity_fields:
        mutated = {cell: [dict(row) for row in rows] for cell, rows in matrix.items()}
        mutated["seen_surface_seen_operand"][0][field_name] = replacement
        with pytest.raises(ValueError):
            sf.validate_named_diagnostic_matrix(mutated)


def test_diagnostic_array_training_plan_is_exactly_three_small_3000_runs() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    plan = sf.diagnostic_array_training_plan()

    assert plan == [
        {"family": "array_json", "model_size": "small", "steps": 3000, "seed": 0},
        {"family": "array_json", "model_size": "small", "steps": 3000, "seed": 1},
        {"family": "array_json", "model_size": "small", "steps": 3000, "seed": 2},
    ]
    sf.validate_diagnostic_array_training_plan(plan)
    with pytest.raises(ValueError, match="exactly three"):
        sf.validate_diagnostic_array_training_plan([*plan, {"family": "hex_copy", "model_size": "small", "steps": 3000, "seed": 0}])
    with pytest.raises(ValueError, match="exactly three"):
        sf.validate_diagnostic_array_training_plan([{**plan[0], "steps": 1500}, *plan[1:]])


def test_diagnostic_step1500_equality_gate_and_batch_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    records = sf.diagnostic_record_sets()["array_train512"]
    assert sf.deterministic_batch_indices(
        record_count=len(records),
        seed=0,
        state_mask=0,
        batch_size=sf.BATCH_SIZE,
        steps=sf.DIAGNOSTIC_STEPS,
    )[: sf.TRAINING_STEPS] == sf.deterministic_batch_indices(
        record_count=len(records),
        seed=0,
        state_mask=0,
        batch_size=sf.BATCH_SIZE,
        steps=sf.TRAINING_STEPS,
    )

    checkpoint = Path("artifacts/phase8_toy_lm_bridge/feasibility_004/array_json__small__seed0/checkpoint_step1500.pt")
    model = sf.load_checkpoint_model(checkpoint, "small", torch.device("cpu"))
    random.seed(1001)
    torch_rng = torch.random.get_rng_state()
    python_rng = random.getstate()
    cuda_rng = tuple(state.clone() for state in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None
    optimizer = make_optimizer(model)
    optimizer_fingerprint = sf.optimizer_state_fingerprint(optimizer)
    monkeypatch.setattr(sf, "build_model", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("gate must not instantiate a model")))
    sf.compare_model_to_checkpoint_step1500(model, checkpoint, "small")
    assert random.getstate() == python_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    if cuda_rng is not None:
        assert all(torch.equal(actual, expected) for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rng, strict=True))
    assert sf.optimizer_state_fingerprint(optimizer) == optimizer_fingerprint
    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(ValueError, match="tensor bytes mismatch"):
        sf.compare_model_to_checkpoint_step1500(model, checkpoint, "small")


def test_diagnostic_step_loop_gate_occurs_after_1500_before_1501() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    events: list[str] = []
    progress: list[int] = []

    def step_callback(step_index: int, _batch: object) -> None:
        events.append(f"step{step_index}")

    def gate_callback() -> None:
        events.append("gate")

    result = sf.run_diagnostic_step_loop_with_gate(
        ((index,) for index in range(sf.TRAINING_STEPS + 1)),
        gate_step=sf.TRAINING_STEPS,
        step_callback=step_callback,
        gate_callback=gate_callback,
        progress_callback=lambda row: progress.append(row["completed_steps"]),
    )
    assert events[-4:] == ["step1499", "step1500", "gate", "step1501"]
    assert progress[-3:] == [1500, 1500, 1501]
    assert result["step_after_gate_executed"] is True
    mismatch_events: list[str] = []
    mismatch_progress: list[int] = []

    def mismatch_gate() -> None:
        mismatch_events.append("gate")
        raise ValueError("mismatch")

    with pytest.raises(ValueError, match="mismatch"):
        sf.run_diagnostic_step_loop_with_gate(
            ((index,) for index in range(sf.TRAINING_STEPS + 1)),
            gate_step=sf.TRAINING_STEPS,
            step_callback=lambda step_index, _batch: mismatch_events.append(f"step{step_index}"),
            gate_callback=mismatch_gate,
            progress_callback=lambda row: mismatch_progress.append(row["completed_steps"]),
        )
    assert mismatch_events[-3:] == ["step1499", "step1500", "gate"]
    assert mismatch_progress[-1] == sf.TRAINING_STEPS


def test_diagnostic_frozen_blobs_record_hashes_and_reused_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    assert sf.validate_diagnostic_record_hashes() == sf.DIAGNOSTIC_RECORD_HASHES
    assert sf.validate_diagnostic_core_blobs(sf.DIAGNOSTIC_INPUT_SOURCE_COMMIT) == sf.DIAGNOSTIC_CORE_BLOBS

    input_root = sf.ARTIFACT_PARENT / "feasibility_004"
    manifest = json.loads((input_root / "manifest.json").read_text())
    core_blob_commits: list[str] = []

    def validate_input_core_blobs(commit: str = "HEAD") -> dict[str, str]:
        core_blob_commits.append(commit)
        if commit != sf.DIAGNOSTIC_INPUT_SOURCE_COMMIT:
            raise AssertionError(f"frozen diagnostic input core blobs checked {commit}, not DIAGNOSTIC_INPUT_SOURCE_COMMIT")
        return sf.DIAGNOSTIC_CORE_BLOBS

    monkeypatch.setattr(sf, "validate_diagnostic_core_blobs", validate_input_core_blobs)
    sf.validate_diagnostic_input_root(input_root, environment=manifest["environment"])
    assert core_blob_commits == [sf.DIAGNOSTIC_INPUT_SOURCE_COMMIT]
    for model_size in ("small", "medium"):
        for seed in (0, 1, 2):
            binding = sf.reused_feasibility_cell_binding(input_root, "array_json", model_size, seed)
            assert binding["family"] == "array_json"
            assert binding["model_size"] == model_size
            assert binding["seed"] == seed
            assert re.fullmatch(r"[0-9a-f]{64}", binding["generations_sha256"])
            assert re.fullmatch(r"[0-9a-f]{64}", binding["checkpoint_sha256"])
    for seed in (0, 1, 2):
        binding = sf.reused_feasibility_cell_binding(input_root, "named_value_json", "medium", seed)
        assert binding["checkpoint_path"].endswith(f"named_value_json__medium__seed{seed}/checkpoint_step1500.pt")


def test_diagnostic_row_metrics_edge_cases_and_teacher_forced_reconstruction() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    tokenizer = ByteTokenizer()
    record = sf.diagnostic_record_sets()["named_eval64"][0]
    full = (*tokenizer.encode_evaluation_prefix(record.prompt), *tokenizer.encode_text(record.answer), EOS_ID)
    metrics = sf.named_row_metrics(record.prompt, record.answer, full, sf.named_value_target_key(record), [value for _key, value in sf.named_roster(record)])
    assert metrics["exact_match"] is True
    assert metrics["has_eos"] is True
    assert metrics["suffix_correct_positions"] == 4
    assert metrics["suffix_first_error"] is None
    named_aggregate = sf.aggregate_named_rows([{**metrics, "template_id": record.template_id, "target_key": sf.named_value_target_key(record)}])
    assert named_aggregate["exact_sequence_matches"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert named_aggregate["template_target_exact_matches"][f"{record.template_id}__{sf.named_value_target_key(record)}"]["numerator"] == 1
    assert named_aggregate["suffix_hamming_distance"]["observation_count"] == 1
    assert named_aggregate["has_eos"]["denominator"] == 1

    wrong = json.dumps(json.loads(record.answer)[:-1] + "f")
    wrong_full = (*tokenizer.encode_evaluation_prefix(record.prompt), *tokenizer.encode_text(wrong), EOS_ID)
    wrong_metrics = sf.named_row_metrics(record.prompt, record.answer, wrong_full, sf.named_value_target_key(record), [value for _key, value in sf.named_roster(record)])
    assert wrong_metrics["target_prefix"] is True
    assert wrong_metrics["suffix_first_error"] == 3
    assert wrong_metrics["suffix_position_denominator"] == 4

    array_prompt = 'Return compact JSON array from chunks: qaaaa | qbbbb | qaaaa'
    array_expected = '["qaaaa","qbbbb","qaaaa"]'
    array_generated = '["qaaaa","qcccc"]'
    array_full = (*tokenizer.encode_evaluation_prefix(array_prompt), *tokenizer.encode_text(array_generated), EOS_ID)
    array_metrics = sf.array_row_metrics(array_prompt, array_expected, array_full)
    assert array_metrics["valid_array_schema"] is True
    assert array_metrics["correct_item_count"] is False
    assert Counter(array_metrics["missing_items"]) == Counter(["qbbbb", "qaaaa"])
    assert array_metrics["extra_items"] == ["qcccc"]
    array_aggregate = sf.aggregate_array_rows([{**array_metrics, "family": "array_json", "template_id": "seq_array_json_eval_0", "item_count": 3, "expected": array_expected}])
    assert array_aggregate["item_count_exact_matches"]["3"]["denominator"] == 1
    assert array_aggregate["template_item_count_exact_matches"]["seq_array_json_eval_0__items3"]["denominator"] == 1
    assert array_aggregate["expected_item_counts"] == {"observation_count": 1, "total": 3, "counts": {"qaaaa": 2, "qbbbb": 1}}
    assert array_aggregate["missing_item_counts"] == {"observation_count": 1, "total": 2, "counts": {"qaaaa": 1, "qbbbb": 1}}
    assert array_aggregate["extra_item_counts"] == {"observation_count": 1, "total": 1, "counts": {"qcccc": 1}}
    assert array_aggregate["response_edit_distance"]["observation_count"] == 1

    invalid_full = (*tokenizer.encode_evaluation_prefix(record.prompt), EOS_ID, EOS_ID)
    invalid = sf.generation_slice_metrics(record.prompt, record.answer, invalid_full)
    assert invalid["generated"] is None
    assert invalid["generation_error"] is not None
    assert invalid["response_edit_distance"] is None
    invalid_utf8_cap = (*tokenizer.encode_evaluation_prefix(record.prompt), 255, *([65] * 63))
    invalid_utf8_metrics = sf.generation_slice_metrics(record.prompt, record.answer, invalid_utf8_cap)
    assert invalid_utf8_metrics["generated"] is None
    assert invalid_utf8_metrics["hit_generation_cap"] is True
    context_prompt = "x" * 190
    context_prefix = tokenizer.encode_evaluation_prefix(context_prompt)
    context_full = (*context_prefix, *([65] * (ByteTokenizer.max_sequence_length - len(context_prefix))))
    assert len(context_full) == ByteTokenizer.max_sequence_length
    context_metrics = sf.generation_slice_metrics(context_prompt, "x", context_full)
    assert context_metrics["hit_context_cap"] is True
    eos_full = (*context_prefix, *([65] * 10), EOS_ID)
    eos_metrics = sf.generation_slice_metrics(context_prompt, "A" * 10, eos_full)
    assert eos_metrics["has_eos"] is True
    assert eos_metrics["hit_generation_cap"] is False
    assert eos_metrics["hit_context_cap"] is False

    class UniformModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.seen_shape: tuple[int, int] | None = None

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            self.seen_shape = tuple(input_ids.shape)
            return torch.zeros((*input_ids.shape, ByteTokenizer.vocab_size), dtype=torch.float32, device=input_ids.device) + self.weight

    model = UniformModel()
    model.eval()
    tf_rows, aggregate = sf.teacher_forced_rows(model, sf.diagnostic_record_sets()["array_eval64"], tokenizer, torch.device("cpu"), split="eval")
    assert model.seen_shape == (64, 256)
    assert len(tf_rows) == 64
    assert [row["record_index"] for row in tf_rows] == list(range(64))
    assert aggregate["selected_token_count"] == sum(row["selected_token_count"] for row in tf_rows)
    assert aggregate["token_accuracy"]["numerator"] == sum(row["correct_token_count"] for row in tf_rows)
    assert aggregate["nll_numerator"] == math.fsum(row["nll_numerator"] for row in tf_rows)
    assert aggregate["loss"]["observation_count"] == aggregate["selected_token_count"]


def test_diagnostic_preflight_refusals_lineage_and_no_root_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    process_state = _snapshot_diagnostic_process_state(sf)
    request.addfinalizer(lambda: _restore_diagnostic_process_state(sf, process_state))
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", tmp_path)
    monkeypatch.setattr(sf, "validate_diagnostic_input_root", lambda input_root, environment=None: {})
    input_root = tmp_path / "feasibility_004"
    output_root = tmp_path / "feasibility_diagnostic_001"
    exact_env = {"PYTHONDONTWRITEBYTECODE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": "."}
    exact_argv = [
        "python",
        "scripts/phase8_sequence_feasibility.py",
        "diagnose-failure",
        "--device",
        "cuda:0",
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
    ]

    with pytest.raises(ValueError, match="cuda:0"):
        sf.validate_diagnostic_cli_contract(device="cpu", input_root=input_root, output_root=output_root, predecessor_diagnostic_roots=(), environ=exact_env)
    assert not output_root.exists()
    with pytest.raises(ValueError, match="repository-relative"):
        sf.validate_diagnostic_cli_contract(device="cuda:0", input_root=input_root, output_root=output_root, predecessor_diagnostic_roots=(), environ=exact_env)
    validate_new_root = sf.validate_new_diagnostic_root
    monkeypatch.setattr(sf, "validate_new_diagnostic_root", lambda input_root, output_root, predecessor_diagnostic_roots=(): None)
    command_input_root = Path("artifacts/phase8_toy_lm_bridge/feasibility_004")
    command_output_root = Path("artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001")
    exact_argv = [
        "python",
        "scripts/phase8_sequence_feasibility.py",
        "diagnose-failure",
        "--device",
        "cuda:0",
        "--input-root",
        str(command_input_root),
        "--output-root",
        str(command_output_root),
    ]
    monkeypatch.setattr(sf, "diagnostic_kernel_argv", lambda: exact_argv)
    sf.validate_diagnostic_cli_contract(
        device="cuda:0",
        input_root=command_input_root,
        output_root=command_output_root,
        predecessor_diagnostic_roots=(),
        environ=exact_env,
    )
    monkeypatch.setattr(sf, "diagnostic_kernel_argv", lambda: [*exact_argv[:-2], str(command_output_root), "--device", "cuda:0"])
    with pytest.raises(ValueError, match="process argv"):
        sf.validate_diagnostic_cli_contract(
            device="cuda:0",
            input_root=command_input_root,
            output_root=command_output_root,
            predecessor_diagnostic_roots=(),
            environ=exact_env,
        )
    monkeypatch.setattr(sf, "diagnostic_kernel_argv", lambda: exact_argv)
    with pytest.raises(ValueError, match="PYTHONPATH"):
        sf.validate_diagnostic_cli_contract(
            device="cuda:0",
            input_root=command_input_root,
            output_root=command_output_root,
            predecessor_diagnostic_roots=(),
            environ={**exact_env, "PYTHONPATH": str(REPO_ROOT)},
        )
    with pytest.raises(ValueError, match="repository-relative"):
        sf.validate_diagnostic_cli_contract(
            device="cuda:0",
            input_root=command_input_root.resolve(),
            output_root=command_output_root.resolve(),
            predecessor_diagnostic_roots=(),
            environ=exact_env,
        )
    monkeypatch.setattr(sf, "validate_new_diagnostic_root", validate_new_root)

    done_root = tmp_path / "feasibility_diagnostic_001"
    done_root.mkdir()
    common = {
        "artifact_class": sf.DIAGNOSTIC_ARTIFACT_CLASS,
        "feasibility_selection_eligible": False,
        "task_010d_authorized": False,
        "protocol": "phase8_feasibility_failure_diagnostic",
        "terminal_status": "DONE",
        "source_commit": "b" * 40,
        "source_provenance": {},
        "exact_command": [],
        "output_root": str(done_root),
        "wall_time_seconds": 0.0,
        "deterministic_flags": {},
        "handoff": {},
        "input_root": {},
        "configuration": {},
        "environment": {},
        "record_hashes": {},
        "core_blobs": {},
        "diagnostic_lineage": [],
        "repair_transition": None,
        "completed_scope": [],
        "partial_scope": [],
        "failure_classification": None,
    }
    (done_root / "summary.json").write_text(json.dumps(common, sort_keys=True) + "\n")
    manifest = {
        **common,
        "file_inventory": sf.diagnostic_inventory(done_root),
    }
    (done_root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    terminal = {
        "artifact_class": sf.DIAGNOSTIC_ARTIFACT_CLASS,
        "feasibility_selection_eligible": False,
        "task_010d_authorized": False,
        "status": "DONE",
        "manifest_path": "manifest.json",
        "manifest_sha256": sf.file_sha256(done_root / "manifest.json"),
            **{key: common[key] for key in (
            "source_commit", "source_provenance", "protocol", "exact_command", "output_root", "wall_time_seconds",
            "deterministic_flags", "handoff", "input_root", "configuration", "environment", "record_hashes",
            "core_blobs", "diagnostic_lineage", "repair_transition", "completed_scope", "partial_scope", "failure_classification",
        )},
        "file_inventory": manifest["file_inventory"],
    }
    (done_root / "DONE.json").write_text(json.dumps(terminal) + "\n")
    monkeypatch.setattr(sf, "validate_diagnostic_common_semantics", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="prior DONE"):
        sf.validate_new_diagnostic_root(input_root, tmp_path / "feasibility_diagnostic_002", (done_root,))
    assert sf.diagnostic_terminal_binding(done_root)["terminal_state"] == "DONE"


def test_diagnostic_failed_terminal_inventory_fields_and_checkpoint_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    input_root = sf.ARTIFACT_PARENT / "feasibility_004"
    root = tmp_path / "feasibility_diagnostic_001"
    root.mkdir()
    (root / "rows.jsonl").write_text("{}\n")
    monkeypatch.setattr(sf, "current_source_commit", lambda: _test_source_commit())
    monkeypatch.setattr(sf, "ignored_source_inputs", lambda: ())
    monkeypatch.setattr(sf, "validate_diagnostic_core_blobs", lambda commit="HEAD": dict(sf.DIAGNOSTIC_CORE_BLOBS))
    preflight_bindings = sf.diagnostic_preflight_bindings(input_root, ())
    monkeypatch.setattr(sf, "current_environment_dict", lambda: {"mutated_after_preflight": True})
    sf.write_diagnostic_terminal(
        root,
        "FAILED",
        input_root=input_root,
        output_root=root,
        predecessor_diagnostic_roots=(),
        completed_scope=({"name": "preflight"},),
        partial_scope=({"name": "array_run", "seed": 0},),
        failure="synthetic failure",
        failure_classification="diagnostic_implementation_defect",
        source_snapshot=sf.SourceSnapshot(commit=_test_source_commit(), status_lines=(), ignored_inputs=()),
        preflight_bindings=preflight_bindings,
        wall_time_seconds=0.0,
    )
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    failed = json.loads((root / "FAILED.json").read_text())
    for record in (manifest, summary, failed):
        assert record["artifact_class"] == sf.DIAGNOSTIC_ARTIFACT_CLASS
        assert record["feasibility_selection_eligible"] is False
        assert record["task_010d_authorized"] is False
        assert record["source_provenance"]["commit"] == _test_source_commit()
        assert record["exact_command"][:3] == ["PYTHONDONTWRITEBYTECODE=1", "CUBLAS_WORKSPACE_CONFIG=:4096:8", "PYTHONPATH=."]
        assert record["configuration"]["diagnostic_steps"] == 3000
        assert record["handoff"]["path"] == sf.DIAGNOSTIC_HANDOFF_PATH
        assert record["input_root"]["path"] == str(input_root)
        assert record["environment"] == preflight_bindings["environment"]
    assert record["record_hashes"] == sf.DIAGNOSTIC_RECORD_HASHES
    assert failed["manifest_sha256"] == sf.file_sha256(root / "manifest.json")
    assert failed["file_inventory"] == manifest["file_inventory"]
    assert {row["path"]: row["role"] for row in manifest["file_inventory"]}["rows.jsonl"] == "retained_rows"
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", tmp_path)
    monkeypatch.setattr(sf, "validate_diagnostic_common_semantics", lambda *args, **kwargs: None)
    assert sf.diagnostic_terminal_binding(root)["terminal_state"] == "FAILED"

    checkpoint = tmp_path / "diagnostic_checkpoint.pt"
    save_checkpoint(
        str(checkpoint),
        build_model("small"),
        metadata={
            "artifact_class": sf.DIAGNOSTIC_ARTIFACT_CLASS,
            "feasibility_selection_eligible": False,
            "task_010d_authorized": False,
            "family": "array_json",
            "model_size": "small",
            "seed": 0,
            "training_steps": 3000,
        },
    )
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)["metadata"]
    assert metadata["artifact_class"] == sf.DIAGNOSTIC_ARTIFACT_CLASS
    assert metadata["feasibility_selection_eligible"] is False
    assert metadata["task_010d_authorized"] is False
    assert sf.classify_diagnostic_failure(OSError("disk unavailable")) == "transient_infrastructure"
    assert sf.classify_diagnostic_failure(ValueError("bad implementation")) == "diagnostic_implementation_defect"


def test_diagnostic_publish_done_requires_exact_paths_and_no_clobber(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    monkeypatch.setattr(sf, "validate_diagnostic_common_semantics", lambda *args, **kwargs: None)

    def make_done_root(root: Path) -> None:
        root.mkdir(parents=True)
        for rel_path in sf.expected_done_diagnostic_paths() - {"summary.json", "manifest.json", "DONE.json"}:
            path = root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".pt":
                path.write_bytes(b"checkpoint")
            elif path.suffix == ".jsonl":
                path.write_text("{}\n")
            else:
                path.write_text("{}\n")
        common = {
            "artifact_class": sf.DIAGNOSTIC_ARTIFACT_CLASS,
            "feasibility_selection_eligible": False,
            "task_010d_authorized": False,
            "protocol": "phase8_feasibility_failure_diagnostic",
            "terminal_status": "DONE",
            "source_commit": "b" * 40,
            "source_provenance": {},
            "exact_command": [],
            "output_root": str(root.with_suffix("")),
            "wall_time_seconds": 0.0,
            "deterministic_flags": {},
            "handoff": {},
            "input_root": {},
            "configuration": {},
            "environment": {},
            "record_hashes": {},
        "core_blobs": {},
        "diagnostic_lineage": [],
        "repair_transition": None,
        "completed_scope": [
                *({"name": "named_cell"} for _ in range(12)),
                *({"name": "array_diagnostic_training"} for _ in range(3)),
                *({"name": "array_baseline_reuse"} for _ in range(6)),
            ],
            "partial_scope": [],
            "failure_classification": None,
        }
        (root / "summary.json").write_text(json.dumps(common, sort_keys=True) + "\n")
        manifest = {**common, "file_inventory": sf.diagnostic_inventory(root)}
        (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
        terminal = {
            "artifact_class": sf.DIAGNOSTIC_ARTIFACT_CLASS,
            "feasibility_selection_eligible": False,
            "task_010d_authorized": False,
            "status": "DONE",
            "manifest_path": "manifest.json",
            "manifest_sha256": sf.file_sha256(root / "manifest.json"),
                **{key: common[key] for key in (
                    "source_commit", "source_provenance", "protocol", "exact_command", "output_root", "wall_time_seconds",
                    "deterministic_flags", "handoff", "input_root", "configuration", "environment", "record_hashes",
                "core_blobs", "diagnostic_lineage", "repair_transition", "completed_scope", "partial_scope", "failure_classification",
            )},
            "file_inventory": manifest["file_inventory"],
        }
        (root / "DONE.json").write_text(json.dumps(terminal, sort_keys=True) + "\n")

    temp_root = tmp_path / "feasibility_diagnostic_001.tmp"
    make_done_root(temp_root)
    target = tmp_path / "feasibility_diagnostic_001"
    target.mkdir()
    (target / "sentinel").write_text("keep\n")
    with pytest.raises(FileExistsError):
        sf.publish_diagnostic_root(temp_root, target, terminal_status="DONE")
    assert (target / "sentinel").read_text() == "keep\n"
    assert temp_root.exists()
    with pytest.raises(sf.DiagnosticPublicationError, match="temporary root is incomplete"):
        sf.publish_diagnostic_root_or_leave_incomplete(temp_root, target, terminal_status="DONE")
    assert not (temp_root / "DONE.json").exists()

    success_temp = tmp_path / "feasibility_diagnostic_002.tmp"
    success_target = tmp_path / "feasibility_diagnostic_002"
    make_done_root(success_temp)
    sf.publish_diagnostic_root(success_temp, success_target, terminal_status="DONE")
    assert success_target.is_dir()
    assert not success_temp.exists()

    bad_root = tmp_path / "bad.tmp"
    make_done_root(bad_root)
    (bad_root / "extra.json").write_text("{}\n")
    with pytest.raises(ValueError, match="inventory|extra files"):
        sf.validate_diagnostic_terminal_root(bad_root, terminal_status="DONE")
    (bad_root / "extra.json").unlink()
    (bad_root / "array_json__small__3000__seed0" / "generations.jsonl").unlink()
    with pytest.raises(ValueError, match="inventory|incomplete"):
        sf.validate_diagnostic_terminal_root(bad_root, terminal_status="DONE")


def test_diagnostic_predecessor_inventory_lineage_and_retry_contracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", tmp_path)
    monkeypatch.setattr(sf, "validate_diagnostic_input_root", lambda input_root, environment=None: {})
    monkeypatch.setattr(sf, "validate_diagnostic_common_semantics", lambda *args, **kwargs: None)
    monkeypatch.setattr(sf, "current_source_commit", lambda: "b" * 40)
    configuration = sf.frozen_diagnostic_configuration()

    def make_failed(root: Path, *, lineage: list[dict[str, object]], source_commit: str, failure_classification: str) -> dict[str, object]:
        root.mkdir()
        (root / "rows.jsonl").write_text("{}\n")
        common = {
            "artifact_class": sf.DIAGNOSTIC_ARTIFACT_CLASS,
            "feasibility_selection_eligible": False,
            "task_010d_authorized": False,
            "terminal_status": "FAILED",
            "protocol": "phase8_feasibility_failure_diagnostic",
            "diagnostic_lineage": lineage,
            "configuration": configuration,
            "source_commit": source_commit,
            "source_provenance": {},
            "exact_command": [],
            "output_root": str(root),
            "wall_time_seconds": 0.0,
            "deterministic_flags": {},
            "handoff": {},
            "input_root": {},
            "environment": {},
            "record_hashes": {},
            "core_blobs": {},
            "completed_scope": [],
            "partial_scope": [],
            "failure_classification": failure_classification,
            "repair_transition": None,
        }
        (root / "summary.json").write_text(json.dumps(common, sort_keys=True) + "\n")
        manifest = {**common, "file_inventory": sf.diagnostic_inventory(root)}
        (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
        terminal = {
            "artifact_class": sf.DIAGNOSTIC_ARTIFACT_CLASS,
            "feasibility_selection_eligible": False,
            "task_010d_authorized": False,
            "status": "FAILED",
            "manifest_path": "manifest.json",
            "manifest_sha256": sf.file_sha256(root / "manifest.json"),
                **{key: common[key] for key in (
                    "source_commit", "source_provenance", "protocol", "exact_command", "output_root", "wall_time_seconds",
                    "deterministic_flags", "handoff", "input_root", "configuration", "environment", "record_hashes",
                "core_blobs", "diagnostic_lineage", "repair_transition", "completed_scope", "partial_scope", "failure_classification",
            )},
            "file_inventory": manifest["file_inventory"],
        }
        (root / "FAILED.json").write_text(json.dumps(terminal, sort_keys=True) + "\n")
        return sf.diagnostic_terminal_binding(root)

    input_root = tmp_path / "feasibility_004"
    first = make_failed(
        tmp_path / "feasibility_diagnostic_001",
        lineage=[],
        source_commit="b" * 40,
        failure_classification="transient_infrastructure",
    )
    assert first["terminal_state"] == "FAILED"
    sf.validate_new_diagnostic_root(input_root, tmp_path / "feasibility_diagnostic_002", (tmp_path / "feasibility_diagnostic_001",))

    tampered = tmp_path / "feasibility_diagnostic_001" / "rows.jsonl"
    tampered.write_text("{\"tampered\":true}\n")
    with pytest.raises(ValueError, match="inventory|checksum"):
        sf.diagnostic_terminal_binding(tmp_path / "feasibility_diagnostic_001")
    tampered.write_text("{}\n")

    escape_manifest = json.loads((tmp_path / "feasibility_diagnostic_001" / "manifest.json").read_text())
    escape_manifest["file_inventory"][0]["path"] = "../escape.jsonl"
    (tmp_path / "feasibility_diagnostic_001" / "manifest.json").write_text(json.dumps(escape_manifest, sort_keys=True) + "\n")
    escape_terminal = json.loads((tmp_path / "feasibility_diagnostic_001" / "FAILED.json").read_text())
    escape_terminal["manifest_sha256"] = sf.file_sha256(tmp_path / "feasibility_diagnostic_001" / "manifest.json")
    (tmp_path / "feasibility_diagnostic_001" / "FAILED.json").write_text(json.dumps(escape_terminal, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="inventory|relative"):
        sf.diagnostic_terminal_binding(tmp_path / "feasibility_diagnostic_001")
    valid_manifest = json.loads((tmp_path / "feasibility_diagnostic_001" / "manifest.json").read_text())
    valid_manifest["file_inventory"] = sf.diagnostic_inventory(tmp_path / "feasibility_diagnostic_001")
    (tmp_path / "feasibility_diagnostic_001" / "manifest.json").write_text(json.dumps(valid_manifest, sort_keys=True) + "\n")
    valid_terminal = json.loads((tmp_path / "feasibility_diagnostic_001" / "FAILED.json").read_text())
    valid_terminal["manifest_sha256"] = sf.file_sha256(tmp_path / "feasibility_diagnostic_001" / "manifest.json")
    (tmp_path / "feasibility_diagnostic_001" / "FAILED.json").write_text(json.dumps(valid_terminal, sort_keys=True) + "\n")

    first_lineage_binding = {key: value for key, value in first.items() if key != "diagnostic_lineage"}
    repaired_second = make_failed(
        tmp_path / "feasibility_diagnostic_002",
        lineage=[first_lineage_binding],
        source_commit="b" * 40,
        failure_classification="diagnostic_implementation_defect",
    )
    assert repaired_second["diagnostic_lineage"] == [first_lineage_binding]
    with pytest.raises(ValueError, match="newly reviewed repair source"):
        sf.validate_new_diagnostic_root(
            input_root,
            tmp_path / "feasibility_diagnostic_003",
            (tmp_path / "feasibility_diagnostic_001", tmp_path / "feasibility_diagnostic_002"),
        )
    monkeypatch.setattr(sf, "current_source_commit", lambda: "c" * 40)
    sf.validate_new_diagnostic_root(
        input_root,
        tmp_path / "feasibility_diagnostic_003",
        (tmp_path / "feasibility_diagnostic_001", tmp_path / "feasibility_diagnostic_002"),
    )


def test_diagnostic_roots_are_rejected_by_feasibility_selection_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", tmp_path)
    diagnostic_root = tmp_path / "feasibility_diagnostic_001"
    diagnostic_root.mkdir()
    with pytest.raises(ValueError, match="feasibility_NNN"):
        sf.validate_feasibility_root_artifacts(diagnostic_root, require_passing=False)
    selection = tmp_path / "feasibility_selection_001.json"
    selection.write_text(
        json.dumps(
            {
                "selected_root": str(diagnostic_root),
                "selected_manifest_sha256": "0" * 64,
                "source_commit": _test_source_commit(),
                "configuration": sf.frozen_configuration(),
                "per_cell_counts": [],
                "pass_decision": True,
                "independent_review_verdict": "ACCEPT",
                "predecessor_roots": [],
                "predecessor_selections": [],
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="feasibility_NNN"):
        sf.validate_selection_record(selection)


def test_diagnostic_array_aggregate_rejects_invalid_row_observation_tampering() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    tokenizer = ByteTokenizer()
    prompt = "Return compact JSON array from chunks: qaaaa | qbbbb"
    expected = '["qaaaa","qbbbb"]'
    valid_full = (*tokenizer.encode_evaluation_prefix(prompt), *tokenizer.encode_text('["qaaaa","qcccc"]'), EOS_ID)
    valid_row = {
        **sf.array_row_metrics(prompt, expected, valid_full),
        "family": "array_json",
        "template_id": "seq_array_json_eval_0",
        "item_count": 2,
        "expected": expected,
    }
    invalid_full = (*tokenizer.encode_evaluation_prefix(prompt), *tokenizer.encode_text('{"not":"array"}'), EOS_ID)
    invalid_row = {
        **sf.array_row_metrics(prompt, expected, invalid_full),
        "family": "array_json",
        "template_id": "seq_array_json_eval_0",
        "item_count": 2,
        "expected": expected,
    }
    aggregate = sf.aggregate_array_rows([valid_row, invalid_row])
    assert aggregate["row_count"] == 2
    assert aggregate["valid_schema_observation_count"] == 1
    assert aggregate["missing_item_counts"] == {"observation_count": 1, "total": 1, "counts": {"qbbbb": 1}}
    assert aggregate["extra_item_counts"] == {"observation_count": 1, "total": 1, "counts": {"qcccc": 1}}
    all_invalid = sf.aggregate_array_rows([invalid_row])
    assert all_invalid["missing_item_counts"] == {"observation_count": 0, "total": None, "counts": None}
    assert all_invalid["positional_exact_items"] == {"numerator": 0, "denominator": 0, "rate": None}
    tampered = {**invalid_row, "missing_items": []}
    with pytest.raises(ValueError, match="Invalid Array schema"):
        sf.aggregate_array_rows([tampered])
    tampered_valid = {**valid_row, "extra_items": None}
    with pytest.raises(ValueError, match="Valid Array schema"):
        sf.aggregate_array_rows([tampered_valid])


def test_diagnostic_reused_array_rows_preserve_current_eval_identity(tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    records = sf.grouped_records()["array_json"]["eval"]
    tokenizer = ByteTokenizer()
    rows = [_generation_row(record, tokenizer, exact_match=False) for record in records]
    path = tmp_path / "generations.jsonl"
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    converted = sf.array_rows_from_reused_generations(
        path,
        comparison_source="feasibility_004_reused",
        model_size="small",
        steps=sf.TRAINING_STEPS,
        seed=0,
    )
    assert len(converted) == sf.EVAL_RECORDS_PER_FAMILY
    sf.validate_array_generation_artifact(
        converted,
        comparison_source="feasibility_004_reused",
        model_size="small",
        steps=sf.TRAINING_STEPS,
        seed=0,
    )
    metric_tampered = [dict(row) for row in converted]
    metric_tampered[0]["exact_match"] = not metric_tampered[0]["exact_match"]
    with pytest.raises(ValueError, match="does not rebuild"):
        sf.validate_array_generation_artifact(
            metric_tampered,
            comparison_source="feasibility_004_reused",
            model_size="small",
            steps=sf.TRAINING_STEPS,
            seed=0,
        )
    for field_name, replacement in (
        ("family", "hex_copy"),
        ("index", 999),
        ("template_id", "wrong-template"),
        ("operand_id", "wrong-operand"),
        ("prompt", "wrong prompt"),
        ("expected", "[]"),
    ):
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0][field_name] = replacement
        path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in tampered_rows) + "\n")
        with pytest.raises(ValueError, match="Generation artifact row|raw_token_ids prefix"):
            sf.array_rows_from_reused_generations(
                path,
                comparison_source="feasibility_004_reused",
                model_size="small",
                steps=sf.TRAINING_STEPS,
                seed=0,
            )


def test_diagnostic_named_matrix_rejects_family_and_cell_relabeling() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    matrix = sf.build_named_diagnostic_matrix()
    family_tampered = {cell: [dict(row) for row in rows] for cell, rows in matrix.items()}
    family_tampered["seen_surface_seen_operand"][0]["family"] = "array_json"
    with pytest.raises(ValueError, match="family named_value_json"):
        sf.validate_named_diagnostic_matrix(family_tampered)
    cell_tampered = {cell: [dict(row) for row in rows] for cell, rows in matrix.items()}
    cell_tampered["seen_surface_seen_operand"][0]["diagnostic_cell"] = "held_surface_seen_operand"
    with pytest.raises(ValueError, match="diagnostic_cell"):
        sf.validate_named_diagnostic_matrix(cell_tampered)
    tokenizer = ByteTokenizer()
    matrix_rows = matrix["seen_surface_seen_operand"]
    generation_rows = []
    for row in matrix_rows:
        raw_ids = [*tokenizer.encode_evaluation_prefix(row["prompt"]), *tokenizer.encode_text(row["expected"]), EOS_ID]
        generation_rows.append(
            {
                **row,
                **sf.named_row_metrics(
                    row["prompt"],
                    row["expected"],
                    raw_ids,
                    row["target_key"],
                    [item["value"] for item in row["roster"]],
                ),
            }
        )
    sf.validate_named_generation_artifact(generation_rows, matrix_rows)
    generation_rows[0] = {**generation_rows[0], "target_prefix": False}
    with pytest.raises(ValueError, match="does not rebuild"):
        sf.validate_named_generation_artifact(generation_rows, matrix_rows)


def test_diagnostic_real_process_command_rejects_direct_main_and_flags(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    process_state = _snapshot_diagnostic_process_state(sf)
    request.addfinalizer(lambda: _restore_diagnostic_process_state(sf, process_state))
    monkeypatch.setattr(sf, "validate_diagnostic_input_root", lambda input_root, environment=None: {})
    monkeypatch.setattr(sf, "validate_new_diagnostic_root", lambda input_root, output_root, predecessor_diagnostic_roots=(): None)
    command_input_root = Path("artifacts/phase8_toy_lm_bridge/feasibility_004")
    command_output_root = Path("artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001")
    exact_env = {"PYTHONDONTWRITEBYTECODE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": "."}
    exact_argv = [
        "python",
        "scripts/phase8_sequence_feasibility.py",
        "diagnose-failure",
        "--device",
        "cuda:0",
        "--input-root",
        str(command_input_root),
        "--output-root",
        str(command_output_root),
    ]
    monkeypatch.setattr(sf, "diagnostic_kernel_argv", lambda: exact_argv)
    sf.validate_diagnostic_cli_contract(
        device="cuda:0",
        input_root=command_input_root,
        output_root=command_output_root,
        predecessor_diagnostic_roots=(),
        environ=exact_env,
    )
    monkeypatch.setattr(sf, "diagnostic_kernel_argv", lambda: ["python", "-O", *exact_argv[1:]])
    with pytest.raises(ValueError, match="process argv"):
        sf.validate_diagnostic_cli_contract(
            device="cuda:0",
            input_root=command_input_root,
            output_root=command_output_root,
            predecessor_diagnostic_roots=(),
            environ=exact_env,
        )
    with pytest.raises(ValueError, match="main\\(argv"):
        sf.main([
            "diagnose-failure",
            "--device",
            "cuda:0",
            "--input-root",
            str(command_input_root),
            "--output-root",
            str(command_output_root),
        ])
    with pytest.raises(ValueError, match="real __main__"):
        sf.run_diagnostic_failure(
            device="cuda:0",
            input_root=command_input_root,
            output_root=command_output_root,
            predecessor_diagnostic_roots=(),
            environ=exact_env,
        )


@pytest.mark.parametrize(
    ("exc", "expected"),
    (
        (MemoryError("host oom"), "transient_infrastructure"),
        (torch.OutOfMemoryError("cuda oom"), "transient_infrastructure"),
        (OSError("filesystem"), "transient_infrastructure"),
        (subprocess.SubprocessError("worker"), "transient_infrastructure"),
        (RuntimeError("bug"), "diagnostic_implementation_defect"),
        (ValueError("bug"), "diagnostic_implementation_defect"),
    ),
)
def test_diagnostic_failure_classification_matrix(exc: Exception, expected: str) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    assert sf.classify_diagnostic_failure(exc) == expected


def test_diagnostic_parameter_count_comparator_is_isolated_from_formal_helper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    assert sf.expected_parameter_count("small") == build_model("small").parameter_count
    calls: list[str] = []

    def forbidden_helper(model_size: str) -> int:
        calls.append(model_size)
        raise AssertionError("diagnostic comparator must not call formal expected_parameter_count")

    checkpoint = Path("artifacts/phase8_toy_lm_bridge/feasibility_004/array_json__small__seed0/checkpoint_step1500.pt")
    model = sf.load_checkpoint_model(checkpoint, "small", torch.device("cpu"))
    monkeypatch.setattr(sf, "expected_parameter_count", forbidden_helper)
    sf.compare_model_to_checkpoint_step1500(model, checkpoint, "small")
    assert calls == []


def test_diagnostic_common_semantic_validator_rejects_provenance_protocol_and_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    source_commit = sf.DECISION_DIAGNOSTIC_ROOT_BINDING["source_commit"]
    output_root = Path("artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001")
    input_root = Path("artifacts/phase8_toy_lm_bridge/feasibility_004")
    input_manifest = json.loads((input_root / "manifest.json").read_text())
    core_blob_commits: list[str] = []

    def validate_manifest_core_blobs(commit: str = "HEAD") -> dict[str, str]:
        core_blob_commits.append(commit)
        return sf.DIAGNOSTIC_CORE_BLOBS

    monkeypatch.setattr(sf, "validate_git_commit_exists", lambda commit: None)
    monkeypatch.setattr(sf, "validate_diagnostic_record_hashes", lambda: sf.DIAGNOSTIC_RECORD_HASHES)
    monkeypatch.setattr(sf, "validate_diagnostic_core_blobs", validate_manifest_core_blobs)
    monkeypatch.setattr(sf, "validate_diagnostic_environment", lambda environment, input_root: None)
    common = {
        "artifact_class": sf.DIAGNOSTIC_ARTIFACT_CLASS,
        "feasibility_selection_eligible": False,
        "task_010d_authorized": False,
        "protocol": "phase8_feasibility_failure_diagnostic",
        "terminal_status": "FAILED",
        "failure": "synthetic",
        "failure_classification": "diagnostic_implementation_defect",
        "source_commit": source_commit,
        "source_provenance": {"commit": source_commit, "status_lines": [], "ignored_inputs": []},
        "exact_command": sf.diagnostic_exact_command(output_root, (), input_root=input_root),
        "output_root": str(output_root),
        "wall_time_seconds": 0.0,
        "deterministic_flags": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONPATH": ".",
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_tf32": False,
            "cudnn_tf32": False,
        },
        "handoff": sf.handoff_binding_for_commit(source_commit),
        "input_root": {
            "path": str(input_root),
            "manifest_sha256": sf.DIAGNOSTIC_INPUT_CHECKSUMS["manifest.json"],
            "summary_sha256": sf.DIAGNOSTIC_INPUT_CHECKSUMS["summary.json"],
            "terminal_sha256": sf.DIAGNOSTIC_INPUT_CHECKSUMS["FAILED.json"],
            "source_commit": sf.DIAGNOSTIC_INPUT_SOURCE_COMMIT,
            "predecessor_roots": input_manifest["predecessor_roots"],
        },
        "diagnostic_lineage": [],
        "configuration": sf.frozen_diagnostic_configuration(),
        "record_hashes": sf.DIAGNOSTIC_RECORD_HASHES,
        "core_blobs": sf.DIAGNOSTIC_CORE_BLOBS,
        "environment": {"python": "x", "platform": "x", "torch": "x", "cuda_available": True, "cuda": "x", "gpu": "x", "gpu_driver": "x"},
        "completed_scope": [],
        "partial_scope": [{"name": "diagnostic_initialization"}],
        "repair_transition": None,
        "file_inventory": [],
        "summary": {"done": False, "completed_scope_count": 0, "partial_scope_count": 1},
    }
    common["partial_scope"][0]["error"] = "synthetic"
    sf.validate_diagnostic_common_semantics(output_root, terminal_status="FAILED", terminal_data=common, manifest_data=common, summary_data=common)
    assert core_blob_commits == [source_commit]
    core_blob_commits.clear()
    mutations = (
        ("protocol", "wrong"),
        ("source_commit", "0" * 40),
        ("source_provenance", {}),
        ("exact_command", []),
        ("configuration", {}),
        ("record_hashes", {}),
        ("core_blobs", {}),
        ("deterministic_flags", {}),
        ("handoff", {}),
        ("input_root", {}),
        ("environment", {}),
    )
    for key, value in mutations:
        tampered = {**common, key: value}
        with pytest.raises(ValueError):
            sf.validate_diagnostic_common_semantics(output_root, terminal_status="FAILED", terminal_data=tampered, manifest_data=tampered, summary_data=tampered)
    with pytest.raises(ValueError, match="environment"):
        sf.validate_diagnostic_environment_schema({"python": "x"})


def test_diagnostic_done_validators_reject_fake_checkpoint_and_duplicate_scope(tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    fake_checkpoint = tmp_path / "checkpoint_step3000.pt"
    fake_checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        sf.validate_diagnostic_checkpoint_3000(fake_checkpoint, seed=0)
    fabricated_scope = [{"name": "array_training_plan", "runs": len(sf.SEEDS)}]
    with pytest.raises(ValueError, match="prefix"):
        sf.validate_diagnostic_completed_scope(fabricated_scope, terminal_status="FAILED")
    duplicate_scope = [
        {"name": "named_matrix_construction", "rows": len(sf.NAMED_DIAGNOSTIC_CELLS) * sf.EVAL_RECORDS_PER_FAMILY},
        {"name": "named_matrix_construction", "rows": len(sf.NAMED_DIAGNOSTIC_CELLS) * sf.EVAL_RECORDS_PER_FAMILY},
    ]
    with pytest.raises(ValueError, match="prefix"):
        sf.validate_diagnostic_completed_scope(duplicate_scope, terminal_status="FAILED")
    completed_matrix = [{"name": "named_matrix_construction", "rows": len(sf.NAMED_DIAGNOSTIC_CELLS) * sf.EVAL_RECORDS_PER_FAMILY}]
    with pytest.raises(ValueError, match="lacks retained artifact"):
        sf.validate_diagnostic_completed_scope_artifacts(tmp_path, completed_matrix)
    (tmp_path / "named_diagnostic_matrix.jsonl").write_text("{}\n")
    sf.validate_diagnostic_completed_scope_artifacts(tmp_path, completed_matrix)
    with pytest.raises(ValueError, match="Named diagnostic matrix|diagnostic_cell"):
        sf.validate_diagnostic_completed_scope_semantics(tmp_path, completed_matrix, tmp_path / "feasibility_004")


def _write_diagnostic_checkpoint_with_trajectory_evidence(
    tmp_path: Path,
    sf: object,
    *,
    seed: int = 0,
    final_loss: float = 1.25,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    step1500_checkpoint = tmp_path / f"checkpoint_step1500_seed{seed}.pt"
    step1500_model = build_historical_model("small")
    sf.save_historical_checkpoint(
        str(step1500_checkpoint),
        step1500_model,
        metadata={"family": "array_json", "model_size": "small", "seed": seed, "training_steps": 1500},
    )
    final_checkpoint = tmp_path / f"checkpoint_step3000_seed{seed}.pt"
    final_model = build_historical_model("small")
    schedule = sf.diagnostic_training_schedule_evidence(seed, sf.TRAIN_RECORDS_PER_FAMILY)
    loss_evidence = sf.loss_scalar_evidence(torch.tensor(final_loss, dtype=torch.float32), final_loss)
    step1500_fingerprint = sf.checkpoint_model_state_fingerprint(step1500_checkpoint)
    evidence = {
        "serialization": sf.DIAGNOSTIC_TRAJECTORY_EVIDENCE_SCHEMA,
        "claimed_seed": seed,
        "family": "array_json",
        "model_size": "small",
        "training_steps": sf.DIAGNOSTIC_STEPS,
        "authorized_new_training_run_count": len(sf.SEEDS),
        "schedule": schedule,
        "initial_model_state_fingerprint": sf.expected_initial_model_state_fingerprint(seed),
        "step1500": {
            "step": sf.TRAINING_STEPS,
            "model_state_fingerprint": step1500_fingerprint,
            "checkpoint_state_fingerprint": step1500_fingerprint,
            "model_matches_checkpoint": True,
            "optimizer_state_fingerprint": sf.optimizer_state_fingerprint(make_optimizer(step1500_model)),
        },
        "trace": {
            "serialization": sf.DIAGNOSTIC_TRAINING_TRACE_SCHEMA,
            "seed": seed,
            "step_count": sf.DIAGNOSTIC_STEPS,
            "schedule_sha256": schedule["sha256"],
            "sha256": "a" * 64,
            "final_loss": loss_evidence,
        },
        "step_count": sf.DIAGNOSTIC_STEPS,
        "step1501_executed": True,
        "final_model_state_fingerprint": sf.model_state_fingerprint(final_model),
        "final_optimizer_state_fingerprint": sf.optimizer_state_fingerprint(make_optimizer(final_model)),
        "retained_final_loss": final_loss,
    }
    sf.save_historical_checkpoint(
        str(final_checkpoint),
        final_model,
        metadata={
            "artifact_class": sf.DIAGNOSTIC_ARTIFACT_CLASS,
            "feasibility_selection_eligible": False,
            "task_010d_authorized": False,
            "family": "array_json",
            "model_size": "small",
            "seed": seed,
            "training_steps": sf.DIAGNOSTIC_STEPS,
            "training_loss": final_loss,
            "step1501_executed": True,
            "training_trajectory_evidence": evidence,
        },
    )
    training_metrics = {
        "final_loss": final_loss,
        "checkpoint_path": f"array_json__small__3000__seed{seed}/checkpoint_step3000.pt",
        "step1501_executed": True,
        "trajectory_evidence": evidence,
    }
    return final_checkpoint, step1500_checkpoint, training_metrics, evidence


def test_diagnostic_trajectory_validation_does_not_call_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    _allow_synthetic_historical_checkpoints(monkeypatch, sf)
    checkpoint, step1500_checkpoint, training_metrics, _evidence = _write_diagnostic_checkpoint_with_trajectory_evidence(tmp_path, sf)

    def forbidden_training(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("validators must not run optimizer training")

    monkeypatch.setattr(sf, "train_array_small_3000_diagnostic", forbidden_training)
    sf.validate_diagnostic_training_trajectory_evidence(
        checkpoint,
        seed=0,
        training_metrics=training_metrics,
        frozen_step1500_checkpoint=step1500_checkpoint,
    )


def test_diagnostic_trajectory_rejects_seed_substitution_with_outer_relabel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    _allow_synthetic_historical_checkpoints(monkeypatch, sf)
    checkpoint, step1500_checkpoint, training_metrics, _evidence = _write_diagnostic_checkpoint_with_trajectory_evidence(tmp_path, sf, seed=0)
    substituted = tmp_path / "checkpoint_step3000_seed1_relabel.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["metadata"] = {**payload["metadata"], "seed": 1}
    torch.save(payload, substituted)
    with pytest.raises(ValueError, match="claimed_seed|schedule|initial"):
        sf.validate_diagnostic_training_trajectory_evidence(
            substituted,
            seed=1,
            training_metrics=training_metrics,
            frozen_step1500_checkpoint=step1500_checkpoint,
        )


def test_diagnostic_trajectory_rejects_unused_checkpoint_tensor_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    _allow_synthetic_historical_checkpoints(monkeypatch, sf)
    checkpoint, step1500_checkpoint, training_metrics, _evidence = _write_diagnostic_checkpoint_with_trajectory_evidence(tmp_path, sf)
    mutated = tmp_path / "checkpoint_step3000_mutated.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model_state_dict"]["position_embedding.weight"][255, 0] += torch.tensor(1.0)
    torch.save(payload, mutated)
    with pytest.raises(ValueError, match="final model-state fingerprint"):
        sf.validate_diagnostic_training_trajectory_evidence(
            mutated,
            seed=0,
            training_metrics=training_metrics,
            frozen_step1500_checkpoint=step1500_checkpoint,
        )


def test_diagnostic_initial_fingerprint_validation_restores_rng_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    _allow_synthetic_historical_checkpoints(monkeypatch, sf)
    checkpoint, step1500_checkpoint, training_metrics, _evidence = _write_diagnostic_checkpoint_with_trajectory_evidence(tmp_path, sf, seed=2)
    random.seed(123456)
    torch.manual_seed(654321)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(111222)
    python_rng = random.getstate()
    torch_rng = torch.random.get_rng_state()
    cuda_rng = tuple(state.clone() for state in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None
    sf.validate_diagnostic_training_trajectory_evidence(
        checkpoint,
        seed=2,
        training_metrics=training_metrics,
        frozen_step1500_checkpoint=step1500_checkpoint,
    )
    assert random.getstate() == python_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    if cuda_rng is not None:
        assert all(torch.equal(actual, expected) for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rng, strict=True))

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    broken_evidence = {
        **payload["metadata"]["training_trajectory_evidence"],
        "initial_model_state_fingerprint": "0" * 64,
    }
    payload["metadata"]["training_trajectory_evidence"] = broken_evidence
    torch.save(payload, checkpoint)
    broken_metrics = {**training_metrics, "trajectory_evidence": broken_evidence}
    with pytest.raises(ValueError, match="initial model-state fingerprint"):
        sf.validate_diagnostic_training_trajectory_evidence(
            checkpoint,
            seed=2,
            training_metrics=broken_metrics,
            frozen_step1500_checkpoint=step1500_checkpoint,
        )
    assert random.getstate() == python_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    if cuda_rng is not None:
        assert all(torch.equal(actual, expected) for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rng, strict=True))


def test_diagnostic_trajectory_and_schedule_mutation_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    _allow_synthetic_historical_checkpoints(monkeypatch, sf)
    checkpoint, step1500_checkpoint, training_metrics, evidence = _write_diagnostic_checkpoint_with_trajectory_evidence(tmp_path, sf)
    trace_mutated_metrics = {
        **training_metrics,
        "trajectory_evidence": {
            **evidence,
            "trace": {**evidence["trace"], "sha256": "b" * 64},
        },
    }
    with pytest.raises(ValueError, match="metadata and metrics"):
        sf.validate_diagnostic_training_trajectory_evidence(
            checkpoint,
            seed=0,
            training_metrics=trace_mutated_metrics,
            frozen_step1500_checkpoint=step1500_checkpoint,
        )

    loss_bytes_mutated = json.loads(json.dumps(evidence))
    loss_bytes_mutated["trace"]["final_loss"]["tensor_bytes_hex"] = "00000000"
    with pytest.raises(ValueError, match="encode the retained float32 loss value"):
        sf.validate_diagnostic_trajectory_evidence(
            loss_bytes_mutated,
            seed=0,
            final_model_state_fingerprint=evidence["final_model_state_fingerprint"],
            expected_final_loss=evidence["retained_final_loss"],
        )

    schedule_mutated = tmp_path / "checkpoint_step3000_schedule_mutated.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    forged_evidence = json.loads(json.dumps(evidence))
    forged_evidence["schedule"] = {**forged_evidence["schedule"], "sha256": "0" * 64}
    payload["metadata"]["training_trajectory_evidence"] = forged_evidence
    torch.save(payload, schedule_mutated)
    forged_metrics = {**training_metrics, "trajectory_evidence": forged_evidence}
    with pytest.raises(ValueError, match="schedule evidence"):
        sf.validate_diagnostic_training_trajectory_evidence(
            schedule_mutated,
            seed=0,
            training_metrics=forged_metrics,
            frozen_step1500_checkpoint=step1500_checkpoint,
        )


def test_diagnostic_completed_scope_semantics_reject_empty_array_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    prefix = tmp_path / "array_json__small__1500__seed0__reused"
    prefix.mkdir()
    for name in ("greedy_metrics_rows.jsonl", "teacher_forced_train_rows.jsonl", "teacher_forced_eval_rows.jsonl"):
        (prefix / name).write_text("")
    (prefix / "metrics.json").write_text("{}\n")
    monkeypatch.setattr(sf, "reused_feasibility_cell_binding", lambda *args: {})
    monkeypatch.setattr(sf, "replay_array_artifacts", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="greedy artifact|wrong row count"):
        sf.validate_diagnostic_completed_scope_semantics(
            tmp_path,
            [{"name": "array_baseline_reuse", "model_size": "small", "steps": sf.TRAINING_STEPS, "seed": 0, "rows": sf.EVAL_RECORDS_PER_FAMILY}],
            tmp_path / "feasibility_004",
        )


def test_diagnostic_publish_final_callback_and_artifact_mutation_block_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")

    def make_bound_root(root: Path) -> None:
        root.mkdir()
        (root / "rows.jsonl").write_text("{}\n")
        common = {
            "file_inventory": sf.diagnostic_inventory(root),
        }
        (root / "manifest.json").write_text(json.dumps(common, sort_keys=True) + "\n")
        terminal = {
            "status": "DONE",
            "manifest_path": "manifest.json",
            "manifest_sha256": sf.file_sha256(root / "manifest.json"),
            "file_inventory": common["file_inventory"],
        }
        (root / "DONE.json").write_text(json.dumps(terminal, sort_keys=True) + "\n")
        manifest = {"file_inventory": sf.diagnostic_inventory(root)}
        (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
        terminal["manifest_sha256"] = sf.file_sha256(root / "manifest.json")
        terminal["file_inventory"] = manifest["file_inventory"]
        (root / "DONE.json").write_text(json.dumps(terminal, sort_keys=True) + "\n")

    callback_temp = tmp_path / "callback.tmp"
    callback_target = tmp_path / "callback"
    make_bound_root(callback_temp)
    monkeypatch.setattr(sf, "validate_diagnostic_terminal_root", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="source changed"):
        sf.publish_diagnostic_root(
            callback_temp,
            callback_target,
            terminal_status="DONE",
            final_callback=lambda: (_ for _ in ()).throw(RuntimeError("source changed")),
        )
    assert callback_temp.exists()
    assert not callback_target.exists()

    mutation_temp = tmp_path / "mutation.tmp"
    mutation_target = tmp_path / "mutation"
    make_bound_root(mutation_temp)

    def mutate_after_semantic_validation(*args: object, **kwargs: object) -> None:
        (mutation_temp / "rows.jsonl").write_text('{"mutated": true}\n')

    monkeypatch.setattr(sf, "validate_diagnostic_terminal_root", mutate_after_semantic_validation)
    with pytest.raises(ValueError, match="inventory|checksum"):
        sf.publish_diagnostic_root(mutation_temp, mutation_target, terminal_status="DONE")
    assert mutation_temp.exists()
    assert not mutation_target.exists()


def test_diagnostic_failed_progress_replaces_stale_generation_and_teacher_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    source_snapshot = sf.SourceSnapshot(commit="b" * 40, status_lines=(), ignored_inputs=())
    preflight = {
        "input_root": str(tmp_path / "feasibility_004"),
        "predecessors": [],
        "diagnostic_lineage": [],
        "repair_transition": None,
        "handoff": {},
        "input_root_binding": {},
        "record_hashes": {},
        "core_blobs": {},
        "environment": {},
    }

    def configure_common(output_root: Path) -> Path:
        monkeypatch.setattr(sf, "require_diagnostic_real_main_context", lambda: None)
        monkeypatch.setattr(sf, "validate_diagnostic_cli_contract", lambda **kwargs: None)
        monkeypatch.setattr(sf, "capture_diagnostic_source_provenance", lambda *args: source_snapshot)
        monkeypatch.setattr(sf, "diagnostic_preflight_bindings", lambda *args: preflight)
        monkeypatch.setattr(sf, "validate_diagnostic_input_root", lambda *args, **kwargs: {})
        monkeypatch.setattr(sf, "verify_diagnostic_preflight_bindings", lambda *args: None)
        monkeypatch.setattr(sf, "verify_source_unchanged", lambda *args, **kwargs: None)
        monkeypatch.setattr(sf, "SEEDS", (0,))
        monkeypatch.setattr(sf, "NAMED_DIAGNOSTIC_CELLS", ())
        monkeypatch.setattr(sf, "MODEL_SIZES", ())
        monkeypatch.setattr(sf, "build_named_diagnostic_matrix", lambda: {})
        monkeypatch.setattr(sf, "publish_diagnostic_root_or_leave_incomplete", lambda *args, **kwargs: None)
        monkeypatch.setattr(sf, "reused_feasibility_cell_binding", lambda *args: {"checkpoint_path": "checkpoint_step1500.pt", "checkpoint_sha256": "a" * 64})
        monkeypatch.setattr(sf, "load_checkpoint_model", lambda *args, **kwargs: torch.nn.Linear(1, 1).eval())
        return output_root.with_name(output_root.name + ".tmp")

    def train_stub(**kwargs: object) -> dict[str, object]:
        callback = kwargs["progress_callback"]
        assert callable(callback)
        callback({"subphase": "array_training", "completed_steps": 3000})
        Path(kwargs["checkpoint_path"]).write_bytes(b"checkpoint")
        return {"final_loss": 1.0, "checkpoint_path": str(kwargs["checkpoint_path"]), "step1501_executed": True}

    monkeypatch.setattr(sf, "train_array_small_3000_diagnostic", train_stub)
    generation_output = tmp_path / "generation"
    generation_temp = configure_common(generation_output)

    def generation_failure(*args: object, **kwargs: object) -> list[dict[str, object]]:
        callback = kwargs["progress_callback"]
        assert callable(callback)
        callback({"subphase": "array_generation", "completed_rows": 7})
        raise RuntimeError("generation boom")

    monkeypatch.setattr(sf, "generate_array_rows", generation_failure)
    with pytest.raises(RuntimeError, match="generation boom"):
        sf.run_diagnostic_failure(device="cuda:0", input_root=tmp_path / "feasibility_004", output_root=generation_output)
    partial = json.loads((generation_temp / "FAILED.json").read_text())["partial_scope"][0]
    assert partial["subphase"] == "array_generation"
    assert partial["completed_rows"] == 7
    assert "completed_steps" not in partial

    teacher_output = tmp_path / "teacher"
    teacher_temp = configure_common(teacher_output)
    monkeypatch.setattr(sf, "generate_array_rows", lambda *args, **kwargs: [])

    def teacher_failure(*args: object, split: str, **kwargs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        callback = kwargs["progress_callback"]
        assert callable(callback)
        callback({"subphase": "teacher_forced", "split": split, "completed_batches": 3})
        if split == "eval":
            raise RuntimeError("teacher boom")
        return [], {"split": split}

    monkeypatch.setattr(sf, "teacher_forced_rows", teacher_failure)
    with pytest.raises(RuntimeError, match="teacher boom"):
        sf.run_diagnostic_failure(device="cuda:0", input_root=tmp_path / "feasibility_004", output_root=teacher_output)
    partial = json.loads((teacher_temp / "FAILED.json").read_text())["partial_scope"][0]
    assert partial["subphase"] == "teacher_forced"
    assert partial["split"] == "eval"
    assert partial["completed_batches"] == 3
    assert "completed_rows" not in partial
    assert "completed_steps" not in partial


def test_diagnostic_early_failed_terminal_uses_backend_before_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    process_state = _snapshot_diagnostic_process_state(sf)
    request.addfinalizer(lambda: _restore_diagnostic_process_state(sf, process_state))
    contract_source = inspect.getsource(sf.validate_diagnostic_cli_contract)
    auth_index = contract_source.index("validate_diagnostic_cli_authorization_contract(")
    backend_index = contract_source.index("configure_diagnostic_deterministic_backend()")
    root_index = contract_source.index("validate_diagnostic_cli_root_contract(")
    assert auth_index < backend_index < root_index
    source = inspect.getsource(sf.run_diagnostic_failure)
    cli_contract_index = source.index("validate_diagnostic_cli_contract(")
    assert source.index("require_diagnostic_real_main_context()") < cli_contract_index
    assert cli_contract_index < source.index("temp_root =")
    for forward_capable_call in (
        "load_checkpoint_model(",
        "generate_named_diagnostic_rows(",
        "teacher_forced_rows(",
        "train_array_small_3000_diagnostic(",
    ):
        assert cli_contract_index < source.index(forward_capable_call)

    input_root = Path("artifacts/phase8_toy_lm_bridge/feasibility_004")
    output_root = Path("artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001")
    source_snapshot = sf.SourceSnapshot(commit="b" * 40, status_lines=(), ignored_inputs=())
    preflight = {
        "input_root": str(input_root),
        "predecessors": [],
        "diagnostic_lineage": [],
        "repair_transition": None,
        "handoff": {},
        "input_root_binding": {},
        "record_hashes": {},
        "core_blobs": {},
        "environment": {},
    }
    exact_argv = [
        "python",
        "scripts/phase8_sequence_feasibility.py",
        "diagnose-failure",
        "--device",
        "cuda:0",
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
    ]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONPATH", ".")
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def current_flags() -> dict[str, object]:
        return {
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        }

    events: list[object] = []
    real_authorization_contract = sf.validate_diagnostic_cli_authorization_contract
    real_set_deterministic_backend = sf.set_deterministic_backend

    def authorization_probe(**kwargs: object) -> None:
        events.append("authorization")
        real_authorization_contract(**kwargs)

    def backend_probe(seed: int) -> None:
        events.append(("backend", seed))
        real_set_deterministic_backend(seed)

    def root_stage_probe(
        *,
        input_root: Path,
        output_root: Path,
        predecessor_diagnostic_roots: tuple[Path, ...],
    ) -> None:
        events.append("root_predecessor_stage")
        assert predecessor_diagnostic_roots == ()
        sf.validate_diagnostic_flags(current_flags())

    def early_matrix_failure() -> dict[str, list[dict[str, object]]]:
        events.append("named_matrix_construction")
        sf.validate_diagnostic_flags(current_flags())
        raise RuntimeError("early backend probe")

    def forbidden_model_work(*args: object, **kwargs: object) -> object:
        raise AssertionError("model work ran before the early catchable failure")

    def publish_probe(
        temp_root: Path,
        target_root: Path,
        *,
        terminal_status: str,
        final_callback: object = None,
    ) -> None:
        assert target_root == output_root
        assert terminal_status == "FAILED"
        events.append("failed_publication")
        for filename in ("summary.json", "manifest.json", "FAILED.json"):
            record = json.loads((temp_root / filename).read_text())
            sf.validate_diagnostic_flags(record["deterministic_flags"])
        failed = json.loads((temp_root / "FAILED.json").read_text())
        assert failed["partial_scope"] == [
            {"name": "named_matrix_construction", "error": "RuntimeError('early backend probe')"}
        ]

    monkeypatch.setattr(sf, "require_diagnostic_real_main_context", lambda: None)
    monkeypatch.setattr(sf, "diagnostic_kernel_argv", lambda: exact_argv)
    monkeypatch.setattr(sf, "validate_diagnostic_cli_authorization_contract", authorization_probe)
    monkeypatch.setattr(sf, "validate_diagnostic_cli_root_contract", root_stage_probe)
    monkeypatch.setattr(sf, "set_deterministic_backend", backend_probe)
    monkeypatch.setattr(sf, "capture_diagnostic_source_provenance", lambda *args: source_snapshot)
    monkeypatch.setattr(sf, "diagnostic_preflight_bindings", lambda *args: preflight)
    monkeypatch.setattr(sf, "verify_diagnostic_preflight_bindings", lambda *args: None)
    monkeypatch.setattr(sf, "verify_source_unchanged", lambda *args, **kwargs: None)
    monkeypatch.setattr(sf, "validate_diagnostic_input_root", lambda *args, **kwargs: {})
    monkeypatch.setattr(sf, "build_named_diagnostic_matrix", early_matrix_failure)
    monkeypatch.setattr(sf, "load_checkpoint_model", forbidden_model_work)
    monkeypatch.setattr(sf, "generate_named_diagnostic_rows", forbidden_model_work)
    monkeypatch.setattr(sf, "teacher_forced_rows", forbidden_model_work)
    monkeypatch.setattr(sf, "generate_array_rows", forbidden_model_work)
    monkeypatch.setattr(sf, "train_array_small_3000_diagnostic", forbidden_model_work)
    monkeypatch.setattr(sf, "publish_diagnostic_root_or_leave_incomplete", publish_probe)

    with pytest.raises(RuntimeError, match="early backend probe"):
        sf.run_diagnostic_failure(device="cuda:0", input_root=input_root, output_root=output_root)

    assert events == [
        "authorization",
        ("backend", sf.DIAGNOSTIC_BACKEND_SEED),
        "root_predecessor_stage",
        "named_matrix_construction",
        "failed_publication",
    ]


def test_diagnostic_native_preflight_retains_head_blob_gate_before_tmp_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    input_root = Path("artifacts/phase8_toy_lm_bridge/feasibility_004")
    output_root = tmp_path / "feasibility_diagnostic_001"
    source_snapshot = sf.SourceSnapshot(commit="b" * 40, status_lines=(), ignored_inputs=())
    core_blob_commits: list[str] = []

    def reject_native_head_gate(commit: str = "HEAD") -> dict[str, str]:
        core_blob_commits.append(commit)
        if commit != "HEAD":
            raise AssertionError(f"native diagnostic preflight checked {commit}, not HEAD")
        raise ValueError("native diagnostic HEAD core blob gate")

    def forbidden_model_work(*args: object, **kwargs: object) -> object:
        raise AssertionError("model or artifact construction ran after rejected native HEAD preflight")

    monkeypatch.setattr(sf, "require_diagnostic_real_main_context", lambda: None)
    monkeypatch.setattr(sf, "validate_diagnostic_cli_contract", lambda **kwargs: None)
    monkeypatch.setattr(sf, "capture_diagnostic_source_provenance", lambda *args: source_snapshot)
    monkeypatch.setattr(sf, "current_source_commit", lambda: source_snapshot.commit)
    monkeypatch.setattr(sf, "handoff_binding_for_commit", lambda source_commit: {})
    monkeypatch.setattr(sf, "current_environment_dict", lambda: {})
    monkeypatch.setattr(sf, "validate_diagnostic_record_hashes", lambda: {})
    monkeypatch.setattr(sf, "validate_diagnostic_core_blobs", reject_native_head_gate)
    monkeypatch.setattr(sf, "build_named_diagnostic_matrix", forbidden_model_work)
    monkeypatch.setattr(sf, "load_checkpoint_model", forbidden_model_work)
    monkeypatch.setattr(sf, "generate_named_diagnostic_rows", forbidden_model_work)
    monkeypatch.setattr(sf, "teacher_forced_rows", forbidden_model_work)
    monkeypatch.setattr(sf, "generate_array_rows", forbidden_model_work)
    monkeypatch.setattr(sf, "train_array_small_3000_diagnostic", forbidden_model_work)

    with pytest.raises(ValueError, match="native diagnostic HEAD core blob gate"):
        sf.run_diagnostic_failure(device="cuda:0", input_root=input_root, output_root=output_root)

    assert core_blob_commits == ["HEAD"]
    assert not output_root.exists()
    assert not output_root.with_name(output_root.name + ".tmp").exists()


def test_diagnostic_retry_preflight_root_stage_runs_after_backend_from_defaults(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    process_state = _snapshot_diagnostic_process_state(sf)
    request.addfinalizer(lambda: _restore_diagnostic_process_state(sf, process_state))
    input_root = Path("artifacts/phase8_toy_lm_bridge/feasibility_004")
    predecessor = Path("artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001")
    output_root = Path("artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_002")
    exact_argv = [
        "python",
        "scripts/phase8_sequence_feasibility.py",
        "diagnose-failure",
        "--device",
        "cuda:0",
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
        "--predecessor-diagnostic-root",
        str(predecessor),
    ]
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONPATH", ".")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    torch.backends.mkldnn.enabled = True
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic
    assert torch.backends.cudnn.benchmark
    assert torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.allow_tf32

    def current_flags() -> dict[str, object]:
        return {
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        }

    def root_stage_probe(
        *,
        input_root: Path,
        output_root: Path,
        predecessor_diagnostic_roots: tuple[Path, ...],
    ) -> None:
        assert input_root == Path("artifacts/phase8_toy_lm_bridge/feasibility_004")
        assert output_root == Path("artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_002")
        assert predecessor_diagnostic_roots == (predecessor,)
        sf.validate_diagnostic_flags(current_flags())
        assert torch.get_num_threads() == 1
        assert torch.backends.mkldnn.enabled is False
        assert torch.is_deterministic_algorithms_warn_only_enabled() is False

    monkeypatch.setattr(sf, "diagnostic_kernel_argv", lambda: exact_argv)
    monkeypatch.setattr(sf, "validate_diagnostic_cli_root_contract", root_stage_probe)

    sf.validate_diagnostic_cli_contract(
        device="cuda:0",
        input_root=input_root,
        output_root=output_root,
        predecessor_diagnostic_roots=(predecessor,),
    )


@pytest.mark.parametrize(
    "original_cublas",
    (":16:8", None),
    ids=("wrong", "missing"),
)
def test_diagnostic_cli_rejects_original_cublas_before_backend_configuration(
    original_cublas: str | None,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    process_state = _snapshot_diagnostic_process_state(sf)
    request.addfinalizer(lambda: _restore_diagnostic_process_state(sf, process_state))
    input_root = Path("artifacts/phase8_toy_lm_bridge/feasibility_004")
    output_root = Path("artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001")
    exact_argv = [
        "python",
        "scripts/phase8_sequence_feasibility.py",
        "diagnose-failure",
        "--device",
        "cuda:0",
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
    ]
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTHONPATH", ".")
    if original_cublas is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", original_cublas)
    torch.backends.mkldnn.enabled = True
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    calls: list[str] = []

    def forbidden_backend_configuration() -> None:
        calls.append("backend")
        raise AssertionError("backend configuration ran before original environment authorization")

    def forbidden_root_stage(**kwargs: object) -> None:
        calls.append("root")
        raise AssertionError("root/predecessor stage ran after rejected original environment")

    monkeypatch.setattr(sf, "diagnostic_kernel_argv", lambda: exact_argv)
    monkeypatch.setattr(sf, "configure_diagnostic_deterministic_backend", forbidden_backend_configuration)
    monkeypatch.setattr(sf, "validate_diagnostic_cli_root_contract", forbidden_root_stage)

    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        sf.validate_diagnostic_cli_contract(
            device="cuda:0",
            input_root=input_root,
            output_root=output_root,
            predecessor_diagnostic_roots=(),
        )

    assert calls == []
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == original_cublas
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.mkldnn.enabled is True
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


def test_diagnostic_flags_reject_each_false_deterministic_state() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    flags = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": ".",
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_tf32": False,
        "cudnn_tf32": False,
    }
    sf.validate_diagnostic_flags(flags)
    flips = {
        "torch_deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": True,
        "cuda_tf32": True,
        "cudnn_tf32": True,
        "PYTHONDONTWRITEBYTECODE": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":16:8",
        "PYTHONPATH": str(REPO_ROOT),
    }
    for key, value in flips.items():
        with pytest.raises(ValueError, match=key):
            sf.validate_diagnostic_flags({**flags, key: value})


def test_diagnostic_failed_partial_scope_rejects_impossible_progress_and_missing_identity() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    failure = "synthetic failure"
    completed = [{"name": "named_matrix_construction", "rows": len(sf.NAMED_DIAGNOSTIC_CELLS) * sf.EVAL_RECORDS_PER_FAMILY}]
    valid = [{
        "name": "named_cell",
        "seed": 0,
        "cell": sf.NAMED_DIAGNOSTIC_CELLS[0],
        "subphase": "named_generation",
        "completed_rows": sf.EVAL_RECORDS_PER_FAMILY,
        "error": failure,
    }]
    sf.validate_diagnostic_partial_scope(valid, completed_scope=completed, terminal_status="FAILED", failure=failure)
    for partial, match in (
        ([{**valid[0], "completed_rows": -1}], "out of range"),
        ([{**valid[0], "completed_rows": sf.EVAL_RECORDS_PER_FAMILY + 1}], "out of range"),
        ([{"name": "named_cell", "subphase": "named_generation", "completed_rows": 1, "error": failure}], "identity"),
        ([valid[0], valid[0]], "exactly one"),
        ([{**valid[0], "error": "different"}], "failure error"),
    ):
        with pytest.raises(ValueError, match=match):
            sf.validate_diagnostic_partial_scope(partial, completed_scope=completed, terminal_status="FAILED", failure=failure)


def test_diagnostic_replay_validators_reject_rows_not_from_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    model = torch.nn.Linear(1, 1)
    model.eval()
    monkeypatch.setattr(sf, "diagnostic_replay_device", lambda: torch.device("cuda:0"))
    monkeypatch.setattr(sf, "load_checkpoint_model", lambda checkpoint_path, model_size, device: model)
    monkeypatch.setattr(sf, "generate_array_rows", lambda *args, **kwargs: [{"row": "replayed"}])
    monkeypatch.setattr(sf, "teacher_forced_rows", lambda *args, split, **kwargs: ([{"split": split}], {"split": split}))
    with pytest.raises(ValueError, match="greedy rows"):
        sf.replay_array_artifacts(
            tmp_path / "checkpoint.pt",
            model_size="small",
            comparison_source="diagnostic_small_3000",
            steps=sf.DIAGNOSTIC_STEPS,
            seed=0,
            retained_greedy_rows=[{"row": "retained"}],
            retained_train_rows=[{"split": "train"}],
            retained_eval_rows=[{"split": "eval"}],
            retained_train_aggregate={"split": "train"},
            retained_eval_aggregate={"split": "eval"},
        )
    monkeypatch.setattr(sf, "generate_array_rows", lambda *args, **kwargs: [{"row": "retained"}])
    sf.replay_array_artifacts(
        tmp_path / "checkpoint.pt",
        model_size="small",
        comparison_source="diagnostic_small_3000",
        steps=sf.DIAGNOSTIC_STEPS,
        seed=0,
        retained_greedy_rows=[{"row": "retained"}],
        retained_train_rows=[{"split": "train"}],
        retained_eval_rows=[{"split": "eval"}],
        retained_train_aggregate={"split": "train"},
        retained_eval_aggregate={"split": "eval"},
    )
    monkeypatch.setattr(sf, "generate_named_diagnostic_rows", lambda *args, **kwargs: [{"row": "replayed"}])
    with pytest.raises(ValueError, match="Named retained rows"):
        sf.replay_named_generation_artifact(tmp_path / "checkpoint.pt", matrix_rows=[], retained_rows=[{"row": "retained"}])


def test_diagnostic_terminal_markers_are_removed_after_post_terminal_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    input_root = tmp_path / "feasibility_004"
    output_root = tmp_path / "feasibility_diagnostic_001"
    source_snapshot = sf.SourceSnapshot(commit="b" * 40, status_lines=(), ignored_inputs=())
    preflight = {
        "input_root": str(input_root),
        "predecessors": [],
        "diagnostic_lineage": [],
        "repair_transition": None,
        "handoff": {},
        "input_root_binding": {},
        "record_hashes": {},
        "core_blobs": {},
        "environment": {},
    }
    monkeypatch.setattr(sf, "require_diagnostic_real_main_context", lambda: None)
    monkeypatch.setattr(sf, "validate_diagnostic_cli_contract", lambda **kwargs: None)
    monkeypatch.setattr(sf, "capture_diagnostic_source_provenance", lambda *args: source_snapshot)
    monkeypatch.setattr(sf, "diagnostic_preflight_bindings", lambda *args: preflight)
    monkeypatch.setattr(sf, "validate_diagnostic_input_root", lambda *args, **kwargs: {})
    monkeypatch.setattr(sf, "SEEDS", ())
    monkeypatch.setattr(sf, "NAMED_DIAGNOSTIC_CELLS", ())
    monkeypatch.setattr(sf, "build_named_diagnostic_matrix", lambda: {})
    monkeypatch.setattr(sf, "validate_diagnostic_array_training_plan", lambda plan: None)
    monkeypatch.setattr(sf, "diagnostic_array_training_plan", lambda: [])
    def publication_stub(*args: object, **kwargs: object) -> None:
        callback = kwargs.get("final_callback")
        if callback is not None:
            callback()

    monkeypatch.setattr(sf, "publish_diagnostic_root_or_leave_incomplete", publication_stub)

    calls = {"source": 0}

    def source_changed_after_done(snapshot: object, *, active_output_root: Path | None = None) -> None:
        calls["source"] += 1
        if calls["source"] == 2:
            raise sf.SourceChangedError("changed after terminal")

    monkeypatch.setattr(sf, "verify_diagnostic_preflight_bindings", lambda *args: None)
    monkeypatch.setattr(sf, "verify_source_unchanged", source_changed_after_done)
    with pytest.raises(sf.DiagnosticPublicationError):
        sf.run_diagnostic_failure(device="cuda:0", input_root=input_root, output_root=output_root)
    assert not (output_root.with_name(output_root.name + ".tmp") / "DONE.json").exists()
    assert not (output_root.with_name(output_root.name + ".tmp") / "FAILED.json").exists()

    second_output = tmp_path / "feasibility_diagnostic_002"
    calls = {"preflight": 0}

    def preflight_changed_after_done(*args: object) -> None:
        calls["preflight"] += 1
        if calls["preflight"] == 2:
            raise ValueError("changed after terminal")

    monkeypatch.setattr(sf, "verify_diagnostic_preflight_bindings", preflight_changed_after_done)
    monkeypatch.setattr(sf, "verify_source_unchanged", lambda *args, **kwargs: None)
    with pytest.raises(sf.DiagnosticPublicationError):
        sf.run_diagnostic_failure(device="cuda:0", input_root=input_root, output_root=second_output)
    assert not (second_output.with_name(second_output.name + ".tmp") / "DONE.json").exists()
    assert not (second_output.with_name(second_output.name + ".tmp") / "FAILED.json").exists()


def test_diagnostic_repair_transition_binds_predecessor_current_and_allowed_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    superseded = "b" * 40
    repaired = "c" * 40
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(sf, "validate_diagnostic_repair_diff_confined", lambda old, new: calls.append((old, new)))
    lineage = [{"source_commit": superseded, "failure_classification": "diagnostic_implementation_defect"}]
    transition = sf.diagnostic_repair_transition(lineage, repaired_source_commit=repaired)
    assert transition == {
        "superseded_source_commit": superseded,
        "repaired_source_commit": repaired,
        "allowed_diff_paths": sorted(sf.DIAGNOSTIC_REPAIR_ALLOWED_DIFF_PATHS),
    }
    sf.validate_diagnostic_repair_transition(transition, lineage=lineage, source_commit=repaired)
    assert calls == [(superseded, repaired), (superseded, repaired)]
    with pytest.raises(ValueError, match="superseded"):
        sf.validate_diagnostic_repair_transition({**transition, "superseded_source_commit": "d" * 40}, lineage=lineage, source_commit=repaired)
    with pytest.raises(ValueError, match="repaired"):
        sf.validate_diagnostic_repair_transition({**transition, "repaired_source_commit": "d" * 40}, lineage=lineage, source_commit=repaired)
    sf.validate_diagnostic_repair_transition(None, lineage=[{"source_commit": superseded, "failure_classification": "transient_infrastructure"}], source_commit=superseded)
    with pytest.raises(ValueError, match="repair_transition=null"):
        sf.validate_diagnostic_repair_transition(transition, lineage=[], source_commit=repaired)


def test_diagnostic_frozen_configuration_expands_protocol_and_training_constants() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    config = sf.frozen_diagnostic_configuration()
    assert config["accepted_protocol_commit"] == sf.DIAGNOSTIC_ACCEPTED_PROTOCOL_COMMIT
    assert config["handoff_blob"] == sf.DIAGNOSTIC_HANDOFF_BLOB
    assert config["tokenizer"]["eos_id"] == EOS_ID
    assert config["batch_size"] == sf.BATCH_SIZE
    assert config["model_configs"]["small"]["parameter_count"] == sf.HISTORICAL_PARAMETER_COUNTS["small"] == build_historical_model("small").parameter_count
    assert config["model_configs"]["medium"]["parameter_count"] == sf.HISTORICAL_PARAMETER_COUNTS["medium"] == build_historical_model("medium").parameter_count
    assert config["optimizer"]["class"] == "torch.optim.AdamW"
    assert config["optimizer"]["learning_rate"] == 0.0003
    assert config["validation"] == {
        "authorized_new_training_runs": 3,
        "validation_training_or_replay_runs": 0,
        "trajectory_evidence_source": "in_run_checkpoint_metadata_and_metrics",
        "selection_evidence": False,
    }


def _d3_env(sf: object) -> dict[str, str]:
    return dict(sf.POSTMORTEM_EXECVE_ENVIRONMENT)


class _D3FakeClosedRepositoryLoader:
    __slots__ = ("_sources", "_sealed")

    def __init__(self) -> None:
        object.__setattr__(self, "_sources", MappingProxyType({"capability_certificate_lab": ("capability_certificate_lab/__init__.py", b"", True)}))
        object.__setattr__(self, "_sealed", True)

    @property
    def sources(self) -> MappingProxyType:
        return self._sources

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise RuntimeError("sealed")
        object.__setattr__(self, name, value)

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> object:
        return None

    def create_module(self, spec: object) -> object:
        return None

    def exec_module(self, module: object) -> None:
        return None


class _D3FakeRepositoryAuthority:
    __slots__ = ("_accepted_commit", "_runner_blob", "_untracked", "_ignored", "_proposal_binding", "_sealed")

    def __init__(
        self,
        *,
        accepted_commit: str,
        runner_blob: str,
        untracked: tuple[str, ...],
        ignored: tuple[str, ...],
        proposal_binding: tuple[str, str, str],
    ) -> None:
        object.__setattr__(self, "_accepted_commit", accepted_commit)
        object.__setattr__(self, "_runner_blob", runner_blob)
        object.__setattr__(self, "_untracked", untracked)
        object.__setattr__(self, "_ignored", ignored)
        object.__setattr__(self, "_proposal_binding", proposal_binding)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise RuntimeError("sealed")
        object.__setattr__(self, name, value)

    def __call__(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self._untracked, self._ignored


def _d3_install_fake_verifier(
    sf: object,
    monkeypatch: pytest.MonkeyPatch,
    *,
    accepted_commit: str = "d" * 40,
    runner_blob: str = "e" * 40,
    untracked: tuple[str, ...] = (),
    ignored: tuple[str, ...] = (),
    authority: object | None = None,
    loader: object | None = None,
) -> object:
    proposal_binding = (sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT, sf.POSTMORTEM_PROPOSAL_PATH, sf.POSTMORTEM_PROPOSAL_BLOB)
    fake_authority = authority or _D3FakeRepositoryAuthority(
        accepted_commit=accepted_commit,
        runner_blob=runner_blob,
        untracked=untracked,
        ignored=ignored,
        proposal_binding=proposal_binding,
    )
    monkeypatch.setattr(sf, "_POSTMORTEM_RETAINED_VERIFIER_BINDING", None)
    monkeypatch.setitem(sf.__dict__, "__phase8_verifier_sha256__", sf.POSTMORTEM_VERIFIER_SHA256)
    monkeypatch.setitem(sf.__dict__, "__phase8_accepted_implementation_commit__", accepted_commit)
    monkeypatch.setitem(sf.__dict__, "__phase8_repository_loader__", loader or _D3FakeClosedRepositoryLoader())
    monkeypatch.setitem(sf.__dict__, "__phase8_verify_repository_unchanged__", fake_authority)
    monkeypatch.setitem(sf.__dict__, "__phase8_runner_blob__", runner_blob)
    monkeypatch.setitem(sf.__dict__, "__phase8_proposal_binding__", proposal_binding)
    monkeypatch.setitem(sf.__dict__, "__phase8_authority_token__", object())
    return fake_authority


def _d3_synthetic_teacher_rows(
    sf: object,
    records: tuple[object, ...],
    *,
    split: str,
    family: str,
    model_size: str,
    seed: int,
    checkpoint_path: str,
    checkpoint_sha256: str,
) -> list[dict[str, object]]:
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "schema_version": sf.POSTMORTEM_SCHEMA_VERSION,
                "family": family,
                "model_size": model_size,
                "seed": seed,
                "split": split,
                "record_index": record.index,
                "template_id": record.template_id,
                "operand_id": record.operand_id,
                "batch_index": index // sf.BATCH_SIZE,
                "row_within_batch": index % sf.BATCH_SIZE,
                "selected_token_count": 1,
                "correct_token_count": 1,
                "sequence_exact": True,
                "nll_numerator_hex": (0.0).hex(),
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha256,
            }
        )
    return rows


def _d3_build_synthetic_postmortem_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, Path, Path]:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    artifact_parent = tmp_path / "artifacts" / "phase8_toy_lm_bridge"
    input_root = artifact_parent / "feasibility_005"
    output_root = artifact_parent / "feasibility_postmortem_001"
    input_root.mkdir(parents=True)
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", artifact_parent)
    monkeypatch.setattr(sf, "POSTMORTEM_REQUIRED_INPUT_ROOT", str(input_root))
    monkeypatch.setattr(sf, "POSTMORTEM_REQUIRED_OUTPUT_ROOT", str(output_root))
    monkeypatch.setattr(
        sf,
        "POSTMORTEM_INPUT_CHECKSUMS",
        {
            "manifest.json": "a" * 64,
            "summary.json": "b" * 64,
            "FAILED.json": "c" * 64,
        },
    )
    tokenizer = ByteTokenizer()
    monkeypatch.setattr(sf, "validate_feasibility_records", lambda records: None)
    records = sf.grouped_records()
    monkeypatch.setattr(sf, "grouped_records", lambda: records)
    cells = _passing_feasibility_cells()
    for cell in cells:
        family = str(cell["family"])
        generation_path = input_root / str(cell["generations_path"])
        checkpoint_path = input_root / str(cell["checkpoint_path"])
        generation_path.parent.mkdir(parents=True, exist_ok=True)
        generation_path.write_text(
            "\n".join(
                json.dumps(_generation_row(record, tokenizer, exact_match=index < int(cell["exact_matches"])), sort_keys=True)
                for index, record in enumerate(records[family]["eval"])
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoint_path.write_bytes(f"checkpoint {family} {cell['model_size']} {cell['seed']}".encode("utf-8"))
    (input_root / "summary.json").write_text("{}\n", encoding="utf-8")
    input_binding = {
        "root": str(input_root),
        "source_commit": sf.POSTMORTEM_INPUT_SOURCE_COMMIT,
        "manifest_path": "manifest.json",
        "manifest_sha256": sf.POSTMORTEM_INPUT_CHECKSUMS["manifest.json"],
        "summary_path": "summary.json",
        "summary_sha256": sf.POSTMORTEM_INPUT_CHECKSUMS["summary.json"],
        "terminal_path": "FAILED.json",
        "terminal_sha256": sf.POSTMORTEM_INPUT_CHECKSUMS["FAILED.json"],
        "configuration": sf.frozen_configuration(),
        "record_hashes": dict(sf.FEASIBILITY_RECORD_HASHES),
        "file_inventory": sf.inventory(input_root),
        "cells": cells,
    }
    proposal_binding = {
        "commit": sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
        "path": sf.POSTMORTEM_PROPOSAL_PATH,
        "blob": sf.POSTMORTEM_PROPOSAL_BLOB,
    }
    accepted_implementation_commit = _test_source_commit()
    implementation_binding = {
        "commit": accepted_implementation_commit,
        "runner_path": sf.POSTMORTEM_RUNNER_PATH,
        "runner_blob": subprocess.run(
            ["git", "rev-parse", "HEAD:scripts/phase8_sequence_feasibility.py"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            cwd=REPO_ROOT,
        ).stdout.strip(),
    }
    teacher_rows: list[dict[str, object]] = []
    taxonomy_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    for cell in cells:
        family = str(cell["family"])
        model_size = str(cell["model_size"])
        seed = int(cell["seed"])
        checkpoint_path = str(cell["checkpoint_path"])
        generation_path = str(cell["generations_path"])
        checkpoint_sha = sf.postmortem_file_sha_from_input_binding(input_binding, checkpoint_path)
        generation_sha = sf.postmortem_file_sha_from_input_binding(input_binding, generation_path)
        generation_rows = sf.validate_generation_artifact(input_root / generation_path, cell)
        train_rows = _d3_synthetic_teacher_rows(
            sf,
            records[family]["train"],
            split="train",
            family=family,
            model_size=model_size,
            seed=seed,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
        )
        eval_rows = _d3_synthetic_teacher_rows(
            sf,
            records[family]["eval"],
            split="eval",
            family=family,
            model_size=model_size,
            seed=seed,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
        )
        teacher_rows.extend(train_rows)
        teacher_rows.extend(eval_rows)
        for record, generation_row in zip(records[family]["eval"], generation_rows, strict=True):
            taxonomy_rows.append(
                sf.postmortem_taxonomy_row(
                    record,
                    generation_row,
                    family=family,
                    model_size=model_size,
                    seed=seed,
                    generation_path=generation_path,
                    generation_sha256=generation_sha,
                )
            )
        checkpoint = {
            "metadata": {
                "family": family,
                "model_size": model_size,
                "seed": seed,
                "training_steps": sf.TRAINING_STEPS,
                "training_loss": 1.25,
                "training_accuracy": 0.5,
            }
        }
        cell_rows.append(
            sf.postmortem_cell_metrics_row(
                cell=cell,
                input_binding=input_binding,
                checkpoint=checkpoint,
                train_teacher_rows=train_rows,
                eval_teacher_rows=eval_rows,
                generation_rows=generation_rows,
            )
        )
    output_root.mkdir()
    sf.write_postmortem_jsonl(output_root / "teacher_forced_rows.jsonl", teacher_rows)
    sf.write_postmortem_jsonl(output_root / "cell_metrics.jsonl", cell_rows)
    sf.write_postmortem_jsonl(output_root / "error_taxonomy.jsonl", taxonomy_rows)
    sf.write_postmortem_json(
        output_root / "summary.json",
        sf.build_postmortem_summary(
            input_binding=input_binding,
            proposal_binding=proposal_binding,
            implementation_binding=implementation_binding,
            cell_rows=cell_rows,
            taxonomy_rows=taxonomy_rows,
        ),
    )
    sf.write_postmortem_json(
        output_root / "manifest.json",
        sf.build_postmortem_manifest(
            output_root,
            input_binding=input_binding,
            proposal_binding=proposal_binding,
            implementation_binding=implementation_binding,
            exact_command=sf.postmortem_exact_command(
                input_root,
                output_root,
                sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
                accepted_implementation_commit,
            ),
            environment=dict(sf.FEASIBILITY_REQUIRED_RUNTIME_ENV),
            deterministic_flags=_current_deterministic_flags(),
            rng_state_contract={key: True for key in sf.POSTMORTEM_RNG_CONTRACT_KEYS},
        ),
    )
    sf.write_postmortem_done(output_root, input_binding=input_binding)
    return sf, input_root, output_root


def _d3_rebuild_postmortem_manifest_done(sf: object, output_root: Path) -> None:
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_inventory"] = sf.postmortem_output_inventory(output_root)
    sf.write_postmortem_json(manifest_path, manifest)
    sf.write_postmortem_done(output_root, input_binding=summary["input_binding"])


def _d3_expected_publication_context(sf: object, output_root: Path) -> dict[str, object]:
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    return {
        "expected_input_binding": manifest["input_binding"],
        "expected_proposal_binding": manifest["proposal_binding"],
        "expected_implementation_binding": manifest["implementation_binding"],
        "expected_source_snapshot": sf.SourceSnapshot(
            commit=manifest["implementation_binding"]["commit"],
            status_lines=(),
            ignored_inputs=(),
            runner_blob=manifest["implementation_binding"]["runner_blob"],
        ),
        "expected_teacher_rows": sf.read_canonical_postmortem_jsonl(output_root / "teacher_forced_rows.jsonl", "test.teacher"),
        "expected_cell_rows": sf.read_canonical_postmortem_jsonl(output_root / "cell_metrics.jsonl", "test.cells"),
        "expected_taxonomy_rows": sf.read_canonical_postmortem_jsonl(output_root / "error_taxonomy.jsonl", "test.taxonomy"),
    }


def _d3_minimal_publication_context(sf: object) -> dict[str, object]:
    return {
        "expected_input_binding": {},
        "expected_proposal_binding": {},
        "expected_implementation_binding": {},
        "expected_source_snapshot": sf.SourceSnapshot(commit="a" * 40, status_lines=(), ignored_inputs=()),
        "expected_teacher_rows": [],
        "expected_cell_rows": [],
        "expected_taxonomy_rows": [],
    }


def _d3_assert_no_canonical_postmortem_terminal_markers(root: Path) -> None:
    assert not (root / "DONE.json").exists()
    assert not (root / "FAILED.json").exists()


def _d3_build_faithful_input_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, Path, Path]:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    artifact_parent = tmp_path / "artifacts" / "phase8_toy_lm_bridge"
    input_root = artifact_parent / "feasibility_005"
    output_root = artifact_parent / "feasibility_postmortem_001"
    input_root.mkdir(parents=True)
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", artifact_parent)
    monkeypatch.setattr(sf, "FEASIBILITY_REQUIRED_ROOT", str(input_root))
    monkeypatch.setattr(sf, "POSTMORTEM_REQUIRED_INPUT_ROOT", str(input_root))
    monkeypatch.setattr(sf, "POSTMORTEM_REQUIRED_OUTPUT_ROOT", str(output_root))
    monkeypatch.setattr(sf, "FEASIBILITY_REQUIRED_PREDECESSOR_ROOTS", ())
    monkeypatch.setattr(sf, "shallow_decision_diagnostic_paths", lambda root: set())
    monkeypatch.setattr(sf, "validate_root_manifest_lineage", lambda *args, **kwargs: None)
    monkeypatch.setattr(sf, "validate_feasibility_records", lambda records: None)
    records = sf.grouped_records()
    monkeypatch.setattr(sf, "grouped_records", lambda: records)
    tokenizer = ByteTokenizer()
    cells = _passing_feasibility_cells()
    for cell in cells:
        family = str(cell["family"])
        model_size = str(cell["model_size"])
        seed = int(cell["seed"])
        cell_dir = input_root / f"{family}__{model_size}__seed{seed}"
        cell_dir.mkdir(parents=True)
        (cell_dir / "generations.jsonl").write_text(
            "\n".join(
                json.dumps(_generation_row(record, tokenizer, exact_match=index < int(cell["exact_matches"])), sort_keys=True)
                for index, record in enumerate(records[family]["eval"])
            )
            + "\n",
            encoding="utf-8",
        )
        save_checkpoint(
            str(cell_dir / "checkpoint_step1500.pt"),
            build_model(model_size),
            metadata={
                "family": family,
                "model_size": model_size,
                "seed": seed,
                "training_steps": sf.TRAINING_STEPS,
                "training_loss": 1.25,
                "training_accuracy": 0.5,
            },
        )
    exact_command = sf.feasibility_exact_command(input_root, (), Path(sf.FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT))
    sf.write_terminal(
        input_root,
        "FAILED",
        cells,
        (),
        (),
        failure="synthetic accepted feasibility_005 failure",
        source_snapshot=sf.SourceSnapshot(commit=sf.POSTMORTEM_INPUT_SOURCE_COMMIT, status_lines=(), ignored_inputs=()),
        decision_diagnostic=sf.DECISION_DIAGNOSTIC_ROOT_BINDING,
        exact_command=exact_command,
        deterministic_flags=_current_deterministic_flags(),
        record_hashes=dict(sf.FEASIBILITY_RECORD_HASHES),
    )
    manifest_path = input_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"] = dict(sf.FEASIBILITY_REQUIRED_RUNTIME_ENV)
    sf.write_json(manifest_path, manifest)
    _rewrite_terminal_manifest_sha(input_root, "FAILED")
    monkeypatch.setattr(
        sf,
        "POSTMORTEM_INPUT_CHECKSUMS",
        {
            "manifest.json": sf.file_sha256(input_root / "manifest.json"),
            "summary.json": sf.file_sha256(input_root / "summary.json"),
            "FAILED.json": sf.file_sha256(input_root / "FAILED.json"),
        },
    )
    return sf, input_root, output_root


def _d3_refresh_faithful_input_checksums(sf: object, input_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = input_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_inventory"] = sf.inventory(input_root)
    sf.write_json(manifest_path, manifest)
    _rewrite_terminal_manifest_sha(input_root, "FAILED")
    monkeypatch.setattr(
        sf,
        "POSTMORTEM_INPUT_CHECKSUMS",
        {
            "manifest.json": sf.file_sha256(input_root / "manifest.json"),
            "summary.json": sf.file_sha256(input_root / "summary.json"),
            "FAILED.json": sf.file_sha256(input_root / "FAILED.json"),
        },
    )


def test_d3_postmortem_cli_contract_and_import_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    input_root = Path(sf.POSTMORTEM_REQUIRED_INPUT_ROOT)
    output_root = Path(sf.POSTMORTEM_REQUIRED_OUTPUT_ROOT)
    accepted_implementation_commit = "d" * 40
    verifier_source = sf.postmortem_verifier_source_text()
    verifier_path = REPO_ROOT / sf.POSTMORTEM_VERIFIER_PATH
    verifier_bytes = verifier_path.read_bytes()
    assert len(verifier_bytes) == sf.POSTMORTEM_VERIFIER_BYTE_COUNT == 39455
    assert sha256(verifier_bytes).hexdigest() == sf.POSTMORTEM_VERIFIER_SHA256
    assert subprocess.run(
        ["git", "hash-object", sf.POSTMORTEM_VERIFIER_PATH],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        cwd=REPO_ROOT,
    ).stdout.strip() == sf.POSTMORTEM_VERIFIER_BLOB
    exact_argv = sf.postmortem_exact_argv(
        input_root,
        output_root,
        sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
        accepted_implementation_commit,
        verifier_source=verifier_source,
    )
    assert len(exact_argv) == 24
    assert exact_argv[0:5] == [sf.POSTMORTEM_EXECUTABLE, "-I", "-B", "-S", "-c"]
    assert exact_argv[5] == verifier_source
    assert exact_argv[7] == sf.POSTMORTEM_VERIFIER_SHA256
    assert exact_argv[9] == exact_argv[23] == accepted_implementation_commit
    assert exact_argv[21] == sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT
    exact_command = sf.postmortem_exact_command(
        input_root,
        output_root,
        sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
        accepted_implementation_commit,
        argv=exact_argv,
    )
    assert tuple(exact_command) == ("executable", "cwd", "environment", "argv")
    assert exact_command["environment"] == sf.POSTMORTEM_EXECVE_ENVIRONMENT
    assert exact_command["environment"] != sf.FEASIBILITY_REQUIRED_RUNTIME_ENV
    assert exact_command["argv"] == exact_argv
    decoded_command = json.loads(json.dumps(exact_command))
    assert decoded_command["argv"][5].encode("utf-8") == verifier_source.encode("utf-8")
    sf.validate_postmortem_exact_command_object(
        decoded_command,
        input_root=input_root,
        output_root=output_root,
        accepted_proposal_commit=sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
        accepted_implementation_commit=accepted_implementation_commit,
        runtime_environment=dict(sf.FEASIBILITY_REQUIRED_RUNTIME_ENV),
    )
    for index in range(13, 24):
        mutated = list(exact_argv)
        mutated[index] = "e" * 40 if index in {9, 23} else f"{mutated[index]}-mutated"
        with pytest.raises(ValueError, match="argv|implementation|proposal|verifier"):
            sf.validate_postmortem_exact_argv(
                mutated,
                input_root=input_root,
                output_root=output_root,
                accepted_proposal_commit=sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
                accepted_implementation_commit=accepted_implementation_commit,
            )
    with pytest.raises(ValueError, match="24"):
        sf.validate_postmortem_exact_argv(
            [*exact_argv, "--extra"],
            input_root=input_root,
            output_root=output_root,
            accepted_proposal_commit=sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
            accepted_implementation_commit=accepted_implementation_commit,
        )
    reordered = list(exact_argv)
    reordered[13], reordered[14] = reordered[14], reordered[13]
    with pytest.raises(ValueError, match="argv"):
        sf.validate_postmortem_exact_argv(
            reordered,
            input_root=input_root,
            output_root=output_root,
            accepted_proposal_commit=sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
            accepted_implementation_commit=accepted_implementation_commit,
        )

    def fake_verifier_context(**kwargs: object) -> None:
        assert kwargs["accepted_implementation_commit"] == accepted_implementation_commit
        assert kwargs["environ"] == _d3_env(sf)

    def assert_expected_new_root(candidate: Path) -> None:
        assert candidate == output_root
        assert candidate.as_posix() == sf.POSTMORTEM_REQUIRED_OUTPUT_ROOT

    original_verifier_context = sf.require_postmortem_verifier_context
    monkeypatch.setattr(sf, "require_postmortem_verifier_context", fake_verifier_context)
    monkeypatch.setattr(sf, "validate_new_postmortem_root", assert_expected_new_root)
    sf.validate_postmortem_cli_contract(
        device="cuda:0",
        input_root=input_root,
        output_root=output_root,
        accepted_proposal_commit=sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
        accepted_implementation_commit=accepted_implementation_commit,
        environ=_d3_env(sf),
    )
    with pytest.raises(ValueError, match="environment PYTHONPATH"):
        monkeypatch.setattr(sf, "require_postmortem_verifier_context", lambda **kwargs: (_ for _ in ()).throw(ValueError("environment PYTHONPATH mismatch")))
        sf.validate_postmortem_cli_contract(
            device="cuda:0",
            input_root=input_root,
            output_root=output_root,
            accepted_proposal_commit=sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
            accepted_implementation_commit=accepted_implementation_commit,
            environ={**_d3_env(sf), "PYTHONPATH": str(REPO_ROOT)},
        )
    monkeypatch.setattr(sf, "require_postmortem_verifier_context", original_verifier_context)
    with pytest.raises(ValueError, match="postmortem-failure"):
        sf.main(
            [
                "postmortem-failure",
                "--device",
                "cuda:0",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--accepted-proposal-commit",
                sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
                "--accepted-implementation-commit",
                accepted_implementation_commit,
            ]
        )
    with pytest.raises(ValueError, match="real __main__"):
        sf.run_postmortem_failure(
            device="cuda:0",
            input_root=input_root,
            output_root=output_root,
            accepted_proposal_commit=sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
            accepted_implementation_commit=accepted_implementation_commit,
            environ=_d3_env(sf),
        )
    with pytest.raises(ValueError, match="feasibility_postmortem_001"):
        sf.validate_postmortem_argument_contract(
            device="cuda:0",
            input_root=input_root,
            output_root=Path("artifacts/phase8_toy_lm_bridge/feasibility_postmortem_002"),
            accepted_proposal_commit=sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
            accepted_implementation_commit=accepted_implementation_commit,
        )
    with pytest.raises(ValueError, match="accepted proposal"):
        sf.validate_postmortem_argument_contract(
            device="cuda:0",
            input_root=input_root,
            output_root=output_root,
            accepted_proposal_commit="0" * 40,
            accepted_implementation_commit=accepted_implementation_commit,
        )


def _d3_exact_verifier_namespace(sf: object) -> dict[str, object]:
    verifier_path = REPO_ROOT / sf.POSTMORTEM_VERIFIER_PATH
    verifier_bytes = verifier_path.read_bytes()
    assert len(verifier_bytes) == 39455
    assert sha256(verifier_bytes).hexdigest() == sf.POSTMORTEM_VERIFIER_SHA256
    namespace: dict[str, object] = {"__name__": "phase8_verifier_non_main_harness"}
    exec(compile(verifier_bytes, str(verifier_path), "exec"), namespace, namespace)
    assert "PinnedGitContext" in namespace
    assert "main" in namespace
    return namespace


def _d3_git(repo: Path, *args: str) -> str:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["LC_ALL"] = "C"
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _d3_build_exact_verifier_repo(repo: Path, sf: object) -> tuple[Path, str, str]:
    repo.mkdir()
    _d3_git(repo, "init")
    _d3_git(repo, "remote", "add", "origin", "git@github.com:johnxye/CapKnow.git")
    (repo / "scripts").mkdir()
    (repo / "scripts/phase8_sequence_feasibility.py").write_text("RUNNER_SENTINEL = 'captured'\n", encoding="utf-8")
    package_root = repo / "capability_certificate_lab"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("VALUE = 'captured-package'\n", encoding="utf-8")
    (package_root / "child.py").write_text("from . import VALUE\nCHILD = VALUE\n", encoding="utf-8")
    proposal_path = repo / sf.POSTMORTEM_PROPOSAL_PATH
    proposal_path.parent.mkdir(parents=True)
    proposal_path.write_text("accepted proposal\n", encoding="utf-8")
    _d3_git(repo, "add", ".")
    _d3_git(repo, "-c", "user.name=Phase8 Test", "-c", "user.email=phase8@example.invalid", "commit", "-m", "fixture")
    (repo / ".git/packed-refs").touch()
    head = _d3_git(repo, "rev-parse", "HEAD")
    proposal_blob = _d3_git(repo, "rev-parse", f"HEAD:{sf.POSTMORTEM_PROPOSAL_PATH}")
    return repo, head, proposal_blob


def _d3_pin_exact_verifier_repo(
    namespace: dict[str, object],
    repo: Path,
    head: str,
    proposal_blob: str,
    sf: object,
) -> object:
    namespace["REPO_ROOT"] = repo
    namespace["GIT_DIR"] = repo / ".git"
    namespace["RUNNER_PATH"] = "scripts/phase8_sequence_feasibility.py"
    namespace["PROPOSAL_COMMIT"] = head
    namespace["PROPOSAL_BLOB"] = proposal_blob
    namespace["PINNED_GIT"] = None
    pinned = namespace["PinnedGitContext"]()
    namespace["PINNED_GIT"] = pinned
    return pinned


def _d3_close_exact_pinned(pinned: object) -> None:
    for attr in ("repo_fd", "git_fd", "objects_fd", "refs_fd", "info_fd", "head_fd", "index_fd", "config_fd", "exclude_fd", "packed_refs_fd"):
        fd = getattr(pinned, attr, None)
        if isinstance(fd, int):
            try:
                os.close(fd)
            except OSError:
                pass


def test_d3_postmortem_exact_verifier_bytes_real_git_authority_and_closed_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    namespace = _d3_exact_verifier_namespace(sf)

    repo = tmp_path / "repo"
    repo, head, proposal_blob = _d3_build_exact_verifier_repo(repo, sf)
    package_root = repo / "capability_certificate_lab"
    pinned = _d3_pin_exact_verifier_repo(namespace, repo, head, proposal_blob, sf)
    try:
        captured, modes, untracked, ignored = namespace["authenticate_repository"](head)
        assert untracked == ()
        assert ignored == ()
        assert captured["scripts/phase8_sequence_feasibility.py"] == b"RUNNER_SENTINEL = 'captured'\n"
        assert namespace["authenticate_proposal"]() == (head, sf.POSTMORTEM_PROPOSAL_PATH, proposal_blob)
        for bad_args in (
            ("cat-file", "--filters", proposal_blob),
            ("cat-file", "--textconv", proposal_blob),
            ("cat-file", "blob", "--batch"),
            ("status", "--porcelain"),
        ):
            with pytest.raises(SystemExit, match="outside the exact allowlist"):
                namespace["git_run"](*bad_args)

        authority = namespace["RepositoryAuthority"](
            head,
            MappingProxyType(dict(captured)),
            MappingProxyType(dict(modes)),
            (head, sf.POSTMORTEM_PROPOSAL_PATH, proposal_blob),
        )
        assert authority() == ((), ())

        loader = namespace["ClosedRepositoryLoader"](captured)
        assert not hasattr(loader, "__dict__")
        with pytest.raises(SystemExit, match="sealed"):
            loader.find_spec = lambda *_args, **_kwargs: None

        saved_modules = {name: module for name, module in sys.modules.items() if name == "capability_certificate_lab" or name.startswith("capability_certificate_lab.")}
        for name in list(saved_modules):
            sys.modules.pop(name, None)
        shadow_dir = tmp_path / "shadow_site"
        (shadow_dir / "capability_certificate_lab").mkdir(parents=True)
        (shadow_dir / "capability_certificate_lab/__init__.py").write_text("VALUE = 'shadow'\n", encoding="utf-8")
        (shadow_dir / "capability_certificate_lab/child.py").write_text("CHILD = 'shadow'\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(shadow_dir))
        sys.meta_path.insert(0, loader)
        try:
            imported_child = importlib.import_module("capability_certificate_lab.child")
            assert imported_child.CHILD == "captured-package"
            (package_root / "child.py").write_text("CHILD = 'mutated-worktree'\n", encoding="utf-8")
            sys.modules.pop("capability_certificate_lab.child", None)
            imported_child = importlib.import_module("capability_certificate_lab.child")
            assert imported_child.CHILD == "captured-package"
            with pytest.raises(ImportError, match="outside the closed verifier set"):
                importlib.import_module("capability_certificate_lab.missing")
        finally:
            sys.meta_path.remove(loader)
            for name in list(sys.modules):
                if name == "capability_certificate_lab" or name.startswith("capability_certificate_lab."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)

        with pytest.raises(SystemExit, match="worktree bytes|captured repository sources"):
            authority()
    finally:
        _d3_close_exact_pinned(pinned)


def test_d3_postmortem_exact_verifier_rejects_git_env_loader_attack_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    namespace = _d3_exact_verifier_namespace(sf)
    base_repo, head, proposal_blob = _d3_build_exact_verifier_repo(tmp_path / "base_repo", sf)
    sentinel = tmp_path / "sentinel_touched"
    sentinel_script = tmp_path / "sentinel.sh"
    sentinel_script.write_text(f"#!/bin/sh\ntouch {sentinel}\nexit 0\n", encoding="utf-8")
    sentinel_script.chmod(0o755)
    global_config = tmp_path / "global_gitconfig"
    global_config.write_text(
        "\n".join(
            (
                "[core]",
                f"\tfsmonitor = {sentinel_script}",
                f"\thooksPath = {tmp_path / 'global-hooks'}",
                f"\tpager = {sentinel_script}",
                '[diff "evil"]',
                f"\ttextconv = {sentinel_script}",
                '[filter "evil"]',
                f"\tsmudge = {sentinel_script}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "inherited-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "inherited-worktree"))
    monkeypatch.setenv("GIT_COMMON_DIR", str(tmp_path / "inherited-common-dir"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "inherited-index"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path / "inherited-objects"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", str(sentinel_script))
    monkeypatch.setenv("GIT_PAGER", str(sentinel_script))
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "0")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "shadow-site"))
    monkeypatch.setenv("PYTHONSTARTUP", str(tmp_path / "startup.py"))
    pinned = _d3_pin_exact_verifier_repo(namespace, base_repo, head, proposal_blob, sf)
    try:
        captured, modes, untracked, ignored = namespace["authenticate_repository"](head)
        assert untracked == ()
        assert ignored == ()
        authority = namespace["RepositoryAuthority"](
            head,
            MappingProxyType(dict(captured)),
            MappingProxyType(dict(modes)),
            namespace["authenticate_proposal"](),
        )
        assert authority() == ((), ())
    finally:
        _d3_close_exact_pinned(pinned)
        namespace["PINNED_GIT"] = None
    assert not sentinel.exists()

    def config_payload(repo: Path) -> str:
        return (repo / ".git/config").read_text(encoding="utf-8")

    def replace_config(repo: Path, payload: str) -> None:
        (repo / ".git/config").write_text(payload, encoding="utf-8")

    def add_core_config(repo: Path, *lines: str) -> None:
        payload = config_payload(repo)
        insertion = "".join(f"\t{line}\n" for line in lines)
        marker = "\tlogallrefupdates = true\n"
        assert marker in payload
        replace_config(repo, payload.replace(marker, marker + insertion, 1))

    def add_remote_config(repo: Path, *lines: str) -> None:
        payload = config_payload(repo)
        marker = "\tfetch = +refs/heads/*:refs/remotes/origin/*\n"
        assert marker in payload
        replace_config(repo, payload.replace(marker, marker + "".join(f"\t{line}\n" for line in lines), 1))

    def expect_repository_reject(name: str, mutator: Callable[[Path], None], match: str) -> None:
        attack_repo = tmp_path / f"attack_{name}"
        shutil.copytree(base_repo, attack_repo, symlinks=True)
        mutator(attack_repo)
        attack_pinned: object | None = None
        try:
            with pytest.raises(SystemExit, match=match):
                attack_pinned = _d3_pin_exact_verifier_repo(namespace, attack_repo, head, proposal_blob, sf)
                namespace["authenticate_repository"](head)
        finally:
            if attack_pinned is not None:
                _d3_close_exact_pinned(attack_pinned)
            namespace["PINNED_GIT"] = None
        assert not sentinel.exists()

    expect_repository_reject(
        "local_fsmonitor_hooks_pager",
        lambda repo: add_core_config(repo, f"fsmonitor = {sentinel_script}", f"hooksPath = {tmp_path / 'hooks'}", f"pager = {sentinel_script}"),
        "exact inert allowlist",
    )
    expect_repository_reject(
        "local_diff_textconv_filter",
        lambda repo: replace_config(
            repo,
            config_payload(repo)
            + f'\n[diff "evil"]\n\ttextconv = {sentinel_script}\n[filter "evil"]\n\tsmudge = {sentinel_script}\n',
        ),
        "exact inert allowlist",
    )
    expect_repository_reject(
        "config_section_case_variant",
        lambda repo: replace_config(repo, config_payload(repo).replace("[core]", "[Core]", 1)),
        "exact inert allowlist",
    )
    expect_repository_reject(
        "config_extra_key_spelling_variant",
        lambda repo: add_core_config(repo, "baree = false"),
        "exact inert allowlist",
    )
    expect_repository_reject(
        "partial_clone_promisor_config",
        lambda repo: add_remote_config(repo, "promisor = true", "partialclonefilter = blob:none"),
        "exact inert allowlist",
    )
    expect_repository_reject(
        "assume_unchanged",
        lambda repo: _d3_git(repo, "update-index", "--assume-unchanged", "scripts/phase8_sequence_feasibility.py"),
        "assume-unchanged|skip-worktree",
    )
    expect_repository_reject(
        "skip_worktree",
        lambda repo: _d3_git(repo, "update-index", "--skip-worktree", "scripts/phase8_sequence_feasibility.py"),
        "assume-unchanged|skip-worktree",
    )
    expect_repository_reject(
        "replacement_refs",
        lambda repo: ((repo / ".git/refs/replace").mkdir(parents=True), (repo / f".git/refs/replace/{head}").write_text(f"{head}\n", encoding="ascii")),
        "replacement refs",
    )
    expect_repository_reject(
        "alternates",
        lambda repo: (repo / ".git/objects/info/alternates").write_text("/tmp/nonexistent-objects\n", encoding="utf-8"),
        "alternate|redirected",
    )
    expect_repository_reject(
        "grafts",
        lambda repo: (repo / ".git/info/grafts").write_text(f"{head}\n", encoding="ascii"),
        "alternate|grafted|redirected",
    )
    expect_repository_reject(
        "promisor_pack",
        lambda repo: (repo / ".git/objects/pack/pack-deadbeef.promisor").write_text("", encoding="utf-8"),
        "promisor",
    )

    option_pinned = _d3_pin_exact_verifier_repo(namespace, base_repo, head, proposal_blob, sf)
    try:
        for bad_args in (
            ("--config-env=core.fsmonitor=EVIL", "rev-parse", "--show-object-format"),
            ("-c", "core.fsmonitor=true", "rev-parse", "--show-object-format"),
            ("cat-file", "blob", proposal_blob, "--"),
        ):
            with pytest.raises(SystemExit, match="outside the exact allowlist"):
                namespace["git_run"](*bad_args)
    finally:
        _d3_close_exact_pinned(option_pinned)
        namespace["PINNED_GIT"] = None

    def expect_plumbing_mutation_reject(name: str, mutator: Callable[[Path], None], match: str) -> None:
        attack_repo = tmp_path / f"race_{name}"
        shutil.copytree(base_repo, attack_repo, symlinks=True)
        attack_pinned = _d3_pin_exact_verifier_repo(namespace, attack_repo, head, proposal_blob, sf)
        subprocess_module = namespace["subprocess"]
        original_run = subprocess_module.run

        def mutating_run(*args: object, **kwargs: object) -> object:
            mutator(attack_repo)
            return original_run(*args, **kwargs)

        try:
            subprocess_module.run = mutating_run
            with pytest.raises(SystemExit, match=match):
                namespace["git_run"]("rev-parse", "--show-object-format")
        finally:
            subprocess_module.run = original_run
            _d3_close_exact_pinned(attack_pinned)
            namespace["PINNED_GIT"] = None
        assert not sentinel.exists()

    def mutate_head_same_size(repo: Path) -> None:
        head_path = repo / ".git/HEAD"
        payload = bytearray(head_path.read_bytes())
        payload[-2] = ord("x") if payload[-2] != ord("x") else ord("y")
        head_path.write_bytes(bytes(payload))

    expect_plumbing_mutation_reject(
        "same_size_head",
        mutate_head_same_size,
        "pinned Git file content changed|Git plumbing failed",
    )
    expect_plumbing_mutation_reject(
        "commondir",
        lambda repo: (repo / ".git/commondir").write_text(".git\n", encoding="utf-8"),
        "redirectable Git control input appeared|directory content changed|pinned Git descriptor identity changed",
    )
    expect_plumbing_mutation_reject(
        "info_attributes",
        lambda repo: (repo / ".git/info/attributes").write_text("* diff=evil\n", encoding="utf-8"),
        "redirectable Git info input appeared|directory content changed|pinned Git descriptor identity changed",
    )


def test_d3_postmortem_permanent_no_selection_no_retry_no_006_no_010d_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    input_root = Path(sf.POSTMORTEM_REQUIRED_INPUT_ROOT)
    output_root = Path(sf.POSTMORTEM_REQUIRED_OUTPUT_ROOT)
    accepted_implementation_commit = "d" * 40
    with pytest.raises(ValueError, match="feasibility_005"):
        sf.validate_postmortem_argument_contract(
            device="cuda:0",
            input_root=Path("artifacts/phase8_toy_lm_bridge/feasibility_006"),
            output_root=output_root,
            accepted_proposal_commit=sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
            accepted_implementation_commit=accepted_implementation_commit,
        )
    with pytest.raises(ValueError, match="retry|alternate|001"):
        sf.require_postmortem_root_path_string(
            "artifacts/phase8_toy_lm_bridge/feasibility_postmortem_002",
            "output_root",
        )
    with pytest.raises(SystemExit):
        sf.parse_args(
            [
                "postmortem-failure",
                "--device",
                "cuda:0",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--accepted-proposal-commit",
                sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
                "--accepted-implementation-commit",
                accepted_implementation_commit,
                "--predecessor-selection",
                "artifacts/phase8_toy_lm_bridge/feasibility_selection_001.json",
            ]
        )
    with pytest.raises(SystemExit):
        sf.parse_args(
            [
                "postmortem-failure",
                "--device",
                "cuda:0",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--accepted-proposal-commit",
                sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
                "--accepted-implementation-commit",
                accepted_implementation_commit,
                "--predecessor-root",
                "artifacts/phase8_toy_lm_bridge/feasibility_004",
            ]
        )

    for field in ("no_verdict_change", "no_010d_authority"):
        sf, _input_root, synthetic_output_root = _d3_build_synthetic_postmortem_root(tmp_path / field, monkeypatch)
        context = _d3_expected_publication_context(sf, synthetic_output_root)
        summary_path = synthetic_output_root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["interpretation_limits"][field] = False
        sf.write_postmortem_json(summary_path, summary)
        _d3_rebuild_postmortem_manifest_done(sf, synthetic_output_root)
        with pytest.raises(ValueError, match="interpretation_limits"):
            sf.validate_postmortem_terminal_root(synthetic_output_root, synthetic_output_root, **context)


def test_d3_postmortem_cli_rejects_non_repo_cwd_before_shallow_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    input_root = Path(sf.POSTMORTEM_REQUIRED_INPUT_ROOT)
    output_root = Path(sf.POSTMORTEM_REQUIRED_OUTPUT_ROOT)
    accepted_implementation_commit = "d" * 40
    copied_parent = tmp_path / "artifacts" / "phase8_toy_lm_bridge"
    copied_parent.mkdir(parents=True)
    (copied_parent / "feasibility_005").mkdir()
    (copied_parent / "feasibility_004").symlink_to(copied_parent / "feasibility_005")
    exact_argv = sf.postmortem_exact_argv(
        input_root,
        output_root,
        sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
        accepted_implementation_commit,
    )
    monkeypatch.setattr(sf, "postmortem_kernel_argv", lambda: exact_argv)
    monkeypatch.setattr(sf, "validate_new_postmortem_root", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("output preflight ran before cwd guard")))
    monkeypatch.setattr(sf, "validate_postmortem_input_shallow", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("shallow validation ran before cwd guard")))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="cwd exactly"):
        sf.validate_postmortem_cli_contract(
            device="cuda:0",
            input_root=input_root,
            output_root=output_root,
            accepted_proposal_commit=sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
            accepted_implementation_commit=accepted_implementation_commit,
            environ=_d3_env(sf),
        )


def test_d3_postmortem_source_cleanliness_uses_only_injected_no_argument_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    accepted_implementation_commit = "d" * 40
    runner_blob = "e" * 40
    input_root = Path(sf.POSTMORTEM_REQUIRED_INPUT_ROOT)
    output_root = Path(sf.POSTMORTEM_REQUIRED_OUTPUT_ROOT)
    repo_root = tmp_path / "repo"
    allowed_relative = "artifacts/phase8_toy_lm_bridge/feasibility_005/manifest.json"
    allowed = repo_root / allowed_relative
    allowed.parent.mkdir(parents=True)
    allowed.write_text("{}", encoding="utf-8")
    (repo_root / ".pytest_cache/v/cache").mkdir(parents=True)
    (repo_root / ".pytest_cache/v/cache/nodeids").write_text("[]", encoding="utf-8")
    (repo_root / "backup").mkdir()
    (repo_root / "backup/phase8-note.txt").write_text("benign", encoding="utf-8")
    allowed_paths = {allowed.resolve()}
    monkeypatch.setattr(sf, "REPO_ROOT", repo_root)
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", repo_root / "artifacts" / "phase8_toy_lm_bridge")
    benign_ignored = (".pytest_cache/v/cache/nodeids", "backup/phase8-note.txt")
    ignored_paths_state = {"paths": benign_ignored}
    calls: list[str] = []

    class DynamicAuthority:
        __slots__ = ()

        def __call__(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
            calls.append("callback")
            return ((allowed_relative,), ignored_paths_state["paths"])

    authority = _d3_install_fake_verifier(
        sf,
        monkeypatch,
        accepted_commit=accepted_implementation_commit,
        runner_blob=runner_blob,
        authority=DynamicAuthority(),
    )
    monkeypatch.setattr(sf, "git_output", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ambient git_output used")))
    monkeypatch.setattr(sf, "current_source_commit", lambda: (_ for _ in ()).throw(AssertionError("ambient HEAD lookup used")))
    monkeypatch.setattr(sf, "ignored_source_inputs", lambda: (_ for _ in ()).throw(AssertionError("ambient ignored-source scan used")))

    snapshot = sf.capture_postmortem_source_provenance(input_root, output_root, allowed_paths=allowed_paths)
    assert snapshot == sf.SourceSnapshot(
        commit=accepted_implementation_commit,
        status_lines=(f"?? {allowed_relative}",),
        ignored_inputs=benign_ignored,
        runner_blob=runner_blob,
        verifier_binding=sf._POSTMORTEM_RETAINED_VERIFIER_BINDING,
    )
    sf.verify_postmortem_source_unchanged(snapshot, active_output_root=output_root.with_name(output_root.name + ".tmp"))
    assert calls == ["callback", "callback"]

    ignored_paths_state["paths"] = (*benign_ignored, "backup/another-benign-note.txt")
    (repo_root / "backup/another-benign-note.txt").write_text("another benign note", encoding="utf-8")
    with pytest.raises(sf.SourceChangedError, match="ignored executable source inputs changed"):
        sf.verify_postmortem_source_unchanged(snapshot, active_output_root=output_root.with_name(output_root.name + ".tmp"))
    ignored_paths_state["paths"] = benign_ignored

    malformed_returns = (
        [(), ()],
        ([], ()),
        ((123,), ()),
        (("../escape",), ()),
        ((), ("/tmp/absolute",)),
        ((), ("not//canonical",)),
    )
    for malformed in malformed_returns:
        class MalformedAuthority:
            __slots__ = ()

            def __call__(self) -> object:
                return malformed

        _d3_install_fake_verifier(
            sf,
            monkeypatch,
            accepted_commit=accepted_implementation_commit,
            runner_blob=runner_blob,
            authority=MalformedAuthority(),
        )
        with pytest.raises(ValueError, match="verifier source callback"):
            sf.call_postmortem_verify_callback()

    dangerous_pyc = repo_root / "scripts/__pycache__/phase8_sequence_feasibility.cpython-313.pyc"
    dangerous_pyc.parent.mkdir(parents=True)
    dangerous_pyc.write_bytes(b"pyc")
    _d3_install_fake_verifier(
        sf,
        monkeypatch,
        accepted_commit=accepted_implementation_commit,
        runner_blob=runner_blob,
        ignored=("scripts/__pycache__/phase8_sequence_feasibility.cpython-313.pyc",),
    )
    with pytest.raises(RuntimeError, match="dangerous ignored import surface"):
        sf.capture_postmortem_source_provenance(input_root, output_root, allowed_paths=set())

    class ArgumentAuthority:
        __slots__ = ()

        def __call__(self, _path: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
            return (), ()

    _d3_install_fake_verifier(
        sf,
        monkeypatch,
        accepted_commit=accepted_implementation_commit,
        runner_blob=runner_blob,
        authority=ArgumentAuthority(),
    )
    with pytest.raises(ValueError, match="no arguments"):
        sf.postmortem_verify_callback()

    alias = repo_root / "artifacts/phase8_toy_lm_bridge/feasibility_005/alias_manifest.json"
    alias.symlink_to(allowed)
    _d3_install_fake_verifier(
        sf,
        monkeypatch,
        accepted_commit=accepted_implementation_commit,
        runner_blob=runner_blob,
        untracked=(str(alias.relative_to(repo_root)),),
    )
    with pytest.raises(RuntimeError, match="non-symlink"):
        sf.capture_postmortem_source_provenance(input_root, output_root, allowed_paths={allowed.resolve()})

    ignored_exec = repo_root / "artifacts/phase8_toy_lm_bridge/ignored_executable"
    ignored_exec.write_text("x", encoding="utf-8")
    ignored_exec.chmod(0o755)
    _d3_install_fake_verifier(
        sf,
        monkeypatch,
        accepted_commit=accepted_implementation_commit,
        runner_blob=runner_blob,
        ignored=(str(ignored_exec.relative_to(repo_root)),),
    )
    with pytest.raises(RuntimeError, match="ignored executable"):
        sf.capture_postmortem_source_provenance(input_root, output_root, allowed_paths=set())


def test_d3_postmortem_retained_verifier_binding_rejects_global_replacement_and_same_object_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    loader = _D3FakeClosedRepositoryLoader()
    authority = _d3_install_fake_verifier(sf, monkeypatch, loader=loader)
    binding = sf.retained_postmortem_verifier_binding()
    assert not hasattr(loader, "__dict__")
    with pytest.raises(RuntimeError, match="sealed"):
        loader.find_spec = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sf.__dict__,
        "__phase8_verify_repository_unchanged__",
        _D3FakeRepositoryAuthority(
            accepted_commit="d" * 40,
            runner_blob="e" * 40,
            untracked=(),
            ignored=(),
            proposal_binding=(sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT, sf.POSTMORTEM_PROPOSAL_PATH, sf.POSTMORTEM_PROPOSAL_BLOB),
        ),
    )
    with pytest.raises(ValueError, match="global|authority changed"):
        sf.verify_postmortem_verifier_binding(binding)

    loader = _D3FakeClosedRepositoryLoader()
    _d3_install_fake_verifier(sf, monkeypatch, loader=loader)
    binding = sf.retained_postmortem_verifier_binding()
    object.__setattr__(loader, "_sources", MappingProxyType({}))
    with pytest.raises(ValueError, match="loader binding|source map"):
        sf.verify_postmortem_verifier_binding(binding)

    authority = _d3_install_fake_verifier(sf, monkeypatch)
    binding = sf.retained_postmortem_verifier_binding()
    object.__setattr__(authority, "_ignored", ("backup/phase8-note.txt",))
    with pytest.raises(ValueError, match="authority binding|sealed state"):
        sf.verify_postmortem_verifier_binding(binding)

    authority = _d3_install_fake_verifier(sf, monkeypatch)
    binding = sf.retained_postmortem_verifier_binding()
    original_call = type(authority).__call__

    def replacement_call(self: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return (), ()

    monkeypatch.setattr(type(authority), "__call__", replacement_call)
    with pytest.raises(ValueError, match="authority binding|code"):
        sf.verify_postmortem_verifier_binding(binding)
    monkeypatch.setattr(type(authority), "__call__", original_call)

    loader = _D3FakeClosedRepositoryLoader()
    _d3_install_fake_verifier(sf, monkeypatch, loader=loader)
    binding = sf.retained_postmortem_verifier_binding()

    def replacement_find_spec(self: object, fullname: str, path: object = None, target: object = None) -> object:
        return None

    monkeypatch.setattr(type(loader), "find_spec", replacement_find_spec)
    with pytest.raises(ValueError, match="loader binding|method"):
        sf.verify_postmortem_verifier_binding(binding)


def test_d3_postmortem_retained_real_verifier_binding_rejects_transitive_helper_state_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    namespace = _d3_exact_verifier_namespace(sf)
    repo, head, proposal_blob = _d3_build_exact_verifier_repo(tmp_path / "repo", sf)
    pinned = _d3_pin_exact_verifier_repo(namespace, repo, head, proposal_blob, sf)
    try:
        captured, modes, untracked, ignored = namespace["authenticate_repository"](head)
        assert untracked == ()
        assert ignored == ()
        proposal_binding = namespace["authenticate_proposal"]()
        authority = namespace["RepositoryAuthority"](
            head,
            MappingProxyType(dict(captured)),
            MappingProxyType(dict(modes)),
            proposal_binding,
        )
        loader = namespace["ClosedRepositoryLoader"](captured)
        runner_blob = namespace["git_blob_oid"](captured["scripts/phase8_sequence_feasibility.py"])
        monkeypatch.setattr(sf, "_POSTMORTEM_RETAINED_VERIFIER_BINDING", None)
        monkeypatch.setattr(sf, "POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT", head)
        monkeypatch.setattr(sf, "POSTMORTEM_PROPOSAL_BLOB", proposal_blob)
        monkeypatch.setitem(sf.__dict__, "__phase8_verifier_sha256__", sf.POSTMORTEM_VERIFIER_SHA256)
        monkeypatch.setitem(sf.__dict__, "__phase8_accepted_implementation_commit__", head)
        monkeypatch.setitem(sf.__dict__, "__phase8_repository_loader__", loader)
        monkeypatch.setitem(sf.__dict__, "__phase8_verify_repository_unchanged__", authority)
        monkeypatch.setitem(sf.__dict__, "__phase8_runner_blob__", runner_blob)
        monkeypatch.setitem(sf.__dict__, "__phase8_proposal_binding__", proposal_binding)
        monkeypatch.setitem(sf.__dict__, "__phase8_authority_token__", object())
        binding = sf.retained_postmortem_verifier_binding()

        original_directory_snapshots = namespace["PINNED_GIT"].directory_snapshots
        namespace["PINNED_GIT"].directory_snapshots = {}
        with pytest.raises(ValueError, match="authority binding|helper|sealed state"):
            sf.verify_postmortem_verifier_binding(binding)
        namespace["PINNED_GIT"].directory_snapshots = original_directory_snapshots
        sf.verify_postmortem_verifier_binding(binding)

        pinned_type = namespace["PinnedGitContext"]
        for method_name in ("revalidate", "git_prefix", "git_environment"):
            original_method = getattr(pinned_type, method_name)

            def replacement_method(self: object, _name: str = method_name) -> object:
                return None

            setattr(pinned_type, method_name, replacement_method)
            try:
                with pytest.raises(ValueError, match="authority binding|helper|code"):
                    sf.verify_postmortem_verifier_binding(binding)
            finally:
                setattr(pinned_type, method_name, original_method)
            sf.verify_postmortem_verifier_binding(binding)

        subprocess_module = namespace["subprocess"]
        original_run = subprocess_module.run

        def replacement_run(*args: object, **kwargs: object) -> object:
            return original_run(*args, **kwargs)

        subprocess_module.run = replacement_run
        try:
            with pytest.raises(ValueError, match="authority binding|helper|code"):
                sf.verify_postmortem_verifier_binding(binding)
        finally:
            subprocess_module.run = original_run
        sf.verify_postmortem_verifier_binding(binding)
    finally:
        _d3_close_exact_pinned(pinned)
        namespace["PINNED_GIT"] = None


def test_d3_postmortem_top_level_prepublication_exception_demote_failure_exits_74(
    tmp_path: Path,
) -> None:
    child = r'''
import sys
from pathlib import Path
sys.path.insert(0, %r)
import scripts.phase8_sequence_feasibility as sf

base = Path(%r)
input_root = base / "input"
output_root = base / "postmortem"
output_root.parent.mkdir(parents=True, exist_ok=True)
input_root.mkdir(parents=True, exist_ok=True)
cell = {
    "family": "hex_copy",
    "model_size": "small",
    "seed": 0,
    "checkpoint_path": "cell/checkpoint.pt",
    "generations_path": "cell/generations.jsonl",
}
key = ("hex_copy", "small", 0)
input_binding = {name: None for name in sf.POSTMORTEM_INPUT_BINDING_KEYS}
input_binding["root"] = str(input_root)
input_binding["cells"] = [cell]
shallow = {
    "allowed_source_paths": set(),
    "manifest_data": {"environment": dict(sf.FEASIBILITY_REQUIRED_RUNTIME_ENV)},
}
deep = {
    "input_binding": input_binding,
    "generation_rows": {key: []},
    "checkpoint_payloads": {key: {}},
    "cells": [cell],
}
sf.validate_postmortem_argument_contract = lambda **kwargs: None
sf.validate_postmortem_cli_contract = lambda **kwargs: None
sf.validate_postmortem_input_shallow = lambda root: shallow
sf.capture_postmortem_source_provenance = lambda *args, **kwargs: sf.SourceSnapshot(
    commit="d" * 40,
    status_lines=(),
    ignored_inputs=(),
    runner_blob="e" * 40,
)
sf.configure_postmortem_deterministic_backend = lambda: None
sf.validate_postmortem_runtime_against_input = lambda manifest: ({}, {})
sf.postmortem_proposal_binding = lambda commit: {}
sf.postmortem_implementation_binding = lambda snapshot: {}
sf.validate_postmortem_input_deep = lambda root, shallow_binding: deep
sf.postmortem_record_sets_rng_neutral = lambda: ({"hex_copy": {"train": (), "eval": ()}}, {"record_reconstruction_global_state_unchanged": True})
sf.snapshot_rng_states = lambda: ("rng",)
sf.require_rng_states_equal = lambda *args, **kwargs: None
sf.validate_new_postmortem_root = lambda output: None
sf.validate_postmortem_current_cells_shallow = lambda cells: list(cells)
sf.postmortem_file_sha_from_input_binding = lambda binding, path: "a" * 64
sf.construct_postmortem_model_from_checkpoint = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("forced model construction failure"))

def fail_demote(root_fd):
    raise sf.FeasibilityPublicationError("forced demotion failure")

sf.demote_postmortem_terminal_markers_at_fd = fail_demote
sf.run_postmortem_failure(
    device="cpu",
    input_root=input_root,
    output_root=output_root,
    accepted_proposal_commit="a" * 40,
    accepted_implementation_commit="b" * 40,
    environ={},
)
raise SystemExit(99)
''' % (str(REPO_ROOT), str(tmp_path / "child_fail_stop"))
    completed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 74, completed.stderr
    assert "Postmortem terminal demotion failed" in completed.stderr


def test_d3_postmortem_preflight_order_stops_before_source_clean_output_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    events: list[str] = []
    accepted_implementation_commit = "d" * 40
    artifact_parent = Path(".pytest_d3_preflight") / tmp_path.name / "phase8_toy_lm_bridge"
    output_root = artifact_parent / "feasibility_postmortem_001"
    input_root = artifact_parent / "feasibility_005"
    monkeypatch.setattr(sf, "POSTMORTEM_REQUIRED_INPUT_ROOT", str(input_root))
    monkeypatch.setattr(sf, "POSTMORTEM_REQUIRED_OUTPUT_ROOT", str(output_root))
    monkeypatch.setattr(sf, "ARTIFACT_PARENT", artifact_parent)
    monkeypatch.setattr(sf, "validate_postmortem_argument_contract", lambda **kwargs: events.append("args"))
    monkeypatch.setattr(sf, "require_postmortem_real_main_context", lambda: events.append("real_main"))
    monkeypatch.setattr(sf, "validate_postmortem_cli_contract", lambda **kwargs: events.append("cli"))
    monkeypatch.setattr(
        sf,
        "validate_postmortem_input_shallow",
        lambda root: events.append("shallow") or {"input_binding": {}, "manifest_data": {}, "allowed_source_paths": set()},
    )

    def dirty_source(*args: object, **kwargs: object) -> object:
        events.append("source_clean")
        raise RuntimeError("synthetic dirty source")

    monkeypatch.setattr(sf, "capture_postmortem_source_provenance", dirty_source)
    for name in (
        "configure_postmortem_deterministic_backend",
        "validate_postmortem_runtime_against_input",
        "validate_postmortem_input_deep",
        "postmortem_record_sets_rng_neutral",
        "construct_postmortem_model_from_checkpoint",
        "write_postmortem_jsonl",
    ):
        monkeypatch.setattr(sf, name, lambda *args, _name=name, **kwargs: (_ for _ in ()).throw(AssertionError(f"{_name} ran before source cleanliness")))

    with pytest.raises(RuntimeError, match="synthetic dirty source"):
        sf.run_postmortem_failure(
            device="cuda:0",
            input_root=input_root,
            output_root=output_root,
            accepted_proposal_commit=sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
            accepted_implementation_commit=accepted_implementation_commit,
            environ=_d3_env(sf),
        )
    assert events == ["args", "cli", "shallow", "source_clean"]
    assert not output_root.exists()
    assert not output_root.with_name(output_root.name + ".tmp").exists()


def test_d3_postmortem_real_shallow_and_deep_input_validation_rejects_evidence_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, input_root, _output_root = _d3_build_faithful_input_root(tmp_path / "valid", monkeypatch)
    shallow = sf.validate_postmortem_input_shallow(input_root)
    deep = sf.validate_postmortem_input_deep(input_root, shallow)
    assert len(deep["generation_rows"]) == 24
    assert len(deep["checkpoint_payloads"]) == 24

    sf, input_root, _output_root = _d3_build_faithful_input_root(tmp_path / "inventory_drift", monkeypatch)
    generation_path = input_root / "hex_copy__small__seed0" / "generations.jsonl"
    generation_path.write_text(generation_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum|file_inventory"):
        sf.validate_postmortem_input_shallow(input_root)

    sf, input_root, _output_root = _d3_build_faithful_input_root(tmp_path / "inventory_binding_drift", monkeypatch)
    manifest_path = input_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_inventory"][0]["sha256"] = "0" * 64
    sf.write_json(manifest_path, manifest)
    _rewrite_terminal_manifest_sha(input_root, "FAILED")
    monkeypatch.setattr(
        sf,
        "POSTMORTEM_INPUT_CHECKSUMS",
        {
            "manifest.json": sf.file_sha256(input_root / "manifest.json"),
            "summary.json": sf.file_sha256(input_root / "summary.json"),
            "FAILED.json": sf.file_sha256(input_root / "FAILED.json"),
        },
    )
    with pytest.raises(ValueError, match="file_inventory|checksum"):
        sf.validate_postmortem_input_shallow(input_root)

    sf, input_root, _output_root = _d3_build_faithful_input_root(tmp_path / "generation_drift", monkeypatch)
    generation_path = input_root / "hex_copy__small__seed0" / "generations.jsonl"
    generation_rows = [json.loads(line) for line in generation_path.read_text(encoding="utf-8").splitlines()]
    generation_rows[0]["prompt"] = f"{generation_rows[0]['prompt']} forged"
    generation_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in generation_rows) + "\n", encoding="utf-8")
    _d3_refresh_faithful_input_checksums(sf, input_root, monkeypatch)
    shallow = sf.validate_postmortem_input_shallow(input_root)
    with pytest.raises(ValueError, match="prompt|frozen evaluation record"):
        sf.validate_postmortem_input_deep(input_root, shallow)

    sf, input_root, _output_root = _d3_build_faithful_input_root(tmp_path / "checkpoint_drift", monkeypatch)
    checkpoint_path = input_root / "hex_copy__small__seed0" / "checkpoint_step1500.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["model_state_dict"]["lm_head.weight"] = checkpoint["model_state_dict"]["lm_head.weight"].clone()
    checkpoint["model_state_dict"]["lm_head.weight"][0, 0] += 1
    torch.save(checkpoint, checkpoint_path)
    _d3_refresh_faithful_input_checksums(sf, input_root, monkeypatch)
    shallow = sf.validate_postmortem_input_shallow(input_root)
    with pytest.raises(ValueError, match="byte-equal"):
        sf.validate_postmortem_input_deep(input_root, shallow)


def test_d3_postmortem_passing_path_call_order_reaches_publication_after_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    events: list[str] = []
    accepted_implementation_commit = "d" * 40
    input_root = tmp_path / "feasibility_005"
    output_root = tmp_path / "feasibility_postmortem_001"
    cell = {
        "family": "hex_copy",
        "model_size": "small",
        "seed": 0,
        "checkpoint_path": "hex_copy__small__seed0/checkpoint_step1500.pt",
        "generations_path": "hex_copy__small__seed0/generations.jsonl",
    }
    records = {
        "hex_copy": {
            "train": (sf.FeasibilityRecord("hex_copy", "train", 0, "t0", "o0", "p", "a"),),
            "eval": (sf.FeasibilityRecord("hex_copy", "eval", 0, "e0", "eo0", "p", "a"),),
        }
    }
    input_binding = {
        "root": str(input_root),
        "source_commit": sf.POSTMORTEM_INPUT_SOURCE_COMMIT,
        "manifest_path": "manifest.json",
        "manifest_sha256": "a" * 64,
        "summary_path": "summary.json",
        "summary_sha256": "b" * 64,
        "terminal_path": "FAILED.json",
        "terminal_sha256": "c" * 64,
        "configuration": sf.frozen_configuration(),
        "record_hashes": dict(sf.FEASIBILITY_RECORD_HASHES),
        "file_inventory": [],
        "cells": [cell],
    }
    monkeypatch.setattr(sf, "validate_postmortem_argument_contract", lambda **kwargs: events.append("args"))
    monkeypatch.setattr(sf, "require_postmortem_real_main_context", lambda: events.append("real_main"))
    monkeypatch.setattr(sf, "require_postmortem_repo_cwd", lambda: events.append("cwd"))
    monkeypatch.setattr(sf, "validate_postmortem_cli_contract", lambda **kwargs: events.append("cli"))
    monkeypatch.setattr(
        sf,
        "validate_postmortem_input_shallow",
        lambda root: events.append("shallow") or {"input_binding": input_binding, "manifest_data": {"environment": "runtime"}, "allowed_source_paths": set()},
    )
    monkeypatch.setattr(sf, "capture_postmortem_source_provenance", lambda *args, **kwargs: events.append("source_clean") or sf.SourceSnapshot(commit="a" * 40, status_lines=(), ignored_inputs=()))
    monkeypatch.setattr(sf, "configure_postmortem_deterministic_backend", lambda: events.append("backend"))
    monkeypatch.setattr(sf, "validate_postmortem_runtime_against_input", lambda manifest: events.append("runtime") or ({"env": True}, {"flags": True}))
    monkeypatch.setattr(sf, "postmortem_proposal_binding", lambda commit: events.append("proposal") or {"proposal": True})
    monkeypatch.setattr(sf, "postmortem_implementation_binding", lambda snapshot: events.append("implementation") or {"implementation": True})
    monkeypatch.setattr(sf, "validate_postmortem_input_deep", lambda root, shallow: events.append("deep") or {"input_binding": input_binding, "generation_rows": {("hex_copy", "small", 0): [{"exact_match": True}]}, "checkpoint_payloads": {("hex_copy", "small", 0): {"metadata": {}}}, "cells": [cell]})
    monkeypatch.setattr(sf, "postmortem_record_sets_rng_neutral", lambda: events.append("records_rng") or (records, {"record_reconstruction_global_state_unchanged": True}))
    monkeypatch.setattr(sf, "validate_new_postmortem_root", lambda output: events.append("tmp"))
    monkeypatch.setattr(sf, "validate_postmortem_current_cells_shallow", lambda cells: [cell])
    monkeypatch.setattr(sf, "postmortem_file_sha_from_input_binding", lambda binding, path: "d" * 64)
    monkeypatch.setattr(sf, "construct_postmortem_model_from_checkpoint", lambda *args, **kwargs: events.append("model") or object())
    monkeypatch.setattr(sf, "postmortem_model_snapshot", lambda model: {})
    monkeypatch.setattr(sf, "require_postmortem_model_unchanged", lambda *args, **kwargs: events.append("model_same"))
    monkeypatch.setattr(sf, "require_rng_states_equal", lambda *args, **kwargs: events.append("rng_same"))

    def teacher_rows(_model: object, _records: object, _tokenizer: object, _device: object, *, split: str, **_kwargs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        events.append(f"forward:{split}")
        return ([{"split": split}], {})

    monkeypatch.setattr(sf, "postmortem_teacher_forced_rows", teacher_rows)
    monkeypatch.setattr(sf, "postmortem_taxonomy_row", lambda *args, **kwargs: events.append("taxonomy") or {"taxonomy": True})
    monkeypatch.setattr(sf, "postmortem_cell_metrics_row", lambda *args, **kwargs: events.append("cell_metrics") or {"cell": True})
    monkeypatch.setattr(sf, "finalize_postmortem_outputs_under_guard", lambda **kwargs: events.append("output_publication"))
    sf.run_postmortem_failure(
        device="cuda:0",
        input_root=input_root,
        output_root=output_root,
        accepted_proposal_commit=sf.POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
        accepted_implementation_commit=accepted_implementation_commit,
        environ=_d3_env(sf),
    )
    ordered = [event for event in events if event in {"args", "cli", "shallow", "source_clean", "backend", "runtime", "proposal", "implementation", "deep", "records_rng", "tmp", "model", "forward:train", "forward:eval", "taxonomy", "cell_metrics", "output_publication"}]
    assert ordered == [
        "args",
        "cli",
        "shallow",
        "source_clean",
        "backend",
        "runtime",
        "proposal",
        "implementation",
        "deep",
        "records_rng",
        "tmp",
        "model",
        "forward:train",
        "forward:eval",
        "taxonomy",
        "cell_metrics",
        "output_publication",
    ]


def test_d3_postmortem_checkpoint_validator_is_rng_safe_and_tie_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    checkpoint_path = tmp_path / "checkpoint_step1500.pt"
    model = build_model("small")
    save_checkpoint(
        str(checkpoint_path),
        model,
        metadata={
            "family": "hex_copy",
            "model_size": "small",
            "seed": 0,
            "training_steps": sf.TRAINING_STEPS,
            "training_loss": 0.25,
            "training_accuracy": 0.75,
        },
    )
    cell = {
        "family": "hex_copy",
        "model_size": "small",
        "seed": 0,
        "parameter_count": sf.FROZEN_PARAMETER_COUNTS["small"],
    }
    before = sf.snapshot_rng_states()
    monkeypatch.setattr(sf, "build_model", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("validator constructed a model")))
    payload = sf.validate_postmortem_checkpoint_payload(checkpoint_path, cell)
    assert payload["metadata"]["training_accuracy"] == 0.75
    assert sf.rng_states_equal(before, sf.snapshot_rng_states())

    mutated = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    mutated["model_state_dict"]["lm_head.weight"] = mutated["model_state_dict"]["lm_head.weight"].clone()
    mutated["model_state_dict"]["lm_head.weight"][0, 0] += 1
    mutated_path = tmp_path / "mutated_checkpoint.pt"
    torch.save(mutated, mutated_path)
    with pytest.raises(ValueError, match="byte-equal"):
        sf.validate_postmortem_checkpoint_payload(mutated_path, cell)


def test_d3_postmortem_constructor_inference_callbacks_and_publication_preserve_all_rng_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    all_records = sf.grouped_records()
    original_batch_size = sf.BATCH_SIZE
    original_train_records_per_family = sf.TRAIN_RECORDS_PER_FAMILY
    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path / "publication", monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    temp_root = output_root.with_name(output_root.name + ".tmp")
    monkeypatch.setattr(sf, "BATCH_SIZE", 2)
    monkeypatch.setattr(sf, "TRAIN_RECORDS_PER_FAMILY", 2)
    checkpoint_path = tmp_path / "checkpoint_step1500.pt"
    save_checkpoint(
        str(checkpoint_path),
        build_model("small"),
        metadata={
            "family": "hex_copy",
            "model_size": "small",
            "seed": 0,
            "training_steps": sf.TRAINING_STEPS,
            "training_loss": 0.25,
            "training_accuracy": 0.75,
        },
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    records = all_records["hex_copy"]["train"][:2]
    tokenizer = ByteTokenizer()
    with sf.PostmortemForbiddenOperationGuard() as guard:
        baseline = sf.snapshot_rng_states()
        model = sf.construct_postmortem_model_from_checkpoint(
            checkpoint,
            model_size="small",
            device=torch.device("cpu"),
            guard=guard,
        )
        assert sf.rng_states_equal(baseline, sf.snapshot_rng_states())
        sf.postmortem_teacher_forced_rows(
            model,
            records,
            tokenizer,
            torch.device("cpu"),
            split="train",
            family="hex_copy",
            model_size="small",
            seed=0,
            checkpoint_path="hex_copy__small__seed0/checkpoint_step1500.pt",
            checkpoint_sha256="d" * 64,
        )
        assert sf.rng_states_equal(baseline, sf.snapshot_rng_states())

        monkeypatch.setattr(sf, "BATCH_SIZE", original_batch_size)
        monkeypatch.setattr(sf, "TRAIN_RECORDS_PER_FAMILY", original_train_records_per_family)
        output_root.rename(temp_root)
        callback_count = 0

        def assert_rng_unchanged_callback() -> None:
            nonlocal callback_count
            callback_count += 1
            assert sf.rng_states_equal(baseline, sf.snapshot_rng_states())

        sf.publish_postmortem_root(
            temp_root,
            output_root,
            **context,
            final_callback=assert_rng_unchanged_callback,
        )
        assert callback_count >= 1
        assert output_root.is_dir()
        assert not temp_root.exists()
        assert sf.rng_states_equal(baseline, sf.snapshot_rng_states())


def test_d3_postmortem_forbidden_operation_guard_rejects_training_generation_rng_and_save(
    tmp_path: Path,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    model = build_model("small")
    prefix = torch.tensor([[BOS_ID, SEP_ID]], dtype=torch.long)
    with sf.PostmortemForbiddenOperationGuard():
        with pytest.raises(ValueError, match="forbidden"):
            sf.make_optimizer(model)
        with pytest.raises(ValueError, match="forbidden"):
            sf.train_text_records()
        with pytest.raises(ValueError, match="forbidden"):
            sf.evaluate_model()
        with pytest.raises(ValueError, match="forbidden"):
            sf.generate_named_diagnostic_rows()
        with pytest.raises(ValueError, match="forbidden"):
            sf.generate_array_rows()
        with pytest.raises(ValueError, match="forbidden"):
            sf.save_checkpoint()
        with pytest.raises(ValueError, match="forbidden"):
            torch.optim.AdamW(model.parameters())
        with pytest.raises(ValueError, match="grad enablement|forbidden"):
            torch.enable_grad()
        with pytest.raises(ValueError, match="grad enablement|forbidden"):
            torch.set_grad_enabled(True)
        with pytest.raises(ValueError, match="forbidden"):
            torch.tensor(1.0, requires_grad=True).backward()
        with pytest.raises(ValueError, match="forbidden"):
            torch.manual_seed(123)
        with pytest.raises(ValueError, match="RNG-state mutation"):
            torch.random.set_rng_state(torch.random.get_rng_state())
        with pytest.raises(ValueError, match="forbidden"):
            torch.save({"x": 1}, tmp_path / "forbidden.pt")
        with pytest.raises(ValueError, match="forbidden"):
            model.greedy_decode(prefix)


def test_d3_postmortem_forbidden_operation_guard_patches_exec_runner_globals_without_sys_modules() -> None:
    runner_name = "json"
    assert runner_name in sys.modules
    namespace: dict[str, object] = {
        "__name__": runner_name,
        "__file__": str(REPO_ROOT / "scripts/phase8_sequence_feasibility.py"),
        "__package__": "scripts",
    }
    assert sys.modules[runner_name].__dict__ is not namespace
    source = (REPO_ROOT / "scripts/phase8_sequence_feasibility.py").read_text(encoding="utf-8")
    exec(compile(source, str(REPO_ROOT / "scripts/phase8_sequence_feasibility.py"), "exec"), namespace)
    assert sys.modules[runner_name].__dict__ is not namespace
    guard_class = namespace["PostmortemForbiddenOperationGuard"]
    original_make_optimizer = namespace["make_optimizer"]
    original_sys_module_make_optimizer = getattr(sys.modules[runner_name], "make_optimizer", None)
    with guard_class():  # type: ignore[operator]
        assert namespace["make_optimizer"] is not original_make_optimizer
        assert getattr(sys.modules[runner_name], "make_optimizer", None) is original_sys_module_make_optimizer
        with pytest.raises(ValueError, match="forbidden training/generation"):
            namespace["make_optimizer"]()
    assert namespace["make_optimizer"] is original_make_optimizer
    assert getattr(sys.modules[runner_name], "make_optimizer", None) is original_sys_module_make_optimizer


def test_d3_postmortem_teacher_forcing_uses_response_labels_hex_and_full_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    monkeypatch.setattr(sf, "BATCH_SIZE", 2)
    monkeypatch.setattr(sf, "TRAIN_RECORDS_PER_FAMILY", 2)
    tokenizer = ByteTokenizer()
    records = (
        sf.FeasibilityRecord("hex_copy", "train", 0, "t0", "o0", "copy a", '"a"'),
        sf.FeasibilityRecord("hex_copy", "train", 1, "t1", "o1", "copy b", '"b"'),
    )

    class PerfectTeacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.forward_calls = 0

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            labels = torch.full_like(input_ids, -100)
            for row_index, row in enumerate(input_ids.tolist()):
                unpadded = list(row)
                while unpadded and unpadded[-1] == PAD_ID:
                    unpadded.pop()
                sep_index = unpadded.index(SEP_ID)
                eos_index = len(unpadded) - 1
                assert unpadded[eos_index] == EOS_ID
                for source_position in range(sep_index, eos_index):
                    labels[row_index, source_position] = input_ids[row_index, source_position + 1]
            logits = torch.full((*input_ids.shape, ByteTokenizer.vocab_size), -3.0, dtype=torch.float32, device=input_ids.device)
            for row_index in range(labels.shape[0]):
                for position, label in enumerate(labels[row_index].tolist()):
                    if label != -100:
                        logits[row_index, position, int(label)] = 2.0
            return logits

    model = PerfectTeacher().eval()
    rows, aggregate = sf.postmortem_teacher_forced_rows(
        model,
        records,
        tokenizer,
        torch.device("cpu"),
        split="train",
        family="hex_copy",
        model_size="small",
        seed=0,
        checkpoint_path="hex_copy__small__seed0/checkpoint_step1500.pt",
        checkpoint_sha256="d" * 64,
    )
    assert len(rows) == 2
    assert rows[0]["batch_index"] == 0
    assert rows[1]["row_within_batch"] == 1
    assert rows[0]["selected_token_count"] == len(tokenizer.encode_text(records[0].answer)) + 1
    assert all(row["sequence_exact"] is True for row in rows)
    assert model.forward_calls == 1

    input_ids = encode_record_batch(
        tuple(TextRecord(record.prompt, record.answer) for record in records),
        tokenizer,
        max_length=ByteTokenizer.max_sequence_length,
    )
    manual_labels = torch.full_like(input_ids, -100)
    manual_selected_positions: list[list[int]] = []
    for row_index, encoded_row in enumerate(input_ids.tolist()):
        unpadded = list(encoded_row)
        while unpadded and unpadded[-1] == PAD_ID:
            unpadded.pop()
        sep_index = unpadded.index(SEP_ID)
        eos_index = len(unpadded) - 1
        assert unpadded[eos_index] == EOS_ID
        row_positions = list(range(sep_index, eos_index))
        assert row_positions
        assert input_ids[row_index, row_positions[-1] + 1].item() == EOS_ID
        manual_selected_positions.append(row_positions)
        for source_position in row_positions:
            manual_labels[row_index, source_position] = input_ids[row_index, source_position + 1]
    oracle_logits = torch.full((*input_ids.shape, ByteTokenizer.vocab_size), -3.0, dtype=torch.float32)
    for row_index, row_positions in enumerate(manual_selected_positions):
        for source_position in row_positions:
            oracle_logits[row_index, source_position, int(manual_labels[row_index, source_position])] = 2.0
    oracle_losses = torch.nn.functional.cross_entropy(
        oracle_logits.reshape(-1, oracle_logits.shape[-1]),
        manual_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(len(records), ByteTokenizer.max_sequence_length)
    assert oracle_losses.dtype == torch.float32
    manual_total_nll = 0.0
    manual_correct_total = 0
    manual_selected_total = 0
    for row_index, row_positions in enumerate(manual_selected_positions):
        expected_hex = math.fsum(float(oracle_losses[row_index, position]) for position in row_positions).hex()
        assert rows[row_index]["nll_numerator_hex"] == expected_hex
        predictions = oracle_logits.argmax(dim=-1)
        expected_correct = sum(
            int(predictions[row_index, position]) == int(manual_labels[row_index, position])
            for position in row_positions
        )
        assert rows[row_index]["selected_token_count"] == len(row_positions)
        assert rows[row_index]["correct_token_count"] == expected_correct
        manual_total_nll = math.fsum((manual_total_nll, float.fromhex(str(rows[row_index]["nll_numerator_hex"]))))
        manual_correct_total += expected_correct
        manual_selected_total += len(row_positions)
    expected_aggregate = {
        "record_count": 2,
        "sequence_exact_numerator": 2,
        "selected_token_count": manual_selected_total,
        "correct_token_count": manual_correct_total,
        "nll_numerator_hex": manual_total_nll.hex(),
        "nll_per_token_hex": (manual_total_nll / manual_selected_total).hex(),
    }
    assert aggregate == expected_aggregate
    assert sf.postmortem_teacher_aggregate(rows) == expected_aggregate
    assert model.forward_calls == 1
    with pytest.raises(ValueError, match="full 64-row batches|cardinalities"):
        sf.postmortem_teacher_forced_rows(
            model,
            records[:1],
            tokenizer,
            torch.device("cpu"),
            split="train",
            family="hex_copy",
            model_size="small",
            seed=0,
            checkpoint_path="hex_copy__small__seed0/checkpoint_step1500.pt",
            checkpoint_sha256="d" * 64,
        )


def test_d3_postmortem_taxonomy_oracle_named_array_fields() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    schema = "phase8_feasibility_postmortem_v1"
    invalid_utf8_error = "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"

    named_record = sf.FeasibilityRecord(
        "named_value_json",
        "eval",
        0,
        "literal_named_template",
        "literal_named_operand",
        "Give JSON string for key red; fields red=red-ca54; blue=blue-e7fb; green=green-6e58; silver=silver-8c9e",
        '"red-ca54"',
        ("red-ca54", "blue-e7fb", "green-6e58", "silver-8c9e"),
    )
    array_record_1 = sf.FeasibilityRecord("array_json", "eval", 0, "literal_array_1", "literal_array_operand_1", "Output compact JSON array from chunks: qe38e", '["qe38e"]', ("qe38e", '["qe38e"]'))
    array_record_2 = sf.FeasibilityRecord("array_json", "eval", 1, "literal_array_2", "literal_array_operand_2", "Print compact JSON array for pieces: qc663 | q410a", '["qc663","q410a"]', ("qc663", "q410a", '["qc663","q410a"]'))
    array_record_3 = sf.FeasibilityRecord("array_json", "eval", 2, "literal_array_3", "literal_array_operand_3", "Make JSON array using these chunks: qa111 | qb222 | qc333", '["qa111","qb222","qc333"]', ("qa111", "qb222", "qc333", '["qa111","qb222","qc333"]'))
    array_record_4 = sf.FeasibilityRecord("array_json", "eval", 3, "literal_array_4", "literal_array_operand_4", "Give JSON array only from chunks: qa101 | qb202 | qc303 | qd404", '["qa101","qb202","qc303","qd404"]', ("qa101", "qb202", "qc303", "qd404", '["qa101","qb202","qc303","qd404"]'))

    def raw_ids(prompt_bytes: bytes, generated_bytes: bytes | None) -> list[int]:
        if generated_bytes is None:
            return [BOS_ID, *prompt_bytes, SEP_ID, 255, EOS_ID]
        return [BOS_ID, *prompt_bytes, SEP_ID, *generated_bytes, EOS_ID]

    named_prompt = b"Give JSON string for key red; fields red=red-ca54; blue=blue-e7fb; green=green-6e58; silver=silver-8c9e"
    array_prompt_1 = b"Output compact JSON array from chunks: qe38e"
    array_prompt_2 = b"Print compact JSON array for pieces: qc663 | q410a"
    array_prompt_3 = b"Make JSON array using these chunks: qa111 | qb222 | qc333"
    array_prompt_4 = b"Give JSON array only from chunks: qa101 | qb202 | qc303 | qd404"

    named_common_exact = {
        "prompt": named_record.prompt,
        "expected_response": '"red-ca54"',
        "generated_response": '"red-ca54"',
        "raw_token_ids": raw_ids(named_prompt, b'"red-ca54"'),
        "generation_error": None,
        "greedy_exact": True,
        "decode_valid": True,
        "has_eos": True,
        "hits_generation_cap": False,
        "hits_context_cap": False,
        "generation_token_count": 11,
        "generation_utf8_bytes": 10,
        "target_utf8_bytes": 10,
        "length_matches": True,
        "hamming_distance": 0,
        "edit_distance": 0,
        "first_error": None,
    }
    named_common_distractor = {
        **named_common_exact,
        "generated_response": '"blue-e7fb"',
        "raw_token_ids": raw_ids(named_prompt, b'"blue-e7fb"'),
        "greedy_exact": False,
        "generation_token_count": 12,
        "generation_utf8_bytes": 11,
        "length_matches": False,
        "hamming_distance": None,
        "edit_distance": 8,
        "first_error": 1,
    }
    named_common_wrong_suffix = {
        **named_common_exact,
        "generated_response": '"red-cXXX"',
        "raw_token_ids": raw_ids(named_prompt, b'"red-cXXX"'),
        "greedy_exact": False,
        "hamming_distance": 3,
        "edit_distance": 3,
        "first_error": 6,
    }
    named_common_invalid_decode = {
        **named_common_exact,
        "generated_response": None,
        "raw_token_ids": raw_ids(named_prompt, None),
        "generation_error": invalid_utf8_error,
        "greedy_exact": False,
        "decode_valid": False,
        "has_eos": False,
        "generation_token_count": 2,
        "generation_utf8_bytes": None,
        "length_matches": False,
        "hamming_distance": None,
        "edit_distance": None,
        "first_error": None,
    }
    named_exact = {
        "surface_source": "held_surface",
        "operand_source": "held_operand",
        "valid_json_string": True,
        "valid_named_grammar": True,
        "target_prefix": True,
        "occurs_in_prompt": True,
        "exact_target_value": True,
        "exact_distractor_value": False,
        "suffix_positional_correct": 4,
        "suffix_positional_total": 4,
        "suffix_hamming_distance": 0,
        "suffix_edit_distance": 0,
        "suffix_first_error": None,
    }
    named_distractor = {
        **named_exact,
        "target_prefix": False,
        "exact_target_value": False,
        "exact_distractor_value": True,
        "suffix_positional_correct": None,
        "suffix_positional_total": None,
        "suffix_hamming_distance": None,
        "suffix_edit_distance": None,
        "suffix_first_error": None,
    }
    named_wrong_suffix = {
        **named_exact,
        "valid_named_grammar": False,
        "occurs_in_prompt": False,
        "exact_target_value": False,
        "suffix_positional_correct": 1,
        "suffix_hamming_distance": 3,
        "suffix_edit_distance": 3,
        "suffix_first_error": 1,
    }
    named_invalid_decode = {
        **named_distractor,
        "valid_json_string": False,
        "valid_named_grammar": False,
        "occurs_in_prompt": False,
        "exact_distractor_value": False,
    }

    def taxonomy_row(
        *,
        family: str,
        model_size: str,
        seed: int,
        record_index: int,
        generation_path: str,
        generation_sha256: str,
        common: dict[str, object],
        named: dict[str, object] | None,
        array: dict[str, object] | None,
    ) -> dict[str, object]:
        return {
            "schema_version": schema,
            "family": family,
            "model_size": model_size,
            "seed": seed,
            "record_index": record_index,
            "generation_path": generation_path,
            "generation_sha256": generation_sha256,
            "common": common,
            "named": named,
            "array": array,
        }

    named_path = "named_value_json__medium__seed0/generations.jsonl"
    named_expected_rows = [
        taxonomy_row(family="named_value_json", model_size="medium", seed=0, record_index=0, generation_path=named_path, generation_sha256="a" * 64, common=named_common_exact, named=named_exact, array=None),
        taxonomy_row(family="named_value_json", model_size="medium", seed=0, record_index=0, generation_path=named_path, generation_sha256="a" * 64, common=named_common_distractor, named=named_distractor, array=None),
        taxonomy_row(family="named_value_json", model_size="medium", seed=0, record_index=0, generation_path=named_path, generation_sha256="a" * 64, common=named_common_wrong_suffix, named=named_wrong_suffix, array=None),
        taxonomy_row(family="named_value_json", model_size="medium", seed=0, record_index=0, generation_path=named_path, generation_sha256="a" * 64, common=named_common_invalid_decode, named=named_invalid_decode, array=None),
    ]
    named_retained = [
        {"raw_token_ids": raw_ids(named_prompt, b'"red-ca54"'), "generated": '"red-ca54"', "generation_error": None, "exact_match": True},
        {"raw_token_ids": raw_ids(named_prompt, b'"blue-e7fb"'), "generated": '"blue-e7fb"', "generation_error": None, "exact_match": False},
        {"raw_token_ids": raw_ids(named_prompt, b'"red-cXXX"'), "generated": '"red-cXXX"', "generation_error": None, "exact_match": False},
        {"raw_token_ids": raw_ids(named_prompt, None), "generated": None, "generation_error": invalid_utf8_error, "exact_match": False},
    ]
    for retained, expected in zip(named_retained, named_expected_rows, strict=True):
        assert sf.postmortem_taxonomy_row(named_record, retained, family="named_value_json", model_size="medium", seed=0, generation_path=named_path, generation_sha256="a" * 64) == expected

    array_common_1 = {
        "prompt": array_record_1.prompt,
        "expected_response": '["qe38e"]',
        "generated_response": '["qe38e"]',
        "raw_token_ids": raw_ids(array_prompt_1, b'["qe38e"]'),
        "generation_error": None,
        "greedy_exact": True,
        "decode_valid": True,
        "has_eos": True,
        "hits_generation_cap": False,
        "hits_context_cap": False,
        "generation_token_count": 10,
        "generation_utf8_bytes": 9,
        "target_utf8_bytes": 9,
        "length_matches": True,
        "hamming_distance": 0,
        "edit_distance": 0,
        "first_error": None,
    }
    array_common_2 = {
        "prompt": array_record_2.prompt,
        "expected_response": '["qc663","q410a"]',
        "generated_response": '["q410a","qc663"]',
        "raw_token_ids": raw_ids(array_prompt_2, b'["q410a","qc663"]'),
        "generation_error": None,
        "greedy_exact": False,
        "decode_valid": True,
        "has_eos": True,
        "hits_generation_cap": False,
        "hits_context_cap": False,
        "generation_token_count": 18,
        "generation_utf8_bytes": 17,
        "target_utf8_bytes": 17,
        "length_matches": True,
        "hamming_distance": 8,
        "edit_distance": 8,
        "first_error": 3,
    }
    array_common_3 = {
        "prompt": array_record_3.prompt,
        "expected_response": '["qa111","qb222","qc333"]',
        "generated_response": None,
        "raw_token_ids": raw_ids(array_prompt_3, None),
        "generation_error": invalid_utf8_error,
        "greedy_exact": False,
        "decode_valid": False,
        "has_eos": False,
        "hits_generation_cap": False,
        "hits_context_cap": False,
        "generation_token_count": 2,
        "generation_utf8_bytes": None,
        "target_utf8_bytes": 25,
        "length_matches": False,
        "hamming_distance": None,
        "edit_distance": None,
        "first_error": None,
    }
    array_common_4 = {
        "prompt": array_record_4.prompt,
        "expected_response": '["qa101","qb202","qc303","qd404"]',
        "generated_response": '["qa101","qb202","qc303"]',
        "raw_token_ids": raw_ids(array_prompt_4, b'["qa101","qb202","qc303"]'),
        "generation_error": None,
        "greedy_exact": False,
        "decode_valid": True,
        "has_eos": True,
        "hits_generation_cap": False,
        "hits_context_cap": False,
        "generation_token_count": 26,
        "generation_utf8_bytes": 25,
        "target_utf8_bytes": 33,
        "length_matches": False,
        "hamming_distance": None,
        "edit_distance": 8,
        "first_error": 24,
    }
    array_1 = {
        "comparison_source": "feasibility_005_retained_eval",
        "training_steps": 1500,
        "target_item_count": 1,
        "valid_json_syntax": True,
        "valid_array_schema": True,
        "correct_item_count": True,
        "positional_exact_count": 1,
        "positional_denominator": 1,
        "missing_items": [],
        "extra_items": [],
        "all_items_copied": True,
        "all_items_exact": True,
        "first_wrong_item_position": None,
    }
    array_2 = {**array_1, "target_item_count": 2, "correct_item_count": True, "positional_exact_count": 0, "positional_denominator": 2, "all_items_exact": False, "first_wrong_item_position": 0}
    array_3 = {**array_1, "target_item_count": 3, "valid_json_syntax": False, "valid_array_schema": False, "correct_item_count": False, "positional_exact_count": None, "positional_denominator": None, "missing_items": None, "extra_items": None, "all_items_copied": False, "all_items_exact": False, "first_wrong_item_position": None}
    array_4 = {**array_1, "target_item_count": 4, "correct_item_count": False, "positional_exact_count": 3, "positional_denominator": 4, "missing_items": ["qd404"], "all_items_exact": False, "first_wrong_item_position": 3}
    array_path = "array_json__small__seed1/generations.jsonl"
    array_expected_rows = [
        taxonomy_row(family="array_json", model_size="small", seed=1, record_index=0, generation_path=array_path, generation_sha256="1" * 64, common=array_common_1, named=None, array=array_1),
        taxonomy_row(family="array_json", model_size="small", seed=1, record_index=1, generation_path=array_path, generation_sha256="1" * 64, common=array_common_2, named=None, array=array_2),
        taxonomy_row(family="array_json", model_size="small", seed=1, record_index=2, generation_path=array_path, generation_sha256="1" * 64, common=array_common_3, named=None, array=array_3),
        taxonomy_row(family="array_json", model_size="small", seed=1, record_index=3, generation_path=array_path, generation_sha256="1" * 64, common=array_common_4, named=None, array=array_4),
    ]
    array_retained = [
        {"raw_token_ids": raw_ids(array_prompt_1, b'["qe38e"]'), "generated": '["qe38e"]', "generation_error": None, "exact_match": True},
        {"raw_token_ids": raw_ids(array_prompt_2, b'["q410a","qc663"]'), "generated": '["q410a","qc663"]', "generation_error": None, "exact_match": False},
        {"raw_token_ids": raw_ids(array_prompt_3, None), "generated": None, "generation_error": invalid_utf8_error, "exact_match": False},
        {"raw_token_ids": raw_ids(array_prompt_4, b'["qa101","qb202","qc303"]'), "generated": '["qa101","qb202","qc303"]', "generation_error": None, "exact_match": False},
    ]
    for record, retained, expected in zip((array_record_1, array_record_2, array_record_3, array_record_4), array_retained, array_expected_rows, strict=True):
        assert sf.postmortem_taxonomy_row(record, retained, family="array_json", model_size="small", seed=1, generation_path=array_path, generation_sha256="1" * 64) == expected

    taxonomy_rows: list[dict[str, object]] = []
    for model_size in ("small", "medium"):
        for seed in (0, 1, 2):
            for prototype in named_expected_rows * 16:
                taxonomy_rows.append({**prototype, "model_size": model_size, "seed": seed, "generation_path": f"named_value_json__{model_size}__seed{seed}/generations.jsonl", "generation_sha256": (str(seed) if seed else "a") * 64})
            for prototype in array_expected_rows * 16:
                taxonomy_rows.append({**prototype, "model_size": model_size, "seed": seed, "generation_path": f"array_json__{model_size}__seed{seed}/generations.jsonl", "generation_sha256": (str(seed) if seed else "a") * 64})

    expected_named_aggregates = [
        {
            "model_size": model_size,
            "seed": seed,
            "denominator": 64,
            "valid_json_string_count": 48,
            "valid_named_grammar_count": 32,
            "target_prefix_count": 32,
            "occurs_in_prompt_count": 32,
            "exact_target_value_count": 16,
            "exact_distractor_value_count": 16,
            "suffix_positional_correct": 80,
            "suffix_positional_denominator": 128,
            "suffix_first_error_histogram": {"1": 16},
            "suffix_first_error_observation_count": 16,
            "eos_present_count": 48,
            "generation_cap_count": 0,
            "target_length_match_count": 32,
            "first_error_histogram": {"1": 16, "6": 16},
            "first_error_observation_count": 32,
            "generation_token_count_histogram": {"2": 16, "11": 32, "12": 16},
            "generation_utf8_bytes_histogram": {"10": 32, "11": 16},
        }
        for model_size in ("small", "medium")
        for seed in (0, 1, 2)
    ]
    expected_array_base = {
        1: {
            "denominator": 16,
            "valid_json_syntax_count": 16,
            "valid_array_schema_count": 16,
            "correct_item_count_count": 16,
            "positional_exact_count": 16,
            "positional_denominator": 16,
            "all_items_copied_count": 16,
            "all_items_exact_count": 16,
            "missing_item_count": 0,
            "extra_item_count": 0,
            "first_wrong_histogram": {},
            "first_wrong_observation_count": 0,
            "greedy_exact_count": 16,
            "eos_present_count": 16,
            "generation_cap_count": 0,
            "target_length_match_count": 16,
            "first_error_histogram": {},
            "first_error_observation_count": 0,
            "generation_token_count_histogram": {"10": 16},
            "generation_utf8_bytes_histogram": {"9": 16},
        },
        2: {
            "denominator": 16,
            "valid_json_syntax_count": 16,
            "valid_array_schema_count": 16,
            "correct_item_count_count": 16,
            "positional_exact_count": 0,
            "positional_denominator": 32,
            "all_items_copied_count": 16,
            "all_items_exact_count": 0,
            "missing_item_count": 0,
            "extra_item_count": 0,
            "first_wrong_histogram": {"0": 16},
            "first_wrong_observation_count": 16,
            "greedy_exact_count": 0,
            "eos_present_count": 16,
            "generation_cap_count": 0,
            "target_length_match_count": 16,
            "first_error_histogram": {"3": 16},
            "first_error_observation_count": 16,
            "generation_token_count_histogram": {"18": 16},
            "generation_utf8_bytes_histogram": {"17": 16},
        },
        3: {
            "denominator": 16,
            "valid_json_syntax_count": 0,
            "valid_array_schema_count": 0,
            "correct_item_count_count": 0,
            "positional_exact_count": 0,
            "positional_denominator": 0,
            "all_items_copied_count": 0,
            "all_items_exact_count": 0,
            "missing_item_count": 0,
            "extra_item_count": 0,
            "first_wrong_histogram": {},
            "first_wrong_observation_count": 0,
            "greedy_exact_count": 0,
            "eos_present_count": 0,
            "generation_cap_count": 0,
            "target_length_match_count": 0,
            "first_error_histogram": {},
            "first_error_observation_count": 0,
            "generation_token_count_histogram": {"2": 16},
            "generation_utf8_bytes_histogram": {},
        },
        4: {
            "denominator": 16,
            "valid_json_syntax_count": 16,
            "valid_array_schema_count": 16,
            "correct_item_count_count": 0,
            "positional_exact_count": 48,
            "positional_denominator": 64,
            "all_items_copied_count": 16,
            "all_items_exact_count": 0,
            "missing_item_count": 16,
            "extra_item_count": 0,
            "first_wrong_histogram": {"3": 16},
            "first_wrong_observation_count": 16,
            "greedy_exact_count": 0,
            "eos_present_count": 16,
            "generation_cap_count": 0,
            "target_length_match_count": 0,
            "first_error_histogram": {"24": 16},
            "first_error_observation_count": 16,
            "generation_token_count_histogram": {"26": 16},
            "generation_utf8_bytes_histogram": {"25": 16},
        },
    }
    expected_array_aggregates = [
        {"model_size": model_size, "seed": seed, "target_item_count": item_count, **expected_array_base[item_count]}
        for model_size in ("small", "medium")
        for seed in (0, 1, 2)
        for item_count in (1, 2, 3, 4)
    ]
    assert sf.postmortem_named_aggregates(taxonomy_rows) == expected_named_aggregates
    assert sf.postmortem_array_aggregates(taxonomy_rows) == expected_array_aggregates
    summary = sf.build_postmortem_summary(
        input_binding={"configuration": {}},
        proposal_binding={},
        implementation_binding={},
        cell_rows=[],
        taxonomy_rows=taxonomy_rows,
    )
    assert summary["named_aggregates"] == expected_named_aggregates
    assert summary["array_aggregates"] == expected_array_aggregates
    assert len(summary["named_aggregates"]) == 6
    assert len(summary["array_aggregates"]) == 24

    teacher_rows = [{"sequence_exact": value} for value in ([True, True, False, False] * 16)]
    generation_rows = [{"exact_match": value} for value in ([True, False, True, False] * 16)]
    assert sf.postmortem_paired_eval_exact(teacher_rows, generation_rows) == {
        "both_exact": 16,
        "tf_only": 16,
        "greedy_only": 16,
        "neither_exact": 16,
        "denominator": 64,
        "count_difference": 0,
    }


def test_d3_postmortem_done_publication_schema_aggregates_and_mutation_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path, monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    sf.validate_postmortem_terminal_root(output_root, output_root, **context)
    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["interpretation_limits"] == {
        "non_evidence": True,
        "no_causal_weight_tying_claim": True,
        "no_verdict_change": True,
        "no_010d_authority": True,
    }
    assert len(summary["cells"]) == 24
    assert len(summary["named_aggregates"]) == 6
    assert len(summary["array_aggregates"]) == 24
    summary["named_aggregates"][0]["denominator"] = 63
    sf.write_postmortem_json(summary_path, summary)
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_inventory"] = sf.postmortem_output_inventory(output_root)
    sf.write_postmortem_json(manifest_path, manifest)
    sf.write_postmortem_done(output_root, input_binding=summary["input_binding"])
    with pytest.raises(ValueError, match="Named aggregates"):
        sf.validate_postmortem_terminal_root(output_root, output_root, **context)


def test_d3_postmortem_terminal_rejects_forged_composite_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        ("input_binding", lambda binding: {**binding, "record_hashes": {**binding["record_hashes"], "array_eval64": "0" * 64}}, "record_hashes|frozen in-memory"),
        ("proposal_binding", lambda binding: {**binding, "blob": "0" * 40}, "proposal_binding"),
        ("implementation_binding", lambda binding: {**binding, "runner_blob": "0" * 40}, "runner_blob"),
        ("implementation_binding", lambda binding: {**binding, "commit": "0" * 40}, "commit"),
    )
    for index, (binding_name, mutate, match) in enumerate(cases):
        sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path / f"case_{index}", monkeypatch)
        context = _d3_expected_publication_context(sf, output_root)
        sf.validate_postmortem_terminal_root(output_root, output_root, **context)
        summary_path = output_root / "summary.json"
        manifest_path = output_root / "manifest.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        replacement = mutate(json.loads(json.dumps(manifest[binding_name])))
        summary[binding_name] = replacement
        manifest[binding_name] = replacement
        if binding_name == "input_binding":
            summary["configuration"] = replacement["configuration"]
            manifest["configuration"] = replacement["configuration"]
        sf.write_postmortem_json(summary_path, summary)
        sf.write_postmortem_json(manifest_path, manifest)
        _d3_rebuild_postmortem_manifest_done(sf, output_root)
        with pytest.raises(ValueError, match=match):
            sf.validate_postmortem_terminal_root(output_root, output_root, **context)


def test_d3_postmortem_terminal_rejects_canonical_byte_and_teacher_row_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path / "canonical_json", monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    sf.validate_postmortem_terminal_root(output_root, output_root, **context)
    summary_path = output_root / "summary.json"
    summary_path.write_bytes(summary_path.read_bytes().replace(b"\n", b" \n", 1))
    with pytest.raises(ValueError, match="canonical JSON bytes"):
        sf.validate_postmortem_terminal_root(output_root, output_root, **context)

    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path / "canonical_jsonl", monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    teacher_path = output_root / "teacher_forced_rows.jsonl"
    first_line, rest = teacher_path.read_bytes().split(b"\n", 1)
    teacher_path.write_bytes(first_line + b" \n" + rest)
    _d3_rebuild_postmortem_manifest_done(sf, output_root)
    with pytest.raises(ValueError, match="canonical JSONL bytes"):
        sf.validate_postmortem_terminal_root(output_root, output_root, **context)

    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path / "negative_zero", monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    summary_path = output_root / "summary.json"
    summary_path.write_bytes(summary_path.read_bytes().replace(b'"training_accuracy": 0.5', b'"training_accuracy": -0.0', 1))
    with pytest.raises(ValueError, match="negative zero"):
        sf.validate_postmortem_terminal_root(output_root, output_root, **context)

    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path / "teacher_expected", monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    bad_context = dict(context)
    bad_teacher_rows = json.loads(json.dumps(context["expected_teacher_rows"]))
    bad_teacher_rows[0]["operand_id"] = "synthetic-forged-operand"
    bad_context["expected_teacher_rows"] = bad_teacher_rows
    with pytest.raises(ValueError, match="teacher_forced_rows changed"):
        sf.validate_postmortem_terminal_root(output_root, output_root, **bad_context)


def test_d3_postmortem_terminal_rejects_schema_type_null_order_cardinality_runtime_and_terminal_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, _input_root, base_output_root = _d3_build_synthetic_postmortem_root(tmp_path / "base", monkeypatch)
    context = _d3_expected_publication_context(sf, base_output_root)
    sf.validate_postmortem_terminal_root(base_output_root, base_output_root, **context)

    def fresh_root(name: str) -> Path:
        candidate = tmp_path / f"{name}_root"
        shutil.copytree(base_output_root, candidate)
        return candidate

    def rewrite_rows(root: Path, relative_path: str, rows: list[dict[str, object]]) -> None:
        sf.write_postmortem_jsonl(root / relative_path, rows)
        _d3_rebuild_postmortem_manifest_done(sf, root)

    root = fresh_root("teacher_closed_schema")
    rows = sf.read_canonical_postmortem_jsonl(root / "teacher_forced_rows.jsonl", "teacher")
    rows[0]["unexpected"] = True
    rewrite_rows(root, "teacher_forced_rows.jsonl", rows)
    with pytest.raises(ValueError, match="closed schema|teacher_forced_rows changed"):
        sf.validate_postmortem_terminal_root(root, base_output_root, **context)

    root = fresh_root("teacher_order")
    rows = sf.read_canonical_postmortem_jsonl(root / "teacher_forced_rows.jsonl", "teacher")
    rows[0], rows[1] = rows[1], rows[0]
    rewrite_rows(root, "teacher_forced_rows.jsonl", rows)
    with pytest.raises(ValueError, match="record_index|teacher_forced_rows changed|mismatch"):
        sf.validate_postmortem_terminal_root(root, base_output_root, **context)

    root = fresh_root("teacher_cardinality")
    rows = sf.read_canonical_postmortem_jsonl(root / "teacher_forced_rows.jsonl", "teacher")
    rewrite_rows(root, "teacher_forced_rows.jsonl", rows[:-1])
    with pytest.raises(ValueError, match="cardinality|teacher_forced_rows changed"):
        sf.validate_postmortem_terminal_root(root, base_output_root, **context)

    root = fresh_root("cell_type")
    rows = sf.read_canonical_postmortem_jsonl(root / "cell_metrics.jsonl", "cell")
    rows[0]["seed"] = "0"
    rewrite_rows(root, "cell_metrics.jsonl", rows)
    with pytest.raises(ValueError, match="seed.*integer|cell metrics changed|closed schema"):
        sf.validate_postmortem_terminal_root(root, base_output_root, **context)

    root = fresh_root("taxonomy_null")
    rows = sf.read_canonical_postmortem_jsonl(root / "error_taxonomy.jsonl", "taxonomy")
    rows[0]["named"] = {}
    rewrite_rows(root, "error_taxonomy.jsonl", rows)
    with pytest.raises(ValueError, match="taxonomy row|error_taxonomy rows changed"):
        sf.validate_postmortem_terminal_root(root, base_output_root, **context)

    root = fresh_root("summary_schema")
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["unexpected"] = True
    sf.write_postmortem_json(summary_path, summary)
    _d3_rebuild_postmortem_manifest_done(sf, root)
    with pytest.raises(ValueError, match="summary.*closed schema"):
        sf.validate_postmortem_terminal_root(root, base_output_root, **context)

    root = fresh_root("summary_null_boundary")
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["interpretation_limits"]["no_010d_authority"] = None
    sf.write_postmortem_json(summary_path, summary)
    _d3_rebuild_postmortem_manifest_done(sf, root)
    with pytest.raises(ValueError, match="interpretation_limits"):
        sf.validate_postmortem_terminal_root(root, base_output_root, **context)

    root = fresh_root("manifest_schema")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected"] = True
    manifest["file_inventory"] = sf.postmortem_output_inventory(root)
    sf.write_postmortem_json(manifest_path, manifest)
    sf.write_postmortem_done(root, input_binding=manifest["input_binding"])
    with pytest.raises(ValueError, match="manifest.*closed schema"):
        sf.validate_postmortem_terminal_root(root, base_output_root, **context)

    root = fresh_root("runtime_environment")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"] = {**manifest["environment"], "forged_runtime": "present"}
    manifest["file_inventory"] = sf.postmortem_output_inventory(root)
    sf.write_postmortem_json(manifest_path, manifest)
    sf.write_postmortem_done(root, input_binding=manifest["input_binding"])
    with pytest.raises(ValueError, match="environment mismatch"):
        sf.validate_postmortem_terminal_root(root, base_output_root, **context)

    root = fresh_root("done_schema")
    done_path = root / "DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["unexpected"] = True
    sf.write_postmortem_json(done_path, done)
    with pytest.raises(ValueError, match="done.*closed schema"):
        sf.validate_postmortem_terminal_root(root, base_output_root, **context)

    root = fresh_root("done_null_row_count")
    done_path = root / "DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["row_counts"]["cell_metrics"] = None
    sf.write_postmortem_json(done_path, done)
    with pytest.raises(ValueError, match="row_counts mismatch"):
        sf.validate_postmortem_terminal_root(root, base_output_root, **context)


def test_d3_postmortem_publication_rejects_coherent_callback_output_rewrite_without_rename_or_terminal_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path, monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    temp_root = output_root.with_name(output_root.name + ".tmp")
    output_root.rename(temp_root)
    assert not output_root.exists()

    def mutate_teacher_rows_after_validation() -> None:
        teacher_path = temp_root / "teacher_forced_rows.jsonl"
        rows = sf.read_canonical_postmortem_jsonl(teacher_path, "callback.teacher")
        rows[0]["operand_id"] = "forged-with-aggregate-preserved"
        sf.write_postmortem_jsonl(teacher_path, rows)
        manifest_path = temp_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["file_inventory"] = sf.postmortem_output_inventory(temp_root)
        sf.write_postmortem_json(manifest_path, manifest)
        sf.write_postmortem_done(temp_root, input_binding=manifest["input_binding"])

    with pytest.raises(sf.FeasibilityPublicationError, match="temporary root is incomplete"):
        sf.publish_postmortem_root_or_leave_incomplete(
            temp_root,
            output_root,
            **context,
            final_callback=mutate_teacher_rows_after_validation,
        )
    assert not output_root.exists()
    assert temp_root.exists()
    assert not (temp_root / "DONE.json").exists()
    assert not (temp_root / "FAILED.json").exists()


def test_d3_postmortem_terminal_root_rejects_extra_directory_and_non_file_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path / "extra_dir", monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    (output_root / "empty_extra_dir").mkdir()
    with pytest.raises(ValueError, match="exactly six root-level entries"):
        sf.validate_postmortem_terminal_root(output_root, output_root, **context)

    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path / "non_file", monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    (output_root / "DONE.json").unlink()
    (output_root / "DONE.json").mkdir()
    with pytest.raises(ValueError, match="regular non-symlink files"):
        sf.validate_postmortem_terminal_root(output_root, output_root, **context)


def test_d3_postmortem_guarded_publication_never_calls_path_based_terminal_validators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path, monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    temp_root = output_root.with_name(output_root.name + ".tmp")
    output_root.rename(temp_root)
    monkeypatch.setattr(
        sf,
        "validate_postmortem_terminal_root",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("path terminal validator used by guarded publish")),
    )
    monkeypatch.setattr(
        sf,
        "validate_postmortem_terminal_inventory_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("path inventory validator used by guarded publish")),
    )
    sf.publish_postmortem_root_or_leave_incomplete(temp_root, output_root, **context)
    assert output_root.is_dir()
    assert not temp_root.exists()
    assert (output_root / "DONE.json").is_file()


def test_d3_postmortem_guarded_publication_binds_fd_across_transient_same_byte_path_swap_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path, monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    temp_root = output_root.with_name(output_root.name + ".tmp")
    output_root.rename(temp_root)
    original_temp = tmp_path / "trusted_original_temp"
    replacement = tmp_path / "same_byte_replacement"
    shutil.copytree(temp_root, replacement)
    callback_ran = False

    def transient_swap_restore() -> None:
        nonlocal callback_ran
        if callback_ran:
            return
        callback_ran = True
        temp_root.rename(original_temp)
        replacement.rename(temp_root)
        temp_root.rename(replacement)
        original_temp.rename(temp_root)

    monkeypatch.setattr(
        sf,
        "validate_postmortem_terminal_root",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("path terminal validator used during transient swap test")),
    )
    monkeypatch.setattr(
        sf,
        "validate_postmortem_terminal_inventory_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("path inventory validator used during transient swap test")),
    )
    sf.publish_postmortem_root_or_leave_incomplete(
        temp_root,
        output_root,
        **context,
        final_callback=transient_swap_restore,
    )
    assert callback_ran is True
    assert output_root.is_dir()
    assert replacement.is_dir()
    assert not (replacement / "FAILED.json").exists()


def test_d3_postmortem_publication_rejects_parent_and_temp_chmod_drift_across_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for drift_target in ("parent", "temp"):
        sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path / drift_target, monkeypatch)
        context = _d3_expected_publication_context(sf, output_root)
        temp_root = output_root.with_name(output_root.name + ".tmp")
        output_root.rename(temp_root)
        parent = temp_root.parent
        original_parent_mode = stat.S_IMODE(parent.stat().st_mode)
        original_temp_mode = stat.S_IMODE(temp_root.stat().st_mode)
        callback_ran = False

        def chmod_after_validation() -> None:
            nonlocal callback_ran
            if callback_ran:
                return
            callback_ran = True
            target = parent if drift_target == "parent" else temp_root
            mode = stat.S_IMODE(target.stat().st_mode)
            target.chmod(mode ^ stat.S_IRGRP)

        try:
            with pytest.raises(sf.FeasibilityPublicationError, match="temporary root is incomplete"):
                sf.publish_postmortem_root_or_leave_incomplete(
                    temp_root,
                    output_root,
                    **context,
                    final_callback=chmod_after_validation,
                )
        finally:
            parent.chmod(original_parent_mode)
            if temp_root.exists() and not temp_root.is_symlink():
                temp_root.chmod(original_temp_mode)
        assert callback_ran is True
        assert not output_root.exists()
        assert not (temp_root / "DONE.json").exists()
        assert not (temp_root / "FAILED.json").exists()


def test_d3_postmortem_publication_rejects_output_chmod_drift_after_atomic_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path, monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    temp_root = output_root.with_name(output_root.name + ".tmp")
    output_root.rename(temp_root)
    real_rename_at = sf.atomic_rename_noreplace_at
    rename_calls: list[tuple[str, str]] = []

    def chmod_output_after_publish_rename(
        source_dir_fd: int,
        source_name: str,
        destination_dir_fd: int,
        destination_name: str,
        destination_label: object,
    ) -> None:
        real_rename_at(source_dir_fd, source_name, destination_dir_fd, destination_name, destination_label)
        rename_calls.append((source_name, destination_name))
        if (source_name, destination_name) == (temp_root.name, output_root.name):
            output_root.chmod(stat.S_IMODE(output_root.stat().st_mode) ^ stat.S_IRGRP)

    monkeypatch.setattr(sf, "atomic_rename_noreplace_at", chmod_output_after_publish_rename)
    with pytest.raises(sf.FeasibilityPublicationError, match="temporary root is incomplete"):
        sf.publish_postmortem_root_or_leave_incomplete(temp_root, output_root, **context)
    assert rename_calls[0] == (temp_root.name, output_root.name)
    assert not output_root.exists()
    assert temp_root.is_dir()
    assert not (temp_root / "DONE.json").exists()
    assert not (temp_root / "FAILED.json").exists()


def test_d3_postmortem_fd_writer_does_not_follow_swapped_temp_root_path(
    tmp_path: Path,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    output_root = tmp_path / "feasibility_postmortem_001"
    guard = sf.create_postmortem_publication_guard(output_root)
    temp_root = guard.temp_root
    original_temp = tmp_path / "trusted_temp_directory"
    external = tmp_path / "external_directory"
    external.mkdir()
    temp_root.rename(original_temp)
    temp_root.symlink_to(external, target_is_directory=True)
    try:
        sf.write_postmortem_json_at(guard, "summary.json", {"sentinel": "trusted-fd"})
    finally:
        guard.close()
    assert temp_root.is_symlink()
    assert json.loads((original_temp / "summary.json").read_text(encoding="utf-8")) == {"sentinel": "trusted-fd"}
    assert not (external / "summary.json").exists()


@pytest.mark.parametrize("marker_kind", ("symlink", "fifo", "socket", "directory", "device"))
def test_d3_postmortem_publication_demotes_nonregular_done_marker_forms_without_final_root_or_external_follow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_kind: str,
) -> None:
    short_socket_base: Path | None = None
    if marker_kind == "socket":
        short_socket_base = Path(tempfile.mkdtemp(prefix="d3sock_"))
    base_root = short_socket_base if short_socket_base is not None else tmp_path / marker_kind
    try:
        sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(base_root, monkeypatch)
        context = _d3_expected_publication_context(sf, output_root)
        temp_root = output_root.with_name(output_root.name + ".tmp")
        output_root.rename(temp_root)
        done_path = temp_root / "DONE.json"
        done_path.unlink()
        external_target = base_root / f"{marker_kind}_external_done_target"
        marker_socket: socket.socket | None = None
        try:
            if marker_kind == "symlink":
                external_target.write_text("external target remains untouched", encoding="utf-8")
                done_path.symlink_to(external_target)
            elif marker_kind == "fifo":
                os.mkfifo(done_path)
            elif marker_kind == "socket":
                marker_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    marker_socket.bind(str(done_path))
                except PermissionError:
                    pytest.skip("socket marker publication fixture requires AF_UNIX bind permission")
            elif marker_kind == "directory":
                done_path.mkdir()
            elif marker_kind == "device":
                try:
                    os.mknod(done_path, stat.S_IFCHR | 0o600, os.makedev(1, 7))
                except PermissionError:
                    pytest.skip("device-node publication fixture requires CAP_MKNOD")
            else:
                raise AssertionError(marker_kind)

            with pytest.raises(sf.FeasibilityPublicationError, match="temporary root is incomplete"):
                sf.publish_postmortem_root_or_leave_incomplete(temp_root, output_root, **context)
        finally:
            if marker_socket is not None:
                marker_socket.close()
        assert not output_root.exists()
        assert temp_root.is_dir()
        assert not (temp_root / "DONE.json").exists()
        assert not (temp_root / "FAILED.json").exists()
        assert any(path.name.startswith(".DONE.json.demoted.") for path in temp_root.iterdir())
        if marker_kind == "symlink":
            assert external_target.read_text(encoding="utf-8") == "external target remains untouched"
    finally:
        if short_socket_base is not None:
            shutil.rmtree(short_socket_base, ignore_errors=True)


def test_d3_postmortem_publication_rejects_temp_root_symlink_swap_without_following_external_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path, monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    temp_root = output_root.with_name(output_root.name + ".tmp")
    output_root.rename(temp_root)
    external_copy = tmp_path / "external_copy"
    original_temp = tmp_path / "original_temp"
    shutil.copytree(temp_root, external_copy)
    callback_ran = False

    def swap_temp_for_external_symlink() -> None:
        nonlocal callback_ran
        if callback_ran:
            return
        callback_ran = True
        temp_root.rename(original_temp)
        temp_root.symlink_to(external_copy, target_is_directory=True)

    with pytest.raises(sf.FeasibilityPublicationError, match="temporary root is incomplete"):
        sf.publish_postmortem_root_or_leave_incomplete(
            temp_root,
            output_root,
            **context,
            final_callback=swap_temp_for_external_symlink,
        )
    assert callback_ran is True
    assert not output_root.exists()
    assert temp_root.is_symlink()
    assert (external_copy / "DONE.json").is_file()
    assert not (original_temp / "DONE.json").exists()
    assert not (original_temp / "FAILED.json").exists()
    assert any(path.name.startswith(".DONE.json.demoted.") for path in original_temp.iterdir())


def test_d3_postmortem_publication_rejects_same_byte_temp_directory_inode_swap_and_removes_new_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path, monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    temp_root = output_root.with_name(output_root.name + ".tmp")
    output_root.rename(temp_root)
    replacement = tmp_path / "same_byte_replacement"
    callback_ran = False

    def swap_temp_for_same_byte_directory() -> None:
        nonlocal callback_ran
        if callback_ran:
            return
        callback_ran = True
        shutil.copytree(temp_root, replacement)
        shutil.rmtree(temp_root)
        replacement.rename(temp_root)

    with pytest.raises(sf.FeasibilityPublicationError, match="temporary root is incomplete"):
        sf.publish_postmortem_root_or_leave_incomplete(
            temp_root,
            output_root,
            **context,
            final_callback=swap_temp_for_same_byte_directory,
        )
    assert callback_ran is True
    assert not output_root.exists()
    assert temp_root.is_dir()
    assert not temp_root.is_symlink()
    assert not (temp_root / "DONE.json").exists()
    assert not (temp_root / "FAILED.json").exists()


def test_d3_postmortem_publication_rolls_back_final_identity_mismatch_without_final_root_or_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf, _input_root, output_root = _d3_build_synthetic_postmortem_root(tmp_path, monkeypatch)
    context = _d3_expected_publication_context(sf, output_root)
    temp_root = output_root.with_name(output_root.name + ".tmp")
    output_root.rename(temp_root)
    rename_calls: list[tuple[str, str]] = []
    real_rename_at = sf.atomic_rename_noreplace_at

    def rename_probe(
        source_dir_fd: int,
        source_name: str,
        destination_dir_fd: int,
        destination_name: str,
        destination_label: object,
    ) -> None:
        rename_calls.append((source_name, destination_name))
        real_rename_at(source_dir_fd, source_name, destination_dir_fd, destination_name, destination_label)

    real_fingerprint_at = sf.postmortem_terminal_fingerprint_at

    def fingerprint_mismatch_after_rename(guard: object) -> tuple[tuple[object, ...], ...]:
        fingerprint = real_fingerprint_at(guard)
        if output_root.exists() and not temp_root.exists():
            return (("__tampered_after_rename__",), *fingerprint[1:])
        return fingerprint

    monkeypatch.setattr(sf, "atomic_rename_noreplace_at", rename_probe)
    monkeypatch.setattr(sf, "postmortem_terminal_fingerprint_at", fingerprint_mismatch_after_rename)
    with pytest.raises(sf.FeasibilityPublicationError, match="temporary root is incomplete"):
        sf.publish_postmortem_root_or_leave_incomplete(temp_root, output_root, **context)
    assert rename_calls[0] == (temp_root.name, output_root.name)
    assert rename_calls[1][0] == "DONE.json"
    assert (output_root.name, temp_root.name) in rename_calls
    assert not output_root.exists()
    assert temp_root.is_dir()
    assert not (temp_root / "DONE.json").exists()
    assert not (temp_root / "FAILED.json").exists()
    assert any(path.name.startswith(".DONE.json.demoted.") for path in temp_root.iterdir())


def test_d3_postmortem_publication_preserves_occupied_temp_collision_failure_without_final_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    temp_root = tmp_path / "feasibility_postmortem_001.tmp"
    output_root = tmp_path / "feasibility_postmortem_001"
    temp_root.mkdir()
    (temp_root / "DONE.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sf, "validate_postmortem_terminal_inventory_snapshot_at", lambda *args, **kwargs: None)
    monkeypatch.setattr(sf, "postmortem_terminal_fingerprint_at", lambda *args, **kwargs: ())
    real_rename_at = sf.atomic_rename_noreplace_at

    def validator_that_creates_collision(guard: object, _output_root: Path, **_kwargs: object) -> None:
        if output_root.exists() and not temp_root.exists():
            temp_root.mkdir()
            raise ValueError("forced post-rename validation failure")

    def fail_collision_preserve(
        source_dir_fd: int,
        source_name: str,
        destination_dir_fd: int,
        destination_name: str,
        destination_label: object,
    ) -> None:
        if source_name == temp_root.name and destination_name.startswith(f".{temp_root.name}.collision."):
            raise sf.FeasibilityPublicationError("forced collision preservation failure")
        real_rename_at(source_dir_fd, source_name, destination_dir_fd, destination_name, destination_label)

    monkeypatch.setattr(sf, "validate_postmortem_terminal_root_at", validator_that_creates_collision)
    monkeypatch.setattr(sf, "atomic_rename_noreplace_at", fail_collision_preserve)
    with pytest.raises(sf.FeasibilityPublicationError, match="temporary root is incomplete"):
        sf.publish_postmortem_root_or_leave_incomplete(temp_root, output_root, **_d3_minimal_publication_context(sf))
    assert not output_root.exists()
    assert temp_root.is_dir()
    _d3_assert_no_canonical_postmortem_terminal_markers(temp_root)


def test_d3_postmortem_publication_uses_quarantine_when_first_rollback_rename_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    temp_root = tmp_path / "feasibility_postmortem_001.tmp"
    output_root = tmp_path / "feasibility_postmortem_001"
    temp_root.mkdir()
    for name in ("DONE.json", "FAILED.json"):
        (temp_root / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sf, "validate_postmortem_terminal_inventory_snapshot_at", lambda *args, **kwargs: None)
    monkeypatch.setattr(sf, "postmortem_terminal_fingerprint_at", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        sf,
        "validate_postmortem_terminal_root_at",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("forced post-rename validation failure")) if output_root.exists() else None,
    )
    real_rename_at = sf.atomic_rename_noreplace_at

    def fail_first_rollback_rename(
        source_dir_fd: int,
        source_name: str,
        destination_dir_fd: int,
        destination_name: str,
        destination_label: object,
    ) -> None:
        if (source_name, destination_name) == (output_root.name, temp_root.name):
            raise OSError("forced first rollback rename failure")
        real_rename_at(source_dir_fd, source_name, destination_dir_fd, destination_name, destination_label)

    monkeypatch.setattr(sf, "atomic_rename_noreplace_at", fail_first_rollback_rename)
    with pytest.raises(sf.FeasibilityPublicationError, match="temporary root is incomplete"):
        sf.publish_postmortem_root_or_leave_incomplete(temp_root, output_root, **_d3_minimal_publication_context(sf))
    quarantines = [path for path in tmp_path.iterdir() if path.name.startswith(".feasibility_postmortem_001.evacuated.")]
    assert len(quarantines) == 1
    assert not output_root.exists()
    assert not temp_root.exists()
    _d3_assert_no_canonical_postmortem_terminal_markers(quarantines[0])


def test_d3_postmortem_publication_irrecoverable_failure_injection_matrix_exits_74(
    tmp_path: Path,
) -> None:
    scenarios = (
        "pre_done_demote_fail",
        "pre_failed_demote_fail",
        "post_done_demote_fail",
        "post_failed_demote_fail",
        "pre_done_absence_fail",
        "pre_failed_absence_fail",
        "post_done_absence_fail",
        "post_failed_absence_fail",
        "rollback_final_absence_fail",
        "quarantine_final_absence_fail",
        "final_quarantine_evacuation_fail",
    )
    child_template = r'''
import os
import sys
from pathlib import Path
sys.path.insert(0, {repo_root!r})
import scripts.phase8_sequence_feasibility as sf

scenario = {scenario!r}
base = Path({base!r})
temp_root = base / "feasibility_postmortem_001.tmp"
output_root = base / "feasibility_postmortem_001"
temp_root.mkdir(parents=True)
for name in ("DONE.json", "FAILED.json"):
    (temp_root / name).write_text("{{}}\n", encoding="utf-8")
context = {{
    "expected_input_binding": {{}},
    "expected_proposal_binding": {{}},
    "expected_implementation_binding": {{}},
    "expected_source_snapshot": sf.SourceSnapshot(commit="a" * 40, status_lines=(), ignored_inputs=()),
    "expected_teacher_rows": [],
    "expected_cell_rows": [],
    "expected_taxonomy_rows": [],
}}
sf.validate_postmortem_terminal_inventory_snapshot_at = lambda *args, **kwargs: None
sf.postmortem_terminal_fingerprint_at = lambda *args, **kwargs: ()

def validate(*args, **kwargs):
    if scenario.startswith("post") or scenario in ("quarantine_final_absence_fail", "rollback_final_absence_fail", "final_quarantine_evacuation_fail"):
        if output_root.exists():
            raise ValueError("forced post-rename validation failure")
    return None

sf.validate_postmortem_terminal_root_at = validate
real_rename_at = sf.atomic_rename_noreplace_at
real_stat = sf.os.stat
demoted_markers = set()

def rename_probe(source_dir_fd, source_name, destination_dir_fd, destination_name, destination_label):
    if scenario.endswith("done_demote_fail") and source_name == "DONE.json":
        raise sf.FeasibilityPublicationError("forced DONE demotion failure")
    if scenario.endswith("failed_demote_fail") and source_name == "FAILED.json":
        raise sf.FeasibilityPublicationError("forced FAILED demotion failure")
    real_rename_at(source_dir_fd, source_name, destination_dir_fd, destination_name, destination_label)
    if source_name in ("DONE.json", "FAILED.json") and destination_name.startswith("." + source_name + ".demoted."):
        demoted_markers.add(source_name)

def rename_probe_with_rollback_failures(source_dir_fd, source_name, destination_dir_fd, destination_name, destination_label):
    if scenario in ("quarantine_final_absence_fail", "final_quarantine_evacuation_fail") and (source_name, destination_name) == (output_root.name, temp_root.name):
        raise OSError("forced rollback rename failure")
    if scenario == "final_quarantine_evacuation_fail" and source_name == output_root.name and destination_name.startswith("." + output_root.name + ".evacuated."):
        raise OSError("forced final quarantine evacuation failure")
    rename_probe(source_dir_fd, source_name, destination_dir_fd, destination_name, destination_label)

def stat_probe(path, *args, **kwargs):
    target = None
    if scenario.endswith("done_absence_fail"):
        target = "DONE.json"
    if scenario.endswith("failed_absence_fail"):
        target = "FAILED.json"
    if target is not None and path == target and kwargs.get("dir_fd") is not None and target in demoted_markers:
        return real_stat(base)
    return real_stat(path, *args, **kwargs)

sf.atomic_rename_noreplace_at = rename_probe_with_rollback_failures
sf.os.stat = stat_probe

if scenario in ("rollback_final_absence_fail", "quarantine_final_absence_fail"):
    def fail_absence(_guard):
        sf.postmortem_fail_stop_74("forced final-root absence verification failure")
    sf.require_postmortem_output_absent_or_fail_stop = fail_absence

callback = (lambda: (_ for _ in ()).throw(RuntimeError("forced pre-rename failure"))) if scenario.startswith("pre_") else None
sf.publish_postmortem_root_or_leave_incomplete(temp_root, output_root, final_callback=callback, **context)
raise SystemExit(99)
'''
    for scenario in scenarios:
        completed = subprocess.run(
            [sys.executable, "-c", child_template.format(repo_root=str(REPO_ROOT), scenario=scenario, base=str(tmp_path / scenario))],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        assert completed.returncode == 74, (scenario, completed.stdout, completed.stderr)


def test_d3_postmortem_publication_guard_covers_callbacks_and_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")
    temp_root = tmp_path / "feasibility_postmortem_001.tmp"
    output_root = tmp_path / "feasibility_postmortem_001"
    temp_root.mkdir()
    monkeypatch.setattr(sf, "validate_postmortem_terminal_root_at", lambda *args, **kwargs: None)
    monkeypatch.setattr(sf, "validate_postmortem_terminal_inventory_snapshot_at", lambda *args, **kwargs: None)
    monkeypatch.setattr(sf, "postmortem_terminal_fingerprint_at", lambda *args, **kwargs: ())
    monkeypatch.setattr(sf, "atomic_rename_noreplace_at", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rename ran after forbidden callback")))
    required_context = {
        "expected_input_binding": {},
        "expected_proposal_binding": {},
        "expected_implementation_binding": {},
        "expected_source_snapshot": sf.SourceSnapshot(commit="a" * 40, status_lines=(), ignored_inputs=()),
        "expected_teacher_rows": [],
        "expected_cell_rows": [],
        "expected_taxonomy_rows": [],
    }
    with sf.PostmortemForbiddenOperationGuard():
        with pytest.raises(ValueError, match="forbidden|RNG-state mutation"):
            sf.publish_postmortem_root(
                temp_root,
                output_root,
                **required_context,
                final_callback=lambda: torch.manual_seed(123),
            )
    output_root.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        sf.publish_postmortem_root(temp_root, output_root, **required_context)


def test_d3_postmortem_model_snapshot_rejects_parameter_and_buffer_mutation() -> None:
    sf = importlib.import_module("scripts.phase8_sequence_feasibility")

    class BufferedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.register_buffer("running", torch.zeros(1))

    model = BufferedModel()
    snapshot = sf.postmortem_model_snapshot(model)
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(ValueError, match="changed"):
        sf.require_postmortem_model_unchanged(model, snapshot, "parameter mutation")
    model = BufferedModel()
    snapshot = sf.postmortem_model_snapshot(model)
    model.running.add_(1)
    with pytest.raises(ValueError, match="changed"):
        sf.require_postmortem_model_unchanged(model, snapshot, "buffer mutation")
