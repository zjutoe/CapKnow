#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import ctypes
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import errno
from functools import lru_cache
from hashlib import sha1, sha256
import inspect
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import platform
import random
import re
import stat
import subprocess
import sys
import time
from typing import Callable, Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and "postmortem-failure" in sys.argv[1:] and "__phase8_verifier_sha256__" not in globals():
    raise SystemExit("postmortem-failure must be launched through the Phase 8 inline verifier, not direct script execution.")
if str(REPO_ROOT) not in sys.path and "__phase8_verifier_sha256__" not in globals():
    sys.path.insert(0, str(REPO_ROOT))

import torch

from capability_certificate_lab.lm_bridge import corpus_generator as cg
from capability_certificate_lab.lm_bridge.model import (
    TIED_MODEL_PROTOCOL_REVISION,
    ToyCausalTransformer,
    TransformerConfig,
    build_historical_model,
    build_model,
    legacy_serialized_config,
    transformer_config,
)
from capability_certificate_lab.lm_bridge.tokenizer import ByteTokenizer
from capability_certificate_lab.lm_bridge.tokenizer import BOS_ID
from capability_certificate_lab.lm_bridge.tokenizer import EOS_ID
from capability_certificate_lab.lm_bridge.tokenizer import PAD_ID
from capability_certificate_lab.lm_bridge.tokenizer import SEP_ID
from capability_certificate_lab.lm_bridge.train import (
    ADAMW_BETAS,
    ADAMW_EPS,
    CORPUS_ORDER_RNG_OFFSET,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MODEL_RNG_OFFSET,
    TextRecord,
    WEIGHT_DECAY,
    contiguous_uint8_bytes,
    deterministic_batch_indices,
    encode_record_batch,
    make_optimizer,
    response_only_labels,
    response_only_loss,
    save_checkpoint,
    set_deterministic_backend,
    train_text_records,
    load_model_from_checkpoint,
    validate_tied_checkpoint_payload,
)


FAMILIES = ("hex_copy", "named_value_json", "boolean_json", "array_json")
MODEL_SIZES = ("small", "medium")
SEEDS = (0, 1, 2)
TRAIN_RECORDS_PER_FAMILY = 512
EVAL_RECORDS_PER_FAMILY = 64
TRAINING_STEPS = 1500
BATCH_SIZE = 64
PASS_THRESHOLD = 52
ROOT_RE = re.compile(r"^feasibility_(\d{3})$")
DIAGNOSTIC_ROOT_RE = re.compile(r"^feasibility_diagnostic_(\d{3})$")
SELECTION_RE = re.compile(r"^feasibility_selection_(\d{3})\.json$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ARTIFACT_PARENT = REPO_ROOT / "artifacts" / "phase8_toy_lm_bridge"
IGNORED_IMPORT_SURFACE_SUFFIXES = frozenset({".py", ".pyc", ".pyo", ".so", ".pth"})
IGNORED_IMPORT_HOOK_STEMS = frozenset({"sitecustomize", "usercustomize"})
HEX_OPERAND_RE = re.compile(r"(?<![0-9A-Fa-f])([0-9a-f]{16})(?![0-9A-Fa-f])", re.IGNORECASE)
NAMED_VALUE_KEYS = ("red", "blue", "green", "silver")
NAMED_VALUE_KEYS_BY_SPLIT = {
    "train": NAMED_VALUE_KEYS,
    "eval": NAMED_VALUE_KEYS,
}
NAMED_VALUE_RE = re.compile(
    r"\b(?:red|blue|green|silver)-[0-9a-f]{4}\b",
    re.IGNORECASE,
)
NAMED_FIELD_RE = re.compile(
    r"\b(red|blue|green|silver)=((?:red|blue|green|silver)-[0-9a-f]{4})\b",
    re.IGNORECASE,
)
ARRAY_ITEM_RE = re.compile(r"\bq[0-9a-f]{4}\b", re.IGNORECASE)
BOOLEAN_LABELS = ("affirm", "reject")
BOOLEAN_LABEL_TRUTH = {"affirm": True, "reject": False}
BOOLEAN_OPERAND_RE = re.compile(r"\b(affirm|reject)-([0-9a-f]{16})\b", re.IGNORECASE)
NAMED_TARGET_RE = re.compile(r"\b(?:key|name)\s+(red|blue|green|silver)\b", re.IGNORECASE)
PROMPT_PLACEHOLDER_RE = re.compile(r"\{[abc]\}")
ACCEPTED_INDEPENDENT_REVIEW_VERDICT = "ACCEPT"
DIAGNOSTIC_ARTIFACT_CLASS = "non_evidence_feasibility_diagnostic"
DIAGNOSTIC_STEPS = 3000
DIAGNOSTIC_INPUT_ROOT_NAME = "feasibility_004"
DIAGNOSTIC_TRAJECTORY_EVIDENCE_SCHEMA = "phase8_array_small_3000_training_trajectory_v1"
DIAGNOSTIC_MODEL_STATE_FINGERPRINT_SCHEMA = "phase8_model_state_fingerprint_v1"
DIAGNOSTIC_BATCH_SCHEDULE_SCHEMA = "phase8_batch_schedule_digest_v1"
DIAGNOSTIC_TRAINING_TRACE_SCHEMA = "phase8_training_trace_digest_v1"
DIAGNOSTIC_INPUT_SOURCE_COMMIT = "3cb75ad550c4357562c0d4d9a9b098bfb2cf66ea"
DIAGNOSTIC_HANDOFF_PATH = "phase8/Task_010C_D1_Feasibility_Failure_Diagnostic.md"
DIAGNOSTIC_ACCEPTED_PROTOCOL_COMMIT = "f2aa335256201672767eac1840e135672b43e046"
DIAGNOSTIC_HANDOFF_BLOB = "b5a45e3a33baf53d4de9c809bc9ab2c59f400e37"
DIAGNOSTIC_REQUIRED_DEVICE = "cuda:0"
DIAGNOSTIC_REQUIRED_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONPATH": ".",
}
FEASIBILITY_REQUIRED_DEVICE = "cuda:0"
FEASIBILITY_REQUIRED_ROOT = "artifacts/phase8_toy_lm_bridge/feasibility_005"
FEASIBILITY_REQUIRED_PREDECESSOR_ROOTS = (
    "artifacts/phase8_toy_lm_bridge/feasibility_001",
    "artifacts/phase8_toy_lm_bridge/feasibility_002",
    "artifacts/phase8_toy_lm_bridge/feasibility_003",
    "artifacts/phase8_toy_lm_bridge/feasibility_004",
)
FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT = "artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001"
FEASIBILITY_REQUIRED_ENV = DIAGNOSTIC_REQUIRED_ENV
FEASIBILITY_REQUIRED_RUNTIME_ENV = {
    "cuda": "13.0",
    "cuda_available": True,
    "gpu": "NVIDIA A800 80GB PCIe",
    "gpu_driver": "590.48.01",
    "platform": "Linux-6.12.0-184.el10.x86_64-x86_64-with-glibc2.39",
    "python": "3.13.9 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 19:16:10) [GCC 11.2.0]",
    "torch": "2.9.1+cu130",
}
DIAGNOSTIC_BACKEND_SEED = 0
DIAGNOSTIC_INPUT_CHECKSUMS = {
    "manifest.json": "19cf2db4df6f6928d0afa7ab8aef8d891e944152e480541f5aaf8702565e8d00",
    "summary.json": "4eaf7b5742161ff6ea5612f7c53668ee71c38f598e18da27a375fafebb6dc506",
    "FAILED.json": "6b3c81d3be78c8e3deeaf991dad4a1e2df41171187ba4e20533799162a46f3ae",
}
DIAGNOSTIC_CORE_BLOBS = {
    "capability_certificate_lab/lm_bridge/tokenizer.py": "79ded37d148d40b0ba4e501881f585d74e971798",
    "capability_certificate_lab/lm_bridge/model.py": "70e57d8ccea8773b99ec797634e3ea32fe0bb9bf",
    "capability_certificate_lab/lm_bridge/train.py": "fe2d6a901c96dc35a5e90b12996d6c01df2f2f56",
}
FROZEN_PARAMETER_COUNTS = {
    "small": 133120,
    "medium": 859392,
}
HISTORICAL_PARAMETER_COUNTS = {
    "small": 149760,
    "medium": 892672,
}
HISTORICAL_FEASIBILITY_CONFIGURATION = {
    "families": list(FAMILIES),
    "model_sizes": list(MODEL_SIZES),
    "seeds": list(SEEDS),
    "train_records_per_family": TRAIN_RECORDS_PER_FAMILY,
    "eval_records_per_family": EVAL_RECORDS_PER_FAMILY,
    "training_steps": TRAINING_STEPS,
    "batch_size": BATCH_SIZE,
    "pass_threshold": PASS_THRESHOLD,
    "context_window_tokens": ByteTokenizer.max_sequence_length,
    "generation_window_tokens": ByteTokenizer.max_generated_tokens,
}
HISTORICAL_FEASIBILITY_ROOTS = {
    "feasibility_001": {
        "source_commit": "949683d7fd20461da97aa439915f432d18da7680",
        "manifest_sha256": "422f31c5110794892412499233406669edea82d271538c77f58e1ce493d8dd88",
        "terminal_sha256": "47187b0e2a2d0d151fd1ea8a19c0809bf7ca9eff5d5d9b401996cff763eb64ab",
        "terminal": "FAILED.json",
    },
    "feasibility_002": {
        "source_commit": "ef893716373b9c83a33e0bfe71e64e1aa93f55bf",
        "manifest_sha256": "408f5737e0c574d355f589ce3180ee436a5a85bfdf74e448ae1575e162fb6151",
        "terminal_sha256": "2d3b15f074c5bf8ccc14ae0b00a3d4a97b58baa705f90c9eade24d9294da2eda",
        "terminal": "FAILED.json",
    },
    "feasibility_003": {
        "source_commit": "530ea0bf96c9eab5fbc94bf3951f5bf712315a20",
        "manifest_sha256": "399aa93cb43a8d086ba2f26a87af25955e782d2d045eadafcce84c763de63c7b",
        "terminal_sha256": "82c5f88e4a4159e767fbeac1e802b9bb1db5f998c79dc937288ce378c471253e",
        "terminal": "FAILED.json",
    },
    "feasibility_004": {
        "source_commit": "3cb75ad550c4357562c0d4d9a9b098bfb2cf66ea",
        "manifest_sha256": "19cf2db4df6f6928d0afa7ab8aef8d891e944152e480541f5aaf8702565e8d00",
        "terminal_sha256": "6b3c81d3be78c8e3deeaf991dad4a1e2df41171187ba4e20533799162a46f3ae",
        "terminal": "FAILED.json",
    },
}
DECISION_DIAGNOSTIC_ROOT_BINDING = {
    "path": FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT,
    "source_commit": "cb49ebdf577df78e97b7748aadc48f8547a70f6a",
    "artifact_class": DIAGNOSTIC_ARTIFACT_CLASS,
    "feasibility_selection_eligible": False,
    "task_010d_authorized": False,
    "manifest_sha256": "5589ac3a1215ea4cf451bd14f75ab64ab53b183548d1d42134e3f52b6ad12241",
    "summary_sha256": "91582eda4629f29436289896c9f3802fe243c2b86a8f622c98c6c97e579a34ac",
    "terminal_sha256": "39f3893627d24162d94f541217f4f754c2e193fe4a61361dacc6156b84826f92",
    "accepted_proposal_commit": "9a767c6708c7c69f5ba98848250afcf50c8c5a6f",
    "independent_review_verdict": ACCEPTED_INDEPENDENT_REVIEW_VERDICT,
}
D1_HISTORICAL_CHECKPOINT_SHA256 = {
    "array_json__small__3000__seed0/checkpoint_step3000.pt": "8311fb311fc9f7b707be4bb6fe82eed19b99c024c31341e9013052903f00365b",
    "array_json__small__3000__seed1/checkpoint_step3000.pt": "b2684d53fde02136053d89090ef1595bb781a749c511f4c0a52a8c1c318d9d0a",
    "array_json__small__3000__seed2/checkpoint_step3000.pt": "709d8c628d4835dfc0fe5ed8f382c80919539207c7ab63135f4c592f6d04ad2f",
}
FEASIBILITY_RECORD_HASHES = {
    "hex_train512": "8798d57c3bdde6693d8046e3a7525687f06976074dc0cceffaf1c674a0f8c390",
    "hex_eval64": "be97a3a2877fa5f8e6c2e85a4213f975ccc89de6b3c61338dff100d1d1f21868",
    "named_train512": "9b55084d281a9420e12a1b6a35c3eb199abd74f4b766a14a833e6241c59564b8",
    "named_eval64": "6cc73c98616430a12a357de99702e8cb4db95258225adab80fd11212aa203d20",
    "boolean_train512": "52b5d74bec65bc903938b8b1993c1bf6df489d9c13659ad62aca76336c2d77f1",
    "boolean_eval64": "3c2ce21e86dd88ce85c6d2927c8ff09eb8f87cecb223bdeb1009debff68dff3b",
    "array_train512": "6bbcae6203dfcf443f9871f30b48c29580fa721317387ff26b757b4066a98e94",
    "array_eval64": "8d504dd63ad2538aedc8195a4f3c0729ed38ec95a078b9c9895fbf93cf8c173a",
}
CURRENT_CELL_KEYS = frozenset({
    "family",
    "model_size",
    "seed",
    "eval_count",
    "exact_matches",
    "passed",
    "generations_path",
    "checkpoint_path",
    "parameter_count",
    "embedding_weight_tying",
    "model_protocol_revision",
})
HISTORICAL_CELL_KEYS = frozenset({
    "family",
    "model_size",
    "seed",
    "eval_count",
    "exact_matches",
    "passed",
    "generations_path",
    "checkpoint_path",
    "parameter_count",
})
CURRENT_MANIFEST_KEYS = frozenset({
    "protocol",
    "terminal_status",
    "failure",
    "source_commit",
    "source_provenance",
    "configuration",
    "decision_diagnostic",
    "exact_command",
    "deterministic_flags",
    "record_hashes",
    "environment",
    "cells",
    "predecessor_roots",
    "predecessor_selections",
    "file_inventory",
})
CURRENT_SUMMARY_KEYS = frozenset({
    "protocol",
    "terminal_status",
    "failure",
    "source",
    "configuration",
    "decision_diagnostic",
    "exact_command",
    "deterministic_flags",
    "record_hashes",
    "cells",
    "summary",
    "predecessor_root_count",
    "predecessor_selection_count",
})
CURRENT_TERMINAL_KEYS = frozenset({
    "status",
    "manifest_path",
    "manifest_sha256",
    "pass_threshold",
    "configuration",
    "decision_diagnostic",
    "exact_command",
    "deterministic_flags",
    "record_hashes",
    "cells",
})
POSTMORTEM_ROOT_RE = re.compile(r"^feasibility_postmortem_(\d{3})$")
POSTMORTEM_TEMP_ROOT_RE = re.compile(r"^feasibility_postmortem_(\d{3})\.tmp$")
POSTMORTEM_SCHEMA_VERSION = "phase8_feasibility_postmortem_v1"
POSTMORTEM_ARTIFACT_CLASS = "non_evidence_feasibility_postmortem"
POSTMORTEM_REQUIRED_INPUT_ROOT = FEASIBILITY_REQUIRED_ROOT
POSTMORTEM_REQUIRED_OUTPUT_ROOT = "artifacts/phase8_toy_lm_bridge/feasibility_postmortem_001"
POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT = "06ee71d958eb1c4cb446ede3d120e99eb64bba97"
POSTMORTEM_PROPOSAL_PATH = "phase8/Task_010C_D3_Feasibility_005_Postmortem_Proposal.md"
POSTMORTEM_PROPOSAL_BLOB = "5ea32f67d078408a6764adbb2d66518f713245d7"
POSTMORTEM_VERIFIER_PATH = "phase8/Task_010C_D3_Postmortem_Verifier.py.txt"
POSTMORTEM_VERIFIER_BLOB = "87ea81df9b2dfdb4f5f5e85dfe5e7340e2b9b6aa"
POSTMORTEM_VERIFIER_SHA256 = "890671e89404a1c172b669cad8200fe926b9c052de0ab32d5c860549dd903432"
POSTMORTEM_VERIFIER_BYTE_COUNT = 34156
POSTMORTEM_EXECUTABLE = "/opt/anaconda3/bin/python"
POSTMORTEM_CWD = "/home/mye/src/llm/CapKnow"
POSTMORTEM_RUNNER_PATH = "scripts/phase8_sequence_feasibility.py"
POSTMORTEM_EXECVE_ENVIRONMENT = {
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONPATH": ".",
}
POSTMORTEM_INPUT_SOURCE_COMMIT = "2dc8c50fab9ece27a07eb0dc04021b78de52895c"
POSTMORTEM_INPUT_CHECKSUMS = {
    "manifest.json": "51b43e3f3eba98b31c3574a9720c0df3f8e2a72c94d22730d70eb8682e8f516b",
    "summary.json": "bfea56b52dfd1ffe094feb0774ce7895f4dd703770451aeda4d899aafdfef785",
    "FAILED.json": "6ac97f081586ca7fb8d7d8297dae4e0c7a3d0314a0567bbc493fce1b5dcc4c08",
}
POSTMORTEM_ROW_COUNTS = {
    "teacher_forced_rows": 13824,
    "cell_metrics": 24,
    "error_taxonomy": 1536,
}
POSTMORTEM_TERMINAL_FILE_NAMES = (
    "DONE.json",
    "cell_metrics.jsonl",
    "error_taxonomy.jsonl",
    "manifest.json",
    "summary.json",
    "teacher_forced_rows.jsonl",
)
POSTMORTEM_TEACHER_ROW_KEYS = frozenset({
    "schema_version",
    "family",
    "model_size",
    "seed",
    "split",
    "record_index",
    "template_id",
    "operand_id",
    "batch_index",
    "row_within_batch",
    "selected_token_count",
    "correct_token_count",
    "sequence_exact",
    "nll_numerator_hex",
    "checkpoint_path",
    "checkpoint_sha256",
})
POSTMORTEM_TEACHER_AGGREGATE_KEYS = frozenset({
    "record_count",
    "sequence_exact_numerator",
    "selected_token_count",
    "correct_token_count",
    "nll_numerator_hex",
    "nll_per_token_hex",
})
POSTMORTEM_CELL_KEYS = frozenset({
    "schema_version",
    "family",
    "model_size",
    "seed",
    "checkpoint_path",
    "checkpoint_sha256",
    "generation_path",
    "generation_sha256",
    "checkpoint_metadata",
    "train_teacher_forced",
    "eval_teacher_forced",
    "retained_greedy_exact",
    "paired_eval_exact",
})
POSTMORTEM_CHECKPOINT_METADATA_KEYS = frozenset({
    "family",
    "model_size",
    "seed",
    "training_steps",
    "training_loss",
    "training_accuracy",
})
POSTMORTEM_TAXONOMY_KEYS = frozenset({
    "schema_version",
    "family",
    "model_size",
    "seed",
    "record_index",
    "generation_path",
    "generation_sha256",
    "common",
    "named",
    "array",
})
POSTMORTEM_COMMON_KEYS = frozenset({
    "prompt",
    "expected_response",
    "generated_response",
    "raw_token_ids",
    "generation_error",
    "greedy_exact",
    "decode_valid",
    "has_eos",
    "hits_generation_cap",
    "hits_context_cap",
    "generation_token_count",
    "generation_utf8_bytes",
    "target_utf8_bytes",
    "length_matches",
    "hamming_distance",
    "edit_distance",
    "first_error",
})
POSTMORTEM_NAMED_KEYS = frozenset({
    "surface_source",
    "operand_source",
    "valid_json_string",
    "valid_named_grammar",
    "target_prefix",
    "occurs_in_prompt",
    "exact_target_value",
    "exact_distractor_value",
    "suffix_positional_correct",
    "suffix_positional_total",
    "suffix_hamming_distance",
    "suffix_edit_distance",
    "suffix_first_error",
})
POSTMORTEM_ARRAY_KEYS = frozenset({
    "comparison_source",
    "training_steps",
    "target_item_count",
    "valid_json_syntax",
    "valid_array_schema",
    "correct_item_count",
    "positional_exact_count",
    "positional_denominator",
    "missing_items",
    "extra_items",
    "all_items_copied",
    "all_items_exact",
    "first_wrong_item_position",
})
POSTMORTEM_SUMMARY_KEYS = frozenset({
    "schema_version",
    "artifact_class",
    "input_binding",
    "proposal_binding",
    "implementation_binding",
    "configuration",
    "row_counts",
    "cells",
    "named_aggregates",
    "array_aggregates",
    "interpretation_limits",
    "terminal_status",
})
POSTMORTEM_MANIFEST_KEYS = frozenset({
    "schema_version",
    "artifact_class",
    "input_binding",
    "proposal_binding",
    "implementation_binding",
    "runner_path",
    "exact_command",
    "environment",
    "deterministic_flags",
    "rng_state_contract",
    "configuration",
    "row_counts",
    "file_inventory",
    "terminal_status",
})
POSTMORTEM_EXACT_COMMAND_KEYS = frozenset({"executable", "cwd", "environment", "argv"})
POSTMORTEM_DONE_KEYS = frozenset({
    "schema_version",
    "status",
    "manifest_path",
    "manifest_sha256",
    "input_manifest_sha256",
    "row_counts",
})
POSTMORTEM_INPUT_BINDING_KEYS = frozenset({
    "root",
    "source_commit",
    "manifest_path",
    "manifest_sha256",
    "summary_path",
    "summary_sha256",
    "terminal_path",
    "terminal_sha256",
    "configuration",
    "record_hashes",
    "file_inventory",
    "cells",
})
POSTMORTEM_PROPOSAL_BINDING_KEYS = frozenset({"commit", "path", "blob"})
POSTMORTEM_IMPLEMENTATION_BINDING_KEYS = frozenset({"commit", "runner_path", "runner_blob"})
POSTMORTEM_RNG_CONTRACT_KEYS = frozenset({
    "record_reconstruction_global_state_unchanged",
    "constructor_scope_restored",
    "post_load_state_unchanged",
    "post_inference_state_unchanged",
})
DIAGNOSTIC_RECORD_HASHES = {
    "named_train512": "9b55084d281a9420e12a1b6a35c3eb199abd74f4b766a14a833e6241c59564b8",
    "named_train_first64": "6806025fa541c4f71d84a2dfce29777e0ac4e056c1bad762aeb10d309f5503ad",
    "named_eval64": "6cc73c98616430a12a357de99702e8cb4db95258225adab80fd11212aa203d20",
    "array_train512": "6bbcae6203dfcf443f9871f30b48c29580fa721317387ff26b757b4066a98e94",
    "array_eval64": "8d504dd63ad2538aedc8195a4f3c0729ed38ec95a078b9c9895fbf93cf8c173a",
}
DIAGNOSTIC_REPAIR_ALLOWED_DIFF_PATHS = frozenset({
    "scripts/phase8_sequence_feasibility.py",
    "tests/test_lm_bridge.py",
})
NAMED_DIAGNOSTIC_CELLS = (
    "seen_surface_seen_operand",
    "held_surface_seen_operand",
    "seen_surface_held_operand",
    "held_surface_held_operand",
)

SCIENTIFIC_MARKERS = (
    *cg.TASK_ORDER,
    "Phase 8",
    "phase8",
    "primitive",
    "capability",
    "certificate",
    "knowledge state",
    "state id",
    "payload",
    "template",
    "MEMORY",
    "SEARCH",
    "FILTER",
    "CONDITION",
)
SCIENTIFIC_PROMPT_SNIPPETS = (
    "A recorded claim has value true. Write that Boolean in compact JSON.",
    "A recorded claim has value false. Write that Boolean in compact JSON.",
    "state 0101",
    "graph A->B",
    "seed 123",
    "model id 7",
    "prerequisite rule table",
)
SCIENTIFIC_LEAK_PATTERNS = (
    re.compile(
        r"\b(?:MEMORY|SEARCH|FILTER|CONDITION|MEMORY_FILTER|FILTER_CONDITION|SEARCH_CONDITION|MEMORY_SEARCH)\b"
    ),
    *(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\b(?:primitive|capability|certificate|knowledge state|graph|prerequisite)\b",
            r"\brule\s+tables?\b",
            r"\b(?:state|model)\s*(?:id|ids|[0-9])\b",
            r"\bseed\s+[0-9]+\b",
            r"\bstate\s+[01]{4}\b",
            r"[A-Z]\s*\+\s*[A-Z]\s*->\s*[A-Z]",
            r"->",
        )
    ),
)

GENERIC_JSON_MARKERS = frozenset({"true", "false", "null", "unable", "[]", "{}"})

PROMPT_SURFACES: dict[str, dict[str, tuple[str, ...]]] = {
    "hex_copy": {
        "train": (
            "Copy these sixteen hex digits from batch {a}: {b}",
            "Send only sixteen hex digits for ticket {a}: {b}",
            "Write sixteen lowercase hex chars after tag {a}: {b}",
            "Return the sixteen hex symbols at slot {a}: {b}",
        ),
        "eval": (
            "Echo these sixteen hex digits from batch {a}: {b}",
            "Give only sixteen hex digits for ticket {a}: {b}",
            "Print sixteen lowercase hex chars after tag {a}: {b}",
            "Output the sixteen hex symbols at slot {a}: {b}",
        ),
    },
    "named_value_json": {
        "train": (
            "Send JSON string for key {a}; fields {b}",
            "Write JSON string for name {a}; roster {b}",
            "Choose key {a}; reply with JSON string from {b}",
            "For key {a}, emit JSON string using list {b}",
        ),
        "eval": (
            "Give JSON string for key {a}; fields {b}",
            "Print JSON string for name {a}; roster {b}",
            "Select key {a}; reply with JSON string from {b}",
            "For key {a}, give JSON string using list {b}",
        ),
    },
    "boolean_json": {
        "train": (
            "Reply JSON bool for lexical mark: {a}.",
            "Convert lexical mark to JSON bool: {a}.",
            "For lexical mark {a}, write JSON bool.",
            "Check lexical mark {a} and return JSON bool.",
        ),
        "eval": (
            "Write JSON bool for lexical mark: {a}.",
            "Produce lexical mark as JSON bool: {a}.",
            "For lexical mark {a}, print JSON bool.",
            "Judge lexical mark {a} and return JSON bool.",
        ),
    },
    "array_json": {
        "train": (
            "Return compact JSON array from chunks: {a}",
            "Write compact JSON array for pieces: {a}",
            "Form JSON array using these chunks: {a}",
            "Emit JSON array only from chunks: {a}",
        ),
        "eval": (
            "Output compact JSON array from chunks: {a}",
            "Print compact JSON array for pieces: {a}",
            "Make JSON array using these chunks: {a}",
            "Give JSON array only from chunks: {a}",
        ),
    },
}


@dataclass(frozen=True)
class FeasibilityRecord:
    family: str
    split: str
    index: int
    template_id: str
    operand_id: str
    prompt: str
    answer: str
    semantic_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptSurfaceSeed:
    prompt: str
    canonical_context: str
    normalized_payload: object


@dataclass(frozen=True)
class SourceSnapshot:
    commit: str
    status_lines: tuple[str, ...]
    ignored_inputs: tuple[object, ...]
    runner_blob: str | None = None


@dataclass
class PostmortemPublicationGuard:
    parent_fd: int
    temp_fd: int
    parent_path: Path
    temp_root: Path
    output_root: Path
    parent_identity: tuple[int, int, int]
    temp_identity: tuple[int, int, int]
    closed: bool = False

    @property
    def temp_name(self) -> str:
        return self.temp_root.name

    @property
    def output_name(self) -> str:
        return self.output_root.name

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for descriptor in (self.temp_fd, self.parent_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass

    def __enter__(self) -> "PostmortemPublicationGuard":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


@dataclass
class FeasibilityValidationContext:
    active_roots: set[Path] = field(default_factory=set)
    validated_roots: set[tuple[Path, bool, str]] = field(default_factory=set)
    inventory_bound_paths: dict[tuple[Path, str], set[Path]] = field(default_factory=dict)
    selection_records: dict[tuple[Path, str], tuple[dict[str, object], dict[str, tuple[str, str, str]]]] = field(default_factory=dict)


class SourceChangedError(RuntimeError):
    pass


class DiagnosticPublicationError(RuntimeError):
    pass


class FeasibilityPublicationError(RuntimeError):
    pass


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def reject_scientific_markers(text: str) -> None:
    cg.leak_check_model_text(text)
    for pattern in SCIENTIFIC_LEAK_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"Feasibility text contains forbidden Phase 8 scientific metadata: {text!r}.")
    upper_text = text.upper()
    for marker in SCIENTIFIC_MARKERS:
        if marker.upper() in upper_text:
            raise ValueError(f"Feasibility text contains forbidden Phase 8 scientific marker: {marker!r}.")
    for marker in SCIENTIFIC_PROMPT_SNIPPETS:
        if marker in text:
            raise ValueError(f"Feasibility text contains forbidden Phase 8 scientific prompt marker: {marker!r}.")
    lower_text = text.casefold()
    for marker in phase8_scientific_identity_markers():
        if marker.casefold() in lower_text:
            raise ValueError(f"Feasibility text contains forbidden Phase 8 scientific identity marker: {marker!r}.")
    for pattern in phase8_scientific_prompt_patterns():
        if pattern.search(text):
            raise ValueError("Feasibility text matches a forbidden Phase 8 scientific prompt template.")


def validate_feasibility_records(records: Sequence[FeasibilityRecord]) -> None:
    if not records:
        raise ValueError("records must be non-empty.")
    for family in {record.family for record in records}:
        validate_prompt_surface_contract(family)
    for record in records:
        reject_scientific_markers(record.prompt)
        reject_scientific_markers(record.answer)
        reject_scientific_markers(record.template_id)
        reject_scientific_markers(record.operand_id)
        for value in semantic_values_for_record(record):
            reject_scientific_markers(value)
    train_templates = {r.template_id.casefold() for r in records if r.split == "train"}
    eval_templates = {r.template_id.casefold() for r in records if r.split == "eval"}
    if train_templates & eval_templates:
        raise ValueError("Feasibility train/evaluation template_id values must be disjoint.")
    train_operands = {r.operand_id.casefold() for r in records if r.split == "train"}
    eval_operands = {r.operand_id.casefold() for r in records if r.split == "eval"}
    if train_operands & eval_operands:
        raise ValueError("Feasibility train/evaluation operand_id values must be disjoint.")
    train = {(r.template_id, r.operand_id, r.prompt, r.answer) for r in records if r.split == "train"}
    eval_ = {(r.template_id, r.operand_id, r.prompt, r.answer) for r in records if r.split == "eval"}
    if train & eval_:
        raise ValueError("Feasibility train/evaluation records must be disjoint.")
    train_semantic = {
        value
        for record in records
        if record.split == "train"
        for value in semantic_values_for_record(record)
    }
    eval_semantic = {
        value
        for record in records
        if record.split == "eval"
        for value in semantic_values_for_record(record)
    }
    overlap = train_semantic & eval_semantic
    if overlap:
        raise ValueError(f"Feasibility train/evaluation semantic values must be disjoint: {sorted(overlap)!r}.")
    validate_paired_length_profiles(records)
    validate_hex_copy_contract(records)
    validate_boolean_label_contract(records)
    validate_named_value_contract(records)
    validate_array_count_contract(records)


def semantic_values_for_record(record: FeasibilityRecord) -> frozenset[str]:
    prompt_and_answer = f"{record.prompt}\n{record.answer}"
    values: set[str] = set()
    if record.family == "hex_copy":
        values.update(value.lower() for value in HEX_OPERAND_RE.findall(prompt_and_answer))
    elif record.family == "named_value_json":
        decoded = json.loads(record.answer)
        if not isinstance(decoded, str):
            raise ValueError("named_value_json answers must be JSON strings.")
        if NAMED_VALUE_RE.fullmatch(decoded) is None:
            raise ValueError("named_value_json answers must use the fixed held-out value grammar.")
        values.add(decoded.casefold())
        values.update(value.casefold() for value in NAMED_VALUE_RE.findall(prompt_and_answer))
        roster = tuple((key.casefold(), value.casefold()) for key, value in NAMED_FIELD_RE.findall(record.prompt))
        if roster:
            values.add("named_roster:" + compact_json(roster))
    elif record.family == "array_json":
        decoded = json.loads(record.answer)
        if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
            raise ValueError("array_json answers must be JSON arrays of strings.")
        normalized = [item.casefold() for item in decoded]
        if not all(ARRAY_ITEM_RE.fullmatch(item) for item in normalized):
            raise ValueError("array_json answers must use the fixed held-out item grammar.")
        values.update(normalized)
        values.add("array:" + compact_json(normalized))
        values.update(value.casefold() for value in ARRAY_ITEM_RE.findall(prompt_and_answer))
    elif record.family == "boolean_json":
        operands = parsed_boolean_operands(record.prompt)
        if not operands:
            raise ValueError("boolean_json prompts must contain a supported lexical Boolean operand.")
        values.update(f"boolean:{label}:{suffix}" for label, suffix in operands)
    else:
        raise ValueError(f"Unknown feasibility family: {record.family!r}.")
    return frozenset(values)


def parsed_boolean_operands(text: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (match.group(1).casefold(), match.group(2).casefold())
        for match in BOOLEAN_OPERAND_RE.finditer(text)
    )


def boolean_operand_truth(label: str, suffix: str) -> bool:
    if not re.fullmatch(r"[0-9a-f]{16}", suffix):
        raise ValueError(f"Unsupported boolean_json suffix: {suffix!r}.")
    try:
        return BOOLEAN_LABEL_TRUTH[label]
    except KeyError as exc:
        raise ValueError(f"Unsupported boolean_json label: {label!r}.") from exc


def validate_paired_length_profiles(records: Sequence[FeasibilityRecord]) -> None:
    for family in {record.family for record in records}:
        train = tuple(sorted((record for record in records if record.family == family and record.split == "train"), key=lambda r: r.index))
        eval_ = tuple(sorted((record for record in records if record.family == family and record.split == "eval"), key=lambda r: r.index))
        paired_count = min(len(train), len(eval_), EVAL_RECORDS_PER_FAMILY)
        if paired_count == 0:
            continue
        train_prompt_lengths = tuple(len(record.prompt.encode("utf-8")) for record in train[:paired_count])
        eval_prompt_lengths = tuple(len(record.prompt.encode("utf-8")) for record in eval_[:paired_count])
        if train_prompt_lengths != eval_prompt_lengths:
            raise ValueError(f"Feasibility paired prompt byte-length profile must match for {family}.")
        train_answer_lengths = tuple(len(record.answer.encode("utf-8")) for record in train[:paired_count])
        eval_answer_lengths = tuple(len(record.answer.encode("utf-8")) for record in eval_[:paired_count])
        if train_answer_lengths != eval_answer_lengths:
            raise ValueError(f"Feasibility paired answer byte-length profile must match for {family}.")


def validate_boolean_label_contract(records: Sequence[FeasibilityRecord]) -> None:
    boolean_records = tuple(record for record in records if record.family == "boolean_json")
    if not boolean_records:
        return
    operand_by_record: dict[FeasibilityRecord, tuple[str, str]] = {}
    for record in boolean_records:
        operands = parsed_boolean_operands(record.prompt)
        if len(operands) != 1:
            raise ValueError("boolean_json prompts must contain exactly one lexical Boolean operand.")
        operand_by_record[record] = operands[0]
        bound_prompt_surface_id(record, a=f"{operands[0][0]}-{operands[0][1]}")
        expected_answer = "true" if boolean_operand_truth(*operands[0]) else "false"
        if record.answer != expected_answer:
            raise ValueError("boolean_json answer must match deterministic lexical label truth.")
    for split in ("train", "eval"):
        split_suffixes = [
            operand_by_record[record][1]
            for record in boolean_records
            if record.split == split
        ]
        if len(set(split_suffixes)) != len(split_suffixes):
            raise ValueError("boolean_json suffixes must be unique within each split.")
    train_suffixes = {
        operand_by_record[record][1]
        for record in boolean_records
        if record.split == "train"
    }
    eval_suffixes = {
        operand_by_record[record][1]
        for record in boolean_records
        if record.split == "eval"
    }
    if train_suffixes & eval_suffixes:
        raise ValueError("boolean_json train/evaluation suffixes must be disjoint independent of label.")
    for split, expected_count in (("train", TRAIN_RECORDS_PER_FAMILY), ("eval", EVAL_RECORDS_PER_FAMILY)):
        split_records = tuple(record for record in boolean_records if record.split == split)
        if len(split_records) != expected_count:
            continue
        if sum(record.answer == "true" for record in split_records) != len(split_records) // 2:
            raise ValueError("boolean_json truth labels must be balanced overall within each split.")
        suffix_tails_by_label = {
            label: sorted(
                operand_by_record[record][1][-1]
                for record in split_records
                if operand_by_record[record][0] == label
            )
            for label in BOOLEAN_LABELS
        }
        if suffix_tails_by_label["affirm"] != suffix_tails_by_label["reject"]:
            raise ValueError("boolean_json suffix-tail distributions must match exactly across labels.")
        by_template: dict[str, list[FeasibilityRecord]] = {}
        for record in split_records:
            by_template.setdefault(record.template_id, []).append(record)
        if len(by_template) != 4:
            raise ValueError("boolean_json must use four templates per split.")
        for template_records in by_template.values():
            true_count = sum(record.answer == "true" for record in template_records)
            false_count = sum(record.answer == "false" for record in template_records)
            if true_count != false_count:
                raise ValueError("boolean_json truth labels must be balanced within each template.")


def validate_hex_copy_contract(records: Sequence[FeasibilityRecord]) -> None:
    for record in records:
        if record.family != "hex_copy":
            continue
        prompt_values = tuple(value.casefold() for value in HEX_OPERAND_RE.findall(record.prompt))
        if len(prompt_values) != 1:
            raise ValueError("hex_copy prompts must contain exactly one sixteen-digit hex operand.")
        value = prompt_values[0]
        if record.answer != value:
            raise ValueError("hex_copy answer must exactly match the prompt operand.")
        bound_prompt_surface_id(record, a=str(record.index % 17), b=value)


def validate_named_value_contract(records: Sequence[FeasibilityRecord]) -> None:
    named_records = tuple(record for record in records if record.family == "named_value_json")
    if not named_records:
        return
    target_by_record = {record: named_value_target_key(record) for record in named_records}
    surface_by_record = {
        record: bound_prompt_surface_id(
            record,
            a=target_by_record[record],
            b=named_value_field_text(record),
        )
        for record in named_records
    }
    for split, expected_count in (("train", TRAIN_RECORDS_PER_FAMILY), ("eval", EVAL_RECORDS_PER_FAMILY)):
        split_records = tuple(record for record in named_records if record.split == split)
        if len(split_records) != expected_count:
            continue
        template_ids = tuple(sorted({record.template_id for record in split_records}))
        if len(template_ids) != 4:
            raise ValueError("named_value_json must use four templates per split.")
        counts: dict[tuple[str, str], int] = {
            (template_id, target): 0
            for template_id in template_ids
            for target in NAMED_VALUE_KEYS
        }
        for record in split_records:
            target = target_by_record[record]
            surface_id = surface_by_record[record]
            counts[(f"seq_named_value_json_{split}_{surface_id}", target)] += 1
        expected_cell_count = expected_count // (len(template_ids) * len(NAMED_VALUE_KEYS))
        if any(count != expected_cell_count for count in counts.values()):
            raise ValueError("named_value_json template×target-key coverage must be exactly balanced in full splits.")


def named_value_target_key(record: FeasibilityRecord) -> str:
    decoded = json.loads(record.answer)
    if not isinstance(decoded, str):
        raise ValueError("named_value_json answers must be JSON strings.")
    target_matches = tuple(match.casefold() for match in NAMED_TARGET_RE.findall(record.prompt))
    if len(target_matches) != 1:
        raise ValueError("named_value_json prompts must contain exactly one target key.")
    target = target_matches[0]
    roster = tuple((key.casefold(), value.casefold()) for key, value in NAMED_FIELD_RE.findall(record.prompt))
    if tuple(key for key, _value in roster) != NAMED_VALUE_KEYS:
        raise ValueError("named_value_json roster must list red, blue, green, silver in order.")
    fields = dict(roster)
    for key, value in roster:
        if not value.startswith(f"{key}-"):
            raise ValueError("named_value_json field values must preserve their key prefix.")
    if decoded.casefold() != fields[target]:
        raise ValueError("named_value_json answer must match the requested target key.")
    return target


def named_value_field_text(record: FeasibilityRecord) -> str:
    roster = tuple((key.casefold(), value.casefold()) for key, value in NAMED_FIELD_RE.findall(record.prompt))
    if tuple(key for key, _value in roster) != NAMED_VALUE_KEYS:
        raise ValueError("named_value_json roster must list red, blue, green, silver in order.")
    return "; ".join(f"{key}={value}" for key, value in roster)


def validate_array_count_contract(records: Sequence[FeasibilityRecord]) -> None:
    array_records = tuple(record for record in records if record.family == "array_json")
    if not array_records:
        return
    count_by_record = {record: array_item_count(record) for record in array_records}
    surface_by_record = {
        record: bound_prompt_surface_id(record, a=" | ".join(json.loads(record.answer)))
        for record in array_records
    }
    for split, expected_count in (("train", TRAIN_RECORDS_PER_FAMILY), ("eval", EVAL_RECORDS_PER_FAMILY)):
        split_records = tuple(record for record in array_records if record.split == split)
        if len(split_records) != expected_count:
            continue
        template_ids = tuple(sorted({record.template_id for record in split_records}))
        if len(template_ids) != 4:
            raise ValueError("array_json must use four templates per split.")
        counts: dict[tuple[str, int], int] = {
            (template_id, item_count): 0
            for template_id in template_ids
            for item_count in range(1, 5)
        }
        for record in split_records:
            item_count = count_by_record[record]
            surface_id = surface_by_record[record]
            counts[(f"seq_array_json_{split}_{surface_id}", item_count)] += 1
        expected_cell_count = expected_count // (len(template_ids) * 4)
        if any(count != expected_cell_count for count in counts.values()):
            raise ValueError("array_json template×item-count coverage must be exactly balanced in full splits.")


def array_item_count(record: FeasibilityRecord) -> int:
    decoded = json.loads(record.answer)
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise ValueError("array_json answers must be JSON arrays of strings.")
    answer_items = tuple(item.casefold() for item in decoded)
    if len(answer_items) not in range(1, 5):
        raise ValueError("array_json answers must contain one to four items.")
    prompt_items = tuple(value.casefold() for value in ARRAY_ITEM_RE.findall(record.prompt))
    if answer_items != prompt_items:
        raise ValueError("array_json prompt items must exactly match answer items.")
    return len(answer_items)


def bound_prompt_surface_id(record: FeasibilityRecord, **format_values: str) -> int:
    try:
        surfaces = PROMPT_SURFACES[record.family][record.split]
    except KeyError as exc:
        raise ValueError("Feasibility records must use a known family and train/eval split.") from exc
    matches = tuple(
        surface_id
        for surface_id, surface in enumerate(surfaces)
        if surface.format(**format_values) == record.prompt
    )
    if len(matches) != 1:
        raise ValueError("Feasibility prompt must match exactly one declared prompt surface.")
    surface_id = matches[0]
    expected_template_id = f"seq_{record.family}_{record.split}_{surface_id}"
    if record.template_id != expected_template_id:
        raise ValueError("Feasibility template_id must identify the actual prompt surface.")
    return surface_id


def validate_prompt_surface_contract(family: str) -> None:
    surfaces = PROMPT_SURFACES.get(family)
    if surfaces is None:
        raise ValueError(f"Missing feasibility prompt surfaces for family: {family!r}.")
    if set(surfaces) != {"train", "eval"}:
        raise ValueError(f"Feasibility prompt surfaces for {family!r} must define train and eval splits.")
    train_surfaces = tuple(surfaces["train"])
    eval_surfaces = tuple(surfaces["eval"])
    if len(train_surfaces) != 4 or len(eval_surfaces) != 4:
        raise ValueError(f"Feasibility prompt surfaces for {family!r} must define four train and four eval forms.")
    if len(set(train_surfaces)) != len(train_surfaces) or len(set(eval_surfaces)) != len(eval_surfaces):
        raise ValueError(f"Feasibility prompt surfaces for {family!r} must be unique within each split.")
    if set(train_surfaces) & set(eval_surfaces):
        raise ValueError(f"Feasibility prompt surface overlap between train and eval for {family!r}.")
    for train_surface, eval_surface in zip(train_surfaces, eval_surfaces, strict=True):
        if prompt_placeholder_sequence(train_surface) != prompt_placeholder_sequence(eval_surface):
            raise ValueError(f"Feasibility prompt surfaces for {family!r} must have paired placeholder structure.")
        if len(train_surface.encode("utf-8")) != len(eval_surface.encode("utf-8")):
            raise ValueError(f"Feasibility prompt surfaces for {family!r} must have paired byte lengths.")
    for surface in (*train_surfaces, *eval_surfaces):
        reject_scientific_markers(surface)
        if "template" in surface.lower():
            raise ValueError("Feasibility prompt surfaces must not expose template markers.")


def prompt_placeholder_sequence(surface: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in PROMPT_PLACEHOLDER_RE.finditer(surface))


@lru_cache(maxsize=1)
def phase8_scientific_identity_markers() -> frozenset[str]:
    markers: set[str] = set()
    for task_id in cg.TRAINING_TASK_ORDER:
        for record in cg.build_split_records("training", task_id, cg.CORPUS_RECORDS_PER_FAMILY["large"]):
            _add_phase8_record_markers(markers, record)
    for probe in cg.build_evaluation_probe_pack():
        _add_phase8_record_markers(markers, probe)
    return frozenset(markers)


@lru_cache(maxsize=1)
def phase8_scientific_prompt_patterns() -> tuple[re.Pattern[str], ...]:
    grouped: dict[tuple[str, str, str | None], list[str]] = {}
    pattern_texts: set[str] = set()
    for task_id in cg.TRAINING_TASK_ORDER:
        for record in cg.build_split_records("training", task_id, cg.CORPUS_RECORDS_PER_FAMILY["large"]):
            grouped.setdefault((record.task_id, record.template_id, record.style), []).append(record.prompt)
            add_record_prompt_pattern(pattern_texts, record)
    for condition in cg.CONDITIONS:
        for record in cg.build_training_corpus(condition, seed=0, state_mask=0, corpus_size="large"):
            grouped.setdefault((record.task_id, record.template_id, record.style), []).append(record.prompt)
            add_record_prompt_pattern(pattern_texts, record)
    for probe in cg.build_evaluation_probe_pack():
        grouped.setdefault((probe.task_id, probe.template_id, probe.style), []).append(probe.prompt)
        add_record_prompt_pattern(pattern_texts, probe)
    add_renderer_prompt_patterns(grouped, pattern_texts)
    patterns: list[re.Pattern[str]] = []
    patterns.extend(re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in sorted(pattern_texts))
    for prompts in grouped.values():
        unique_prompts = sorted(set(prompts))
        if len(unique_prompts) < 2:
            continue
        pattern = generalized_prompt_pattern(unique_prompts[0], unique_prompts[-1])
        if pattern is not None:
            patterns.append(pattern)
    return tuple(patterns)


def add_renderer_prompt_patterns(
    grouped: dict[tuple[str, str, str | None], list[str]],
    pattern_texts: set[str],
) -> None:
    for task_id in cg.TASK_ORDER:
        for style in renderer_prompt_styles(task_id):
            for suffix in ("a", "b", "c", "d"):
                template_id = renderer_template_id(task_id, style, suffix)
                for context in renderer_prompt_contexts(task_id):
                    prompt = cg.render_prompt(task_id, context, template_id, style)
                    grouped.setdefault((task_id, template_id, style), []).append(prompt)
                    add_record_prompt_pattern(
                        pattern_texts,
                        PromptSurfaceSeed(
                            prompt=prompt,
                            canonical_context=cg.canonical_context_bytes(context).decode("utf-8"),
                            normalized_payload=cg.normalized_payload_tuple(task_id, context),
                        ),
                    )


def renderer_prompt_styles(task_id: str) -> tuple[str, ...]:
    if task_id in {"MEMORY", "SEARCH", "FILTER", "CONDITION"}:
        return ("train", "neutral")
    if task_id in {"MEMORY_FILTER", "FILTER_CONDITION", "SEARCH_CONDITION"}:
        return ("train_explicit", "train_indirect", "explicit", "indirect", "generic")
    if task_id == "MEMORY_SEARCH":
        return ("explicit", "indirect", "generic")
    raise AssertionError(f"Unhandled Phase 8 task: {task_id!r}")


def renderer_template_id(task_id: str, style: str, suffix: str) -> str:
    if style == "train":
        return f"{task_id.lower()}__train_{suffix}"
    if style == "neutral":
        return f"neutral_eval__neutral_{suffix}"
    return f"{task_id.lower()}__{style}_{suffix}"


def renderer_prompt_contexts(task_id: str) -> tuple[dict[str, object], ...]:
    if task_id == "MEMORY":
        return ({"memory": {"alpha": "bravo"}, "key": "alpha"},)
    if task_id in {"SEARCH", "FILTER", "FILTER_CONDITION", "SEARCH_CONDITION"}:
        return ({"items": ["alpha", "bravo", "charlie"], "target": "bravo"},)
    if task_id == "CONDITION":
        return ({"condition": True}, {"condition": False})
    if task_id in {"MEMORY_FILTER", "MEMORY_SEARCH"}:
        return ({"memory": {"alpha": "bravo"}, "key": "alpha", "items": ["bravo", "charlie"]},)
    raise AssertionError(f"Unhandled Phase 8 task: {task_id!r}")


def add_record_prompt_pattern(patterns: set[str], record: object) -> None:
    pattern = record_prompt_surface_pattern(record)
    if pattern is not None:
        patterns.add(pattern)


def record_prompt_surface_pattern(record: object) -> str | None:
    prompt = getattr(record, "prompt", None)
    if not isinstance(prompt, str):
        return None
    values: set[str] = set()
    canonical_context = getattr(record, "canonical_context", None)
    if isinstance(canonical_context, str):
        context_data = json.loads(canonical_context)
        _collect_string_leaves(context_data, values)
        _collect_composite_prompt_values(context_data, values)
    normalized_payload = getattr(record, "normalized_payload", None)
    if normalized_payload is not None:
        _collect_string_leaves(normalized_payload, values)
    escaped = re.escape(prompt)
    for value in sorted((value for value in values if len(value) >= 2), key=len, reverse=True):
        escaped = escaped.replace(re.escape(value), ".*?")
    if ".*?" not in escaped:
        return None
    return escaped


def _collect_composite_prompt_values(value: object, values: set[str]) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _collect_composite_prompt_values(nested, values)
    elif isinstance(value, list):
        if value and all(isinstance(item, str) for item in value):
            values.add(", ".join(value))
        for nested in value:
            _collect_composite_prompt_values(nested, values)


def generalized_prompt_pattern(left: str, right: str) -> re.Pattern[str] | None:
    left_tokens = re.findall(r"\s+|[^\s]+", left)
    right_tokens = re.findall(r"\s+|[^\s]+", right)
    matcher = SequenceMatcher(a=left_tokens, b=right_tokens, autojunk=False)
    parts: list[str] = []
    last_left = 0
    last_right = 0
    literal_chars = 0
    for block in matcher.get_matching_blocks():
        if block.a > last_left or block.b > last_right:
            parts.append(".*?")
        if block.size:
            literal = "".join(left_tokens[block.a : block.a + block.size])
            literal_chars += len(literal.strip())
            parts.append(re.escape(literal))
        last_left = block.a + block.size
        last_right = block.b + block.size
    if literal_chars < 24 or not parts:
        return None
    return re.compile("".join(parts), re.IGNORECASE | re.DOTALL)


def _add_phase8_record_markers(markers: set[str], record: object) -> None:
    for field_name in ("task_id", "template_id", "payload_id", "prompt"):
        value = getattr(record, field_name, None)
        if isinstance(value, str) and value:
            markers.add(value)
    canonical_context = getattr(record, "canonical_context", None)
    if isinstance(canonical_context, str):
        _collect_string_leaves(json.loads(canonical_context), markers)
    normalized_payload = getattr(record, "normalized_payload", None)
    if normalized_payload is not None:
        _collect_string_leaves(normalized_payload, markers)
    answer = getattr(record, "answer", None)
    if isinstance(answer, str):
        try:
            _collect_string_leaves(json.loads(answer), markers)
        except json.JSONDecodeError:
            _maybe_add_marker(answer, markers)


def _collect_string_leaves(value: object, markers: set[str]) -> None:
    if isinstance(value, str):
        _maybe_add_marker(value, markers)
    elif isinstance(value, dict):
        for key, nested in value.items():
            _maybe_add_marker(str(key), markers)
            _collect_string_leaves(nested, markers)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _collect_string_leaves(nested, markers)


def _maybe_add_marker(value: str, markers: set[str]) -> None:
    if len(value) >= 4 and value not in GENERIC_JSON_MARKERS:
        markers.add(value)


def build_family_records(family: str) -> tuple[FeasibilityRecord, ...]:
    if family not in FAMILIES:
        raise ValueError(f"Unknown feasibility family: {family!r}.")
    used_payload_suffixes: set[str] = set()
    return (
        *_family_split(family, "train", TRAIN_RECORDS_PER_FAMILY, used_payload_suffixes),
        *_family_split(family, "eval", EVAL_RECORDS_PER_FAMILY, used_payload_suffixes),
    )


def _family_split(
    family: str,
    split: str,
    count: int,
    used_payload_suffixes: set[str],
) -> tuple[FeasibilityRecord, ...]:
    base = 10_000 if split == "eval" else 0
    return tuple(
        _make_record(family, split, base + index, index, used_payload_suffixes)
        for index in range(count)
    )


def _make_record(
    family: str,
    split: str,
    operand_number: int,
    index: int,
    used_payload_suffixes: set[str],
) -> FeasibilityRecord:
    rng = random.Random(730000 + 10000 * FAMILIES.index(family) + operand_number)
    template_id = f"seq_{family}_{split}_{index % 4}"
    operand_id = f"seq_operand_{family}_{operand_number:05d}"
    surface = PROMPT_SURFACES[family][split][index % 4]
    if family == "hex_copy":
        value = f"{rng.getrandbits(64):016x}"
        prompt = surface.format(a=index % 17, b=value)
        answer = value
        semantic_values = (value,)
    elif family == "named_value_json":
        keys = NAMED_VALUE_KEYS_BY_SPLIT[split]
        target = keys[(index // 4) % len(keys)]
        fields = {key: f"{key}-{make_unique_payload_suffix(rng, used_payload_suffixes)}" for key in keys}
        field_text = "; ".join(f"{key}={fields[key]}" for key in keys)
        prompt = surface.format(a=target, b=field_text)
        answer = compact_json(fields[target])
        semantic_values = tuple(fields[key] for key in keys)
    elif family == "boolean_json":
        operand, truth = make_boolean_operand(rng, index)
        prompt = surface.format(a=operand)
        answer = "true" if truth else "false"
        semantic_values = (operand,)
    elif family == "array_json":
        items = [
            f"q{make_unique_payload_suffix(rng, used_payload_suffixes)}"
            for _ in range(1 + (index // 4) % 4)
        ]
        prompt = surface.format(a=" | ".join(items))
        answer = compact_json(items)
        semantic_values = (*items, answer)
    else:
        raise AssertionError("unreachable")
    return FeasibilityRecord(
        family=family,
        split=split,
        index=index,
        template_id=template_id,
        operand_id=operand_id,
        prompt=prompt,
        answer=answer,
        semantic_values=semantic_values,
    )


def make_boolean_operand(rng: random.Random, index: int) -> tuple[str, bool]:
    label = BOOLEAN_LABELS[(index // 4) % len(BOOLEAN_LABELS)]
    balanced_tail = ((index // 8) * 4 + index % 4) % 16
    suffix = f"{rng.getrandbits(60):015x}{balanced_tail:x}"
    return f"{label}-{suffix}", BOOLEAN_LABEL_TRUTH[label]


def make_unique_payload_suffix(rng: random.Random, used_suffixes: set[str]) -> str:
    while True:
        suffix = f"{rng.getrandbits(16):04x}"
        if suffix not in used_suffixes and not set(suffix) <= {"0", "1"}:
            used_suffixes.add(suffix)
            return suffix


def grouped_records() -> dict[str, dict[str, tuple[FeasibilityRecord, ...]]]:
    result: dict[str, dict[str, tuple[FeasibilityRecord, ...]]] = {}
    for family in FAMILIES:
        records = build_family_records(family)
        validate_feasibility_records(records)
        result[family] = {
            "train": tuple(record for record in records if record.split == "train"),
            "eval": tuple(record for record in records if record.split == "eval"),
        }
    return result


def evaluate_model(model, records: Sequence[FeasibilityRecord], tokenizer: ByteTokenizer, device: torch.device) -> tuple[int, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    correct = 0
    for record in records:
        prefix = tokenizer.encode_evaluation_prefix(record.prompt)
        prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
        generated = model.greedy_decode(prefix_tensor)
        full_ids = tuple(int(token) for token in generated[0].detach().cpu().tolist())
        generation_error = None
        try:
            decoded: str | None = tokenizer.decode_generated_response(full_ids)
        except (UnicodeDecodeError, ValueError) as exc:
            decoded = None
            generation_error = f"{type(exc).__name__}: {exc}"
        match = decoded == record.answer
        correct += int(match)
        rows.append(
            {
                "family": record.family,
                "index": record.index,
                "template_id": record.template_id,
                "operand_id": record.operand_id,
                "prompt": record.prompt,
                "expected": record.answer,
                "generated": decoded,
                "raw_token_ids": list(full_ids),
                "invalid_generation": generation_error is not None,
                "generation_error": generation_error,
                "exact_match": match,
            }
        )
    return correct, rows


def current_feasibility_cell_identities() -> tuple[tuple[str, str, int], ...]:
    return tuple((family, model_size, seed) for family in FAMILIES for model_size in MODEL_SIZES for seed in SEEDS)


def validate_current_feasibility_cells(
    cells: object,
    *,
    require_complete: bool,
    require_all_pass: bool,
) -> tuple[dict[str, object], ...]:
    if not isinstance(cells, list):
        raise ValueError("Feasibility cells must be a JSON list.")
    expected_sequence = current_feasibility_cell_identities()
    parsed: list[tuple[str, str, int]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("Feasibility cell entries must be JSON objects.")
        if set(cell) != CURRENT_CELL_KEYS:
            raise ValueError("Feasibility cell must use the exact current tied schema.")
        seed = require_exact_int(cell["seed"], "seed")
        family = require_exact_str(cell["family"], "family")
        model_size = require_exact_str(cell["model_size"], "model_size")
        key = (family, model_size, seed)
        if key not in expected_sequence:
            raise ValueError(f"Unexpected feasibility cell: {key!r}.")
        if key in parsed:
            raise ValueError(f"Duplicate feasibility cell: {key!r}.")
        parsed.append(key)
        eval_count = require_exact_int(cell["eval_count"], "eval_count")
        exact_matches = require_exact_int(cell["exact_matches"], "exact_matches")
        passed_value = require_exact_bool(cell["passed"], "passed")
        if eval_count != EVAL_RECORDS_PER_FAMILY:
            raise ValueError("Every feasibility cell must evaluate 64 records.")
        if not 0 <= exact_matches <= eval_count:
            raise ValueError("Feasibility exact_matches must satisfy 0 <= exact_matches <= eval_count.")
        if passed_value is not (exact_matches >= PASS_THRESHOLD):
            raise ValueError("Feasibility cell passed must equal exact_matches >= the independent 52/64 requirement.")
        if require_all_pass and (exact_matches < PASS_THRESHOLD or passed_value is not True):
            raise ValueError("Every feasibility cell must pass the independent 52/64 requirement.")
        parameter_count = require_exact_int(cell.get("parameter_count"), "parameter_count")
        if parameter_count != expected_parameter_count(model_size):
            raise ValueError("Feasibility cell parameter_count does not match the frozen model configuration.")
        if cell.get("embedding_weight_tying") is not True:
            raise ValueError("Feasibility cell embedding_weight_tying must be true for the current tied revision.")
        if cell.get("model_protocol_revision") != TIED_MODEL_PROTOCOL_REVISION:
            raise ValueError("Feasibility cell model_protocol_revision must be phase8_tied_io_v1.")
        require_canonical_relative_path(cell.get("generations_path"), "generations_path")
        require_canonical_relative_path(cell.get("checkpoint_path"), "checkpoint_path")
    observed_sequence = tuple(parsed)
    if require_complete:
        if observed_sequence != expected_sequence:
            missing = set(expected_sequence) - set(observed_sequence)
            raise ValueError(f"Missing feasibility cells: {sorted(missing)!r}.")
    elif observed_sequence != expected_sequence[: len(observed_sequence)]:
        raise ValueError("Partial feasibility cells must be a deterministic prefix of the 24-cell matrix.")
    return tuple(cells)


def validate_cell_counts(cells: Sequence[dict[str, object]]) -> None:
    validate_current_feasibility_cells(list(cells), require_complete=True, require_all_pass=True)


@lru_cache(maxsize=None)
def expected_parameter_count(model_size: str) -> int:
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unknown model_size for parameter_count validation: {model_size!r}.")
    return build_model(model_size).parameter_count


def frozen_diagnostic_configuration() -> dict[str, object]:
    return {
        "artifact_class": DIAGNOSTIC_ARTIFACT_CLASS,
        "feasibility_selection_eligible": False,
        "task_010d_authorized": False,
        "accepted_protocol_commit": DIAGNOSTIC_ACCEPTED_PROTOCOL_COMMIT,
        "handoff_path": DIAGNOSTIC_HANDOFF_PATH,
        "handoff_blob": DIAGNOSTIC_HANDOFF_BLOB,
        "named_cells": list(NAMED_DIAGNOSTIC_CELLS),
        "array_new_training": diagnostic_array_training_plan(),
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "formal_training_steps": TRAINING_STEPS,
        "validation": {
            "authorized_new_training_runs": 3,
            "validation_training_or_replay_runs": 0,
            "trajectory_evidence_source": "in_run_checkpoint_metadata_and_metrics",
            "selection_evidence": False,
        },
        "batch_size": BATCH_SIZE,
        "pass_threshold": PASS_THRESHOLD,
        "tokenizer": {
            "pad_id": PAD_ID,
            "bos_id": BOS_ID,
            "sep_id": SEP_ID,
            "eos_id": EOS_ID,
            "vocab_size": ByteTokenizer.vocab_size,
            "max_sequence_length": ByteTokenizer.max_sequence_length,
            "max_generated_tokens": ByteTokenizer.max_generated_tokens,
        },
        "model_configs": {
            "small": {**legacy_serialized_config("small"), "parameter_count": HISTORICAL_PARAMETER_COUNTS["small"]},
            "medium": {**legacy_serialized_config("medium"), "parameter_count": HISTORICAL_PARAMETER_COUNTS["medium"]},
        },
        "optimizer": {
            "class": "torch.optim.AdamW",
            "learning_rate": LEARNING_RATE,
            "betas": list(ADAMW_BETAS),
            "eps": ADAMW_EPS,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "model_rng_offset": MODEL_RNG_OFFSET,
            "corpus_order_rng_offset": CORPUS_ORDER_RNG_OFFSET,
        },
    }


def frozen_configuration() -> dict[str, object]:
    return {
        "model_protocol_revision": TIED_MODEL_PROTOCOL_REVISION,
        "embedding_weight_tying": True,
        "device": FEASIBILITY_REQUIRED_DEVICE,
        "families": list(FAMILIES),
        "model_sizes": list(MODEL_SIZES),
        "seeds": list(SEEDS),
        "train_records_per_family": TRAIN_RECORDS_PER_FAMILY,
        "eval_records_per_family": EVAL_RECORDS_PER_FAMILY,
        "training_steps": TRAINING_STEPS,
        "batch_size": BATCH_SIZE,
        "pass_threshold": PASS_THRESHOLD,
        "context_window_tokens": ByteTokenizer.max_sequence_length,
        "generation_window_tokens": ByteTokenizer.max_generated_tokens,
        "record_hashes": dict(FEASIBILITY_RECORD_HASHES),
        "model_parameter_counts": dict(FROZEN_PARAMETER_COUNTS),
    }


def require_exact_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"Feasibility cell {field_name} must be an exact JSON integer.")
    return value


def require_exact_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"Feasibility cell {field_name} must be a JSON boolean.")
    return value


def require_exact_str(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"Feasibility cell {field_name} must be a JSON string.")
    return value


def canonical_record_set_sha256(records: Sequence[FeasibilityRecord]) -> str:
    payload = b"".join(
        (json.dumps(asdict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )
    return sha256(payload).hexdigest()


def diagnostic_record_sets() -> dict[str, tuple[FeasibilityRecord, ...]]:
    named = build_family_records("named_value_json")
    array = build_family_records("array_json")
    named_train = tuple(record for record in named if record.split == "train")
    named_eval = tuple(record for record in named if record.split == "eval")
    array_train = tuple(record for record in array if record.split == "train")
    array_eval = tuple(record for record in array if record.split == "eval")
    return {
        "named_train512": named_train,
        "named_train_first64": named_train[:EVAL_RECORDS_PER_FAMILY],
        "named_eval64": named_eval,
        "array_train512": array_train,
        "array_eval64": array_eval,
    }


def feasibility_record_sets() -> dict[str, tuple[FeasibilityRecord, ...]]:
    records_by_family = {family: build_family_records(family) for family in FAMILIES}
    result: dict[str, tuple[FeasibilityRecord, ...]] = {}
    for family, records in records_by_family.items():
        train = tuple(record for record in records if record.split == "train")
        eval_ = tuple(record for record in records if record.split == "eval")
        prefix = {
            "hex_copy": "hex",
            "named_value_json": "named",
            "boolean_json": "boolean",
            "array_json": "array",
        }[family]
        result[f"{prefix}_train512"] = train
        result[f"{prefix}_eval64"] = eval_
    return result


def validate_feasibility_record_hashes() -> dict[str, str]:
    actual = {name: canonical_record_set_sha256(records) for name, records in feasibility_record_sets().items()}
    if actual != FEASIBILITY_RECORD_HASHES:
        raise ValueError(f"Feasibility record hash mismatch: expected {FEASIBILITY_RECORD_HASHES!r}, got {actual!r}.")
    return actual


def validate_diagnostic_record_hashes() -> dict[str, str]:
    actual = {name: canonical_record_set_sha256(records) for name, records in diagnostic_record_sets().items()}
    if actual != DIAGNOSTIC_RECORD_HASHES:
        raise ValueError(f"Diagnostic record hash mismatch: expected {DIAGNOSTIC_RECORD_HASHES!r}, got {actual!r}.")
    return actual


def named_roster(record: FeasibilityRecord) -> tuple[tuple[str, str], ...]:
    roster = tuple((key.casefold(), value.casefold()) for key, value in NAMED_FIELD_RE.findall(record.prompt))
    if tuple(key for key, _value in roster) != NAMED_VALUE_KEYS:
        raise ValueError("named_value_json roster must list red, blue, green, silver in order.")
    return roster


def render_named_surface_from_operand(record: FeasibilityRecord, surface_split: str) -> tuple[str, str]:
    if record.family != "named_value_json":
        raise ValueError("Named diagnostic rows require named_value_json records.")
    if surface_split not in {"train", "eval"}:
        raise ValueError("Named diagnostic surface_split must be train or eval.")
    surface_index = record.index % 4
    target = named_value_target_key(record)
    field_text = "; ".join(f"{key}={value}" for key, value in named_roster(record))
    prompt = PROMPT_SURFACES["named_value_json"][surface_split][surface_index].format(a=target, b=field_text)
    template_id = f"seq_named_value_json_{surface_split}_{surface_index}"
    return prompt, template_id


def build_named_diagnostic_matrix() -> dict[str, list[dict[str, object]]]:
    sets = diagnostic_record_sets()
    sources = {
        "seen": sets["named_train_first64"],
        "held": sets["named_eval64"],
    }
    cell_plan = {
        "seen_surface_seen_operand": ("train", "seen"),
        "held_surface_seen_operand": ("eval", "seen"),
        "seen_surface_held_operand": ("train", "held"),
        "held_surface_held_operand": ("eval", "held"),
    }
    matrix: dict[str, list[dict[str, object]]] = {}
    for cell, (surface_split, operand_group) in cell_plan.items():
        rows: list[dict[str, object]] = []
        for row_index, source_record in enumerate(sources[operand_group]):
            prompt, template_id = render_named_surface_from_operand(source_record, surface_split)
            rows.append(
                {
                    "family": "named_value_json",
                    "diagnostic_cell": cell,
                    "row_index": row_index,
                    "operand_source_split": source_record.split,
                    "operand_source_index": source_record.index,
                    "operand_id": source_record.operand_id,
                    "source_template_id": source_record.template_id,
                    "surface_source_split": surface_split,
                    "surface_index": source_record.index % 4,
                    "template_id": template_id,
                    "target_key": named_value_target_key(source_record),
                    "roster": [{"key": key, "value": value} for key, value in named_roster(source_record)],
                    "prompt": prompt,
                    "expected": source_record.answer,
                }
            )
        matrix[cell] = rows
    validate_named_diagnostic_matrix(matrix)
    return matrix


def validate_named_diagnostic_matrix(matrix: dict[str, list[dict[str, object]]]) -> None:
    if tuple(matrix.keys()) != NAMED_DIAGNOSTIC_CELLS:
        raise ValueError("Named diagnostic matrix must contain the four frozen cells in canonical order.")
    sets = diagnostic_record_sets()
    expected_sources = {
        "train": sets["named_train_first64"],
        "eval": sets["named_eval64"],
    }
    for cell, rows in matrix.items():
        if len(rows) != EVAL_RECORDS_PER_FAMILY:
            raise ValueError("Every Named diagnostic cell must contain exactly 64 rows.")
        counts: Counter[tuple[str, str]] = Counter()
        for row_index, row in enumerate(rows):
            if row.get("family") != "named_value_json":
                raise ValueError("Named diagnostic rows must bind family named_value_json.")
            if row.get("diagnostic_cell") != cell:
                raise ValueError("Named diagnostic row diagnostic_cell must match the outer cell.")
            if row.get("row_index") != row_index:
                raise ValueError("Named diagnostic rows must retain deterministic row_index order.")
            operand_split = require_exact_str(row.get("operand_source_split"), "operand_source_split")
            surface_split = require_exact_str(row.get("surface_source_split"), "surface_source_split")
            if cell.endswith("seen_operand") and operand_split != "train":
                raise ValueError("Seen-operand cells must retain train operand source metadata.")
            if cell.endswith("held_operand") and operand_split != "eval":
                raise ValueError("Held-operand cells must retain eval operand source metadata.")
            if cell.startswith("seen_surface") and surface_split != "train":
                raise ValueError("Seen-surface cells must render train surfaces.")
            if cell.startswith("held_surface") and surface_split != "eval":
                raise ValueError("Held-surface cells must render eval surfaces.")
            source_record = expected_sources[operand_split][row_index]
            if row.get("operand_source_index") != source_record.index:
                raise ValueError("Named diagnostic operand_source_index does not match the frozen source record.")
            if row.get("operand_id") != source_record.operand_id:
                raise ValueError("Named diagnostic operand_id does not match the frozen source record.")
            if row.get("source_template_id") != source_record.template_id:
                raise ValueError("Named diagnostic source_template_id does not match the frozen source record.")
            if row.get("expected") != source_record.answer:
                raise ValueError("Named diagnostic expected answer does not match the frozen source record.")
            if row.get("target_key") != named_value_target_key(source_record):
                raise ValueError("Named diagnostic target_key does not match the frozen source record.")
            expected_roster = [{"key": key, "value": value} for key, value in named_roster(source_record)]
            if row.get("roster") != expected_roster:
                raise ValueError("Named diagnostic roster does not match the frozen source record.")
            expected_prompt, expected_template_id = render_named_surface_from_operand(source_record, surface_split)
            if row.get("prompt") != expected_prompt:
                raise ValueError("Named diagnostic prompt does not match the frozen rendered surface/source pair.")
            target_key = require_exact_str(row.get("target_key"), "target_key")
            template_id = require_exact_str(row.get("template_id"), "template_id")
            if template_id != expected_template_id:
                raise ValueError("Named diagnostic template_id does not match the frozen rendered surface.")
            if template_id != f"seq_named_value_json_{surface_split}_{row['surface_index']}":
                raise ValueError("Named diagnostic template_id must identify only the rendered surface.")
            counts[(template_id, target_key)] += 1
            prompt = require_exact_str(row.get("prompt"), "prompt")
            expected = require_exact_str(row.get("expected"), "expected")
            source_like = FeasibilityRecord(
                family="named_value_json",
                split=surface_split,
                index=require_exact_int(row.get("operand_source_index"), "operand_source_index"),
                template_id=template_id,
                operand_id=require_exact_str(row.get("operand_id"), "operand_id"),
                prompt=prompt,
                answer=expected,
            )
            bound_prompt_surface_id(source_like, a=target_key, b=named_value_field_text(source_like))
            other_split = "eval" if surface_split == "train" else "train"
            other_prompt, _other_template = render_named_surface_from_operand(
                FeasibilityRecord(
                    family="named_value_json",
                    split=operand_split,
                    index=require_exact_int(row.get("operand_source_index"), "operand_source_index"),
                    template_id=require_exact_str(row.get("source_template_id"), "source_template_id"),
                    operand_id=require_exact_str(row.get("operand_id"), "operand_id"),
                    prompt=prompt if operand_split == surface_split else render_named_surface_from_operand(source_like, operand_split)[0],
                    answer=expected,
                ),
                other_split,
            )
            if len(prompt.encode("utf-8")) != len(other_prompt.encode("utf-8")):
                raise ValueError("Named diagnostic paired train/eval prompts must retain byte-length equality.")
        if any(count != 4 for count in counts.values()) or len(counts) != 16:
            raise ValueError("Named diagnostic template×target-key balance must be exactly four in every cell.")


def utf8_bytes_or_none(value: str | None) -> bytes | None:
    return None if value is None else value.encode("utf-8")


def byte_hamming_distance(left: str, right: str) -> int | None:
    left_bytes = left.encode("utf-8")
    right_bytes = right.encode("utf-8")
    if len(left_bytes) != len(right_bytes):
        return None
    return sum(a != b for a, b in zip(left_bytes, right_bytes, strict=True))


def byte_edit_distance(left: str, right: str) -> int:
    a = left.encode("utf-8")
    b = right.encode("utf-8")
    previous = list(range(len(b) + 1))
    for index_a, byte_a in enumerate(a, start=1):
        current = [index_a]
        for index_b, byte_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[index_b] + 1,
                    current[index_b - 1] + 1,
                    previous[index_b - 1] + int(byte_a != byte_b),
                )
            )
        previous = current
    return previous[-1]


def byte_first_error(left: str, right: str) -> int | None:
    left_bytes = left.encode("utf-8")
    right_bytes = right.encode("utf-8")
    for index, (left_byte, right_byte) in enumerate(zip(left_bytes, right_bytes)):
        if left_byte != right_byte:
            return index
    if len(left_bytes) == len(right_bytes):
        return None
    return min(len(left_bytes), len(right_bytes))


def generation_slice_metrics(prompt: str, expected: str, raw_token_ids: Sequence[int]) -> dict[str, object]:
    tokenizer = ByteTokenizer()
    prefix = tuple(tokenizer.encode_evaluation_prefix(prompt))
    full_ids = tuple(int(token) for token in raw_token_ids)
    if full_ids[: len(prefix)] != prefix:
        raise ValueError("Diagnostic raw_token_ids prefix does not exactly encode the retained prompt.")
    generation_slice = full_ids[len(prefix) :]
    decoded_error = None
    try:
        decoded: str | None = tokenizer.decode_generated_response(full_ids)
    except (UnicodeDecodeError, ValueError) as exc:
        decoded = None
        decoded_error = f"{type(exc).__name__}: {exc}"
    has_eos = decoded_error is None and generation_slice.count(EOS_ID) == 1 and generation_slice[-1:] == (EOS_ID,)
    hit_generation_cap = not has_eos and len(generation_slice) == ByteTokenizer.max_generated_tokens
    hit_context_cap = not has_eos and len(full_ids) == ByteTokenizer.max_sequence_length
    return {
        "generated": decoded,
        "generation_error": decoded_error,
        "exact_match": decoded == expected,
        "raw_token_ids": list(full_ids),
        "generation_token_count_including_eos": len(generation_slice),
        "decoded_response_utf8_bytes": None if decoded is None else len(decoded.encode("utf-8")),
        "has_eos": has_eos,
        "hit_generation_cap": hit_generation_cap,
        "hit_context_cap": hit_context_cap,
        "response_hamming_distance": None if decoded is None else byte_hamming_distance(decoded, expected),
        "response_edit_distance": None if decoded is None else byte_edit_distance(decoded, expected),
        "response_first_error": None if decoded is None else byte_first_error(decoded, expected),
    }


def named_row_metrics(prompt: str, expected: str, raw_token_ids: Sequence[int], target_key: str, roster_values: Sequence[str]) -> dict[str, object]:
    base = generation_slice_metrics(prompt, expected, raw_token_ids)
    generated = base["generated"]
    parsed = None
    valid_json_string = False
    valid_named_grammar = False
    if isinstance(generated, str):
        try:
            parsed = json.loads(generated)
            valid_json_string = isinstance(parsed, str)
            valid_named_grammar = valid_json_string and NAMED_VALUE_RE.fullmatch(parsed) is not None
        except json.JSONDecodeError:
            parsed = None
    target_prefix = isinstance(parsed, str) and parsed.startswith(f"{target_key}-")
    expected_value = json.loads(expected)
    suffix = parsed[len(target_key) + 1 :] if target_prefix else None
    expected_suffix = expected_value[len(target_key) + 1 :]
    suffix_first_error = None if suffix is None else byte_first_error(suffix, expected_suffix)
    suffix_hamming = None if suffix is None or len(suffix.encode("utf-8")) != 4 else byte_hamming_distance(suffix, expected_suffix)
    suffix_edit = None if suffix is None else byte_edit_distance(suffix, expected_suffix)
    suffix_correct_positions = None
    if suffix is not None:
        suffix_bytes = suffix.encode("utf-8")
        expected_bytes = expected_suffix.encode("utf-8")
        suffix_correct_positions = sum(
            index < len(suffix_bytes) and suffix_bytes[index] == expected_bytes[index]
            for index in range(4)
        )
    base.update(
        {
            "valid_json_string": valid_json_string,
            "valid_named_grammar": valid_named_grammar,
            "target_prefix": target_prefix,
            "exact_target_value": parsed == expected_value,
            "exact_distractor_value": isinstance(parsed, str) and parsed in set(roster_values) - {expected_value},
            "occurs_in_prompt": isinstance(parsed, str) and parsed in prompt,
            "suffix_correct_positions": suffix_correct_positions,
            "suffix_position_denominator": 4 if target_prefix else 0,
            "suffix_hamming_distance": suffix_hamming,
            "suffix_edit_distance": suffix_edit,
            "suffix_first_error": suffix_first_error,
        }
    )
    return base


def array_row_metrics(prompt: str, expected: str, raw_token_ids: Sequence[int]) -> dict[str, object]:
    base = generation_slice_metrics(prompt, expected, raw_token_ids)
    generated = base["generated"]
    expected_items = json.loads(expected)
    syntax_valid = False
    schema_valid = False
    correct_item_count = False
    positional_exact_items = None
    missing_items = None
    extra_items = None
    if isinstance(generated, str):
        try:
            parsed = json.loads(generated)
            syntax_valid = True
            schema_valid = isinstance(parsed, list) and all(isinstance(item, str) for item in parsed)
        except json.JSONDecodeError:
            parsed = None
        if schema_valid:
            correct_item_count = len(parsed) == len(expected_items)  # type: ignore[arg-type]
            positional_exact_items = sum(1 for got, want in zip(parsed, expected_items) if got == want)  # type: ignore[arg-type]
            missing_items = list((Counter(expected_items) - Counter(parsed)).elements())  # type: ignore[arg-type]
            extra_items = list((Counter(parsed) - Counter(expected_items)).elements())  # type: ignore[arg-type]
    base.update(
        {
            "valid_json_syntax": syntax_valid,
            "valid_array_schema": schema_valid,
            "correct_item_count": correct_item_count,
            "positional_exact_items": positional_exact_items,
            "missing_items": missing_items,
            "extra_items": extra_items,
        }
    )
    return base


def count_rate(numerator: int, denominator: int) -> dict[str, object]:
    return {"numerator": numerator, "denominator": denominator, "rate": None if denominator == 0 else numerator / denominator}


def mean_value(total: float, count: int) -> dict[str, object]:
    return {"observation_count": count, "mean": None if count == 0 else total / count}


def mean_from_rows(rows: Sequence[dict[str, object]], field_name: str) -> dict[str, object]:
    values = [row[field_name] for row in rows if row.get(field_name) is not None]
    if not all(type(value) in {int, float} and math.isfinite(float(value)) for value in values):
        raise ValueError(f"Aggregate mean field {field_name} contains non-finite or non-numeric values.")
    return mean_value(math.fsum(float(value) for value in values), len(values))


def histogram_from_values(values: Iterable[object]) -> dict[str, object]:
    observed = [value for value in values if value is not None]
    return {
        "observation_count": len(observed),
        "histogram": {str(key): value for key, value in sorted(Counter(observed).items(), key=lambda item: str(item[0]))},
    }


def aggregate_named_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    exact = sum(1 for row in rows if row["exact_match"] is True)
    target_prefix_rows = [row for row in rows if row["target_prefix"] is True]
    suffix_num = sum(int(row["suffix_correct_positions"]) for row in target_prefix_rows)
    suffix_den = 4 * len(target_prefix_rows)
    template_target_exact: dict[str, dict[str, object]] = {}
    for (template_id, target_key), group_count in sorted(Counter((row["template_id"], row["target_key"]) for row in rows).items()):
        group_exact = sum(1 for row in rows if row["template_id"] == template_id and row["target_key"] == target_key and row["exact_match"] is True)
        template_target_exact[f"{template_id}__{target_key}"] = count_rate(group_exact, group_count)
    return {
        "row_count": len(rows),
        "exact_sequence_matches": count_rate(exact, len(rows)),
        "valid_json_string": count_rate(sum(1 for row in rows if row["valid_json_string"] is True), len(rows)),
        "valid_named_grammar": count_rate(sum(1 for row in rows if row["valid_named_grammar"] is True), len(rows)),
        "target_prefix": count_rate(len(target_prefix_rows), len(rows)),
        "exact_target_value": count_rate(sum(1 for row in rows if row["exact_target_value"] is True), len(rows)),
        "exact_distractor_value": count_rate(sum(1 for row in rows if row["exact_distractor_value"] is True), len(rows)),
        "occurs_in_prompt": count_rate(sum(1 for row in rows if row["occurs_in_prompt"] is True), len(rows)),
        "template_target_exact_matches": template_target_exact,
        "suffix_position_accuracy": count_rate(suffix_num, suffix_den),
        "suffix_hamming_distance": mean_from_rows(rows, "suffix_hamming_distance"),
        "suffix_edit_distance": mean_from_rows(rows, "suffix_edit_distance"),
        "suffix_first_error_histogram": histogram_from_values(row["suffix_first_error"] for row in rows),
        "has_eos": count_rate(sum(1 for row in rows if row["has_eos"] is True), len(rows)),
        "generation_token_count_including_eos": mean_from_rows(rows, "generation_token_count_including_eos"),
        "decoded_response_utf8_bytes": mean_from_rows(rows, "decoded_response_utf8_bytes"),
        "hit_generation_cap": count_rate(sum(1 for row in rows if row["hit_generation_cap"] is True), len(rows)),
        "hit_context_cap": count_rate(sum(1 for row in rows if row["hit_context_cap"] is True), len(rows)),
    }


def aggregate_array_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    for index, row in enumerate(rows):
        if row.get("family") != "array_json":
            raise ValueError(f"Array aggregate row {index} must bind family array_json.")
        valid_schema = require_exact_bool(row.get("valid_array_schema"), f"array_rows[{index}].valid_array_schema")
        require_exact_bool(row.get("valid_json_syntax"), f"array_rows[{index}].valid_json_syntax")
        require_exact_bool(row.get("correct_item_count"), f"array_rows[{index}].correct_item_count")
        expected_items = json.loads(require_exact_str(row.get("expected"), f"array_rows[{index}].expected"))
        if not isinstance(expected_items, list) or not all(isinstance(item, str) for item in expected_items):
            raise ValueError("Array aggregate expected values must be JSON string arrays.")
        require_exact_int(row.get("item_count"), f"array_rows[{index}].item_count")
        if row["item_count"] != len(expected_items):
            raise ValueError("Array aggregate item_count does not match expected JSON array length.")
        missing_items = row.get("missing_items")
        extra_items = row.get("extra_items")
        positional_exact_items = row.get("positional_exact_items")
        if valid_schema:
            if not isinstance(missing_items, list) or not all(isinstance(item, str) for item in missing_items):
                raise ValueError("Valid Array schema rows must record missing_items as a string list.")
            if not isinstance(extra_items, list) or not all(isinstance(item, str) for item in extra_items):
                raise ValueError("Valid Array schema rows must record extra_items as a string list.")
            if type(positional_exact_items) is not int or not 0 <= positional_exact_items <= row["item_count"]:
                raise ValueError("Valid Array schema rows must record positional_exact_items as an in-range integer.")
        else:
            if missing_items is not None or extra_items is not None or positional_exact_items is not None:
                raise ValueError("Invalid Array schema rows must record missing/extra/positional item observations as null.")
    item_count_exact: dict[str, dict[str, object]] = {}
    for item_count, group_count in sorted(Counter(row["item_count"] for row in rows).items()):
        group_exact = sum(1 for row in rows if row["item_count"] == item_count and row["exact_match"] is True)
        item_count_exact[str(item_count)] = count_rate(group_exact, group_count)
    template_item_exact: dict[str, dict[str, object]] = {}
    for (template_id, item_count), group_count in sorted(Counter((row["template_id"], row["item_count"]) for row in rows).items()):
        group_exact = sum(1 for row in rows if row["template_id"] == template_id and row["item_count"] == item_count and row["exact_match"] is True)
        template_item_exact[f"{template_id}__items{item_count}"] = count_rate(group_exact, group_count)
    expected_items = Counter(
        item
        for row in rows
        for item in json.loads(require_exact_str(row["expected"], "array.expected"))
    )
    valid_schema_rows = [row for row in rows if row["valid_array_schema"] is True]
    missing_items = Counter(item for row in valid_schema_rows for item in row["missing_items"])
    extra_items = Counter(item for row in valid_schema_rows for item in row["extra_items"])
    positional_den = sum(int(row["item_count"]) for row in rows if row["positional_exact_items"] is not None)
    positional_num = sum(int(row["positional_exact_items"]) for row in rows if row["positional_exact_items"] is not None)
    return {
        "row_count": len(rows),
        "valid_schema_observation_count": len(valid_schema_rows),
        "greedy_exact_matches": count_rate(sum(1 for row in rows if row["exact_match"] is True), len(rows)),
        "valid_json_syntax": count_rate(sum(1 for row in rows if row["valid_json_syntax"] is True), len(rows)),
        "valid_array_schema": count_rate(sum(1 for row in rows if row["valid_array_schema"] is True), len(rows)),
        "correct_item_count": count_rate(sum(1 for row in rows if row["correct_item_count"] is True), len(rows)),
        "item_count_exact_matches": item_count_exact,
        "template_item_count_exact_matches": template_item_exact,
        "positional_exact_items": count_rate(positional_num, positional_den),
        "expected_item_counts": {
            "observation_count": len(rows),
            "total": sum(expected_items.values()),
            "counts": dict(sorted(expected_items.items())),
        },
        "missing_item_counts": {
            "observation_count": len(valid_schema_rows),
            "total": sum(missing_items.values()) if valid_schema_rows else None,
            "counts": dict(sorted(missing_items.items())) if valid_schema_rows else None,
        },
        "extra_item_counts": {
            "observation_count": len(valid_schema_rows),
            "total": sum(extra_items.values()) if valid_schema_rows else None,
            "counts": dict(sorted(extra_items.items())) if valid_schema_rows else None,
        },
        "response_hamming_distance": mean_from_rows(rows, "response_hamming_distance"),
        "response_edit_distance": mean_from_rows(rows, "response_edit_distance"),
        "response_first_error_histogram": histogram_from_values(row["response_first_error"] for row in rows),
        "has_eos": count_rate(sum(1 for row in rows if row["has_eos"] is True), len(rows)),
        "generation_token_count_including_eos": mean_from_rows(rows, "generation_token_count_including_eos"),
        "decoded_response_utf8_bytes": mean_from_rows(rows, "decoded_response_utf8_bytes"),
        "hit_generation_cap": count_rate(sum(1 for row in rows if row["hit_generation_cap"] is True), len(rows)),
        "hit_context_cap": count_rate(sum(1 for row in rows if row["hit_context_cap"] is True), len(rows)),
    }


def teacher_forced_rows(
    model: torch.nn.Module,
    records: Sequence[FeasibilityRecord],
    tokenizer: ByteTokenizer,
    device: torch.device,
    *,
    split: str,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(records) % BATCH_SIZE != 0:
        raise ValueError("Teacher-forced evaluation forbids partial batches.")
    if model.training:
        raise ValueError("Teacher-forced evaluation requires model.eval() before entry.")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Teacher-forced evaluation requires float32 model parameters.")
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch_index in range(0, len(records), BATCH_SIZE):
            if progress_callback is not None:
                progress_callback({"subphase": "teacher_forced", "split": split, "completed_batches": batch_index // BATCH_SIZE})
            batch_records = records[batch_index : batch_index + BATCH_SIZE]
            input_ids = encode_record_batch(tuple(TextRecord(record.prompt, record.answer) for record in batch_records), tokenizer, max_length=ByteTokenizer.max_sequence_length)
            if tuple(input_ids.shape) != (BATCH_SIZE, ByteTokenizer.max_sequence_length) or input_ids.dtype != torch.int64:
                raise ValueError("Teacher-forced input batch must have int64 shape [64, 256].")
            input_ids = input_ids.to(device)
            logits = model(input_ids)
            if logits.dtype != torch.float32:
                raise ValueError("Teacher-forced logits must be float32.")
            labels = response_only_labels(input_ids)
            if labels.dtype != torch.int64:
                raise ValueError("Teacher-forced labels must be int64.")
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(BATCH_SIZE, ByteTokenizer.max_sequence_length)
            if losses.dtype != torch.float32:
                raise ValueError("Teacher-forced unreduced loss tensor must be float32.")
            predictions = logits.argmax(dim=-1)
            for row_within_batch, record in enumerate(batch_records):
                selected_positions = [position for position, label in enumerate(labels[row_within_batch].tolist()) if label != -100]
                selected_count = len(selected_positions)
                correct_count = sum(
                    int(predictions[row_within_batch, position].item()) == int(labels[row_within_batch, position].item())
                    for position in selected_positions
                )
                nll_values = [float(losses[row_within_batch, position].detach().cpu()) for position in selected_positions]
                rows.append(
                    {
                        "split": split,
                        "record_index": record.index,
                        "template_id": record.template_id,
                        "operand_id": record.operand_id,
                        "batch_index": batch_index // BATCH_SIZE,
                        "row_within_batch": row_within_batch,
                        "selected_token_count": selected_count,
                        "correct_token_count": correct_count,
                        "sequence_exact": correct_count == selected_count,
                        "nll_numerator": math.fsum(nll_values),
                    }
                )
            if progress_callback is not None:
                progress_callback({"subphase": "teacher_forced", "split": split, "completed_batches": batch_index // BATCH_SIZE + 1})
    total_selected = sum(int(row["selected_token_count"]) for row in rows)
    total_correct = sum(int(row["correct_token_count"]) for row in rows)
    total_nll = math.fsum(float(row["nll_numerator"]) for row in rows)
    aggregate = {
        "sequence_exact": count_rate(sum(1 for row in rows if row["sequence_exact"] is True), len(rows)),
        "token_accuracy": count_rate(total_correct, total_selected),
        "loss": {"observation_count": total_selected, "mean": None if total_selected == 0 else total_nll / total_selected},
        "nll_numerator": total_nll,
        "selected_token_count": total_selected,
    }
    return rows, aggregate


def load_checkpoint_model(checkpoint_path: Path, model_size: str, device: torch.device) -> torch.nn.Module:
    validate_historical_checkpoint_allowlist(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_historical_model(model_size)
    if checkpoint.get("config") != legacy_serialized_config(model_size):
        raise ValueError("Historical checkpoint config mismatch.")
    if checkpoint.get("parameter_count") != HISTORICAL_PARAMETER_COUNTS[model_size]:
        raise ValueError("Historical checkpoint parameter_count mismatch.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def generate_named_diagnostic_rows(
    model: torch.nn.Module,
    matrix_rows: Sequence[dict[str, object]],
    device: torch.device,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    tokenizer = ByteTokenizer()
    output_rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for completed_rows, row in enumerate(matrix_rows):
            if progress_callback is not None:
                progress_callback({"subphase": "named_generation", "completed_rows": completed_rows})
            prompt = require_exact_str(row["prompt"], "prompt")
            expected = require_exact_str(row["expected"], "expected")
            prefix = tokenizer.encode_evaluation_prefix(prompt)
            prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
            generated = model.greedy_decode(prefix_tensor)
            raw_ids = tuple(int(token) for token in generated[0].detach().cpu().tolist())
            roster_values = [require_exact_str(item["value"], "roster.value") for item in row["roster"]]  # type: ignore[index]
            output_rows.append(
                {
                    **row,
                    **named_row_metrics(
                        prompt,
                        expected,
                        raw_ids,
                        require_exact_str(row["target_key"], "target_key"),
                        roster_values,
                    ),
                }
            )
            if progress_callback is not None:
                progress_callback({"subphase": "named_generation", "completed_rows": completed_rows + 1})
    return output_rows


def generate_array_rows(
    model: torch.nn.Module,
    records: Sequence[FeasibilityRecord],
    device: torch.device,
    *,
    comparison_source: str,
    model_size: str,
    steps: int,
    seed: int,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    tokenizer = ByteTokenizer()
    output_rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for row_index, record in enumerate(records):
            if progress_callback is not None:
                progress_callback({"subphase": "array_generation", "completed_rows": row_index})
            prefix = tokenizer.encode_evaluation_prefix(record.prompt)
            prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
            generated = model.greedy_decode(prefix_tensor)
            raw_ids = tuple(int(token) for token in generated[0].detach().cpu().tolist())
            output_rows.append(
                {
                    "family": "array_json",
                    "comparison_source": comparison_source,
                    "model_size": model_size,
                    "steps": steps,
                    "seed": seed,
                    "row_index": row_index,
                    "source_split": record.split,
                    "source_index": record.index,
                    "template_id": record.template_id,
                    "operand_id": record.operand_id,
                    "prompt": record.prompt,
                    "expected": record.answer,
                    "item_count": array_item_count(record),
                    **array_row_metrics(record.prompt, record.answer, raw_ids),
                }
            )
            if progress_callback is not None:
                progress_callback({"subphase": "array_generation", "completed_rows": row_index + 1})
    return output_rows


def array_rows_from_reused_generations(
    generations_path: Path,
    *,
    comparison_source: str,
    model_size: str,
    steps: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    records = grouped_records()["array_json"]["eval"]
    raw_rows = validate_generation_rows(
        generations_path,
        {
            "family": "array_json",
            "eval_count": EVAL_RECORDS_PER_FAMILY,
            "exact_matches": sum(
                1
                for line in generations_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("exact_match") is True
            ),
        },
        current_record_for_row=lambda index: records[index],
    )
    for row_index, (raw_row, record) in enumerate(zip(raw_rows, records, strict=True)):
        rows.append(
            {
                "family": "array_json",
                "comparison_source": comparison_source,
                "model_size": model_size,
                "steps": steps,
                "seed": seed,
                "row_index": row_index,
                "source_split": "eval",
                "source_index": record.index,
                "template_id": record.template_id,
                "operand_id": record.operand_id,
                "prompt": record.prompt,
                "expected": record.answer,
                "item_count": array_item_count(record),
                **array_row_metrics(record.prompt, record.answer, raw_row["raw_token_ids"]),
            }
        )
    return rows


def array_item_template_counts(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[f"{row['template_id']}__items{row['item_count']}"] += 1
    denominator = len(rows)
    return {key: count_rate(value, denominator) for key, value in sorted(counts.items())}


def diagnostic_source_clean_allowed_paths(
    input_root: Path,
    predecessor_diagnostic_roots: Sequence[Path],
) -> set[Path]:
    allowed: set[Path] = set()
    for number in range(1, 5):
        root = input_root.with_name(f"feasibility_{number:03d}")
        if root.exists():
            allowed.update(inventory_bound_root_paths(root))
    for predecessor in predecessor_diagnostic_roots:
        terminal, _terminal_data, manifest, _manifest_sha = load_diagnostic_terminal_binding(predecessor)
        allowed.add(terminal.resolve())
        allowed.add(manifest.resolve())
        manifest_data = json.loads(manifest.read_text())
        for row in manifest_data.get("file_inventory", []):
            if isinstance(row, dict) and isinstance(row.get("path"), str):
                allowed.add((predecessor / row["path"]).resolve())
    return allowed


def capture_diagnostic_source_provenance(
    input_root: Path,
    output_root: Path,
    predecessor_diagnostic_roots: Sequence[Path],
) -> SourceSnapshot:
    require_diagnostic_artifact_location(output_root, "output_root")
    allowed = diagnostic_source_clean_allowed_paths(input_root, predecessor_diagnostic_roots)
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        cwd=REPO_ROOT,
    ).stdout.splitlines()
    for line in status:
        code = line[:2]
        rel = line[3:]
        candidate = (Path(rel) if Path(rel).is_absolute() else REPO_ROOT / rel).resolve()
        if code != "??":
            raise RuntimeError(f"Tracked or staged source change blocks diagnostic run: {line}")
        if candidate not in allowed:
            raise RuntimeError(f"Untracked file is not an exact supplied diagnostic/feasibility binding: {line}")
    ignored_inputs = ignored_source_inputs()
    if ignored_inputs:
        raise RuntimeError("Ignored executable source input blocks diagnostic run.")
    return SourceSnapshot(commit=current_source_commit(), status_lines=tuple(status), ignored_inputs=())


def current_environment_dict() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_driver": gpu_driver_version(),
    }


def current_deterministic_flags() -> dict[str, object]:
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


def validate_current_deterministic_flags(value: object) -> None:
    validate_diagnostic_flags(value)


def configure_feasibility_deterministic_backend() -> None:
    set_deterministic_backend(0)
    validate_current_deterministic_flags(current_deterministic_flags())


def validate_feasibility_environment() -> None:
    for key, expected in FEASIBILITY_REQUIRED_ENV.items():
        if os.environ.get(key) != expected:
            raise ValueError(f"run environment {key} must exactly equal {expected!r}.")
    if current_environment_dict() != FEASIBILITY_REQUIRED_RUNTIME_ENV:
        raise ValueError("run environment dictionary does not match the frozen cuda:0 A800 environment.")


def validate_diagnostic_core_blobs(commit: str = "HEAD") -> dict[str, str]:
    actual = {
        path: git_output(["git", "rev-parse", f"{commit}:{path}"])
        for path in DIAGNOSTIC_CORE_BLOBS
    }
    if actual != DIAGNOSTIC_CORE_BLOBS:
        raise ValueError(f"Frozen core Git blob mismatch: expected {DIAGNOSTIC_CORE_BLOBS!r}, got {actual!r}.")
    return actual


def validate_diagnostic_input_root(
    input_root: Path,
    *,
    environment: dict[str, object] | None = None,
) -> dict[str, object]:
    require_canonical_path_string(str(input_root), "input_root", ROOT_RE)
    require_artifact_location(input_root, "input_root", ROOT_RE)
    if input_root.name != DIAGNOSTIC_INPUT_ROOT_NAME:
        raise ValueError("Diagnostic input root must be the frozen feasibility_004 root.")
    for relative_path, expected_sha in DIAGNOSTIC_INPUT_CHECKSUMS.items():
        path = input_root / relative_path
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ValueError(f"Frozen feasibility_004 {relative_path} checksum mismatch.")
    validate_feasibility_root_artifacts(input_root, require_passing=False)
    manifest_data = json.loads((input_root / "manifest.json").read_text())
    if manifest_data.get("source_commit") != DIAGNOSTIC_INPUT_SOURCE_COMMIT:
        raise ValueError("Frozen feasibility_004 source commit mismatch.")
    if manifest_data.get("terminal_status") != "FAILED":
        raise ValueError("Frozen feasibility_004 must be a FAILED root.")
    actual_environment = current_environment_dict() if environment is None else environment
    if actual_environment != manifest_data.get("environment"):
        raise ValueError("Diagnostic execution environment must exactly match feasibility_004.")
    validate_diagnostic_core_blobs(DIAGNOSTIC_INPUT_SOURCE_COMMIT)
    validate_diagnostic_record_hashes()
    return manifest_data


def diagnostic_root_number(root: Path) -> int:
    match = DIAGNOSTIC_ROOT_RE.match(root.name)
    if match is None:
        raise ValueError("Diagnostic root basename must be immutable numbered form feasibility_diagnostic_NNN.")
    number = int(match.group(1))
    if number <= 0:
        raise ValueError("Diagnostic root numbering starts at feasibility_diagnostic_001.")
    return number


def require_diagnostic_artifact_location(path: Path, field_name: str) -> None:
    require_canonical_path_string(str(path), field_name, DIAGNOSTIC_ROOT_RE)
    parent_path = ARTIFACT_PARENT if ARTIFACT_PARENT.is_absolute() else REPO_ROOT / ARTIFACT_PARENT
    artifact_path = path if path.is_absolute() else REPO_ROOT / path
    reject_existing_symlink_component(parent_path, f"{field_name} artifact parent")
    reject_existing_symlink_component(artifact_path, field_name)
    parent = parent_path.resolve(strict=False)
    resolved = artifact_path.resolve(strict=False)
    if resolved.parent != parent:
        raise ValueError(f"{field_name} must be located directly under {ARTIFACT_PARENT.as_posix()}.")


def diagnostic_terminal_binding(root: Path) -> dict[str, object]:
    require_diagnostic_artifact_location(root, "predecessor_diagnostic_root")
    terminal, terminal_data, manifest, manifest_sha = load_diagnostic_terminal_binding(root)
    return {
        "path": str(DECISION_DIAGNOSTIC_ROOT_BINDING["path"]),
        "terminal_state": terminal.stem,
        "terminal_sha256": file_sha256(terminal),
        "manifest_sha256": manifest_sha,
        "source_commit": terminal_data.get("source_commit"),
        "configuration": terminal_data.get("configuration"),
        "failure_classification": terminal_data.get("failure_classification"),
        "repair_transition": terminal_data.get("repair_transition"),
        "diagnostic_lineage": terminal_data.get("diagnostic_lineage", []),
    }


def validate_diagnostic_source_provenance_fields(value: object, source_commit: str) -> None:
    if not isinstance(value, dict) or set(value) != {"commit", "status_lines", "ignored_inputs"}:
        raise ValueError("Diagnostic source_provenance must use the exact commit/status_lines/ignored_inputs schema.")
    if value.get("commit") != source_commit:
        raise ValueError("Diagnostic source_provenance commit does not match source_commit.")
    status_lines = value.get("status_lines")
    if not isinstance(status_lines, list) or not all(isinstance(line, str) for line in status_lines):
        raise ValueError("Diagnostic source_provenance status_lines must be a JSON string list.")
    if any(not line.startswith("?? ") for line in status_lines):
        raise ValueError("Diagnostic source_provenance must not declare tracked source changes.")
    if value.get("ignored_inputs") != []:
        raise ValueError("Diagnostic source_provenance ignored_inputs must be exactly empty.")


def handoff_binding_for_commit(source_commit: str) -> dict[str, object]:
    return {
        "path": DIAGNOSTIC_HANDOFF_PATH,
        "sha256": file_sha256(REPO_ROOT / DIAGNOSTIC_HANDOFF_PATH),
        "accepted_protocol_commit": DIAGNOSTIC_ACCEPTED_PROTOCOL_COMMIT,
        "accepted_protocol_blob": DIAGNOSTIC_HANDOFF_BLOB,
        "current_source_blob": git_output(["git", "rev-parse", f"{source_commit}:{DIAGNOSTIC_HANDOFF_PATH}"]),
    }


def validate_diagnostic_handoff(value: object, *, source_commit: str) -> None:
    required = {"path", "sha256", "accepted_protocol_commit", "accepted_protocol_blob", "current_source_blob"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Diagnostic handoff must bind exactly path, sha256, accepted commit/blob, and current blob.")
    if value.get("path") != DIAGNOSTIC_HANDOFF_PATH:
        raise ValueError("Diagnostic handoff path does not match the frozen handoff.")
    if value.get("accepted_protocol_commit") != DIAGNOSTIC_ACCEPTED_PROTOCOL_COMMIT:
        raise ValueError("Diagnostic handoff accepted protocol commit mismatch.")
    accepted_blob = git_output(["git", "rev-parse", f"{DIAGNOSTIC_ACCEPTED_PROTOCOL_COMMIT}:{DIAGNOSTIC_HANDOFF_PATH}"])
    if accepted_blob != DIAGNOSTIC_HANDOFF_BLOB or value.get("accepted_protocol_blob") != DIAGNOSTIC_HANDOFF_BLOB:
        raise ValueError("Diagnostic handoff accepted protocol blob mismatch.")
    current_blob = git_output(["git", "rev-parse", f"{source_commit}:{DIAGNOSTIC_HANDOFF_PATH}"])
    if value.get("current_source_blob") != current_blob:
        raise ValueError("Diagnostic handoff current source blob mismatch.")
    handoff_path = REPO_ROOT / DIAGNOSTIC_HANDOFF_PATH
    if not handoff_path.is_file() or value.get("sha256") != file_sha256(handoff_path):
        raise ValueError("Diagnostic handoff checksum mismatch.")


def validate_diagnostic_input_binding(value: object) -> Path:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "manifest_sha256",
        "summary_sha256",
        "terminal_sha256",
        "source_commit",
        "predecessor_roots",
    }:
        raise ValueError("Diagnostic input_root must bind the exact frozen input schema.")
    input_root = Path(require_canonical_path_string(value.get("path"), "input_root.path", ROOT_RE))
    require_artifact_location(input_root, "input_root.path", ROOT_RE)
    if input_root.name != DIAGNOSTIC_INPUT_ROOT_NAME:
        raise ValueError("Diagnostic input_root path must be feasibility_004.")
    if value.get("source_commit") != DIAGNOSTIC_INPUT_SOURCE_COMMIT:
        raise ValueError("Diagnostic input_root source_commit mismatch.")
    expected = {
        "manifest_sha256": DIAGNOSTIC_INPUT_CHECKSUMS["manifest.json"],
        "summary_sha256": DIAGNOSTIC_INPUT_CHECKSUMS["summary.json"],
        "terminal_sha256": DIAGNOSTIC_INPUT_CHECKSUMS["FAILED.json"],
    }
    for key, expected_sha in expected.items():
        if value.get(key) != expected_sha:
            raise ValueError(f"Diagnostic input_root {key} does not match the frozen feasibility_004 binding.")
    paths = {
        "manifest_sha256": input_root / "manifest.json",
        "summary_sha256": input_root / "summary.json",
        "terminal_sha256": input_root / "FAILED.json",
    }
    if not all(path.is_file() and not path.is_symlink() for path in paths.values()):
        raise ValueError("Diagnostic input_root must retain all frozen feasibility_004 binding files.")
    actual = {key: file_sha256(path) for key, path in paths.items()}
    if actual != expected:
        raise ValueError("Diagnostic input_root files changed from the frozen feasibility_004 binding.")
    predecessor_roots = value.get("predecessor_roots")
    input_manifest = json.loads(paths["manifest_sha256"].read_text())
    if not isinstance(predecessor_roots, list) or predecessor_roots != input_manifest.get("predecessor_roots"):
        raise ValueError("Diagnostic input_root predecessor_roots must bind the frozen lineage closure list.")
    return input_root


def validate_diagnostic_flags(value: object) -> None:
    required = {
        "PYTHONDONTWRITEBYTECODE",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONPATH",
        "torch_deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_tf32",
        "cudnn_tf32",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Diagnostic deterministic_flags must bind the complete frozen flag schema.")
    for key, expected in DIAGNOSTIC_REQUIRED_ENV.items():
        if value.get(key) != expected:
            raise ValueError(f"Diagnostic deterministic flag {key} mismatch.")
    if not all(type(value.get(key)) is bool for key in required - set(DIAGNOSTIC_REQUIRED_ENV)):
        raise ValueError("Diagnostic deterministic backend flags must be JSON booleans.")
    expected_backend = {
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_tf32": False,
        "cudnn_tf32": False,
    }
    for key, expected in expected_backend.items():
        if value.get(key) is not expected:
            raise ValueError(f"Diagnostic deterministic backend flag {key} must be exactly {expected!r}.")


def configure_diagnostic_deterministic_backend() -> None:
    set_deterministic_backend(DIAGNOSTIC_BACKEND_SEED)


def validate_diagnostic_environment_schema(value: object) -> None:
    required = {"python", "platform", "torch", "cuda_available", "cuda", "gpu", "gpu_driver"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Diagnostic environment must bind the complete execution schema.")
    for key in ("python", "platform", "torch"):
        if type(value.get(key)) is not str or not value[key]:
            raise ValueError(f"Diagnostic environment {key} must be a non-empty JSON string.")
    if type(value.get("cuda_available")) is not bool:
        raise ValueError("Diagnostic environment cuda_available must be a JSON boolean.")
    cuda_fields = ("cuda", "gpu", "gpu_driver")
    if value["cuda_available"] is True:
        if any(type(value.get(key)) is not str or not value[key] for key in cuda_fields):
            raise ValueError("CUDA diagnostic environment fields must be non-empty JSON strings.")
    elif any(value.get(key) is not None for key in cuda_fields):
        raise ValueError("Non-CUDA diagnostic environment fields must be JSON null.")


def validate_diagnostic_environment(value: object, input_root: Path) -> None:
    validate_diagnostic_environment_schema(value)
    manifest_path = input_root / "manifest.json"
    if manifest_path.is_file():
        expected_environment = json.loads(manifest_path.read_text()).get("environment")
        if value != expected_environment:
            raise ValueError("Diagnostic environment does not exactly match the frozen feasibility_004 manifest.")


def expected_diagnostic_completed_scopes() -> list[dict[str, object]]:
    return [
        {"name": "named_matrix_construction", "rows": len(NAMED_DIAGNOSTIC_CELLS) * EVAL_RECORDS_PER_FAMILY},
        *(
            {"name": "named_cell", "seed": seed, "cell": cell, "rows": EVAL_RECORDS_PER_FAMILY}
            for seed in SEEDS
            for cell in NAMED_DIAGNOSTIC_CELLS
        ),
        {"name": "array_training_plan", "runs": len(SEEDS)},
        *(
            {
                "name": "array_baseline_reuse",
                "model_size": model_size,
                "steps": TRAINING_STEPS,
                "seed": seed,
                "rows": EVAL_RECORDS_PER_FAMILY,
            }
            for model_size in MODEL_SIZES
            for seed in SEEDS
        ),
        *(
            {
                "name": "array_diagnostic_training",
                "model_size": "small",
                "steps": DIAGNOSTIC_STEPS,
                "seed": seed,
                "rows": EVAL_RECORDS_PER_FAMILY,
                "checkpoint_path": f"array_json__small__3000__seed{seed}/checkpoint_step3000.pt",
            }
            for seed in SEEDS
        ),
    ]


def validate_diagnostic_completed_scope_row(row: dict[str, object], template: dict[str, object]) -> None:
    name = template["name"]
    if row.get("name") != name:
        raise ValueError("Diagnostic completed_scope is not a canonical operation prefix.")
    if name == "named_matrix_construction":
        if row != template:
            raise ValueError("Diagnostic Named matrix scope must bind exactly 256 constructed rows.")
    elif name == "array_training_plan":
        if row != template:
            raise ValueError("Diagnostic Array plan scope must bind exactly three planned runs.")
    elif name == "named_cell":
        if set(row) != {"name", "seed", "cell", "rows", "checkpoint_reused_from", "checkpoint_sha256"}:
            raise ValueError("Diagnostic Named scopes must use the exact frozen schema.")
        for key in ("seed", "cell", "rows"):
            if row.get(key) != template[key]:
                raise ValueError("Diagnostic Named scope identity or row count mismatch.")
        require_canonical_path_string(row.get("checkpoint_reused_from"), "completed_scope.checkpoint_reused_from", re.compile(r"^checkpoint_step1500\.pt$"))
        checkpoint_sha = require_exact_str(row.get("checkpoint_sha256"), "completed_scope.checkpoint_sha256")
        if re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha) is None:
            raise ValueError("Diagnostic Named checkpoint_sha256 must be a lowercase SHA-256.")
    elif name == "array_baseline_reuse":
        if row != template:
            raise ValueError("Diagnostic reused Array scope identity/schema/count mismatch.")
    elif name == "array_diagnostic_training":
        if row != template:
            raise ValueError("Diagnostic new Array scope identity/schema/count/checkpoint mismatch.")
    else:
        raise ValueError("Diagnostic completed_scope contains an unknown operation.")


def validate_diagnostic_completed_scope(value: object, *, terminal_status: str) -> None:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("Diagnostic completed_scope must be a JSON object list.")
    expected = expected_diagnostic_completed_scopes()
    if len(value) > len(expected):
        raise ValueError("Diagnostic completed_scope contains fabricated extra scopes.")
    if terminal_status == "DONE" and len(value) != len(expected):
        raise ValueError("DONE diagnostic completed_scope forbids omissions.")
    for row, template in zip(value, expected, strict=False):
        validate_diagnostic_completed_scope_row(row, template)
    if len({compact_json(row) for row in value}) != len(value):
        raise ValueError("Diagnostic completed_scope contains duplicate scopes.")


def expected_diagnostic_partial_base(completed_count: int) -> list[dict[str, object]]:
    expected = expected_diagnostic_completed_scopes()
    if completed_count >= len(expected):
        return [{"name": "terminal_publication"}]
    next_scope = expected[completed_count]
    candidates: list[dict[str, object]] = []
    if completed_count == 0:
        candidates.append({"name": "diagnostic_initialization"})
    if next_scope["name"] == "named_cell" and next_scope["cell"] == NAMED_DIAGNOSTIC_CELLS[0]:
        candidates.append({"name": "named_checkpoint_reuse", "seed": next_scope["seed"]})
    candidates.append({key: value for key, value in next_scope.items() if key not in {"rows", "runs", "checkpoint_path"}})
    return candidates


def progress_without_error(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key != "error"}


def validate_completed_rows_progress(row: dict[str, object], *, limit: int, field_name: str) -> None:
    completed = require_exact_int(row.get(field_name), f"partial_scope.{field_name}")
    if not 0 <= completed <= limit:
        raise ValueError(f"Diagnostic partial_scope {field_name} is out of range.")


def validate_diagnostic_partial_scope(
    value: object,
    *,
    completed_scope: object,
    terminal_status: str,
    failure: object,
) -> None:
    if terminal_status == "DONE":
        if value != []:
            raise ValueError("DONE diagnostic partial_scope must be exactly empty.")
        return
    if not isinstance(completed_scope, list) or not all(isinstance(row, dict) for row in completed_scope):
        raise ValueError("FAILED diagnostic completed_scope must be a JSON object list before partial validation.")
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError("FAILED diagnostic partial_scope must contain exactly one final active scope object.")
    row = value[0]
    if row.get("error") != failure:
        raise ValueError("FAILED diagnostic final partial scope must retain the exact failure error.")
    active = progress_without_error(row)
    candidates = expected_diagnostic_partial_base(len(completed_scope))
    identity_keys = {"name", "seed", "cell", "model_size", "steps"}
    active_identity = {key: value for key, value in active.items() if key in identity_keys}
    if active_identity not in candidates:
        raise ValueError("FAILED diagnostic partial_scope does not match the next legal operation identity.")
    progress_keys = set(active) - set(active_identity)
    if not progress_keys:
        return
    name = active_identity["name"]
    subphase = active.get("subphase")
    if subphase == "array_training" and name == "array_diagnostic_training":
        if progress_keys != {"subphase", "completed_steps"}:
            raise ValueError("Array training partial_scope must bind exactly completed_steps progress.")
        validate_completed_rows_progress(active, limit=DIAGNOSTIC_STEPS, field_name="completed_steps")
    elif subphase in {"array_generation", "named_generation"}:
        expected_name = "array_diagnostic_training" if subphase == "array_generation" else "named_cell"
        if name != expected_name or progress_keys != {"subphase", "completed_rows"}:
            raise ValueError("Generation partial_scope must bind the correct operation identity and completed_rows.")
        validate_completed_rows_progress(active, limit=EVAL_RECORDS_PER_FAMILY, field_name="completed_rows")
    elif subphase == "teacher_forced" and name in {"array_baseline_reuse", "array_diagnostic_training"}:
        if progress_keys != {"subphase", "split", "completed_batches"}:
            raise ValueError("Teacher-forced partial_scope must bind split and completed_batches.")
        split = active.get("split")
        if split == "train":
            limit = TRAIN_RECORDS_PER_FAMILY // BATCH_SIZE
        elif split == "eval":
            limit = EVAL_RECORDS_PER_FAMILY // BATCH_SIZE
        else:
            raise ValueError("Teacher-forced partial_scope split must be train or eval.")
        validate_completed_rows_progress(active, limit=limit, field_name="completed_batches")
    else:
        raise ValueError("FAILED diagnostic partial_scope progress is not legal for the active operation.")


def validate_diagnostic_artifact_roles(root: Path, file_inventory: object) -> None:
    if not isinstance(file_inventory, list):
        raise ValueError("Diagnostic file_inventory must be a JSON list.")
    for row in file_inventory:
        if not isinstance(row, dict):
            raise ValueError("Diagnostic file_inventory entries must be JSON objects.")
        path = require_canonical_relative_path(row.get("path"), "diagnostic.file_inventory.path")
        role = require_exact_str(row.get("role"), "diagnostic.file_inventory.role")
        if path == "summary.json" and role != "summary":
            raise ValueError("Diagnostic summary.json must have role summary.")
        if path.endswith(".pt") and role != "checkpoint":
            raise ValueError("Diagnostic checkpoints must have role checkpoint.")
        if path.endswith(".jsonl") and role != "retained_rows":
            raise ValueError("Diagnostic JSONL rows must have role retained_rows.")
        if path not in {"summary.json"} and not path.endswith((".pt", ".jsonl")) and role != "diagnostic_artifact":
            raise ValueError("Diagnostic non-row artifacts must have role diagnostic_artifact.")


def validate_diagnostic_completed_scope_artifacts(root: Path, completed_scope: object) -> None:
    if not isinstance(completed_scope, list) or not all(isinstance(row, dict) for row in completed_scope):
        raise ValueError("Diagnostic completed_scope must be valid before artifact consistency checks.")
    required_paths: set[str] = set()
    for row in completed_scope:
        name = row["name"]
        if name == "named_matrix_construction":
            required_paths.add("named_diagnostic_matrix.jsonl")
        elif name == "array_training_plan":
            required_paths.add("array_training_plan.json")
        elif name == "named_cell":
            prefix = f"named_value_json__medium__seed{row['seed']}__{row['cell']}"
            required_paths.update({f"{prefix}/generations.jsonl", f"{prefix}/metrics.json"})
        elif name == "array_baseline_reuse":
            prefix = f"array_json__{row['model_size']}__1500__seed{row['seed']}__reused"
            required_paths.update({
                f"{prefix}/greedy_metrics_rows.jsonl",
                f"{prefix}/teacher_forced_train_rows.jsonl",
                f"{prefix}/teacher_forced_eval_rows.jsonl",
                f"{prefix}/metrics.json",
            })
        elif name == "array_diagnostic_training":
            prefix = f"array_json__small__3000__seed{row['seed']}"
            required_paths.update({
                f"{prefix}/checkpoint_step3000.pt",
                f"{prefix}/generations.jsonl",
                f"{prefix}/teacher_forced_train_rows.jsonl",
                f"{prefix}/teacher_forced_eval_rows.jsonl",
                f"{prefix}/metrics.json",
            })
    for relative_path in sorted(required_paths):
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Diagnostic completed_scope lacks retained artifact {relative_path!r}.")


def read_jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"{path} contains a blank JSONL row at line {line_number}.")
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path} rows must be JSON objects.")
            rows.append(row)
    return rows


def validate_diagnostic_checkpoint_3000(path: Path, *, seed: int) -> None:
    validate_historical_checkpoint_allowlist(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Diagnostic 3000 checkpoint is not a valid PyTorch checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("Diagnostic 3000 checkpoint must be a mapping.")
    if checkpoint.get("config") != legacy_serialized_config("small"):
        raise ValueError("Diagnostic 3000 checkpoint config mismatch.")
    if checkpoint.get("parameter_count") != HISTORICAL_PARAMETER_COUNTS["small"]:
        raise ValueError("Diagnostic 3000 checkpoint parameter_count mismatch.")
    metadata = checkpoint.get("metadata")
    expected_metadata = {
        "artifact_class": DIAGNOSTIC_ARTIFACT_CLASS,
        "feasibility_selection_eligible": False,
        "task_010d_authorized": False,
        "family": "array_json",
        "model_size": "small",
        "seed": seed,
        "training_steps": DIAGNOSTIC_STEPS,
        "step1501_executed": True,
    }
    if not isinstance(metadata, dict):
        raise ValueError("Diagnostic 3000 checkpoint metadata must be a mapping.")
    expected_metadata_keys = {*expected_metadata, "training_loss", "training_trajectory_evidence"}
    if set(metadata) != expected_metadata_keys:
        raise ValueError("Diagnostic 3000 checkpoint metadata must use the exact trajectory-evidence schema.")
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"Diagnostic 3000 checkpoint metadata {key} mismatch.")
    training_loss = metadata.get("training_loss")
    if type(training_loss) not in {int, float} or not math.isfinite(float(training_loss)):
        raise ValueError("Diagnostic 3000 checkpoint training_loss must be finite.")
    state = checkpoint.get("model_state_dict")
    if model_state_schema_from_state_dict(state) != isolated_model_state_schema("small"):
        raise ValueError("Diagnostic 3000 checkpoint state_dict dtype/shape schema mismatch.")
    validate_diagnostic_trajectory_evidence(
        metadata.get("training_trajectory_evidence"),
        seed=seed,
        final_model_state_fingerprint=model_state_fingerprint_from_state_dict(state),
        expected_final_loss=training_loss,
    )


def validate_loss_scalar_evidence(value: object, *, expected_value: object, field_name: str) -> None:
    if not isinstance(value, dict) or set(value) != {"value", "repr", "tensor_dtype", "tensor_shape", "tensor_bytes_hex"}:
        raise ValueError(f"{field_name} must use the exact loss scalar evidence schema.")
    if type(value.get("value")) not in {int, float} or not math.isfinite(float(value["value"])):
        raise ValueError(f"{field_name}.value must be finite.")
    if float(value["value"]) != float(expected_value):
        raise ValueError(f"{field_name}.value does not match retained final_loss.")
    if value.get("repr") != repr(float(value["value"])):
        raise ValueError(f"{field_name}.repr does not match the retained Python float representation.")
    if value.get("tensor_dtype") != "torch.float32" or value.get("tensor_shape") != []:
        raise ValueError(f"{field_name} must bind a scalar float32 loss tensor.")
    tensor_bytes_hex = value.get("tensor_bytes_hex")
    if not isinstance(tensor_bytes_hex, str) or not re.fullmatch(r"[0-9a-f]+", tensor_bytes_hex) or len(tensor_bytes_hex) != 8:
        raise ValueError(f"{field_name}.tensor_bytes_hex must bind exactly one float32 scalar.")
    expected_tensor_bytes_hex = tensor_bytes(torch.tensor(float(value["value"]), dtype=torch.float32)).hex()
    if tensor_bytes_hex != expected_tensor_bytes_hex:
        raise ValueError(f"{field_name}.tensor_bytes_hex does not encode the retained float32 loss value.")


def validate_diagnostic_trajectory_evidence(
    evidence: object,
    *,
    seed: int,
    final_model_state_fingerprint: str,
    expected_final_loss: object,
    frozen_step1500_model_state_fingerprint: str | None = None,
) -> None:
    expected_keys = {
        "serialization",
        "claimed_seed",
        "family",
        "model_size",
        "training_steps",
        "authorized_new_training_run_count",
        "schedule",
        "initial_model_state_fingerprint",
        "step1500",
        "trace",
        "step_count",
        "step1501_executed",
        "final_model_state_fingerprint",
        "final_optimizer_state_fingerprint",
        "retained_final_loss",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_keys:
        raise ValueError("Diagnostic training trajectory evidence must use the exact schema.")
    if evidence.get("serialization") != DIAGNOSTIC_TRAJECTORY_EVIDENCE_SCHEMA:
        raise ValueError("Diagnostic training trajectory evidence serialization mismatch.")
    if evidence.get("claimed_seed") != seed:
        raise ValueError("Diagnostic training trajectory claimed_seed mismatch.")
    if evidence.get("family") != "array_json" or evidence.get("model_size") != "small":
        raise ValueError("Diagnostic training trajectory family/model_size mismatch.")
    if evidence.get("training_steps") != DIAGNOSTIC_STEPS or evidence.get("step_count") != DIAGNOSTIC_STEPS:
        raise ValueError("Diagnostic training trajectory step count mismatch.")
    if evidence.get("authorized_new_training_run_count") != len(SEEDS):
        raise ValueError("Diagnostic training trajectory authorized run count mismatch.")
    if evidence.get("step1501_executed") is not True:
        raise ValueError("Diagnostic training trajectory must bind step1501 execution.")
    retained_final_loss = evidence.get("retained_final_loss")
    if type(retained_final_loss) not in {int, float} or not math.isfinite(float(retained_final_loss)):
        raise ValueError("Diagnostic training trajectory retained_final_loss must be finite.")
    if float(retained_final_loss) != float(expected_final_loss):
        raise ValueError("Diagnostic training trajectory retained_final_loss mismatch.")
    schedule = evidence.get("schedule")
    expected_schedule = diagnostic_training_schedule_evidence(seed, TRAIN_RECORDS_PER_FAMILY)
    if schedule != expected_schedule:
        raise ValueError("Diagnostic training trajectory schedule evidence mismatch.")
    initial_fingerprint = evidence.get("initial_model_state_fingerprint")
    if initial_fingerprint != expected_initial_model_state_fingerprint(seed):
        raise ValueError("Diagnostic training trajectory initial model-state fingerprint mismatch.")
    step1500 = evidence.get("step1500")
    step1500_keys = {
        "step",
        "model_state_fingerprint",
        "checkpoint_state_fingerprint",
        "model_matches_checkpoint",
        "optimizer_state_fingerprint",
    }
    if not isinstance(step1500, dict) or set(step1500) != step1500_keys:
        raise ValueError("Diagnostic training trajectory step1500 evidence must use the exact schema.")
    if step1500.get("step") != TRAINING_STEPS:
        raise ValueError("Diagnostic training trajectory step1500 step mismatch.")
    step1500_model_fingerprint = require_sha256_hex(step1500.get("model_state_fingerprint"), "trajectory.step1500.model_state_fingerprint")
    step1500_checkpoint_fingerprint = require_sha256_hex(step1500.get("checkpoint_state_fingerprint"), "trajectory.step1500.checkpoint_state_fingerprint")
    require_sha256_hex(step1500.get("optimizer_state_fingerprint"), "trajectory.step1500.optimizer_state_fingerprint")
    if step1500.get("model_matches_checkpoint") is not True or step1500_model_fingerprint != step1500_checkpoint_fingerprint:
        raise ValueError("Diagnostic training trajectory step1500 equality gate was not bound as passed.")
    if frozen_step1500_model_state_fingerprint is not None and step1500_checkpoint_fingerprint != frozen_step1500_model_state_fingerprint:
        raise ValueError("Diagnostic training trajectory step1500 checkpoint fingerprint mismatch.")
    trace = evidence.get("trace")
    trace_keys = {"serialization", "seed", "step_count", "schedule_sha256", "sha256", "final_loss"}
    if not isinstance(trace, dict) or set(trace) != trace_keys:
        raise ValueError("Diagnostic training trajectory trace must use the exact schema.")
    if trace.get("serialization") != DIAGNOSTIC_TRAINING_TRACE_SCHEMA:
        raise ValueError("Diagnostic training trajectory trace serialization mismatch.")
    if trace.get("seed") != seed or trace.get("step_count") != DIAGNOSTIC_STEPS:
        raise ValueError("Diagnostic training trajectory trace seed/step_count mismatch.")
    if trace.get("schedule_sha256") != expected_schedule["sha256"]:
        raise ValueError("Diagnostic training trajectory trace schedule digest mismatch.")
    require_sha256_hex(trace.get("sha256"), "trajectory.trace.sha256")
    validate_loss_scalar_evidence(trace.get("final_loss"), expected_value=retained_final_loss, field_name="trajectory.trace.final_loss")
    if require_sha256_hex(evidence.get("final_model_state_fingerprint"), "trajectory.final_model_state_fingerprint") != final_model_state_fingerprint:
        raise ValueError("Diagnostic training trajectory final model-state fingerprint mismatch.")
    require_sha256_hex(evidence.get("final_optimizer_state_fingerprint"), "trajectory.final_optimizer_state_fingerprint")


def validate_diagnostic_training_trajectory_evidence(
    checkpoint_path: Path,
    *,
    seed: int,
    training_metrics: object,
    frozen_step1500_checkpoint: Path,
) -> None:
    validate_diagnostic_checkpoint_3000(checkpoint_path, seed=seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("metadata"), dict):
        raise ValueError("Diagnostic trajectory checkpoint metadata is invalid.")
    metadata = checkpoint["metadata"]  # type: ignore[index]
    evidence = metadata.get("training_trajectory_evidence")
    if not isinstance(training_metrics, dict):
        raise ValueError("Diagnostic trajectory training metrics must be a mapping.")
    if training_metrics.get("trajectory_evidence") != evidence:
        raise ValueError("Diagnostic trajectory evidence mismatch between checkpoint metadata and metrics.")
    final_loss = training_metrics.get("final_loss")
    if type(final_loss) not in {int, float} or not math.isfinite(float(final_loss)):
        raise ValueError("Diagnostic trajectory training metrics final_loss must be finite.")
    if float(final_loss) != float(metadata.get("training_loss")):
        raise ValueError("Diagnostic trajectory training metrics final_loss does not bind checkpoint metadata.")
    if training_metrics.get("step1501_executed") is not True:
        raise ValueError("Diagnostic trajectory training metrics must bind step1501 execution.")
    validate_diagnostic_trajectory_evidence(
        evidence,
        seed=seed,
        final_model_state_fingerprint=model_state_fingerprint_from_state_dict(checkpoint.get("model_state_dict")),
        expected_final_loss=final_loss,
        frozen_step1500_model_state_fingerprint=checkpoint_model_state_fingerprint(frozen_step1500_checkpoint),
    )


def diagnostic_replay_device() -> torch.device:
    device = torch.device(DIAGNOSTIC_REQUIRED_DEVICE)
    if device.type != "cuda" or device.index != 0:
        raise ValueError("Diagnostic replay validation must run on cuda:0.")
    return device


def feasibility_replay_device() -> torch.device:
    if not torch.cuda.is_available():
        raise ValueError("Current feasibility checkpoint replay requires cuda:0; CPU fallback is forbidden.")
    device = torch.device(FEASIBILITY_REQUIRED_DEVICE)
    if device.type != "cuda" or device.index != 0:
        raise ValueError("Current feasibility replay validation must run on cuda:0.")
    return device


def validate_replay_model_state(model: torch.nn.Module) -> None:
    if model.training:
        raise ValueError("Diagnostic replay requires model.eval().")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Diagnostic replay requires float32 model parameters.")


def replay_named_generation_artifact(
    checkpoint_path: Path,
    *,
    matrix_rows: Sequence[dict[str, object]],
    retained_rows: Sequence[dict[str, object]],
) -> None:
    device = diagnostic_replay_device()
    model = load_checkpoint_model(checkpoint_path, "medium", device)
    validate_replay_model_state(model)
    replayed_rows = generate_named_diagnostic_rows(model, matrix_rows, device)
    if replayed_rows != list(retained_rows):
        raise ValueError("DONE Named retained rows are not exact outputs of the bound checkpoint.")


def replay_array_artifacts(
    checkpoint_path: Path,
    *,
    model_size: str,
    comparison_source: str,
    steps: int,
    seed: int,
    retained_greedy_rows: Sequence[dict[str, object]],
    retained_train_rows: Sequence[dict[str, object]],
    retained_eval_rows: Sequence[dict[str, object]],
    retained_train_aggregate: object,
    retained_eval_aggregate: object,
) -> None:
    device = diagnostic_replay_device()
    tokenizer = ByteTokenizer()
    records = grouped_records()["array_json"]
    model = load_checkpoint_model(checkpoint_path, model_size, device)
    validate_replay_model_state(model)
    replayed_greedy_rows = generate_array_rows(
        model,
        records["eval"],
        device,
        comparison_source=comparison_source,
        model_size=model_size,
        steps=steps,
        seed=seed,
    )
    if replayed_greedy_rows != list(retained_greedy_rows):
        raise ValueError("DONE Array retained greedy rows are not exact outputs of the bound checkpoint.")
    replayed_train_rows, replayed_train_aggregate = teacher_forced_rows(
        model,
        records["train"],
        tokenizer,
        device,
        split="train",
    )
    replayed_eval_rows, replayed_eval_aggregate = teacher_forced_rows(
        model,
        records["eval"],
        tokenizer,
        device,
        split="eval",
    )
    if replayed_train_rows != list(retained_train_rows) or replayed_train_aggregate != retained_train_aggregate:
        raise ValueError("DONE Array retained train teacher-forced rows are not exact outputs of the bound checkpoint.")
    if replayed_eval_rows != list(retained_eval_rows) or replayed_eval_aggregate != retained_eval_aggregate:
        raise ValueError("DONE Array retained eval teacher-forced rows are not exact outputs of the bound checkpoint.")


def validate_teacher_forced_artifact(rows: Sequence[dict[str, object]], aggregate: object, *, split: str, expected_count: int) -> None:
    if len(rows) != expected_count:
        raise ValueError("Diagnostic teacher-forced row artifact has the wrong row count.")
    records = grouped_records()["array_json"][split]
    for index, row in enumerate(rows):
        if set(row) != {
            "split", "record_index", "template_id", "operand_id", "batch_index", "row_within_batch",
            "selected_token_count", "correct_token_count", "sequence_exact", "nll_numerator",
        }:
            raise ValueError("Diagnostic teacher-forced rows must use the exact retained schema.")
        if row.get("split") != split:
            raise ValueError("Diagnostic teacher-forced row split mismatch.")
        if row.get("record_index") != index:
            raise ValueError("Diagnostic teacher-forced rows must be ordered by record_index.")
        record = records[index]
        if row.get("template_id") != record.template_id or row.get("operand_id") != record.operand_id:
            raise ValueError("Diagnostic teacher-forced row identity does not match the frozen record.")
        if require_exact_int(row.get("batch_index"), "teacher_forced.batch_index") != index // BATCH_SIZE:
            raise ValueError("Diagnostic teacher-forced batch_index mismatch.")
        if require_exact_int(row.get("row_within_batch"), "teacher_forced.row_within_batch") != index % BATCH_SIZE:
            raise ValueError("Diagnostic teacher-forced row_within_batch mismatch.")
        selected_count = require_exact_int(row.get("selected_token_count"), "teacher_forced.selected_token_count")
        correct_count = require_exact_int(row.get("correct_token_count"), "teacher_forced.correct_token_count")
        sequence_exact = require_exact_bool(row.get("sequence_exact"), "teacher_forced.sequence_exact")
        if selected_count <= 0 or not 0 <= correct_count <= selected_count or sequence_exact != (correct_count == selected_count):
            raise ValueError("Diagnostic teacher-forced selected/correct/sequence-exact fields are inconsistent.")
        nll_numerator = row.get("nll_numerator")
        if type(nll_numerator) not in {int, float} or not math.isfinite(float(nll_numerator)) or nll_numerator < 0:
            raise ValueError("Diagnostic teacher-forced nll_numerator must be finite and non-negative.")
    expected_aggregate = {
        "sequence_exact": count_rate(sum(1 for row in rows if row["sequence_exact"] is True), len(rows)),
        "token_accuracy": count_rate(
            sum(int(row["correct_token_count"]) for row in rows),
            sum(int(row["selected_token_count"]) for row in rows),
        ),
        "loss": mean_value(
            math.fsum(float(row["nll_numerator"]) for row in rows),
            sum(int(row["selected_token_count"]) for row in rows),
        ),
        "nll_numerator": math.fsum(float(row["nll_numerator"]) for row in rows),
        "selected_token_count": sum(int(row["selected_token_count"]) for row in rows),
    }
    if aggregate != expected_aggregate:
        raise ValueError("Diagnostic teacher-forced aggregate does not rebuild from retained rows.")


def validate_named_generation_artifact(
    rows: Sequence[dict[str, object]],
    matrix_rows: Sequence[dict[str, object]],
) -> None:
    if len(rows) != EVAL_RECORDS_PER_FAMILY:
        raise ValueError("DONE Named generation artifact must retain exactly 64 rows.")
    for row_index, (row, matrix_row) in enumerate(zip(rows, matrix_rows, strict=True)):
        roster = matrix_row["roster"]
        roster_values = [require_exact_str(item["value"], "named_generation.roster.value") for item in roster]  # type: ignore[index]
        raw_token_ids = row.get("raw_token_ids")
        if not isinstance(raw_token_ids, list) or not all(type(token) is int for token in raw_token_ids):
            raise ValueError("DONE Named generation raw_token_ids must be JSON integers.")
        expected_row = {
            **matrix_row,
            **named_row_metrics(
                require_exact_str(matrix_row["prompt"], "named_generation.prompt"),
                require_exact_str(matrix_row["expected"], "named_generation.expected"),
                raw_token_ids,
                require_exact_str(matrix_row["target_key"], "named_generation.target_key"),
                roster_values,
            ),
        }
        if row != expected_row or row.get("row_index") != row_index:
            raise ValueError("DONE Named generation row does not rebuild from the frozen matrix and raw tokens.")


def validate_array_generation_artifact(
    rows: Sequence[dict[str, object]],
    *,
    comparison_source: str,
    model_size: str,
    steps: int,
    seed: int,
) -> None:
    records = grouped_records()["array_json"]["eval"]
    if len(rows) != EVAL_RECORDS_PER_FAMILY:
        raise ValueError("DONE Array greedy artifact must retain exactly 64 rows.")
    for row_index, (row, record) in enumerate(zip(rows, records, strict=True)):
        raw_token_ids = row.get("raw_token_ids")
        if not isinstance(raw_token_ids, list) or not all(type(token) is int for token in raw_token_ids):
            raise ValueError("DONE Array generation raw_token_ids must be JSON integers.")
        expected_row = {
            "family": "array_json",
            "comparison_source": comparison_source,
            "model_size": model_size,
            "steps": steps,
            "seed": seed,
            "row_index": row_index,
            "source_split": record.split,
            "source_index": record.index,
            "template_id": record.template_id,
            "operand_id": record.operand_id,
            "prompt": record.prompt,
            "expected": record.answer,
            "item_count": array_item_count(record),
            **array_row_metrics(record.prompt, record.answer, raw_token_ids),
        }
        if row != expected_row:
            raise ValueError("DONE Array generation row does not rebuild from the frozen eval record and raw tokens.")


def diagnostic_named_matrix_by_cell(root: Path) -> dict[str, list[dict[str, object]]]:
    matrix_rows = read_jsonl_rows(root / "named_diagnostic_matrix.jsonl")
    matrix: dict[str, list[dict[str, object]]] = {cell: [] for cell in NAMED_DIAGNOSTIC_CELLS}
    for row in matrix_rows:
        cell = require_exact_str(row.get("diagnostic_cell"), "named_matrix.diagnostic_cell")
        if cell not in matrix:
            raise ValueError("Named diagnostic matrix contains an unexpected cell.")
        matrix[cell].append(row)
    validate_named_diagnostic_matrix(matrix)
    return matrix


def validate_diagnostic_named_scope_semantics(root: Path, row: dict[str, object], input_root: Path, matrix: dict[str, list[dict[str, object]]]) -> None:
    seed = require_exact_int(row.get("seed"), "completed_scope.seed")
    cell = require_exact_str(row.get("cell"), "completed_scope.cell")
    prefix = root / f"named_value_json__medium__seed{seed}__{cell}"
    rows = read_jsonl_rows(prefix / "generations.jsonl")
    validate_named_generation_artifact(rows, matrix[cell])
    metrics = json.loads((prefix / "metrics.json").read_text())
    if metrics != aggregate_named_rows(rows):
        raise ValueError("Diagnostic Named metrics do not rebuild from retained rows.")
    reused_named = reused_feasibility_cell_binding(input_root, "named_value_json", "medium", seed)
    if row.get("checkpoint_reused_from") != reused_named["checkpoint_path"]:
        raise ValueError("Diagnostic Named completed scope checkpoint path mismatch.")
    if row.get("checkpoint_sha256") != reused_named["checkpoint_sha256"]:
        raise ValueError("Diagnostic Named completed scope checkpoint checksum mismatch.")
    replay_named_generation_artifact(
        Path(str(reused_named["checkpoint_path"])),
        matrix_rows=matrix[cell],
        retained_rows=rows,
    )


def validate_diagnostic_array_plan_scope_semantics(root: Path) -> None:
    plan = json.loads((root / "array_training_plan.json").read_text())
    validate_diagnostic_array_training_plan(plan)


def validate_diagnostic_reused_array_scope_semantics(root: Path, row: dict[str, object], input_root: Path) -> None:
    model_size = require_exact_str(row.get("model_size"), "completed_scope.model_size")
    seed = require_exact_int(row.get("seed"), "completed_scope.seed")
    prefix = root / f"array_json__{model_size}__1500__seed{seed}__reused"
    greedy_rows = read_jsonl_rows(prefix / "greedy_metrics_rows.jsonl")
    validate_array_generation_artifact(
        greedy_rows,
        comparison_source="feasibility_004_reused",
        model_size=model_size,
        steps=TRAINING_STEPS,
        seed=seed,
    )
    train_rows = read_jsonl_rows(prefix / "teacher_forced_train_rows.jsonl")
    eval_rows = read_jsonl_rows(prefix / "teacher_forced_eval_rows.jsonl")
    metrics = json.loads((prefix / "metrics.json").read_text())
    expected_metric_keys = {
        "greedy", "teacher_forced_train", "teacher_forced_eval", "template_item_counts",
        "reused_generations_path", "reused_generations_sha256", "reused_checkpoint_path", "reused_checkpoint_sha256",
    }
    if set(metrics) != expected_metric_keys:
        raise ValueError("Diagnostic reused Array metrics must use the exact frozen schema.")
    validate_teacher_forced_artifact(train_rows, metrics.get("teacher_forced_train"), split="train", expected_count=TRAIN_RECORDS_PER_FAMILY)
    validate_teacher_forced_artifact(eval_rows, metrics.get("teacher_forced_eval"), split="eval", expected_count=EVAL_RECORDS_PER_FAMILY)
    if metrics.get("greedy") != aggregate_array_rows(greedy_rows):
        raise ValueError("Diagnostic reused Array greedy aggregate does not rebuild from retained rows.")
    if metrics.get("template_item_counts") != array_item_template_counts(greedy_rows):
        raise ValueError("Diagnostic reused Array template_item_counts do not rebuild from retained rows.")
    reused = reused_feasibility_cell_binding(input_root, "array_json", model_size, seed)
    for key in ("reused_generations_path", "reused_generations_sha256", "reused_checkpoint_path", "reused_checkpoint_sha256"):
        source_key = key.removeprefix("reused_")
        if metrics.get(key) != reused[source_key]:
            raise ValueError(f"Diagnostic reused Array {key} does not match feasibility_004.")
    replay_array_artifacts(
        Path(str(reused["checkpoint_path"])),
        model_size=model_size,
        comparison_source="feasibility_004_reused",
        steps=TRAINING_STEPS,
        seed=seed,
        retained_greedy_rows=greedy_rows,
        retained_train_rows=train_rows,
        retained_eval_rows=eval_rows,
        retained_train_aggregate=metrics.get("teacher_forced_train"),
        retained_eval_aggregate=metrics.get("teacher_forced_eval"),
    )


def validate_diagnostic_new_array_scope_semantics(root: Path, row: dict[str, object], input_root: Path) -> None:
    seed = require_exact_int(row.get("seed"), "completed_scope.seed")
    prefix = root / f"array_json__small__3000__seed{seed}"
    checkpoint_path = prefix / "checkpoint_step3000.pt"
    validate_diagnostic_checkpoint_3000(checkpoint_path, seed=seed)
    greedy_rows = read_jsonl_rows(prefix / "generations.jsonl")
    validate_array_generation_artifact(
        greedy_rows,
        comparison_source="diagnostic_small_3000",
        model_size="small",
        steps=DIAGNOSTIC_STEPS,
        seed=seed,
    )
    train_rows = read_jsonl_rows(prefix / "teacher_forced_train_rows.jsonl")
    eval_rows = read_jsonl_rows(prefix / "teacher_forced_eval_rows.jsonl")
    metrics = json.loads((prefix / "metrics.json").read_text())
    expected_metric_keys = {
        "training", "greedy", "teacher_forced_train", "teacher_forced_eval", "template_item_counts",
        "step1500_checkpoint_reused_for_equality_gate", "step1500_checkpoint_sha256",
    }
    if set(metrics) != expected_metric_keys:
        raise ValueError("Diagnostic new Array metrics must use the exact frozen schema.")
    training = metrics.get("training")
    if not isinstance(training, dict) or set(training) != {"final_loss", "checkpoint_path", "step1501_executed", "trajectory_evidence"}:
        raise ValueError("Diagnostic new Array training metrics must use the exact frozen schema.")
    if training.get("step1501_executed") is not True:
        raise ValueError("Diagnostic new Array metrics must bind nonselection 3000-step training metadata.")
    final_loss = training.get("final_loss")
    if type(final_loss) not in {int, float} or not math.isfinite(float(final_loss)):
        raise ValueError("Diagnostic new Array final_loss must be finite.")
    expected_checkpoint_path = f"array_json__small__3000__seed{seed}/checkpoint_step3000.pt"
    if training.get("checkpoint_path") != expected_checkpoint_path or row.get("checkpoint_path") != expected_checkpoint_path:
        raise ValueError("Diagnostic new Array training checkpoint_path mismatch.")
    validate_teacher_forced_artifact(train_rows, metrics.get("teacher_forced_train"), split="train", expected_count=TRAIN_RECORDS_PER_FAMILY)
    validate_teacher_forced_artifact(eval_rows, metrics.get("teacher_forced_eval"), split="eval", expected_count=EVAL_RECORDS_PER_FAMILY)
    if metrics.get("greedy") != aggregate_array_rows(greedy_rows):
        raise ValueError("Diagnostic new Array greedy aggregate does not rebuild from retained rows.")
    if metrics.get("template_item_counts") != array_item_template_counts(greedy_rows):
        raise ValueError("Diagnostic new Array template_item_counts do not rebuild from retained rows.")
    reused = reused_feasibility_cell_binding(input_root, "array_json", "small", seed)
    if metrics.get("step1500_checkpoint_reused_for_equality_gate") != reused["checkpoint_path"]:
        raise ValueError("Diagnostic new Array step-1500 equality checkpoint path mismatch.")
    if metrics.get("step1500_checkpoint_sha256") != reused["checkpoint_sha256"]:
        raise ValueError("Diagnostic new Array step-1500 equality checkpoint checksum mismatch.")
    validate_diagnostic_training_trajectory_evidence(
        checkpoint_path,
        seed=seed,
        training_metrics=training,
        frozen_step1500_checkpoint=Path(str(reused["checkpoint_path"])),
    )
    replay_array_artifacts(
        checkpoint_path,
        model_size="small",
        comparison_source="diagnostic_small_3000",
        steps=DIAGNOSTIC_STEPS,
        seed=seed,
        retained_greedy_rows=greedy_rows,
        retained_train_rows=train_rows,
        retained_eval_rows=eval_rows,
        retained_train_aggregate=metrics.get("teacher_forced_train"),
        retained_eval_aggregate=metrics.get("teacher_forced_eval"),
    )


def validate_diagnostic_completed_scope_semantics(root: Path, completed_scope: object, input_root: Path) -> None:
    if not isinstance(completed_scope, list) or not all(isinstance(row, dict) for row in completed_scope):
        raise ValueError("Diagnostic completed_scope must be valid before semantic artifact checks.")
    matrix: dict[str, list[dict[str, object]]] | None = None
    for row in completed_scope:
        name = row["name"]
        if name == "named_matrix_construction":
            matrix = diagnostic_named_matrix_by_cell(root)
        elif name == "array_training_plan":
            validate_diagnostic_array_plan_scope_semantics(root)
        elif name == "named_cell":
            if matrix is None:
                matrix = diagnostic_named_matrix_by_cell(root)
            validate_diagnostic_named_scope_semantics(root, row, input_root, matrix)
        elif name == "array_baseline_reuse":
            validate_diagnostic_reused_array_scope_semantics(root, row, input_root)
        elif name == "array_diagnostic_training":
            validate_diagnostic_new_array_scope_semantics(root, row, input_root)


def validate_diagnostic_done_artifacts(root: Path) -> None:
    manifest_data = json.loads((root / "manifest.json").read_text())
    input_root = validate_diagnostic_input_binding(manifest_data.get("input_root"))
    completed_scope = manifest_data.get("completed_scope")
    validate_diagnostic_completed_scope(completed_scope, terminal_status="DONE")
    validate_diagnostic_completed_scope_semantics(root, completed_scope, input_root)


def validate_diagnostic_common_semantics(
    root: Path,
    *,
    terminal_status: str,
    terminal_data: dict[str, object],
    manifest_data: dict[str, object],
    summary_data: dict[str, object],
) -> None:
    for name, data in (("terminal", terminal_data), ("manifest", manifest_data), ("summary", summary_data)):
        if data.get("protocol") != "phase8_feasibility_failure_diagnostic":
            raise ValueError(f"Diagnostic {name} protocol mismatch.")
    source_commit = validate_git_sha(manifest_data.get("source_commit"), "source_commit")
    validate_git_commit_exists(source_commit)
    validate_diagnostic_source_provenance_fields(manifest_data.get("source_provenance"), source_commit)
    if manifest_data.get("configuration") != frozen_diagnostic_configuration():
        raise ValueError("Diagnostic configuration does not match the frozen diagnostic protocol.")
    if manifest_data.get("record_hashes") != validate_diagnostic_record_hashes():
        raise ValueError("Diagnostic record_hashes mismatch.")
    if manifest_data.get("core_blobs") != validate_diagnostic_core_blobs(source_commit):
        raise ValueError("Diagnostic core_blobs mismatch.")
    input_root = validate_diagnostic_input_binding(manifest_data.get("input_root"))
    validate_diagnostic_environment_schema(manifest_data.get("environment"))
    validate_diagnostic_environment(manifest_data.get("environment"), input_root)
    validate_diagnostic_flags(manifest_data.get("deterministic_flags"))
    validate_diagnostic_handoff(manifest_data.get("handoff"), source_commit=source_commit)
    declared_root = Path(require_canonical_path_string(manifest_data.get("output_root"), "output_root", DIAGNOSTIC_ROOT_RE))
    require_diagnostic_artifact_location(declared_root, "output_root")
    allowed_validation_roots = {
        declared_root.resolve(),
        declared_root.with_name(declared_root.name + ".tmp").resolve(),
    }
    if root.resolve() not in allowed_validation_roots:
        raise ValueError("Diagnostic output_root does not bind the validated root.")
    expected_command = diagnostic_exact_command(
        declared_root,
        tuple(Path(require_exact_str(binding.get("path"), "diagnostic_lineage.path")) for binding in manifest_data.get("diagnostic_lineage", [])),
        input_root=input_root,
    )
    if manifest_data.get("exact_command") != expected_command:
        raise ValueError("Diagnostic exact_command does not match the authorized command.")
    lineage = manifest_data.get("diagnostic_lineage")
    if not isinstance(lineage, list) or not all(isinstance(row, dict) for row in lineage):
        raise ValueError("Diagnostic lineage must be a JSON object list.")
    validate_diagnostic_repair_transition(
        manifest_data.get("repair_transition"),
        lineage=lineage,
        source_commit=source_commit,
    )
    if manifest_data.get("terminal_status") != terminal_status or summary_data.get("terminal_status") != terminal_status:
        raise ValueError("Diagnostic manifest/summary terminal_status mismatch.")
    wall_time_seconds = manifest_data.get("wall_time_seconds")
    if type(wall_time_seconds) not in {int, float} or not math.isfinite(float(wall_time_seconds)) or wall_time_seconds < 0:
        raise ValueError("Diagnostic wall_time_seconds must be finite and non-negative.")
    expected_summary = {
        "done": terminal_status == "DONE",
        "completed_scope_count": len(manifest_data.get("completed_scope", [])),
        "partial_scope_count": len(manifest_data.get("partial_scope", [])),
    }
    if summary_data.get("summary") != expected_summary:
        raise ValueError("Diagnostic summary counts do not match retained scope records.")
    if summary_data.get("failure") != manifest_data.get("failure"):
        raise ValueError("Diagnostic summary and manifest disagree on failure.")
    if terminal_status == "DONE":
        if manifest_data.get("failure") is not None or manifest_data.get("failure_classification") is not None:
            raise ValueError("DONE diagnostic must not record a failure or failure classification.")
        if manifest_data.get("partial_scope") != []:
            raise ValueError("DONE diagnostic partial_scope must be exactly empty.")
    else:
        if type(manifest_data.get("failure")) is not str or not manifest_data.get("failure"):
            raise ValueError("FAILED diagnostic must record a non-empty error string.")
        if manifest_data.get("failure_classification") not in {"transient_infrastructure", "diagnostic_implementation_defect"}:
            raise ValueError("FAILED diagnostic must bind an allowed failure classification.")
        partial_scope = manifest_data.get("partial_scope")
        if not isinstance(partial_scope, list) or not partial_scope or not all(isinstance(row, dict) for row in partial_scope):
            raise ValueError("FAILED diagnostic must retain at least one partial scope object.")
        if partial_scope[-1].get("error") != manifest_data.get("failure"):
            raise ValueError("FAILED diagnostic final partial scope must retain the exact failure error.")
    validate_diagnostic_completed_scope(manifest_data.get("completed_scope"), terminal_status=terminal_status)
    validate_diagnostic_partial_scope(
        manifest_data.get("partial_scope"),
        completed_scope=manifest_data.get("completed_scope"),
        terminal_status=terminal_status,
        failure=manifest_data.get("failure"),
    )
    validate_diagnostic_artifact_roles(root, manifest_data.get("file_inventory"))
    validate_diagnostic_completed_scope_artifacts(root, manifest_data.get("completed_scope"))
    validate_diagnostic_completed_scope_semantics(root, manifest_data.get("completed_scope"), input_root)


def load_diagnostic_terminal_binding(root: Path) -> tuple[Path, dict[str, object], Path, str]:
    done = root / "DONE.json"
    failed = root / "FAILED.json"
    existing = [path for path in (done, failed) if path.exists()]
    if len(existing) > 1:
        raise ValueError("Diagnostic root has both DONE.json and FAILED.json.")
    if not existing:
        raise ValueError("Diagnostic root lacks DONE.json or FAILED.json.")
    terminal = existing[0]
    manifest = root / "manifest.json"
    if not manifest.exists():
        raise ValueError("Diagnostic root lacks manifest.json.")
    manifest_sha = file_sha256(manifest)
    terminal_data = json.loads(terminal.read_text())
    manifest_data = json.loads(manifest.read_text())
    summary = root / "summary.json"
    if not summary.is_file():
        raise ValueError("Diagnostic root lacks summary.json.")
    summary_data = json.loads(summary.read_text())
    for data_name, data in (("terminal", terminal_data), ("manifest", manifest_data)):
        if data.get("artifact_class") != DIAGNOSTIC_ARTIFACT_CLASS:
            raise ValueError(f"Diagnostic {data_name} artifact_class mismatch.")
        if data.get("feasibility_selection_eligible") is not False:
            raise ValueError(f"Diagnostic {data_name} must be non-selection.")
        if data.get("task_010d_authorized") is not False:
            raise ValueError(f"Diagnostic {data_name} must not authorize Task 010D.")
    for data_name, data in (("summary", summary_data),):
        if data.get("artifact_class") != DIAGNOSTIC_ARTIFACT_CLASS:
            raise ValueError(f"Diagnostic {data_name} artifact_class mismatch.")
        if data.get("feasibility_selection_eligible") is not False:
            raise ValueError(f"Diagnostic {data_name} must be non-selection.")
        if data.get("task_010d_authorized") is not False:
            raise ValueError(f"Diagnostic {data_name} must not authorize Task 010D.")
    if terminal_data.get("status") != terminal.stem or manifest_data.get("terminal_status") != terminal.stem:
        raise ValueError("Diagnostic terminal status mismatch.")
    if terminal_data.get("manifest_path") != "manifest.json":
        raise ValueError("Diagnostic terminal manifest_path must be manifest.json.")
    if terminal_data.get("manifest_sha256") != manifest_sha:
        raise ValueError("Diagnostic terminal does not bind manifest checksum.")
    if terminal_data.get("diagnostic_lineage") != manifest_data.get("diagnostic_lineage"):
        raise ValueError("Diagnostic terminal and manifest lineage mismatch.")
    if terminal.stem == "DONE" and terminal_data.get("failure_classification") is not None:
        raise ValueError("DONE diagnostic terminal must not carry a failure classification.")
    if terminal.stem == "FAILED" and terminal_data.get("failure_classification") not in {"transient_infrastructure", "diagnostic_implementation_defect"}:
        raise ValueError("FAILED diagnostic terminal must bind an allowed failure classification.")
    if terminal.stem == "FAILED" and terminal_data.get("error") != manifest_data.get("failure"):
        raise ValueError("FAILED diagnostic terminal error does not match the manifest failure.")
    if terminal.stem == "DONE" and "error" in terminal_data:
        raise ValueError("DONE diagnostic terminal must not contain an error.")
    expected_inventory = diagnostic_inventory(root)
    file_inventory = manifest_data.get("file_inventory")
    if file_inventory != expected_inventory:
        raise ValueError("Diagnostic manifest file_inventory does not exactly match current root inventory.")
    if terminal_data.get("file_inventory") != file_inventory:
        raise ValueError("Diagnostic terminal does not bind the exact manifest inventory.")
    shared_fields = (
        "source_commit",
        "source_provenance",
        "protocol",
        "exact_command",
        "output_root",
        "wall_time_seconds",
        "deterministic_flags",
        "handoff",
        "input_root",
        "configuration",
        "environment",
        "record_hashes",
        "core_blobs",
        "diagnostic_lineage",
        "repair_transition",
        "completed_scope",
        "partial_scope",
        "failure_classification",
    )
    for record_name, record in (("summary", summary_data), ("manifest", manifest_data), ("terminal", terminal_data)):
        missing = [field_name for field_name in shared_fields if field_name not in record]
        if missing:
            raise ValueError(f"Diagnostic {record_name} is missing required provenance fields: {missing!r}.")
    for field_name in shared_fields:
        if summary_data.get(field_name) != manifest_data.get(field_name):
            raise ValueError(f"Diagnostic summary and manifest disagree on {field_name}.")
        if terminal_data.get(field_name) != manifest_data.get(field_name):
            raise ValueError(f"Diagnostic terminal and manifest disagree on {field_name}.")
    validate_diagnostic_common_semantics(
        root,
        terminal_status=terminal.stem,
        terminal_data=terminal_data,
        manifest_data=manifest_data,
        summary_data=summary_data,
    )
    resolved_root = root.resolve()
    seen_paths: set[str] = set()
    for index, row in enumerate(file_inventory):
        if not isinstance(row, dict):
            raise ValueError("Diagnostic file_inventory entries must be JSON objects.")
        rel_path = require_canonical_relative_path(row.get("path"), f"file_inventory[{index}].path")
        if rel_path in seen_paths:
            raise ValueError("Diagnostic file_inventory must not contain duplicate paths.")
        seen_paths.add(rel_path)
        raw_candidate = root / rel_path
        if raw_candidate.is_symlink():
            raise ValueError("Diagnostic file_inventory must not bind symlink files.")
        candidate = raw_candidate.resolve()
        if resolved_root not in candidate.parents:
            raise ValueError("Diagnostic file_inventory path escapes its root.")
        if not candidate.is_file():
            raise ValueError("Diagnostic file_inventory path does not exist.")
        if file_sha256(candidate) != row.get("sha256") or candidate.stat().st_size != row.get("bytes"):
            raise ValueError("Diagnostic file_inventory checksum or byte count mismatch.")
        require_exact_str(row.get("role"), f"file_inventory[{index}].role")
    lineage = terminal_data.get("diagnostic_lineage")
    if not isinstance(lineage, list):
        raise ValueError("Diagnostic lineage must be a JSON list.")
    lineage_paths: list[str] = []
    for index, binding in enumerate(lineage):
        if not isinstance(binding, dict):
            raise ValueError("Diagnostic lineage entries must be JSON objects.")
        lineage_path = require_exact_str(binding.get("path"), f"diagnostic_lineage[{index}].path")
        if Path(lineage_path).resolve() == root.resolve():
            raise ValueError("Diagnostic lineage contains a cycle.")
        lineage_paths.append(lineage_path)
        require_exact_str(binding.get("terminal_sha256"), f"diagnostic_lineage[{index}].terminal_sha256")
        require_exact_str(binding.get("manifest_sha256"), f"diagnostic_lineage[{index}].manifest_sha256")
    if len(lineage_paths) != len(set(lineage_paths)):
        raise ValueError("Diagnostic lineage contains duplicate roots.")
    return terminal, terminal_data, manifest, manifest_sha


def flattened_diagnostic_lineage(predecessor_diagnostic_roots: Sequence[Path]) -> list[dict[str, object]]:
    lineage: list[dict[str, object]] = []
    for predecessor in predecessor_diagnostic_roots:
        binding = diagnostic_terminal_binding(predecessor)
        expected_prior = lineage.copy()
        if binding.get("diagnostic_lineage") != expected_prior:
            raise ValueError("Diagnostic predecessor lineage is broken or forged.")
        binding_without_nested = {key: value for key, value in binding.items() if key != "diagnostic_lineage"}
        lineage.append(binding_without_nested)
    paths = [str(binding["path"]) for binding in lineage]
    if len(paths) != len(set(paths)):
        raise ValueError("Diagnostic lineage contains a cycle.")
    return lineage


def validate_diagnostic_repair_diff_confined(superseded_source_commit: str, repaired_source_commit: str) -> None:
    names = [
        line for line in git_output(["git", "diff", "--name-only", f"{superseded_source_commit}..{repaired_source_commit}"]).splitlines()
        if line
    ]
    if not names:
        raise ValueError("Implementation-defect diagnostic repair must bind a non-empty source diff.")
    unexpected = sorted(set(names) - DIAGNOSTIC_REPAIR_ALLOWED_DIFF_PATHS)
    if unexpected:
        raise ValueError(f"Implementation-defect diagnostic repair diff touches forbidden paths: {unexpected!r}.")


def diagnostic_repair_transition(
    predecessor_bindings: Sequence[dict[str, object]],
    *,
    repaired_source_commit: str,
) -> dict[str, object] | None:
    if not predecessor_bindings:
        return None
    immediate = predecessor_bindings[-1]
    if immediate.get("failure_classification") != "diagnostic_implementation_defect":
        return None
    superseded_source_commit = validate_git_sha(immediate.get("source_commit"), "repair_transition.superseded_source_commit")
    repaired_source_commit = validate_git_sha(repaired_source_commit, "repair_transition.repaired_source_commit")
    if superseded_source_commit == repaired_source_commit:
        raise ValueError("Implementation-defect diagnostic repair transition requires a new source commit.")
    validate_diagnostic_repair_diff_confined(superseded_source_commit, repaired_source_commit)
    return {
        "superseded_source_commit": superseded_source_commit,
        "repaired_source_commit": repaired_source_commit,
        "allowed_diff_paths": sorted(DIAGNOSTIC_REPAIR_ALLOWED_DIFF_PATHS),
    }


def validate_diagnostic_repair_transition(
    value: object,
    *,
    lineage: Sequence[dict[str, object]],
    source_commit: str,
) -> None:
    if not lineage:
        if value is not None:
            raise ValueError("Initial diagnostic run must use repair_transition=null.")
        return
    immediate = lineage[-1]
    if immediate.get("failure_classification") != "diagnostic_implementation_defect":
        if value is not None:
            raise ValueError("Transient diagnostic retry must use repair_transition=null.")
        return
    if not isinstance(value, dict) or set(value) != {"superseded_source_commit", "repaired_source_commit", "allowed_diff_paths"}:
        raise ValueError("Implementation-defect diagnostic retry must bind the exact repair_transition schema.")
    superseded = validate_git_sha(value.get("superseded_source_commit"), "repair_transition.superseded_source_commit")
    repaired = validate_git_sha(value.get("repaired_source_commit"), "repair_transition.repaired_source_commit")
    if superseded != immediate.get("source_commit"):
        raise ValueError("repair_transition superseded_source_commit does not match the immediate predecessor.")
    if repaired != source_commit:
        raise ValueError("repair_transition repaired_source_commit does not match the current source.")
    if value.get("allowed_diff_paths") != sorted(DIAGNOSTIC_REPAIR_ALLOWED_DIFF_PATHS):
        raise ValueError("repair_transition allowed_diff_paths mismatch.")
    validate_diagnostic_repair_diff_confined(superseded, repaired)


def diagnostic_preflight_bindings(input_root: Path, predecessor_diagnostic_roots: Sequence[Path]) -> dict[str, object]:
    input_manifest = json.loads((input_root / "manifest.json").read_text())
    input_binding = {
        relative_path: file_sha256(input_root / relative_path)
        for relative_path in sorted(DIAGNOSTIC_INPUT_CHECKSUMS)
    }
    predecessor_bindings = [diagnostic_terminal_binding(root) for root in predecessor_diagnostic_roots]
    lineage = [{key: value for key, value in binding.items() if key != "diagnostic_lineage"} for binding in predecessor_bindings]
    source_commit = current_source_commit()
    return {
        "input_root": str(input_root),
        "input": input_binding,
        "input_root_binding": {
            "path": str(input_root),
            "manifest_sha256": input_binding["manifest.json"],
            "summary_sha256": input_binding["summary.json"],
            "terminal_sha256": input_binding["FAILED.json"],
            "source_commit": input_manifest.get("source_commit"),
            "predecessor_roots": input_manifest.get("predecessor_roots"),
        },
        "predecessors": predecessor_bindings,
        "diagnostic_lineage": lineage,
        "repair_transition": diagnostic_repair_transition(predecessor_bindings, repaired_source_commit=source_commit),
        "handoff": handoff_binding_for_commit(source_commit),
        "environment": current_environment_dict(),
        "record_hashes": validate_diagnostic_record_hashes(),
        "core_blobs": validate_diagnostic_core_blobs(),
    }


def verify_diagnostic_preflight_bindings(bindings: dict[str, object], input_root: Path, predecessor_diagnostic_roots: Sequence[Path]) -> None:
    validate_diagnostic_input_root(input_root)
    if bindings != diagnostic_preflight_bindings(input_root, predecessor_diagnostic_roots):
        raise ValueError("Diagnostic input or predecessor binding changed after preflight.")


def validate_new_diagnostic_root(
    input_root: Path,
    output_root: Path,
    predecessor_diagnostic_roots: Sequence[Path] = (),
) -> None:
    validate_diagnostic_input_root(input_root)
    require_diagnostic_artifact_location(output_root, "output_root")
    output_number = diagnostic_root_number(output_root)
    predecessor_numbers: list[int] = []
    lineage: list[dict[str, object]] = []
    current_commit = current_source_commit()
    frozen_configuration_value = frozen_diagnostic_configuration()
    predecessor_bindings: list[dict[str, object]] = []
    for predecessor in predecessor_diagnostic_roots:
        require_diagnostic_artifact_location(predecessor, "predecessor_diagnostic_root")
        number = diagnostic_root_number(predecessor)
        if predecessor_numbers and number <= predecessor_numbers[-1]:
            raise ValueError("Predecessor diagnostic roots must be in strictly ascending order.")
        terminal, _terminal_data, _manifest, _manifest_sha = load_diagnostic_terminal_binding(predecessor)
        if terminal.stem == "DONE":
            raise ValueError("A prior DONE diagnostic root forbids later diagnostic roots.")
        predecessor_numbers.append(number)
        binding = diagnostic_terminal_binding(predecessor)
        if binding.get("diagnostic_lineage") != lineage:
            raise ValueError("Diagnostic predecessor lineage is broken or forged.")
        if binding.get("configuration") != frozen_configuration_value:
            raise ValueError("Diagnostic predecessor configuration mismatch.")
        predecessor_source = require_exact_str(binding.get("source_commit"), "predecessor.source_commit")
        if GIT_SHA_RE.fullmatch(predecessor_source) is None:
            raise ValueError("Diagnostic predecessor source_commit must be a full lowercase Git SHA.")
        predecessor_bindings.append(binding)
        lineage.append({key: value for key, value in binding.items() if key != "diagnostic_lineage"})
    if predecessor_bindings:
        immediate = predecessor_bindings[-1]
        failure_classification = immediate.get("failure_classification")
        if failure_classification == "transient_infrastructure" and immediate.get("source_commit") != current_commit:
            raise ValueError("Transient diagnostic retry must use the identical accepted source.")
        if failure_classification == "diagnostic_implementation_defect" and immediate.get("source_commit") == current_commit:
            raise ValueError("Implementation-defect diagnostic retry requires a newly reviewed repair source.")
    if predecessor_numbers != list(range(1, output_number)):
        raise ValueError("Diagnostic predecessor roots must be complete and continuous before the output root.")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic root: {output_root}")
    temp_root = output_root.with_name(output_root.name + ".tmp")
    if temp_root.exists() or temp_root.is_symlink():
        raise FileExistsError(f"Temporary diagnostic root already exists: {temp_root}")


def validate_diagnostic_cli_authorization_contract(
    *,
    device: str,
    input_root: Path,
    output_root: Path,
    predecessor_diagnostic_roots: Sequence[Path],
    environ: dict[str, str] | None = None,
) -> None:
    if device != DIAGNOSTIC_REQUIRED_DEVICE:
        raise ValueError("diagnose-failure only supports device string cuda:0.")
    if any(path.is_absolute() for path in (input_root, output_root, *predecessor_diagnostic_roots)):
        raise ValueError("diagnose-failure authorized command requires canonical repository-relative artifact paths.")
    expected_argv = [
        "python",
        "scripts/phase8_sequence_feasibility.py",
        "diagnose-failure",
        "--device",
        DIAGNOSTIC_REQUIRED_DEVICE,
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
    ]
    for predecessor in predecessor_diagnostic_roots:
        expected_argv.extend(["--predecessor-diagnostic-root", str(predecessor)])
    validate_diagnostic_process_argv(diagnostic_kernel_argv(), expected_argv)
    actual_env = os.environ if environ is None else environ
    for key, expected_value in DIAGNOSTIC_REQUIRED_ENV.items():
        if actual_env.get(key) != expected_value:
            raise ValueError(f"diagnose-failure environment {key} must exactly equal {expected_value!r}.")


def validate_diagnostic_cli_root_contract(
    *,
    input_root: Path,
    output_root: Path,
    predecessor_diagnostic_roots: Sequence[Path],
) -> None:
    if output_root.name == "feasibility_diagnostic_001" and predecessor_diagnostic_roots:
        raise ValueError("Initial diagnostic command must not include predecessor roots.")
    validate_new_diagnostic_root(input_root, output_root, predecessor_diagnostic_roots)


def validate_diagnostic_cli_contract(
    *,
    device: str,
    input_root: Path,
    output_root: Path,
    predecessor_diagnostic_roots: Sequence[Path],
    environ: dict[str, str] | None = None,
) -> None:
    validate_diagnostic_cli_authorization_contract(
        device=device,
        input_root=input_root,
        output_root=output_root,
        predecessor_diagnostic_roots=predecessor_diagnostic_roots,
        environ=environ,
    )
    configure_diagnostic_deterministic_backend()
    validate_diagnostic_cli_root_contract(
        input_root=input_root,
        output_root=output_root,
        predecessor_diagnostic_roots=predecessor_diagnostic_roots,
    )


def diagnostic_kernel_argv() -> list[str]:
    cmdline = Path("/proc/self/cmdline")
    try:
        raw = cmdline.read_bytes()
    except OSError as exc:
        raise ValueError("diagnose-failure cannot authorize without Linux /proc/self/cmdline.") from exc
    if not raw or not raw.endswith(b"\0"):
        raise ValueError("diagnose-failure kernel argv is unavailable or malformed.")
    parts = raw[:-1].split(b"\0")
    if not parts or any(part == b"" for part in parts):
        raise ValueError("diagnose-failure kernel argv is malformed.")
    try:
        return [os.fsdecode(part) for part in parts]
    except UnicodeDecodeError as exc:
        raise ValueError("diagnose-failure kernel argv is not decodable.") from exc


def validate_diagnostic_process_argv(raw_argv: Sequence[str], expected_argv: Sequence[str]) -> None:
    if list(raw_argv) != list(expected_argv):
        raise ValueError(f"diagnose-failure process argv must exactly match the authorized kernel command: {list(expected_argv)!r}.")
    if any(arg in {"-O", "-OO", "-B"} or arg.startswith("-X") for arg in raw_argv):
        raise ValueError("diagnose-failure forbids Python optimization, bytecode, or implementation flags.")


def require_diagnostic_real_main_context() -> None:
    if __name__ != "__main__":
        raise ValueError("diagnose-failure must execute from this file's real __main__ process context.")


def require_feasibility_real_main_context() -> None:
    if __name__ != "__main__":
        raise ValueError("run must execute from this file's real __main__ process context.")


def diagnostic_exact_command(
    output_root: Path,
    predecessor_diagnostic_roots: Sequence[Path],
    *,
    input_root: Path,
) -> list[str]:
    argv = [
        "python",
        "scripts/phase8_sequence_feasibility.py",
        "diagnose-failure",
        "--device",
        DIAGNOSTIC_REQUIRED_DEVICE,
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
    ]
    for predecessor in predecessor_diagnostic_roots:
        argv.extend(["--predecessor-diagnostic-root", str(predecessor)])
    return [
        "PYTHONDONTWRITEBYTECODE=1",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "PYTHONPATH=.",
        *argv,
    ]


def feasibility_exact_argv(
    root: Path,
    predecessor_roots: Sequence[Path],
    decision_diagnostic_root: Path,
) -> list[str]:
    argv = [
        "python",
        "scripts/phase8_sequence_feasibility.py",
        "run",
        "--device",
        FEASIBILITY_REQUIRED_DEVICE,
        "--root",
        str(root),
    ]
    for predecessor in predecessor_roots:
        argv.extend(["--predecessor-root", str(predecessor)])
    argv.extend(["--decision-diagnostic-root", str(decision_diagnostic_root)])
    return argv


def feasibility_exact_command(
    root: Path,
    predecessor_roots: Sequence[Path],
    decision_diagnostic_root: Path,
) -> list[str]:
    return [
        "PYTHONDONTWRITEBYTECODE=1",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "PYTHONPATH=.",
        *feasibility_exact_argv(root, predecessor_roots, decision_diagnostic_root),
    ]


def validate_feasibility_process_argv(raw_argv: Sequence[str], expected_argv: Sequence[str]) -> None:
    if list(raw_argv) != list(expected_argv):
        raise ValueError(f"run process argv must exactly match the authorized command: {list(expected_argv)!r}.")
    if any(arg in {"-O", "-OO", "-B"} or arg.startswith("-X") for arg in raw_argv):
        raise ValueError("run forbids Python optimization, bytecode, or implementation flags.")


def validate_feasibility_cli_contract(
    *,
    device: str,
    root: Path,
    predecessor_roots: Sequence[Path],
    predecessor_selections: Sequence[Path],
    decision_diagnostic_root: Path,
) -> None:
    if device != FEASIBILITY_REQUIRED_DEVICE:
        raise ValueError("run only supports device string cuda:0.")
    if predecessor_selections:
        raise ValueError("run forbids predecessor selections for feasibility_005.")
    if str(root) != FEASIBILITY_REQUIRED_ROOT:
        raise ValueError("run root must be exactly artifacts/phase8_toy_lm_bridge/feasibility_005.")
    if tuple(str(path) for path in predecessor_roots) != FEASIBILITY_REQUIRED_PREDECESSOR_ROOTS:
        raise ValueError("run predecessor roots must be exactly feasibility_001..004 in numerical order.")
    if str(decision_diagnostic_root) != FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT:
        raise ValueError("run decision diagnostic root must be exactly feasibility_diagnostic_001.")
    validate_feasibility_process_argv(
        diagnostic_kernel_argv(),
        feasibility_exact_argv(root, predecessor_roots, decision_diagnostic_root),
    )
    validate_feasibility_environment()


def reused_feasibility_cell_binding(input_root: Path, family: str, model_size: str, seed: int) -> dict[str, object]:
    manifest_path = input_root / "manifest.json"
    manifest_environment = json.loads(manifest_path.read_text()).get("environment") if manifest_path.is_file() else None
    manifest = validate_diagnostic_input_root(input_root, environment=manifest_environment)
    cells = manifest.get("cells")
    if not isinstance(cells, list):
        raise ValueError("feasibility_004 manifest cells must be a list.")
    matches = [
        cell for cell in cells
        if cell.get("family") == family and cell.get("model_size") == model_size and cell.get("seed") == seed
    ]
    if len(matches) != 1:
        raise ValueError("Frozen feasibility_004 reused cell binding is missing or duplicated.")
    cell = dict(matches[0])
    generation_path = input_root / require_canonical_relative_path(cell["generations_path"], "generations_path")
    checkpoint_path = input_root / require_canonical_relative_path(cell["checkpoint_path"], "checkpoint_path")
    cell["generations_sha256"] = file_sha256(generation_path)
    cell["checkpoint_sha256"] = file_sha256(checkpoint_path)
    cell["generations_path"] = str(generation_path)
    cell["checkpoint_path"] = str(checkpoint_path)
    return cell


def diagnostic_array_training_plan() -> list[dict[str, object]]:
    return [
        {"family": "array_json", "model_size": "small", "steps": DIAGNOSTIC_STEPS, "seed": seed}
        for seed in SEEDS
    ]


def validate_diagnostic_array_training_plan(plan: Sequence[dict[str, object]]) -> None:
    expected = diagnostic_array_training_plan()
    if list(plan) != expected:
        raise ValueError("Diagnostic must train exactly three array_json small/3000 runs with seeds 0,1,2.")


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def update_canonical_digest(digest: object, value: object) -> None:
    digest.update(canonical_json_bytes(value))  # type: ignore[attr-defined]
    digest.update(b"\n")  # type: ignore[attr-defined]


def require_sha256_hex(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest.")
    return value


def model_state_schema_from_state_dict(state: object) -> dict[str, dict[str, object]]:
    if not isinstance(state, dict):
        raise ValueError("Model state fingerprint requires a state_dict mapping.")
    schema: dict[str, dict[str, object]] = {}
    for key in sorted(state):
        if not isinstance(key, str):
            raise ValueError("Model state fingerprint keys must be strings.")
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Model state fingerprint values must be tensors.")
        schema[key] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    return schema


def model_state_fingerprint_from_state_dict(state: object) -> str:
    if not isinstance(state, dict):
        raise ValueError("Model state fingerprint requires a state_dict mapping.")
    digest = sha256()
    update_canonical_digest(
        digest,
        {
            "serialization": DIAGNOSTIC_MODEL_STATE_FINGERPRINT_SCHEMA,
            "tensor_count": len(state),
        },
    )
    for key in sorted(state):
        if not isinstance(key, str):
            raise ValueError("Model state fingerprint keys must be strings.")
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Model state fingerprint values must be tensors.")
        raw = tensor_bytes(tensor)
        update_canonical_digest(
            digest,
            {
                "key": key,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "byte_count": len(raw),
            },
        )
        digest.update(raw)
        digest.update(b"\n")
    return digest.hexdigest()


def model_state_fingerprint(model: torch.nn.Module) -> str:
    return model_state_fingerprint_from_state_dict(model.state_dict())


def checkpoint_model_state_fingerprint(checkpoint_path: Path) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint model-state fingerprint requires a checkpoint mapping.")
    return model_state_fingerprint_from_state_dict(checkpoint.get("model_state_dict"))


def snapshot_rng_states() -> tuple[object, torch.Tensor, tuple[torch.Tensor, ...] | None]:
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return python_state, torch_state.clone(), cuda_states


def restore_rng_states(states: tuple[object, torch.Tensor, tuple[torch.Tensor, ...] | None]) -> None:
    python_state, torch_state, cuda_states = states
    random.setstate(python_state)
    torch.random.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(list(cuda_states))


def isolated_model_state_schema(model_size: str) -> dict[str, dict[str, object]]:
    rng_states = snapshot_rng_states()
    try:
        return model_state_schema_from_state_dict(build_historical_model(model_size).state_dict())
    finally:
        restore_rng_states(rng_states)


def expected_initial_model_state_fingerprint(seed: int) -> str:
    rng_states = snapshot_rng_states()
    try:
        model_rng_seed = MODEL_RNG_OFFSET + seed
        random.seed(model_rng_seed)
        torch.manual_seed(model_rng_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_rng_seed)
        return model_state_fingerprint(build_historical_model("small"))
    finally:
        restore_rng_states(rng_states)


def optimizer_state_fingerprint(optimizer: torch.optim.Optimizer) -> str:
    payload = json.dumps(_optimizer_state_payload(optimizer.state_dict()), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def _optimizer_state_payload(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": sha256(tensor_bytes(value)).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): _optimizer_state_payload(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_optimizer_state_payload(nested) for nested in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def diagnostic_training_batches(seed: int, record_count: int) -> tuple[tuple[int, ...], ...]:
    batches = deterministic_batch_indices(
        record_count=record_count,
        seed=seed,
        state_mask=0,
        batch_size=BATCH_SIZE,
        steps=DIAGNOSTIC_STEPS,
    )
    if batches[:TRAINING_STEPS] != deterministic_batch_indices(
        record_count=record_count,
        seed=seed,
        state_mask=0,
        batch_size=BATCH_SIZE,
        steps=TRAINING_STEPS,
    ):
        raise ValueError("Diagnostic 3000-step batch stream does not extend the frozen 1500-step stream.")
    return batches


def diagnostic_batch_schedule_digest(
    batches: Sequence[Sequence[int]],
    *,
    seed: int,
    record_count: int,
) -> str:
    digest = sha256()
    update_canonical_digest(
        digest,
        {
            "serialization": DIAGNOSTIC_BATCH_SCHEDULE_SCHEMA,
            "seed": seed,
            "state_mask": 0,
            "record_count": record_count,
            "batch_size": BATCH_SIZE,
            "step_count": DIAGNOSTIC_STEPS,
            "batch_count": len(batches),
        },
    )
    for step_index, batch in enumerate(batches, start=1):
        batch_indices = [require_exact_int(index, "diagnostic.batch_index") for index in batch]
        if len(batch_indices) != BATCH_SIZE:
            raise ValueError("Diagnostic schedule batches must retain exact batch size.")
        if any(index < 0 or index >= record_count for index in batch_indices):
            raise ValueError("Diagnostic schedule batch index out of range.")
        update_canonical_digest(
            digest,
            {
                "step": step_index,
                "batch_indices": batch_indices,
            },
        )
    return digest.hexdigest()


def diagnostic_training_schedule_evidence(seed: int, record_count: int) -> dict[str, object]:
    batches = diagnostic_training_batches(seed, record_count)
    return {
        "serialization": DIAGNOSTIC_BATCH_SCHEDULE_SCHEMA,
        "seed": seed,
        "state_mask": 0,
        "record_count": record_count,
        "batch_size": BATCH_SIZE,
        "step_count": DIAGNOSTIC_STEPS,
        "batch_count": len(batches),
        "sha256": diagnostic_batch_schedule_digest(batches, seed=seed, record_count=record_count),
    }


def new_diagnostic_training_trace_digest(seed: int, schedule: dict[str, object]) -> object:
    digest = sha256()
    update_canonical_digest(
        digest,
        {
            "serialization": DIAGNOSTIC_TRAINING_TRACE_SCHEMA,
            "seed": seed,
            "step_count": DIAGNOSTIC_STEPS,
            "schedule_sha256": schedule["sha256"],
        },
    )
    return digest


def loss_scalar_evidence(loss_tensor: torch.Tensor, loss_value: float) -> dict[str, object]:
    if loss_tensor.numel() != 1:
        raise ValueError("Diagnostic training trace loss tensor must be scalar.")
    raw = tensor_bytes(loss_tensor)
    return {
        "value": loss_value,
        "repr": repr(loss_value),
        "tensor_dtype": str(loss_tensor.dtype),
        "tensor_shape": list(loss_tensor.shape),
        "tensor_bytes_hex": raw.hex(),
    }


def update_diagnostic_training_trace(
    digest: object,
    *,
    step_index: int,
    batch: Sequence[int],
    loss_tensor: torch.Tensor,
    loss_value: float,
) -> None:
    batch_indices = [require_exact_int(index, "diagnostic.trace_batch_index") for index in batch]
    update_canonical_digest(
        digest,
        {
            "step": step_index,
            "batch_indices": batch_indices,
            "loss": loss_scalar_evidence(loss_tensor, loss_value),
        },
    )


def compare_model_to_checkpoint_step1500(model: torch.nn.Module, checkpoint_path: Path, model_size: str) -> None:
    validate_historical_checkpoint_allowlist(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("config") != legacy_serialized_config(model_size):
        raise ValueError("Step-1500 equality gate config mismatch.")
    if checkpoint.get("parameter_count") != HISTORICAL_PARAMETER_COUNTS[model_size]:
        raise ValueError("Step-1500 equality gate parameter_count mismatch.")
    expected_state = checkpoint.get("model_state_dict")
    current_state = model.state_dict()
    if not isinstance(expected_state, dict) or set(expected_state) != set(current_state):
        raise ValueError("Step-1500 equality gate state_dict keys mismatch.")
    for key in sorted(current_state):
        expected_tensor = expected_state[key]
        current_tensor = current_state[key].detach().cpu()
        if not isinstance(expected_tensor, torch.Tensor):
            raise ValueError("Step-1500 equality gate checkpoint state contains a non-tensor.")
        if expected_tensor.dtype != current_tensor.dtype:
            raise ValueError(f"Step-1500 equality gate dtype mismatch for {key}.")
        if tuple(expected_tensor.shape) != tuple(current_tensor.shape):
            raise ValueError(f"Step-1500 equality gate shape mismatch for {key}.")
        if tensor_bytes(expected_tensor) != tensor_bytes(current_tensor):
            raise ValueError(f"Step-1500 equality gate tensor bytes mismatch for {key}.")


def run_diagnostic_step_loop_with_gate(
    batches: Sequence[Sequence[int]],
    *,
    gate_step: int,
    step_callback: Callable[[int, Sequence[int]], object],
    gate_callback: Callable[[], None],
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    if gate_step <= 0:
        raise ValueError("Diagnostic gate_step must be positive.")
    final_result: object = None
    step_after_gate_executed = False
    for step_index, batch in enumerate(batches, start=1):
        if step_index == gate_step + 1:
            step_after_gate_executed = True
        if progress_callback is not None:
            progress_callback({"subphase": "array_training", "completed_steps": step_index - 1})
        final_result = step_callback(step_index, batch)
        if progress_callback is not None:
            progress_callback({"subphase": "array_training", "completed_steps": step_index})
        if step_index == gate_step:
            gate_callback()
    return {"final_result": final_result, "step_after_gate_executed": step_after_gate_executed}


def save_historical_checkpoint(path: str, model: torch.nn.Module, *, metadata: dict[str, object]) -> None:
    model_size = require_exact_str(metadata.get("model_size"), "metadata.model_size")
    if model_size not in MODEL_SIZES:
        raise ValueError("Historical checkpoint model_size must be small or medium.")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": legacy_serialized_config(model_size),
            "parameter_count": HISTORICAL_PARAMETER_COUNTS[model_size],
            "metadata": metadata,
        },
        path,
    )


def train_array_small_3000_diagnostic(
    *,
    seed: int,
    train_records: Sequence[FeasibilityRecord],
    tokenizer: ByteTokenizer,
    device: torch.device,
    frozen_step1500_checkpoint: Path,
    checkpoint_path: Path,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    step1500_hook: Callable[[torch.nn.Module], None] | None = None,
) -> dict[str, object]:
    if device.type != "cuda" or device.index != 0:
        raise ValueError("Array diagnostic training must run on cuda:0.")
    set_deterministic_backend(seed)
    model = build_historical_model("small")
    model.to(device)
    model.train()
    optimizer = make_optimizer(model)
    initial_model_state_fingerprint = model_state_fingerprint(model)
    batches = diagnostic_training_batches(seed, len(train_records))
    schedule_evidence = diagnostic_training_schedule_evidence(seed, len(train_records))
    trace_digest = new_diagnostic_training_trace_digest(seed, schedule_evidence)
    final_loss = float("nan")
    final_loss_evidence: dict[str, object] | None = None
    step1500_evidence: dict[str, object] | None = None
    def train_step(_step_index: int, batch: Sequence[int]) -> float:
        nonlocal final_loss, final_loss_evidence
        batch_records = [TextRecord(train_records[index].prompt, train_records[index].answer) for index in batch]
        input_ids = encode_record_batch(batch_records, tokenizer).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = response_only_loss(logits, input_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        loss_cpu = loss.detach().cpu()
        final_loss = float(loss_cpu)
        final_loss_evidence = loss_scalar_evidence(loss_cpu, final_loss)
        update_diagnostic_training_trace(
            trace_digest,
            step_index=_step_index,
            batch=batch,
            loss_tensor=loss_cpu,
            loss_value=final_loss,
        )
        return final_loss

    def gate_step1500() -> None:
        nonlocal step1500_evidence
        rng_state = random.getstate()
        torch_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(device)
        live_state_fingerprint = model_state_fingerprint(model)
        checkpoint_state_fingerprint = checkpoint_model_state_fingerprint(frozen_step1500_checkpoint)
        optimizer_state = optimizer_state_fingerprint(optimizer)
        compare_model_to_checkpoint_step1500(model, frozen_step1500_checkpoint, "small")
        step1500_evidence = {
            "step": TRAINING_STEPS,
            "model_state_fingerprint": live_state_fingerprint,
            "checkpoint_state_fingerprint": checkpoint_state_fingerprint,
            "model_matches_checkpoint": True,
            "optimizer_state_fingerprint": optimizer_state,
        }
        if step1500_hook is not None:
            step1500_hook(model)
        if random.getstate() != rng_state:
            raise ValueError("Step-1500 equality snapshot changed Python RNG state.")
        if not torch.equal(torch.random.get_rng_state(), torch_rng_state):
            raise ValueError("Step-1500 equality snapshot changed Torch RNG state.")
        if not torch.equal(torch.cuda.get_rng_state(device), cuda_rng_state):
            raise ValueError("Step-1500 equality snapshot changed CUDA RNG state.")
        if optimizer_state_fingerprint(optimizer) != optimizer_state:
            raise ValueError("Step-1500 equality snapshot changed optimizer state.")
        if model_state_fingerprint(model) != live_state_fingerprint:
            raise ValueError("Step-1500 equality snapshot changed model state.")

    loop_result = run_diagnostic_step_loop_with_gate(
        batches,
        gate_step=TRAINING_STEPS,
        step_callback=train_step,
        gate_callback=gate_step1500,
        progress_callback=progress_callback,
    )
    step1501_executed = loop_result["step_after_gate_executed"] is True
    if step1500_evidence is None:
        raise ValueError("Diagnostic 3000-step training did not bind step-1500 trajectory evidence.")
    if final_loss_evidence is None:
        raise ValueError("Diagnostic 3000-step training did not bind final loss evidence.")
    trajectory_evidence = {
        "serialization": DIAGNOSTIC_TRAJECTORY_EVIDENCE_SCHEMA,
        "claimed_seed": seed,
        "family": "array_json",
        "model_size": "small",
        "training_steps": DIAGNOSTIC_STEPS,
        "authorized_new_training_run_count": len(SEEDS),
        "schedule": schedule_evidence,
        "initial_model_state_fingerprint": initial_model_state_fingerprint,
        "step1500": step1500_evidence,
        "trace": {
            "serialization": DIAGNOSTIC_TRAINING_TRACE_SCHEMA,
            "seed": seed,
            "step_count": DIAGNOSTIC_STEPS,
            "schedule_sha256": schedule_evidence["sha256"],
            "sha256": trace_digest.hexdigest(),  # type: ignore[attr-defined]
            "final_loss": final_loss_evidence,
        },
        "step_count": DIAGNOSTIC_STEPS,
        "step1501_executed": step1501_executed,
        "final_model_state_fingerprint": model_state_fingerprint(model),
        "final_optimizer_state_fingerprint": optimizer_state_fingerprint(optimizer),
        "retained_final_loss": final_loss,
    }
    save_historical_checkpoint(
        str(checkpoint_path),
        model,
        metadata={
            "artifact_class": DIAGNOSTIC_ARTIFACT_CLASS,
            "feasibility_selection_eligible": False,
            "task_010d_authorized": False,
            "family": "array_json",
            "model_size": "small",
            "seed": seed,
            "training_steps": DIAGNOSTIC_STEPS,
            "training_loss": final_loss,
            "step1501_executed": step1501_executed,
            "training_trajectory_evidence": trajectory_evidence,
        },
    )
    if not step1501_executed:
        raise ValueError("Diagnostic 3000-step training did not prove execution of step 1501.")
    return {
        "final_loss": final_loss,
        "checkpoint_path": str(checkpoint_path),
        "step1501_executed": True,
        "trajectory_evidence": trajectory_evidence,
    }


def diagnostic_inventory_role(rel_path: str) -> str:
    role = "summary" if rel_path == "summary.json" else "diagnostic_artifact"
    if rel_path.endswith(".pt"):
        role = "checkpoint"
    elif rel_path.endswith(".jsonl"):
        role = "retained_rows"
    return role


def diagnostic_inventory(root: Path) -> list[dict[str, object]]:
    rows = []
    for row in inventory(root):
        rows.append({**row, "role": diagnostic_inventory_role(str(row["path"]))})
    return rows


def expected_done_diagnostic_paths() -> set[str]:
    paths = {"named_diagnostic_matrix.jsonl", "array_training_plan.json", "summary.json", "manifest.json", "DONE.json"}
    for seed in SEEDS:
        for cell in NAMED_DIAGNOSTIC_CELLS:
            prefix = f"named_value_json__medium__seed{seed}__{cell}"
            paths.add(f"{prefix}/generations.jsonl")
            paths.add(f"{prefix}/metrics.json")
    for model_size in MODEL_SIZES:
        for seed in SEEDS:
            prefix = f"array_json__{model_size}__1500__seed{seed}__reused"
            paths.add(f"{prefix}/greedy_metrics_rows.jsonl")
            paths.add(f"{prefix}/teacher_forced_train_rows.jsonl")
            paths.add(f"{prefix}/teacher_forced_eval_rows.jsonl")
            paths.add(f"{prefix}/metrics.json")
    for seed in SEEDS:
        prefix = f"array_json__small__3000__seed{seed}"
        paths.add(f"{prefix}/checkpoint_step3000.pt")
        paths.add(f"{prefix}/generations.jsonl")
        paths.add(f"{prefix}/teacher_forced_train_rows.jsonl")
        paths.add(f"{prefix}/teacher_forced_eval_rows.jsonl")
        paths.add(f"{prefix}/metrics.json")
    return paths


def validate_diagnostic_terminal_root(root: Path, *, terminal_status: str) -> None:
    terminals = [path.name for path in (root / "DONE.json", root / "FAILED.json") if path.exists()]
    if terminals != [f"{terminal_status}.json"]:
        raise ValueError("Diagnostic root must contain exactly one terminal marker matching the requested status.")
    manifest = root / "manifest.json"
    summary = root / "summary.json"
    if not manifest.is_file() or not summary.is_file():
        raise ValueError("Diagnostic root must contain manifest.json and summary.json before publish.")
    manifest_data = json.loads(manifest.read_text())
    summary_data = json.loads(summary.read_text())
    terminal_data = json.loads((root / f"{terminal_status}.json").read_text())
    for name, data in (("manifest", manifest_data), ("summary", summary_data), ("terminal", terminal_data)):
        if data.get("artifact_class") != DIAGNOSTIC_ARTIFACT_CLASS:
            raise ValueError(f"Diagnostic {name} missing non-evidence artifact_class.")
        if data.get("feasibility_selection_eligible") is not False:
            raise ValueError(f"Diagnostic {name} must be ineligible for feasibility selection.")
        if data.get("task_010d_authorized") is not False:
            raise ValueError(f"Diagnostic {name} must not authorize Task 010D.")
    expected_inventory = diagnostic_inventory(root)
    if manifest_data.get("file_inventory") != expected_inventory:
        raise ValueError("Diagnostic manifest file_inventory does not exactly match the published root.")
    if terminal_data.get("status") != terminal_status:
        raise ValueError("Diagnostic terminal status does not match its filename.")
    if terminal_data.get("manifest_path") != "manifest.json":
        raise ValueError("Diagnostic terminal manifest_path must be manifest.json.")
    if terminal_data.get("manifest_sha256") != file_sha256(manifest):
        raise ValueError("Diagnostic terminal does not bind the exact manifest checksum.")
    if terminal_data.get("file_inventory") != expected_inventory:
        raise ValueError("Diagnostic terminal does not bind the exact manifest inventory.")
    if terminal_status == "FAILED" and terminal_data.get("error") != manifest_data.get("failure"):
        raise ValueError("FAILED diagnostic terminal error does not match the manifest failure.")
    if terminal_status == "DONE" and "error" in terminal_data:
        raise ValueError("DONE diagnostic terminal must not contain an error.")
    shared_fields = (
        "source_commit",
        "source_provenance",
        "protocol",
        "exact_command",
        "output_root",
        "wall_time_seconds",
        "deterministic_flags",
        "handoff",
        "input_root",
        "diagnostic_lineage",
        "configuration",
        "record_hashes",
        "core_blobs",
        "environment",
        "repair_transition",
        "completed_scope",
        "partial_scope",
        "failure_classification",
    )
    for record_name, record in (("summary", summary_data), ("manifest", manifest_data), ("terminal", terminal_data)):
        missing = [field_name for field_name in shared_fields if field_name not in record]
        if missing:
            raise ValueError(f"Diagnostic {record_name} is missing required provenance fields: {missing!r}.")
    for field_name in shared_fields:
        if summary_data.get(field_name) != manifest_data.get(field_name):
            raise ValueError(f"Diagnostic summary and manifest disagree on {field_name}.")
        if terminal_data.get(field_name) != manifest_data.get(field_name):
            raise ValueError(f"Diagnostic terminal and manifest disagree on {field_name}.")
    validate_diagnostic_common_semantics(
        root,
        terminal_status=terminal_status,
        terminal_data=terminal_data,
        manifest_data=manifest_data,
        summary_data=summary_data,
    )
    if terminal_status == "DONE":
        actual_paths = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
        expected_paths = expected_done_diagnostic_paths()
        if actual_paths != expected_paths:
            missing = sorted(expected_paths - actual_paths)
            extra = sorted(actual_paths - expected_paths)
            raise ValueError(f"DONE diagnostic root has incomplete or extra files: missing={missing!r}, extra={extra!r}.")
        completed = manifest_data.get("completed_scope")
        if not isinstance(completed, list):
            raise ValueError("DONE diagnostic manifest completed_scope must be a list.")
        if sum(1 for row in completed if isinstance(row, dict) and row.get("name") == "named_cell") != 12:
            raise ValueError("DONE diagnostic must complete exactly 12 Named cells.")
        if sum(1 for row in completed if isinstance(row, dict) and row.get("name") == "array_diagnostic_training") != 3:
            raise ValueError("DONE diagnostic must complete exactly three new Array runs.")
        if sum(1 for row in completed if isinstance(row, dict) and row.get("name") == "array_baseline_reuse") != 6:
            raise ValueError("DONE diagnostic must bind exactly six reused Array baselines.")


def validate_diagnostic_terminal_inventory_snapshot(root: Path, *, terminal_status: str) -> None:
    terminals = [path.name for path in (root / "DONE.json", root / "FAILED.json") if path.exists()]
    if terminals != [f"{terminal_status}.json"]:
        raise ValueError("Diagnostic root must contain exactly one terminal marker matching the requested status.")
    manifest = root / "manifest.json"
    terminal = root / f"{terminal_status}.json"
    if not manifest.is_file() or not terminal.is_file():
        raise ValueError("Diagnostic root lacks terminal inventory files.")
    manifest_data = json.loads(manifest.read_text())
    terminal_data = json.loads(terminal.read_text())
    expected_inventory = diagnostic_inventory(root)
    if manifest_data.get("file_inventory") != expected_inventory:
        raise ValueError("Diagnostic final manifest inventory does not exactly match current root files.")
    if terminal_data.get("file_inventory") != expected_inventory:
        raise ValueError("Diagnostic final terminal inventory does not exactly match current root files.")
    if terminal_data.get("manifest_sha256") != file_sha256(manifest):
        raise ValueError("Diagnostic final terminal manifest checksum mismatch.")
    for row in expected_inventory:
        path = root / require_canonical_relative_path(row.get("path"), "diagnostic.final_inventory.path")
        if row.get("sha256") != file_sha256(path) or row.get("bytes") != path.stat().st_size:
            raise ValueError("Diagnostic final inventory checksum or byte count mismatch.")


def publish_diagnostic_root(
    temp_root: Path,
    output_root: Path,
    *,
    terminal_status: str,
    final_callback: Callable[[], None] | None = None,
) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic root: {output_root}")
    validate_diagnostic_terminal_root(temp_root, terminal_status=terminal_status)
    if final_callback is not None:
        final_callback()
    validate_diagnostic_terminal_inventory_snapshot(temp_root, terminal_status=terminal_status)
    if final_callback is not None:
        final_callback()
    atomic_rename_noreplace(temp_root, output_root)


def publish_diagnostic_root_or_leave_incomplete(
    temp_root: Path,
    output_root: Path,
    *,
    terminal_status: str,
    final_callback: Callable[[], None] | None = None,
) -> None:
    try:
        publish_diagnostic_root(temp_root, output_root, terminal_status=terminal_status, final_callback=final_callback)
    except Exception as exc:
        remove_diagnostic_terminal_markers(temp_root)
        raise DiagnosticPublicationError(
            f"Diagnostic publication failed without overwriting {output_root}; the temporary root is incomplete."
        ) from exc


def validate_current_feasibility_terminal_root(
    temp_root: Path,
    output_root: Path,
    *,
    terminal_status: str,
    predecessor_roots: Sequence[Path],
    predecessor_selections: Sequence[Path],
    decision_diagnostic_root: Path,
    source_snapshot: SourceSnapshot,
    allowed_source_paths: set[Path],
) -> None:
    terminals = [path.name for path in (temp_root / "DONE.json", temp_root / "FAILED.json") if path.exists()]
    if terminals != [f"{terminal_status}.json"]:
        raise ValueError("Current feasibility temp root must contain exactly one terminal marker matching the requested status.")
    manifest = temp_root / "manifest.json"
    summary = temp_root / "summary.json"
    terminal = temp_root / f"{terminal_status}.json"
    if not manifest.is_file() or not summary.is_file() or not terminal.is_file():
        raise ValueError("Current feasibility temp root must contain manifest.json, summary.json, and its terminal marker before publish.")
    manifest_data = json.loads(manifest.read_text())
    summary_data = json.loads(summary.read_text())
    terminal_data = json.loads(terminal.read_text())
    expected_terminal_keys = set(CURRENT_TERMINAL_KEYS)
    if terminal_status == "FAILED" and manifest_data.get("failure") is not None:
        expected_terminal_keys.add("error")
    if set(manifest_data) != CURRENT_MANIFEST_KEYS:
        raise ValueError("Current manifest must use the exact tied feasibility schema.")
    if set(summary_data) != CURRENT_SUMMARY_KEYS:
        raise ValueError("Current summary must use the exact tied feasibility schema.")
    if set(terminal_data) != expected_terminal_keys:
        raise ValueError("Current terminal must use the exact tied feasibility schema.")
    if terminal_data.get("status") != terminal_status:
        raise ValueError("Current feasibility terminal status does not match its filename.")
    if manifest_data.get("terminal_status") != terminal_status or summary_data.get("terminal_status") != terminal_status:
        raise ValueError("Current feasibility manifest/summary terminal_status mismatch.")
    if terminal_data.get("manifest_path") != "manifest.json":
        raise ValueError("Current feasibility terminal manifest_path must be manifest.json.")
    if terminal_data.get("manifest_sha256") != file_sha256(manifest):
        raise ValueError("Current feasibility terminal does not bind the exact manifest checksum.")
    if terminal_status == "DONE":
        if manifest_data.get("failure") is not None or summary_data.get("failure") is not None or "error" in terminal_data:
            raise ValueError("DONE current feasibility root must not record a failure.")
    elif terminal_data.get("error") != manifest_data.get("failure"):
        raise ValueError("FAILED current feasibility terminal error does not match the manifest failure.")
    if summary_data.get("failure") != manifest_data.get("failure"):
        raise ValueError("Current feasibility summary and manifest disagree on failure.")

    cells = manifest_data.get("cells")
    if terminal_data.get("cells") != cells:
        raise ValueError("Current feasibility terminal cells do not match manifest cells.")
    if summary_data.get("cells") != cells:
        raise ValueError("Current feasibility summary cells do not match manifest cells.")
    complete_matrix = isinstance(cells, list) and len(cells) == len(current_feasibility_cell_identities())
    require_all_pass = terminal_status == "DONE"
    validate_current_feasibility_cells(
        cells,
        require_complete=terminal_status == "DONE" or complete_matrix,
        require_all_pass=require_all_pass,
    )
    if terminal_status == "DONE" and not complete_matrix:
        raise ValueError("DONE current feasibility root must contain the complete 24-cell matrix.")
    if terminal_status == "FAILED" and not complete_matrix and manifest_data.get("failure") is None:
        raise ValueError("Partial FAILED current feasibility root must record an operational failure.")

    validate_summary_aggregates(summary_data, cells, terminal_status, manifest_data.get("failure"))
    expected_configuration = frozen_configuration()
    expected_command = feasibility_exact_command(output_root, predecessor_roots, decision_diagnostic_root)
    for record_name, record in (("manifest", manifest_data), ("summary", summary_data), ("terminal", terminal_data)):
        if record.get("configuration") != expected_configuration:
            raise ValueError(f"Current {record_name} configuration mismatch.")
        if record.get("decision_diagnostic") != DECISION_DIAGNOSTIC_ROOT_BINDING:
            raise ValueError(f"Current {record_name} decision_diagnostic mismatch.")
        if record.get("record_hashes") != FEASIBILITY_RECORD_HASHES:
            raise ValueError(f"Current {record_name} record_hashes mismatch.")
        if record.get("exact_command") != expected_command:
            raise ValueError(f"Current {record_name} exact_command mismatch.")
        validate_current_deterministic_flags(record.get("deterministic_flags"))
    if manifest_data.get("environment") != FEASIBILITY_REQUIRED_RUNTIME_ENV:
        raise ValueError("Current manifest environment does not match the frozen cuda:0 A800 environment.")
    if manifest_data.get("source_commit") != source_snapshot.commit:
        raise ValueError("Current manifest source_commit does not match the clean-source snapshot.")
    validate_source_provenance(manifest_data.get("source_provenance"), source_snapshot.commit, allowed_source_paths)
    summary_source = summary_data.get("source")
    if not isinstance(summary_source, dict) or set(summary_source) != {"commit", "script", "ignored_inputs"}:
        raise ValueError("Current summary source must use the exact schema.")
    if summary_source.get("commit") != source_snapshot.commit:
        raise ValueError("Current summary source commit does not match the clean-source snapshot.")
    if summary_source.get("script") != "scripts/phase8_sequence_feasibility.py":
        raise ValueError("Current summary source script mismatch.")
    if summary_source.get("ignored_inputs") != list(source_snapshot.ignored_inputs):
        raise ValueError("Current summary source ignored_inputs do not match the clean-source snapshot.")
    if manifest_data.get("predecessor_selections") != [selection_binding(path) for path in predecessor_selections]:
        raise ValueError("Current manifest predecessor_selections mismatch.")
    expected_predecessor_roots = complete_predecessor_root_bindings(predecessor_roots, predecessor_selections)
    if manifest_data.get("predecessor_roots") != expected_predecessor_roots:
        raise ValueError("Current manifest predecessor_roots do not bind the complete predecessor lineage.")

    inventory_paths = validate_root_file_inventory(temp_root, manifest_data)
    expected_files = {"summary.json"}
    for cell in cells:
        generation_path = require_canonical_relative_path(cell["generations_path"], "generations_path")
        checkpoint_path = require_canonical_relative_path(cell["checkpoint_path"], "checkpoint_path")
        expected_files.add(generation_path)
        expected_files.add(checkpoint_path)
    if inventory_paths != expected_files:
        raise ValueError("Current manifest file_inventory contains files outside the completed-cell artifact set.")
    actual_files = {str(path.relative_to(temp_root)) for path in temp_root.rglob("*") if path.is_file()}
    expected_root_files = {*expected_files, "manifest.json", f"{terminal_status}.json"}
    if actual_files != expected_root_files:
        missing = sorted(expected_root_files - actual_files)
        extra = sorted(actual_files - expected_root_files)
        raise ValueError(f"Current feasibility temp root has incomplete or extra files: missing={missing!r}, extra={extra!r}.")
    for cell in cells:
        generation_rows = validate_generation_artifact(temp_root / str(cell["generations_path"]), cell)
        validate_checkpoint_artifact(temp_root / str(cell["checkpoint_path"]), cell)
        validate_checkpoint_replays_generations(temp_root / str(cell["checkpoint_path"]), cell, generation_rows)


def validate_current_feasibility_terminal_inventory_snapshot(temp_root: Path, *, terminal_status: str) -> None:
    terminals = [path.name for path in (temp_root / "DONE.json", temp_root / "FAILED.json") if path.exists()]
    if terminals != [f"{terminal_status}.json"]:
        raise ValueError("Current feasibility temp root must contain exactly one terminal marker matching the requested status.")
    manifest = temp_root / "manifest.json"
    terminal = temp_root / f"{terminal_status}.json"
    if not manifest.is_file() or not terminal.is_file():
        raise ValueError("Current feasibility temp root lacks terminal inventory files.")
    manifest_data = json.loads(manifest.read_text())
    terminal_data = json.loads(terminal.read_text())
    expected_inventory = inventory(temp_root)
    if manifest_data.get("file_inventory") != expected_inventory:
        raise ValueError("Current feasibility final manifest inventory does not exactly match current root files.")
    if terminal_data.get("manifest_sha256") != file_sha256(manifest):
        raise ValueError("Current feasibility final terminal manifest checksum mismatch.")
    for row in expected_inventory:
        path = temp_root / require_canonical_relative_path(row.get("path"), "current_feasibility.final_inventory.path")
        if row.get("sha256") != file_sha256(path) or row.get("bytes") != path.stat().st_size:
            raise ValueError("Current feasibility final inventory checksum or byte count mismatch.")


def publish_current_feasibility_root(
    temp_root: Path,
    output_root: Path,
    *,
    terminal_status: str,
    predecessor_roots: Sequence[Path],
    predecessor_selections: Sequence[Path],
    decision_diagnostic_root: Path,
    source_snapshot: SourceSnapshot,
    allowed_source_paths: set[Path],
    final_callback: Callable[[], None] | None = None,
) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing feasibility root: {output_root}")
    validate_current_feasibility_terminal_root(
        temp_root,
        output_root,
        terminal_status=terminal_status,
        predecessor_roots=predecessor_roots,
        predecessor_selections=predecessor_selections,
        decision_diagnostic_root=decision_diagnostic_root,
        source_snapshot=source_snapshot,
        allowed_source_paths=allowed_source_paths,
    )
    if final_callback is not None:
        final_callback()
    validate_current_feasibility_terminal_inventory_snapshot(temp_root, terminal_status=terminal_status)
    if final_callback is not None:
        final_callback()
    atomic_rename_noreplace(temp_root, output_root)


def publish_current_feasibility_root_or_leave_incomplete(
    temp_root: Path,
    output_root: Path,
    *,
    terminal_status: str,
    predecessor_roots: Sequence[Path],
    predecessor_selections: Sequence[Path],
    decision_diagnostic_root: Path,
    source_snapshot: SourceSnapshot,
    allowed_source_paths: set[Path],
    final_callback: Callable[[], None] | None = None,
) -> None:
    try:
        publish_current_feasibility_root(
            temp_root,
            output_root,
            terminal_status=terminal_status,
            predecessor_roots=predecessor_roots,
            predecessor_selections=predecessor_selections,
            decision_diagnostic_root=decision_diagnostic_root,
            source_snapshot=source_snapshot,
            allowed_source_paths=allowed_source_paths,
            final_callback=final_callback,
        )
    except Exception as exc:
        remove_diagnostic_terminal_markers(temp_root)
        raise FeasibilityPublicationError(
            f"Current feasibility publication failed without overwriting {output_root}; the temporary root is incomplete."
        ) from exc


def remove_diagnostic_terminal_markers(root: Path) -> None:
    try:
        root_metadata = root.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        return
    for name in ("DONE.json", "FAILED.json"):
        marker = root / name
        try:
            marker_metadata = marker.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(marker_metadata.st_mode):
            marker.unlink()


def atomic_rename_noreplace(source: Path, destination: Path) -> None:
    atomic_rename_noreplace_at(-100, source, -100, destination, destination)


def atomic_rename_noreplace_at(
    source_dir_fd: int,
    source_name: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    destination_dir_fd: int,
    destination_name: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    destination_label: object,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic no-clobber diagnostic publication requires Linux renameat2.")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        source_dir_fd if source_dir_fd is not None else at_fdcwd,
        os.fsencode(source_name),
        destination_dir_fd if destination_dir_fd is not None else at_fdcwd,
        os.fsencode(destination_name),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, "Refusing to overwrite existing diagnostic root", destination_label)
    raise OSError(error_number, os.strerror(error_number), destination_label)


def write_diagnostic_terminal(
    root: Path,
    terminal_status: str,
    *,
    input_root: Path,
    output_root: Path,
    predecessor_diagnostic_roots: Sequence[Path],
    completed_scope: Sequence[dict[str, object]],
    partial_scope: Sequence[dict[str, object]] = (),
    failure: str | None = None,
    failure_classification: str | None = None,
    source_snapshot: SourceSnapshot,
    preflight_bindings: dict[str, object],
    wall_time_seconds: float,
) -> None:
    if terminal_status not in {"DONE", "FAILED"}:
        raise ValueError("Diagnostic terminal_status must be DONE or FAILED.")
    if terminal_status == "FAILED" and failure_classification not in {"transient_infrastructure", "diagnostic_implementation_defect"}:
        raise ValueError("FAILED diagnostics require a frozen failure classification.")
    if not math.isfinite(wall_time_seconds) or wall_time_seconds < 0:
        raise ValueError("Diagnostic wall_time_seconds must be finite and non-negative.")
    snapshot = source_snapshot
    source_commit = snapshot.commit
    if preflight_bindings.get("input_root") != str(input_root):
        raise ValueError("Diagnostic terminal preflight input root mismatch.")
    expected_predecessors = [str(path) for path in predecessor_diagnostic_roots]
    observed_predecessors = [binding.get("path") for binding in preflight_bindings.get("predecessors", [])]
    if observed_predecessors != expected_predecessors:
        raise ValueError("Diagnostic terminal preflight predecessor roots mismatch.")
    lineage = preflight_bindings.get("diagnostic_lineage")
    if not isinstance(lineage, list):
        raise ValueError("Diagnostic terminal preflight lineage must be a JSON list.")
    common = {
        "artifact_class": DIAGNOSTIC_ARTIFACT_CLASS,
        "feasibility_selection_eligible": False,
        "task_010d_authorized": False,
        "protocol": "phase8_feasibility_failure_diagnostic",
        "terminal_status": terminal_status,
        "failure": failure,
        "failure_classification": failure_classification,
        "source_commit": source_commit,
        "source_provenance": asdict(snapshot),
        "exact_command": diagnostic_exact_command(
            output_root,
            predecessor_diagnostic_roots,
            input_root=input_root,
        ),
        "output_root": str(output_root),
        "wall_time_seconds": wall_time_seconds,
        "deterministic_flags": {
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        },
        "handoff": preflight_bindings.get("handoff"),
        "input_root": preflight_bindings.get("input_root_binding"),
        "diagnostic_lineage": lineage,
        "repair_transition": preflight_bindings.get("repair_transition"),
        "configuration": frozen_diagnostic_configuration(),
        "record_hashes": preflight_bindings.get("record_hashes"),
        "core_blobs": preflight_bindings.get("core_blobs"),
        "environment": preflight_bindings.get("environment"),
        "completed_scope": list(completed_scope),
        "partial_scope": list(partial_scope),
    }
    summary = {
        **common,
        "summary": {
            "done": terminal_status == "DONE",
            "completed_scope_count": len(completed_scope),
            "partial_scope_count": len(partial_scope),
        },
    }
    write_json(root / "summary.json", summary)
    manifest = {**common, "file_inventory": diagnostic_inventory(root)}
    write_json(root / "manifest.json", manifest)
    manifest_sha = file_sha256(root / "manifest.json")
    terminal = {
        "artifact_class": DIAGNOSTIC_ARTIFACT_CLASS,
        "feasibility_selection_eligible": False,
        "task_010d_authorized": False,
        "status": terminal_status,
        "manifest_path": "manifest.json",
        "manifest_sha256": manifest_sha,
        "source_commit": source_commit,
        "source_provenance": asdict(snapshot),
        "protocol": common["protocol"],
        "exact_command": common["exact_command"],
        "output_root": common["output_root"],
        "wall_time_seconds": wall_time_seconds,
        "deterministic_flags": common["deterministic_flags"],
        "handoff": common["handoff"],
        "input_root": common["input_root"],
        "configuration": common["configuration"],
        "environment": common["environment"],
        "record_hashes": common["record_hashes"],
        "core_blobs": common["core_blobs"],
        "diagnostic_lineage": lineage,
        "repair_transition": common["repair_transition"],
        "completed_scope": list(completed_scope),
        "partial_scope": list(partial_scope),
        "failure_classification": failure_classification,
        "file_inventory": manifest["file_inventory"],
    }
    if failure is not None:
        terminal["error"] = failure
        terminal["failure_classification"] = failure_classification
    write_json(root / f"{terminal_status}.json", terminal)


def classify_diagnostic_failure(error: Exception) -> str:
    if isinstance(error, (MemoryError, OSError, subprocess.SubprocessError, torch.OutOfMemoryError)):
        return "transient_infrastructure"
    return "diagnostic_implementation_defect"


def run_diagnostic_failure(
    *,
    device: str,
    input_root: Path,
    output_root: Path,
    predecessor_diagnostic_roots: Sequence[Path] = (),
    environ: dict[str, str] | None = None,
) -> None:
    require_diagnostic_real_main_context()
    validate_diagnostic_cli_contract(
        device=device,
        input_root=input_root,
        output_root=output_root,
        predecessor_diagnostic_roots=predecessor_diagnostic_roots,
        environ=environ,
    )
    source_snapshot = capture_diagnostic_source_provenance(input_root, output_root, predecessor_diagnostic_roots)
    preflight_bindings = diagnostic_preflight_bindings(input_root, predecessor_diagnostic_roots)
    start_time = time.monotonic()
    temp_root = output_root.with_name(output_root.name + ".tmp")
    temp_root.mkdir(parents=True)
    completed_scope: list[dict[str, object]] = []
    partial_scope: list[dict[str, object]] = []
    active_scope: dict[str, object] | None = {"name": "diagnostic_initialization"}
    def update_active_progress(progress: dict[str, object]) -> None:
        nonlocal active_scope
        identity_keys = {"name", "seed", "cell", "model_size", "steps"}
        base = {
            key: value
            for key, value in (active_scope or {"name": "diagnose_failure"}).items()
            if key in identity_keys
        }
        active_scope = {**base, **progress}
    def final_publication_callback() -> None:
        verify_diagnostic_preflight_bindings(preflight_bindings, input_root, predecessor_diagnostic_roots)
        verify_source_unchanged(source_snapshot, active_output_root=temp_root)
    try:
        validate_diagnostic_input_root(input_root)
        tokenizer = ByteTokenizer()
        target_device = torch.device(device)
        records = grouped_records()
        active_scope = {"name": "named_matrix_construction"}
        named_matrix = build_named_diagnostic_matrix()
        write_jsonl(
            temp_root / "named_diagnostic_matrix.jsonl",
            (row for cell in NAMED_DIAGNOSTIC_CELLS for row in named_matrix[cell]),
        )
        completed_scope.append({"name": "named_matrix_construction", "rows": 4 * EVAL_RECORDS_PER_FAMILY})
        active_scope = None
        for seed in SEEDS:
            active_scope = {"name": "named_checkpoint_reuse", "seed": seed}
            reused_named = reused_feasibility_cell_binding(input_root, "named_value_json", "medium", seed)
            model = load_checkpoint_model(Path(str(reused_named["checkpoint_path"])), "medium", target_device)
            for cell in NAMED_DIAGNOSTIC_CELLS:
                active_scope = {"name": "named_cell", "seed": seed, "cell": cell}
                cell_dir = temp_root / f"named_value_json__medium__seed{seed}__{cell}"
                cell_dir.mkdir()
                named_rows = generate_named_diagnostic_rows(
                    model,
                    named_matrix[cell],
                    target_device,
                    progress_callback=update_active_progress,
                )
                write_jsonl(cell_dir / "generations.jsonl", named_rows)
                write_json(cell_dir / "metrics.json", aggregate_named_rows(named_rows))
                completed_scope.append(
                    {
                        "name": "named_cell",
                        "seed": seed,
                        "cell": cell,
                        "rows": len(named_rows),
                        "checkpoint_reused_from": reused_named["checkpoint_path"],
                        "checkpoint_sha256": reused_named["checkpoint_sha256"],
                    }
                )
                active_scope = None
        active_scope = {"name": "array_training_plan"}
        array_plan = diagnostic_array_training_plan()
        validate_diagnostic_array_training_plan(array_plan)
        write_json(temp_root / "array_training_plan.json", array_plan)
        completed_scope.append({"name": "array_training_plan", "runs": len(array_plan)})
        active_scope = None
        for model_size in MODEL_SIZES:
            for seed in SEEDS:
                active_scope = {
                    "name": "array_baseline_reuse",
                    "model_size": model_size,
                    "steps": TRAINING_STEPS,
                    "seed": seed,
                }
                reused_array = reused_feasibility_cell_binding(input_root, "array_json", model_size, seed)
                model = load_checkpoint_model(Path(str(reused_array["checkpoint_path"])), model_size, target_device)
                baseline_dir = temp_root / f"array_json__{model_size}__1500__seed{seed}__reused"
                baseline_dir.mkdir()
                array_rows = array_rows_from_reused_generations(
                    Path(str(reused_array["generations_path"])),
                    comparison_source="feasibility_004_reused",
                    model_size=model_size,
                    steps=TRAINING_STEPS,
                    seed=seed,
                )
                write_jsonl(baseline_dir / "greedy_metrics_rows.jsonl", array_rows)
                train_tf_rows, train_tf_aggregate = teacher_forced_rows(
                    model,
                    records["array_json"]["train"],
                    tokenizer,
                    target_device,
                    split="train",
                    progress_callback=update_active_progress,
                )
                eval_tf_rows, eval_tf_aggregate = teacher_forced_rows(
                    model,
                    records["array_json"]["eval"],
                    tokenizer,
                    target_device,
                    split="eval",
                    progress_callback=update_active_progress,
                )
                write_jsonl(baseline_dir / "teacher_forced_train_rows.jsonl", train_tf_rows)
                write_jsonl(baseline_dir / "teacher_forced_eval_rows.jsonl", eval_tf_rows)
                write_json(
                    baseline_dir / "metrics.json",
                    {
                        "greedy": aggregate_array_rows(array_rows),
                        "teacher_forced_train": train_tf_aggregate,
                        "teacher_forced_eval": eval_tf_aggregate,
                        "template_item_counts": array_item_template_counts(array_rows),
                        "reused_generations_path": reused_array["generations_path"],
                        "reused_generations_sha256": reused_array["generations_sha256"],
                        "reused_checkpoint_path": reused_array["checkpoint_path"],
                        "reused_checkpoint_sha256": reused_array["checkpoint_sha256"],
                    },
                )
                completed_scope.append(
                    {
                        "name": "array_baseline_reuse",
                        "model_size": model_size,
                        "steps": TRAINING_STEPS,
                        "seed": seed,
                        "rows": len(array_rows),
                    }
                )
                active_scope = None
        for row in array_plan:
            seed = require_exact_int(row["seed"], "seed")
            active_scope = {
                "name": "array_diagnostic_training",
                "model_size": "small",
                "steps": DIAGNOSTIC_STEPS,
                "seed": seed,
            }
            reused_small = reused_feasibility_cell_binding(input_root, "array_json", "small", seed)
            run_dir = temp_root / f"array_json__small__3000__seed{seed}"
            run_dir.mkdir()
            train_result = train_array_small_3000_diagnostic(
                seed=seed,
                train_records=records["array_json"]["train"],
                tokenizer=tokenizer,
                device=target_device,
                frozen_step1500_checkpoint=Path(str(reused_small["checkpoint_path"])),
                checkpoint_path=run_dir / "checkpoint_step3000.pt",
                progress_callback=update_active_progress,
            )
            train_result["checkpoint_path"] = str((run_dir / "checkpoint_step3000.pt").relative_to(temp_root))
            model = load_checkpoint_model(run_dir / "checkpoint_step3000.pt", "small", target_device)
            array_rows = generate_array_rows(
                model,
                records["array_json"]["eval"],
                target_device,
                comparison_source="diagnostic_small_3000",
                model_size="small",
                steps=DIAGNOSTIC_STEPS,
                seed=seed,
                progress_callback=update_active_progress,
            )
            write_jsonl(run_dir / "generations.jsonl", array_rows)
            train_tf_rows, train_tf_aggregate = teacher_forced_rows(
                model,
                records["array_json"]["train"],
                tokenizer,
                target_device,
                split="train",
                progress_callback=update_active_progress,
            )
            eval_tf_rows, eval_tf_aggregate = teacher_forced_rows(
                model,
                records["array_json"]["eval"],
                tokenizer,
                target_device,
                split="eval",
                progress_callback=update_active_progress,
            )
            write_jsonl(run_dir / "teacher_forced_train_rows.jsonl", train_tf_rows)
            write_jsonl(run_dir / "teacher_forced_eval_rows.jsonl", eval_tf_rows)
            write_json(
                run_dir / "metrics.json",
                {
                    "training": train_result,
                    "greedy": aggregate_array_rows(array_rows),
                    "teacher_forced_train": train_tf_aggregate,
                    "teacher_forced_eval": eval_tf_aggregate,
                    "template_item_counts": array_item_template_counts(array_rows),
                    "step1500_checkpoint_reused_for_equality_gate": reused_small["checkpoint_path"],
                    "step1500_checkpoint_sha256": reused_small["checkpoint_sha256"],
                },
            )
            completed_scope.append(
                {
                    "name": "array_diagnostic_training",
                    "model_size": "small",
                    "steps": DIAGNOSTIC_STEPS,
                    "seed": seed,
                    "rows": len(array_rows),
                    "checkpoint_path": str((run_dir / "checkpoint_step3000.pt").relative_to(temp_root)),
                }
            )
            active_scope = None
        active_scope = {"name": "terminal_publication"}
        verify_diagnostic_preflight_bindings(preflight_bindings, input_root, predecessor_diagnostic_roots)
        verify_source_unchanged(source_snapshot, active_output_root=temp_root)
        write_diagnostic_terminal(
            temp_root,
            "DONE",
            input_root=input_root,
            output_root=output_root,
            predecessor_diagnostic_roots=predecessor_diagnostic_roots,
            completed_scope=completed_scope,
            partial_scope=partial_scope,
            source_snapshot=source_snapshot,
            preflight_bindings=preflight_bindings,
            wall_time_seconds=time.monotonic() - start_time,
        )
        try:
            publish_diagnostic_root_or_leave_incomplete(
                temp_root,
                output_root,
                terminal_status="DONE",
                final_callback=final_publication_callback,
            )
        except Exception as exc:
            remove_diagnostic_terminal_markers(temp_root)
            raise DiagnosticPublicationError("Diagnostic DONE terminal failed post-construction validation; temporary root is incomplete.") from exc
    except (SourceChangedError, DiagnosticPublicationError):
        remove_diagnostic_terminal_markers(temp_root)
        raise
    except Exception as exc:
        verify_diagnostic_preflight_bindings(preflight_bindings, input_root, predecessor_diagnostic_roots)
        verify_source_unchanged(source_snapshot, active_output_root=temp_root)
        partial_scope.append({**(active_scope or {"name": "diagnose_failure"}), "error": repr(exc)})
        write_diagnostic_terminal(
            temp_root,
            "FAILED",
            input_root=input_root,
            output_root=output_root,
            predecessor_diagnostic_roots=predecessor_diagnostic_roots,
            completed_scope=completed_scope,
            partial_scope=partial_scope,
            failure=repr(exc),
            failure_classification=classify_diagnostic_failure(exc),
            source_snapshot=source_snapshot,
            preflight_bindings=preflight_bindings,
            wall_time_seconds=time.monotonic() - start_time,
        )
        try:
            publish_diagnostic_root_or_leave_incomplete(
                temp_root,
                output_root,
                terminal_status="FAILED",
                final_callback=final_publication_callback,
            )
        except Exception as post_terminal_exc:
            remove_diagnostic_terminal_markers(temp_root)
            raise DiagnosticPublicationError("Diagnostic FAILED terminal failed post-construction validation; temporary root is incomplete.") from post_terminal_exc
        raise


def run_suite(root: Path, predecessor_roots: Sequence[Path], predecessor_selections: Sequence[Path]) -> None:
    raise TypeError("run_suite requires an explicit device and decision_diagnostic_root under the D2 protocol.")


def run_current_suite(
    root: Path,
    predecessor_roots: Sequence[Path],
    predecessor_selections: Sequence[Path],
    *,
    device: str,
    decision_diagnostic_root: Path,
) -> None:
    require_feasibility_real_main_context()
    validate_feasibility_cli_contract(
        device=device,
        root=root,
        predecessor_roots=predecessor_roots,
        predecessor_selections=predecessor_selections,
        decision_diagnostic_root=decision_diagnostic_root,
    )
    validate_current_run_root_contract(root, predecessor_roots, predecessor_selections, decision_diagnostic_root)
    shallow_allowed_paths = shallow_current_run_allowed_paths(predecessor_roots, decision_diagnostic_root)
    source_snapshot = capture_source_provenance(root, predecessor_roots, predecessor_selections, allowed_paths=shallow_allowed_paths)
    configure_feasibility_deterministic_backend()
    record_hashes = validate_feasibility_record_hashes()
    for predecessor_root in predecessor_roots:
        validate_feasibility_root_artifacts(predecessor_root, require_passing=False)
    decision_diagnostic_binding = validate_decision_diagnostic_binding(decision_diagnostic_root, deep=True)
    temp_root = root.with_name(root.name + ".tmp")
    if temp_root.exists():
        raise FileExistsError(f"Temporary feasibility root already exists: {temp_root}")
    temp_root.mkdir(parents=True)
    cells: list[dict[str, object]] = []

    def final_publication_callback() -> None:
        verify_source_unchanged(source_snapshot, active_output_root=temp_root)

    try:
        records = grouped_records()
        tokenizer = ByteTokenizer()
        target_device = torch.device(device)
        for family in FAMILIES:
            train_records = tuple(TextRecord(r.prompt, r.answer) for r in records[family]["train"])
            eval_records = records[family]["eval"]
            for model_size in MODEL_SIZES:
                for seed in SEEDS:
                    set_deterministic_backend(seed)
                    model = build_model(model_size)
                    if model.lm_head.weight is not model.token_embedding.weight:
                        raise ValueError("Current feasibility model must tie lm_head.weight to token_embedding.weight.")
                    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
                        raise ValueError("Current feasibility model must be initialized completely on CPU before cuda:0 transfer.")
                    model.to(target_device)
                    result = train_text_records(
                        model,
                        train_records,
                        seed=seed,
                        state_mask=0,
                        steps=TRAINING_STEPS,
                        batch_size=BATCH_SIZE,
                        tokenizer=tokenizer,
                        device=target_device,
                    )
                    exact_matches, generations = evaluate_model(model, eval_records, tokenizer, target_device)
                    cell_dir = temp_root / f"{family}__{model_size}__seed{seed}"
                    cell_dir.mkdir()
                    write_jsonl(cell_dir / "generations.jsonl", generations)
                    save_checkpoint(
                        str(cell_dir / "checkpoint_step1500.pt"),
                        model,
                        metadata={
                            "family": family,
                            "model_size": model_size,
                            "seed": seed,
                            "training_steps": TRAINING_STEPS,
                            "training_loss": result.final_loss,
                            "training_accuracy": result.training_accuracy,
                        },
                    )
                    cells.append(
                        {
                            "family": family,
                            "model_size": model_size,
                            "seed": seed,
                            "eval_count": EVAL_RECORDS_PER_FAMILY,
                            "exact_matches": exact_matches,
                            "passed": exact_matches >= PASS_THRESHOLD,
                            "generations_path": str(cell_dir.relative_to(temp_root) / "generations.jsonl"),
                            "checkpoint_path": str(cell_dir.relative_to(temp_root) / "checkpoint_step1500.pt"),
                            "parameter_count": model.parameter_count,
                            "embedding_weight_tying": True,
                            "model_protocol_revision": TIED_MODEL_PROTOCOL_REVISION,
                        }
                    )
        terminal_status = "DONE" if all(cell["passed"] for cell in cells) else "FAILED"
        failure = None if terminal_status == "DONE" else "complete feasibility matrix did not satisfy all 24 pass thresholds."
        try:
            validate_current_feasibility_cells(
                cells,
                require_complete=True,
                require_all_pass=terminal_status == "DONE",
            )
            verify_source_unchanged(source_snapshot, active_output_root=temp_root)
            write_terminal(
                temp_root,
                terminal_status,
                cells,
                predecessor_roots,
                predecessor_selections,
                failure=failure,
                source_snapshot=source_snapshot,
                decision_diagnostic=decision_diagnostic_binding,
                exact_command=feasibility_exact_command(root, predecessor_roots, decision_diagnostic_root),
                deterministic_flags=current_deterministic_flags(),
                record_hashes=record_hashes,
            )
            publish_current_feasibility_root_or_leave_incomplete(
                temp_root,
                root,
                terminal_status=terminal_status,
                predecessor_roots=predecessor_roots,
                predecessor_selections=predecessor_selections,
                decision_diagnostic_root=decision_diagnostic_root,
                source_snapshot=source_snapshot,
                allowed_source_paths=shallow_allowed_paths,
                final_callback=final_publication_callback,
            )
        except (SourceChangedError, FeasibilityPublicationError):
            remove_diagnostic_terminal_markers(temp_root)
            raise
        except Exception as terminal_exc:
            remove_diagnostic_terminal_markers(temp_root)
            raise FeasibilityPublicationError("Current feasibility terminal publication failed; temporary root is incomplete.") from terminal_exc
    except (SourceChangedError, FeasibilityPublicationError):
        remove_diagnostic_terminal_markers(temp_root)
        raise
    except Exception as exc:
        verify_source_unchanged(source_snapshot, active_output_root=temp_root)
        try:
            write_terminal(
                temp_root,
                "FAILED",
                cells,
                predecessor_roots,
                predecessor_selections,
                failure=repr(exc),
                source_snapshot=source_snapshot,
                decision_diagnostic=decision_diagnostic_binding,
                exact_command=feasibility_exact_command(root, predecessor_roots, decision_diagnostic_root),
                deterministic_flags=current_deterministic_flags(),
                record_hashes=record_hashes,
            )
            publish_current_feasibility_root_or_leave_incomplete(
                temp_root,
                root,
                terminal_status="FAILED",
                predecessor_roots=predecessor_roots,
                predecessor_selections=predecessor_selections,
                decision_diagnostic_root=decision_diagnostic_root,
                source_snapshot=source_snapshot,
                allowed_source_paths=shallow_allowed_paths,
                final_callback=final_publication_callback,
            )
        except Exception as post_terminal_exc:
            remove_diagnostic_terminal_markers(temp_root)
            raise FeasibilityPublicationError("Current feasibility FAILED terminal failed post-construction validation; temporary root is incomplete.") from post_terminal_exc
        raise


def validate_new_root(
    root: Path,
    predecessor_roots: Sequence[Path] = (),
    predecessor_selections: Sequence[Path] = (),
) -> None:
    context = FeasibilityValidationContext()
    require_canonical_path_string(str(root), "root", ROOT_RE)
    require_artifact_location(root, "root", ROOT_RE)
    root_number = feasibility_root_number(root)
    direct_numbers: list[int] = []
    for predecessor_root in predecessor_roots:
        predecessor_number = feasibility_root_number(predecessor_root)
        if direct_numbers and predecessor_number <= direct_numbers[-1]:
            raise ValueError("Feasibility predecessor roots must be in strictly ascending root-number order.")
        direct_numbers.append(predecessor_number)
        terminal_binding(predecessor_root, context=context)
    selected_numbers: list[int] = []
    for selection in predecessor_selections:
        data = validate_selection_record(selection, context=context)
        selected_numbers.append(feasibility_root_number(Path(str(data["selected_root"]))))
    expected_previous_numbers = list(range(1, root_number))
    observed_numbers = sorted({*direct_numbers, *selected_numbers})
    if observed_numbers != expected_previous_numbers:
        raise ValueError(
            f"Feasibility predecessor roots must be complete and continuous before {root.name!r}; "
            f"expected {expected_previous_numbers!r}, got {observed_numbers!r}."
        )
    expected_number = len(expected_previous_numbers) + 1
    if root_number != expected_number:
        raise ValueError(
            f"Feasibility root must use the next numbered root feasibility_{expected_number:03d}; "
            f"got {root.name!r}."
        )
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing feasibility root: {root}")


def validate_current_run_root_contract(
    root: Path,
    predecessor_roots: Sequence[Path],
    predecessor_selections: Sequence[Path],
    decision_diagnostic_root: Path,
) -> None:
    require_canonical_path_string(str(root), "root", ROOT_RE)
    require_artifact_location(root, "root", ROOT_RE)
    if str(root) != FEASIBILITY_REQUIRED_ROOT:
        raise ValueError("Current D2 run root must be exactly artifacts/phase8_toy_lm_bridge/feasibility_005.")
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing feasibility root: {root}")
    temp_root = root.with_name(root.name + ".tmp")
    if temp_root.exists() or temp_root.is_symlink():
        raise FileExistsError(f"Temporary feasibility root already exists: {temp_root}")
    if predecessor_selections:
        raise ValueError("Current D2 run forbids predecessor selections.")
    if tuple(str(path) for path in predecessor_roots) != FEASIBILITY_REQUIRED_PREDECESSOR_ROOTS:
        raise ValueError("Current D2 run requires exactly predecessor roots feasibility_001..004 in numerical order.")
    if str(decision_diagnostic_root) != FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT:
        raise ValueError("Current D2 run requires exactly feasibility_diagnostic_001 as decision diagnostic.")


def feasibility_root_number(root: Path) -> int:
    match = ROOT_RE.match(root.name)
    if match is None:
        raise ValueError("Feasibility root basename must be immutable numbered form feasibility_NNN.")
    number = int(match.group(1))
    if number <= 0:
        raise ValueError("Feasibility root numbering starts at feasibility_001.")
    return number


def require_artifact_location(path: Path, field_name: str, basename_pattern: re.Pattern[str]) -> None:
    require_canonical_path_string(str(path), field_name, basename_pattern)
    parent_path = ARTIFACT_PARENT if ARTIFACT_PARENT.is_absolute() else REPO_ROOT / ARTIFACT_PARENT
    artifact_path = path if path.is_absolute() else REPO_ROOT / path
    reject_existing_symlink_component(parent_path, f"{field_name} artifact parent")
    reject_existing_symlink_component(artifact_path, field_name)
    parent = parent_path.resolve(strict=False)
    resolved = artifact_path.resolve(strict=False)
    if resolved.parent != parent:
        raise ValueError(f"{field_name} must be located directly under {ARTIFACT_PARENT.as_posix()}.")


def reject_existing_symlink_component(path: Path, field_name: str) -> None:
    absolute = path if path.is_absolute() else REPO_ROOT / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{field_name} must not contain symlink path components.")


def validation_context(context: FeasibilityValidationContext | None = None) -> FeasibilityValidationContext:
    return context if context is not None else FeasibilityValidationContext()


def validate_source_clean(root: Path, predecessor_roots: Sequence[Path], predecessor_selections: Sequence[Path]) -> SourceSnapshot:
    return capture_source_provenance(root, predecessor_roots, predecessor_selections)


def capture_source_provenance(
    root: Path,
    predecessor_roots: Sequence[Path],
    predecessor_selections: Sequence[Path],
    *,
    allowed_paths: set[Path] | None = None,
) -> SourceSnapshot:
    require_artifact_location(root, "root", ROOT_RE)
    allowed = source_clean_allowed_paths(predecessor_roots, predecessor_selections) if allowed_paths is None else allowed_paths
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        cwd=REPO_ROOT,
    ).stdout.splitlines()
    for line in status:
        code = line[:2]
        rel = line[3:]
        rel_path = Path(rel)
        candidate = (rel_path if rel_path.is_absolute() else REPO_ROOT / rel_path).resolve()
        if code != "??":
            raise RuntimeError(f"Tracked or staged source change blocks feasibility run: {line}")
        if candidate not in allowed:
            raise RuntimeError(f"Untracked file is not an exact supplied predecessor/root binding: {line}")
    commit = git_output(["git", "rev-parse", "HEAD"])
    validate_git_sha(commit, "source_commit")
    ignored_inputs = ignored_source_inputs()
    if ignored_inputs:
        paths = ", ".join(str(row["path"]) for row in ignored_inputs[:5])
        suffix = " ..." if len(ignored_inputs) > 5 else ""
        raise RuntimeError(f"Ignored executable source input blocks feasibility run: {paths}{suffix}")
    return SourceSnapshot(
        commit=commit,
        status_lines=tuple(status),
        ignored_inputs=(),
    )


def shallow_inventory_bound_paths_from_rows(root: Path, rows: object, *, schema_mode: str) -> set[Path]:
    if not isinstance(rows, list):
        raise ValueError("Shallow inventory authorization requires a file_inventory list.")
    if schema_mode == "historical":
        expected_keys = {"path", "sha256", "bytes"}
    elif schema_mode == "diagnostic":
        expected_keys = {"path", "sha256", "bytes", "role"}
    else:
        raise ValueError("Shallow inventory authorization requires explicit schema_mode 'historical' or 'diagnostic'.")
    allowed: set[Path] = set()
    root_resolved = root.resolve()
    seen_paths: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Shallow file_inventory entries must be JSON objects.")
        if set(row) != expected_keys:
            raise ValueError("Shallow file_inventory entries do not match the exact schema for the requested mode.")
        rel_path = require_canonical_relative_path(row.get("path"), f"file_inventory[{index}].path")
        if rel_path in seen_paths:
            raise ValueError("Shallow file_inventory must not contain duplicate paths.")
        seen_paths.add(rel_path)
        if schema_mode == "diagnostic":
            role = require_exact_str(row.get("role"), f"file_inventory[{index}].role")
            if role != diagnostic_inventory_role(rel_path):
                raise ValueError("Shallow diagnostic file_inventory role mismatch.")
        raw_candidate = root / rel_path
        if raw_candidate.is_symlink():
            raise ValueError("Shallow file_inventory must not bind symlink files.")
        candidate = raw_candidate.resolve()
        if root_resolved not in candidate.parents:
            raise ValueError("Shallow file_inventory path escapes its root.")
        if not candidate.is_file():
            raise ValueError("Shallow file_inventory path does not exist as a file.")
        expected_sha = require_exact_str(row.get("sha256"), f"file_inventory[{index}].sha256")
        expected_bytes = row.get("bytes")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ValueError(f"file_inventory[{index}].bytes must be a non-negative integer.")
        if candidate.stat().st_size != expected_bytes:
            raise ValueError("Shallow file_inventory byte count mismatch.")
        if file_sha256(candidate) != expected_sha:
            raise ValueError("Shallow file_inventory checksum mismatch.")
        allowed.add(candidate)
    expected_inventory = inventory(root) if schema_mode == "historical" else diagnostic_inventory(root)
    if sorted(rows, key=lambda row: row["path"]) != expected_inventory:
        raise ValueError("Shallow file_inventory does not exactly match root files.")
    return allowed


def shallow_inventory_bound_root_paths(root: Path) -> set[Path]:
    require_canonical_path_string(str(root), "predecessor_root.path", ROOT_RE)
    require_artifact_location(root, "predecessor_root.path", ROOT_RE)
    if str(root) not in FEASIBILITY_REQUIRED_PREDECESSOR_ROOTS:
        raise ValueError("Current feasibility run may shallow-authorize only the four frozen predecessor roots.")
    terminal, _terminal_data, manifest, manifest_sha = load_terminal_binding(root)
    manifest_data = json.loads(manifest.read_text())
    source_commit = validate_git_sha(manifest_data.get("source_commit"), "source_commit")
    if not validate_historical_feasibility_identity(
        root,
        source_commit=source_commit,
        manifest_sha=manifest_sha,
        terminal_path=terminal,
    ):
        raise ValueError("Predecessor root is not an exact historical D2 allowlist root.")
    if manifest_data.get("configuration") != HISTORICAL_FEASIBILITY_CONFIGURATION:
        raise ValueError("Historical predecessor root configuration mismatch.")
    return {
        manifest.resolve(),
        terminal.resolve(),
        *shallow_inventory_bound_paths_from_rows(root, manifest_data.get("file_inventory"), schema_mode="historical"),
    }


def shallow_decision_diagnostic_paths(root: Path) -> set[Path]:
    binding = validate_decision_diagnostic_binding(root, deep=False)
    if binding != DECISION_DIAGNOSTIC_ROOT_BINDING:
        raise ValueError("Decision diagnostic binding does not match the frozen D2 decision object.")
    manifest = root / "manifest.json"
    summary = root / "summary.json"
    terminal = root / "DONE.json"
    manifest_data = json.loads(manifest.read_text())
    return {
        manifest.resolve(),
        summary.resolve(),
        terminal.resolve(),
        *shallow_inventory_bound_paths_from_rows(root, manifest_data.get("file_inventory"), schema_mode="diagnostic"),
    }


def shallow_current_run_allowed_paths(predecessor_roots: Sequence[Path], decision_diagnostic_root: Path) -> set[Path]:
    allowed: set[Path] = set()
    for root in predecessor_roots:
        allowed.update(shallow_inventory_bound_root_paths(root))
    allowed.update(shallow_decision_diagnostic_paths(decision_diagnostic_root))
    return allowed


def verify_source_unchanged(snapshot: SourceSnapshot, *, active_output_root: Path | None = None) -> None:
    current = git_output(["git", "rev-parse", "HEAD"])
    if current != snapshot.commit:
        raise SourceChangedError("Source HEAD changed during feasibility run before terminal publication.")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        cwd=REPO_ROOT,
    ).stdout.splitlines()
    filtered_status = tuple(line for line in status if not is_active_output_status_line(line, active_output_root))
    if filtered_status != snapshot.status_lines:
        raise SourceChangedError("Source worktree status changed during feasibility run before terminal publication.")
    if ignored_source_inputs() != snapshot.ignored_inputs:
        raise SourceChangedError("Ignored executable source inputs changed during feasibility run before terminal publication.")


def ignored_import_surface_candidate(path: Path, *, is_symlink: bool, link_target: str | None) -> bool:
    if path.is_absolute():
        raise ValueError("Ignored source scan requires repository-relative paths.")
    relative = PurePosixPath(path.as_posix())
    if "__pycache__" in relative.parts:
        return True
    if path.suffix in IGNORED_IMPORT_SURFACE_SUFFIXES:
        return True
    if path.name in IGNORED_IMPORT_HOOK_STEMS or path.stem in IGNORED_IMPORT_HOOK_STEMS:
        return True
    if is_symlink:
        if link_target is None:
            return True
        target = PurePosixPath(link_target)
        if "__pycache__" in target.parts:
            return True
        if target.suffix in IGNORED_IMPORT_SURFACE_SUFFIXES:
            return True
        if target.name in IGNORED_IMPORT_HOOK_STEMS or target.stem in IGNORED_IMPORT_HOOK_STEMS:
            return True
        return True
    return False


def sha256_regular_file_no_follow(path: Path, expected_stat: os.stat_result | None = None) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError("Source snapshot can hash only regular non-symlink files.")
        if expected_stat is not None:
            expected_identity = (expected_stat.st_dev, expected_stat.st_ino, stat.S_IFMT(expected_stat.st_mode), expected_stat.st_size)
            observed_identity = (observed.st_dev, observed.st_ino, stat.S_IFMT(observed.st_mode), observed.st_size)
            if observed_identity != expected_identity:
                raise ValueError("Source snapshot file identity changed while hashing.")
        digest = sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(fd)


def symlink_target_sha256(path: Path) -> tuple[str, int]:
    target = os.readlink(path)
    payload = os.fsencode(target)
    return sha256(payload).hexdigest(), len(payload)


def ignored_source_input_snapshot(path: Path) -> dict[str, object] | None:
    if path.is_absolute():
        raise ValueError("Ignored source snapshot requires repository-relative paths.")
    absolute = REPO_ROOT / path
    try:
        metadata = absolute.lstat()
    except FileNotFoundError:
        return None
    is_symlink = stat.S_ISLNK(metadata.st_mode)
    link_target = os.readlink(absolute) if is_symlink else None
    if not ignored_import_surface_candidate(path, is_symlink=is_symlink, link_target=link_target):
        return None
    if is_symlink:
        digest, byte_count = symlink_target_sha256(absolute)
    elif stat.S_ISREG(metadata.st_mode):
        digest = sha256_regular_file_no_follow(absolute, metadata)
        byte_count = metadata.st_size
    else:
        raise ValueError("Ignored import-surface input must be a regular file or symlink.")
    return {"path": path.as_posix(), "sha256": digest, "bytes": int(byte_count)}


def ignored_source_inputs() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    completed = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        cwd=REPO_ROOT,
    )
    for rel in completed.stdout.split("\0"):
        if not rel:
            continue
        path = Path(rel)
        snapshot = ignored_source_input_snapshot(path)
        if snapshot is not None:
            rows.append(snapshot)
    return tuple(sorted(rows, key=lambda row: str(row["path"])))


def is_active_output_status_line(line: str, active_output_root: Path | None) -> bool:
    if active_output_root is None or not line.startswith("?? "):
        return False
    rel = line[3:]
    candidate = Path(rel)
    candidate_path = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    active_path = active_output_root if active_output_root.is_absolute() else REPO_ROOT / active_output_root
    try:
        candidate_resolved = candidate_path.resolve(strict=False)
        active_resolved = active_path.resolve(strict=False)
    except OSError:
        return False
    return candidate_resolved == active_resolved or active_resolved in candidate_resolved.parents


def source_clean_allowed_paths(
    predecessor_roots: Sequence[Path],
    predecessor_selections: Sequence[Path],
    *,
    context: FeasibilityValidationContext | None = None,
) -> set[Path]:
    context = validation_context(context)
    allowed: set[Path] = set()
    for selection in predecessor_selection_paths_from_selections(predecessor_selections, context=context):
        allowed.add(selection.resolve())
    for predecessor_root in (
        *predecessor_roots,
        *predecessor_root_paths_from_selections(predecessor_selections, context=context),
    ):
        allowed.update(inventory_bound_root_paths(predecessor_root, context=context))
    return allowed


def inventory_bound_root_paths(
    root: Path,
    *,
    context: FeasibilityValidationContext | None = None,
) -> set[Path]:
    context = validation_context(context)
    require_canonical_path_string(str(root), "predecessor_root.path", ROOT_RE)
    require_artifact_location(root, "predecessor_root.path", ROOT_RE)
    resolved_root = root.resolve()
    current_commit = current_source_commit()
    cache_key = (resolved_root, current_commit)
    cached = context.inventory_bound_paths.get(cache_key)
    if cached is not None:
        return set(cached)
    validate_feasibility_root_artifacts(root, require_passing=False, context=context)
    terminal, _terminal_data, manifest, _manifest_sha = load_terminal_binding(root)
    manifest_data = json.loads(manifest.read_text())
    file_inventory = manifest_data.get("file_inventory")
    if not isinstance(file_inventory, list):
        raise ValueError("Predecessor manifest must contain a file_inventory list.")
    allowed = {manifest.resolve(), terminal.resolve()}
    root_resolved = root.resolve()
    seen_paths: set[str] = set()
    for index, row in enumerate(file_inventory):
        if not isinstance(row, dict):
            raise ValueError("Predecessor manifest file_inventory entries must be JSON objects.")
        rel_path = require_canonical_relative_path(row.get("path"), f"file_inventory[{index}].path")
        if rel_path in seen_paths:
            raise ValueError("Predecessor manifest file_inventory must not contain duplicate paths.")
        seen_paths.add(rel_path)
        raw_candidate = root / rel_path
        if raw_candidate.is_symlink():
            raise ValueError("Predecessor manifest file_inventory must not bind symlink files.")
        candidate = raw_candidate.resolve()
        if root_resolved not in candidate.parents:
            raise ValueError("Predecessor manifest file_inventory path escapes its root.")
        if not candidate.is_file():
            raise ValueError("Predecessor manifest file_inventory path does not exist as a file.")
        expected_sha = require_exact_str(row.get("sha256"), f"file_inventory[{index}].sha256")
        expected_bytes = row.get("bytes")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ValueError(f"file_inventory[{index}].bytes must be a non-negative integer.")
        if candidate.stat().st_size != expected_bytes:
            raise ValueError("Predecessor manifest file_inventory byte count mismatch.")
        if file_sha256(candidate) != expected_sha:
            raise ValueError("Predecessor manifest file_inventory checksum mismatch.")
        allowed.add(candidate)
    context.inventory_bound_paths[cache_key] = set(allowed)
    return allowed


def build_manifest(
    root: Path,
    cells: Sequence[dict[str, object]],
    predecessor_roots: Sequence[Path],
    predecessor_selections: Sequence[Path],
    terminal_status: str,
    failure: str | None = None,
    source_snapshot: SourceSnapshot | None = None,
    decision_diagnostic: dict[str, object] | None = None,
    exact_command: Sequence[str] | None = None,
    deterministic_flags: dict[str, object] | None = None,
    record_hashes: dict[str, str] | None = None,
) -> dict[str, object]:
    source_snapshot = source_snapshot or unchecked_source_snapshot()
    return {
        "protocol": "phase8_sequence_feasibility",
        "terminal_status": terminal_status,
        "failure": failure,
        "source_commit": source_snapshot.commit,
        "source_provenance": {
            "commit": source_snapshot.commit,
            "status_lines": list(source_snapshot.status_lines),
            "ignored_inputs": list(source_snapshot.ignored_inputs),
        },
        "configuration": frozen_configuration(),
        "decision_diagnostic": decision_diagnostic,
        "exact_command": list(exact_command) if exact_command is not None else None,
        "deterministic_flags": deterministic_flags,
        "record_hashes": record_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_driver": gpu_driver_version(),
        },
        "cells": list(cells),
        "predecessor_roots": complete_predecessor_root_bindings(predecessor_roots, predecessor_selections),
        "predecessor_selections": [selection_binding(path) for path in predecessor_selections],
        "file_inventory": inventory(root),
    }


def gpu_driver_version() -> str | None:
    if not torch.cuda.is_available():
        return None
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        return None
    first_line = completed.stdout.strip().splitlines()
    return first_line[0].strip() if first_line else None


def build_summary(
    cells: Sequence[dict[str, object]],
    predecessor_roots: Sequence[Path],
    predecessor_selections: Sequence[Path],
    terminal_status: str,
    *,
    failure: str | None = None,
    source_snapshot: SourceSnapshot | None = None,
    decision_diagnostic: dict[str, object] | None = None,
    exact_command: Sequence[str] | None = None,
    deterministic_flags: dict[str, object] | None = None,
    record_hashes: dict[str, str] | None = None,
) -> dict[str, object]:
    source_snapshot = source_snapshot or unchecked_source_snapshot()
    passed_cells = sum(1 for cell in cells if cell.get("passed") is True)
    return {
        "protocol": "phase8_sequence_feasibility",
        "terminal_status": terminal_status,
        "failure": failure,
        "source": {
            "commit": source_snapshot.commit,
            "script": "scripts/phase8_sequence_feasibility.py",
            "ignored_inputs": list(source_snapshot.ignored_inputs),
        },
        "configuration": frozen_configuration(),
        "decision_diagnostic": decision_diagnostic,
        "exact_command": list(exact_command) if exact_command is not None else None,
        "deterministic_flags": deterministic_flags,
        "record_hashes": record_hashes,
        "cells": list(cells),
        "summary": {
            "total_cells": len(cells),
            "passed_cells": passed_cells,
            "failed_cells": len(cells) - passed_cells,
            "all_cells_passed": len(cells) == len(FAMILIES) * len(MODEL_SIZES) * len(SEEDS) and passed_cells == len(cells),
        },
        "predecessor_root_count": len(predecessor_roots),
        "predecessor_selection_count": len(predecessor_selections),
    }


def write_terminal(
    root: Path,
    terminal_status: str,
    cells: Sequence[dict[str, object]],
    predecessor_roots: Sequence[Path],
    predecessor_selections: Sequence[Path],
    *,
    failure: str | None = None,
    source_snapshot: SourceSnapshot | None = None,
    decision_diagnostic: dict[str, object] | None = None,
    exact_command: Sequence[str] | None = None,
    deterministic_flags: dict[str, object] | None = None,
    record_hashes: dict[str, str] | None = None,
) -> None:
    if terminal_status not in {"DONE", "FAILED"}:
        raise ValueError(f"Unknown terminal status: {terminal_status!r}.")
    write_json(
        root / "summary.json",
        build_summary(
            cells,
            predecessor_roots,
            predecessor_selections,
            terminal_status,
            failure=failure,
            source_snapshot=source_snapshot,
            decision_diagnostic=decision_diagnostic,
            exact_command=exact_command,
            deterministic_flags=deterministic_flags,
            record_hashes=record_hashes,
        ),
    )
    manifest = build_manifest(
        root,
        cells,
        predecessor_roots,
        predecessor_selections,
        terminal_status,
        failure,
        source_snapshot=source_snapshot,
        decision_diagnostic=decision_diagnostic,
        exact_command=exact_command,
        deterministic_flags=deterministic_flags,
        record_hashes=record_hashes,
    )
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    manifest_sha = file_sha256(manifest_path)
    terminal = {
        "status": terminal_status,
        "manifest_path": "manifest.json",
        "manifest_sha256": manifest_sha,
        "pass_threshold": PASS_THRESHOLD,
        "configuration": frozen_configuration(),
        "decision_diagnostic": decision_diagnostic,
        "exact_command": list(exact_command) if exact_command is not None else None,
        "deterministic_flags": deterministic_flags,
        "record_hashes": record_hashes,
        "cells": list(cells),
    }
    if failure is not None:
        terminal["error"] = failure
    write_json(root / f"{terminal_status}.json", terminal)


def terminal_binding(
    root: Path,
    *,
    context: FeasibilityValidationContext | None = None,
) -> dict[str, object]:
    require_canonical_path_string(str(root), "predecessor_root.path", ROOT_RE)
    require_artifact_location(root, "predecessor_root.path", ROOT_RE)
    validate_feasibility_root_artifacts(root, require_passing=False, context=context)
    terminal, terminal_data, manifest, manifest_sha = load_terminal_binding(root)
    return {
        "path": str(root),
        "terminal_state": terminal.stem,
        "terminal_sha256": file_sha256(terminal),
        "manifest_sha256": manifest_sha,
    }


def load_terminal_binding(root: Path) -> tuple[Path, dict[str, object], Path, str]:
    done = root / "DONE.json"
    failed = root / "FAILED.json"
    existing_terminals = [path for path in (done, failed) if path.exists()]
    if len(existing_terminals) > 1:
        raise ValueError(f"Root has both DONE.json and FAILED.json terminal markers: {root}")
    if not existing_terminals:
        raise ValueError(f"Root lacks DONE.json or FAILED.json: {root}")
    terminal = existing_terminals[0]
    manifest = root / "manifest.json"
    if not manifest.exists():
        raise ValueError(f"Root lacks manifest.json: {root}")
    manifest_sha = file_sha256(manifest)
    terminal_data = json.loads(terminal.read_text())
    manifest_data = json.loads(manifest.read_text())
    if terminal_data.get("status") != terminal.stem:
        raise ValueError("Terminal marker status does not match its filename.")
    if manifest_data.get("terminal_status") != terminal.stem:
        raise ValueError("Manifest terminal_status does not match the terminal marker.")
    if terminal_data.get("manifest_sha256") != manifest_sha:
        raise ValueError("Terminal marker does not bind the manifest checksum.")
    if terminal_data.get("manifest_path") != "manifest.json":
        raise ValueError("Terminal marker manifest_path must be manifest.json.")
    if terminal_data.get("pass_threshold") != PASS_THRESHOLD:
        raise ValueError("Terminal marker pass_threshold does not match the frozen threshold.")
    if terminal.stem == "DONE" and "error" in terminal_data:
        raise ValueError("DONE terminal marker must not contain an error field.")
    if terminal.stem == "FAILED" and terminal_data.get("error") != manifest_data.get("failure"):
        raise ValueError("FAILED terminal error does not match manifest failure.")
    return terminal, terminal_data, manifest, manifest_sha


def exact_artifact_root_path(name: str) -> str:
    return f"artifacts/phase8_toy_lm_bridge/{name}"


def matches_repo_artifact_path(path: Path, relative_path: str) -> bool:
    candidate = path if path.is_absolute() else REPO_ROOT / path
    expected = REPO_ROOT / relative_path
    return candidate.resolve(strict=False) == expected.resolve(strict=False)


def validate_historical_feasibility_identity(
    root: Path,
    *,
    source_commit: str,
    manifest_sha: str,
    terminal_path: Path,
) -> bool:
    expected = HISTORICAL_FEASIBILITY_ROOTS.get(root.name)
    if expected is None:
        return False
    if not matches_repo_artifact_path(root, exact_artifact_root_path(root.name)):
        return False
    if source_commit != expected["source_commit"]:
        raise ValueError(f"Historical {root.name} source commit mismatch.")
    if manifest_sha != expected["manifest_sha256"]:
        raise ValueError(f"Historical {root.name} manifest checksum mismatch.")
    if terminal_path.name != expected["terminal"]:
        raise ValueError(f"Historical {root.name} terminal path mismatch.")
    if file_sha256(terminal_path) != expected["terminal_sha256"]:
        raise ValueError(f"Historical {root.name} terminal checksum mismatch.")
    return True


def historical_feasibility_checkpoint_allowed(path: Path) -> bool:
    for parent in path.parents:
        if ROOT_RE.match(parent.name) is None:
            continue
        if not matches_repo_artifact_path(parent, exact_artifact_root_path(parent.name)):
            return False
        terminal, _terminal_data, manifest, manifest_sha = load_terminal_binding(parent)
        manifest_data = json.loads(manifest.read_text())
        source_commit = validate_git_sha(manifest_data.get("source_commit"), "source_commit")
        if not validate_historical_feasibility_identity(
            parent,
            source_commit=source_commit,
            manifest_sha=manifest_sha,
            terminal_path=terminal,
        ):
            return False
        rel_path = path.relative_to(parent).as_posix()
        file_inventory = manifest_data.get("file_inventory")
        if not isinstance(file_inventory, list):
            raise ValueError("Historical feasibility manifest must bind file_inventory.")
        matching = [row for row in file_inventory if isinstance(row, dict) and row.get("path") == rel_path]
        if len(matching) != 1:
            raise ValueError("Historical checkpoint path is not bound by its manifest inventory.")
        row = matching[0]
        if row.get("sha256") != file_sha256(path) or row.get("bytes") != path.stat().st_size:
            raise ValueError("Historical checkpoint inventory checksum or size mismatch.")
        return True
    return False


def validate_decision_diagnostic_binding(root: Path, *, deep: bool) -> dict[str, object]:
    if not matches_repo_artifact_path(root, str(DECISION_DIAGNOSTIC_ROOT_BINDING["path"])):
        raise ValueError("Decision diagnostic root path mismatch.")
    require_diagnostic_artifact_location(root, "decision_diagnostic.path")
    for relative, key in (("manifest.json", "manifest_sha256"), ("summary.json", "summary_sha256"), ("DONE.json", "terminal_sha256")):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Decision diagnostic missing canonical {relative}.")
        if file_sha256(path) != DECISION_DIAGNOSTIC_ROOT_BINDING[key]:
            raise ValueError(f"Decision diagnostic {relative} checksum mismatch.")
    manifest_data = json.loads((root / "manifest.json").read_text())
    summary_data = json.loads((root / "summary.json").read_text())
    terminal_data = json.loads((root / "DONE.json").read_text())
    for data_name, data in (("manifest", manifest_data), ("summary", summary_data), ("terminal", terminal_data)):
        if data.get("artifact_class") != DIAGNOSTIC_ARTIFACT_CLASS:
            raise ValueError(f"Decision diagnostic {data_name} artifact_class mismatch.")
        if data.get("feasibility_selection_eligible") is not False:
            raise ValueError(f"Decision diagnostic {data_name} must not be selection-eligible.")
        if data.get("task_010d_authorized") is not False:
            raise ValueError(f"Decision diagnostic {data_name} must not authorize Task 010D.")
    if manifest_data.get("source_commit") != DECISION_DIAGNOSTIC_ROOT_BINDING["source_commit"]:
        raise ValueError("Decision diagnostic source_commit mismatch.")
    file_inventory = manifest_data.get("file_inventory")
    if not isinstance(file_inventory, list):
        raise ValueError("Decision diagnostic must bind a complete file_inventory.")
    actual_inventory = diagnostic_inventory(root)
    if file_inventory != actual_inventory or terminal_data.get("file_inventory") != file_inventory:
        raise ValueError("Decision diagnostic inventory mismatch.")
    if deep:
        load_diagnostic_terminal_binding(root)
    return {
        "path": str(DECISION_DIAGNOSTIC_ROOT_BINDING["path"]),
        "source_commit": manifest_data.get("source_commit"),
        "artifact_class": DIAGNOSTIC_ARTIFACT_CLASS,
        "feasibility_selection_eligible": False,
        "task_010d_authorized": False,
        "manifest_sha256": DECISION_DIAGNOSTIC_ROOT_BINDING["manifest_sha256"],
        "summary_sha256": DECISION_DIAGNOSTIC_ROOT_BINDING["summary_sha256"],
        "terminal_sha256": DECISION_DIAGNOSTIC_ROOT_BINDING["terminal_sha256"],
        "accepted_proposal_commit": DECISION_DIAGNOSTIC_ROOT_BINDING["accepted_proposal_commit"],
        "independent_review_verdict": ACCEPTED_INDEPENDENT_REVIEW_VERDICT,
    }


def d1_historical_checkpoint_allowed(path: Path) -> bool:
    root = REPO_ROOT / FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT
    candidate = path if path.is_absolute() else REPO_ROOT / path
    try:
        rel_path = candidate.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return False
    expected_sha = D1_HISTORICAL_CHECKPOINT_SHA256.get(rel_path)
    if expected_sha is None:
        return False
    validate_decision_diagnostic_binding(root, deep=False)
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError("D1 historical checkpoint path must be a regular file.")
    if file_sha256(candidate) != expected_sha:
        raise ValueError("D1 historical checkpoint checksum mismatch.")
    manifest_data = json.loads((root / "manifest.json").read_text())
    file_inventory = manifest_data.get("file_inventory")
    if not isinstance(file_inventory, list):
        raise ValueError("D1 diagnostic manifest must bind file_inventory.")
    matching = [row for row in file_inventory if isinstance(row, dict) and row.get("path") == rel_path]
    if len(matching) != 1:
        raise ValueError("D1 historical checkpoint path is not bound by diagnostic inventory.")
    row = matching[0]
    if row.get("sha256") != expected_sha or row.get("bytes") != candidate.stat().st_size:
        raise ValueError("D1 historical checkpoint inventory mismatch.")
    return True


def validate_historical_checkpoint_allowlist(path: Path) -> None:
    if historical_feasibility_checkpoint_allowed(path):
        return
    if d1_historical_checkpoint_allowed(path):
        return
    raise ValueError("Historical checkpoint is outside the exact D2 legacy allowlist.")


def validate_feasibility_root_artifacts(
    root: Path,
    *,
    require_passing: bool,
    context: FeasibilityValidationContext | None = None,
) -> None:
    context = validation_context(context)
    require_artifact_location(root, "selected_root" if require_passing else "predecessor_root.path", ROOT_RE)
    if not root.is_dir():
        raise ValueError(f"Feasibility root is not a directory: {root}")
    resolved_root = root.resolve()
    current_commit = current_source_commit()
    cache_key = (resolved_root, require_passing, current_commit)
    if cache_key in context.validated_roots or (not require_passing and (resolved_root, True, current_commit) in context.validated_roots):
        return
    if resolved_root in context.active_roots:
        raise ValueError(f"Feasibility root lineage contains a cycle at {root}.")
    context.active_roots.add(resolved_root)
    try:
        _validate_feasibility_root_artifacts(
            root,
            require_passing=require_passing,
            context=context,
            current_commit=current_commit,
        )
    finally:
        context.active_roots.remove(resolved_root)
    context.validated_roots.add(cache_key)
    if require_passing:
        context.validated_roots.add((resolved_root, False, current_commit))


def _validate_feasibility_root_artifacts(
    root: Path,
    *,
    require_passing: bool,
    context: FeasibilityValidationContext,
    current_commit: str,
) -> None:
    require_artifact_location(root, "selected_root" if require_passing else "predecessor_root.path", ROOT_RE)
    if not root.is_dir():
        raise ValueError(f"Feasibility root is not a directory: {root}")
    terminal, terminal_data, manifest, manifest_sha = load_terminal_binding(root)
    manifest_data = json.loads(manifest.read_text())
    summary = root / "summary.json"
    if not summary.is_file():
        raise ValueError("Feasibility root must contain summary.json.")
    summary_data = json.loads(summary.read_text())

    if require_passing and terminal.stem != "DONE":
        raise ValueError("Selection record must bind a passing DONE root.")
    if manifest_data.get("protocol") != "phase8_sequence_feasibility":
        raise ValueError("Manifest protocol is not phase8_sequence_feasibility.")
    if summary_data.get("protocol") != "phase8_sequence_feasibility":
        raise ValueError("Summary protocol is not phase8_sequence_feasibility.")
    source_commit = validate_git_sha(manifest_data.get("source_commit"), "source_commit")
    validate_git_commit_exists(source_commit)
    historical_protocol = validate_historical_feasibility_identity(
        root,
        source_commit=source_commit,
        manifest_sha=manifest_sha,
        terminal_path=terminal,
    )
    if historical_protocol and require_passing:
        raise ValueError("Historical untied feasibility roots cannot be selected as passing current roots.")
    if not historical_protocol:
        expected_terminal_keys = set(CURRENT_TERMINAL_KEYS)
        if terminal.stem == "FAILED":
            expected_terminal_keys.add("error")
        if set(manifest_data) != CURRENT_MANIFEST_KEYS:
            raise ValueError("Current manifest must use the exact tied feasibility schema.")
        if set(summary_data) != CURRENT_SUMMARY_KEYS:
            raise ValueError("Current summary must use the exact tied feasibility schema.")
        if set(terminal_data) != expected_terminal_keys:
            raise ValueError("Current terminal must use the exact tied feasibility schema.")
    validate_root_manifest_lineage(root, manifest_data, context=context)
    validate_source_provenance(
        manifest_data.get("source_provenance"),
        source_commit,
        source_provenance_allowed_paths(manifest_data, context=context),
    )
    if summary_data.get("source", {}).get("commit") != source_commit:
        raise ValueError("Summary source commit does not match manifest source_commit.")
    if summary_data.get("source", {}).get("ignored_inputs") != manifest_data["source_provenance"]["ignored_inputs"]:
        raise ValueError("Summary source ignored_inputs do not match manifest source_provenance.")
    expected_configuration = HISTORICAL_FEASIBILITY_CONFIGURATION if historical_protocol else frozen_configuration()
    if manifest_data.get("configuration") != expected_configuration:
        raise ValueError("Manifest configuration does not match the frozen feasibility schema.")
    if summary_data.get("configuration") != expected_configuration:
        raise ValueError("Summary configuration does not match the frozen feasibility schema.")
    if not historical_protocol:
        for record_name, record in (("manifest", manifest_data), ("summary", summary_data), ("terminal", terminal_data)):
            if record.get("configuration") != expected_configuration:
                raise ValueError(f"Current {record_name} configuration mismatch.")
            if record.get("decision_diagnostic") != DECISION_DIAGNOSTIC_ROOT_BINDING:
                raise ValueError(f"Current {record_name} decision_diagnostic mismatch.")
            if record.get("record_hashes") != FEASIBILITY_RECORD_HASHES:
                raise ValueError(f"Current {record_name} record_hashes mismatch.")
            validate_current_deterministic_flags(record.get("deterministic_flags"))
        if manifest_data.get("environment") != FEASIBILITY_REQUIRED_RUNTIME_ENV:
            raise ValueError("Current manifest environment does not match the frozen cuda:0 A800 environment.")
        predecessor_paths = tuple(
            Path(require_canonical_path_string(binding.get("path"), "predecessor_root.path", ROOT_RE))
            for binding in manifest_data.get("predecessor_roots", [])
            if isinstance(binding, dict)
        )
        expected_command = feasibility_exact_command(
            root,
            predecessor_paths,
            Path(FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT),
        )
        for record_name, record in (("manifest", manifest_data), ("summary", summary_data), ("terminal", terminal_data)):
            if record.get("exact_command") != expected_command:
                raise ValueError(f"Current {record_name} exact_command mismatch.")
    elif any(key in terminal_data for key in ("configuration", "decision_diagnostic", "exact_command", "deterministic_flags", "record_hashes")):
        raise ValueError("Historical feasibility terminals must not be migrated to the current D2 terminal schema.")
    if manifest_data.get("terminal_status") != terminal.stem:
        raise ValueError("Manifest terminal_status does not match the terminal marker.")
    if summary_data.get("terminal_status") != terminal.stem:
        raise ValueError("Summary terminal_status does not match the terminal marker.")
    cells = manifest_data.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Manifest cells must be a JSON list.")
    if terminal_data.get("cells") != cells:
        raise ValueError("Terminal marker cells do not match manifest cells.")
    if summary_data.get("cells") != cells:
        raise ValueError("Summary cells do not match manifest cells.")
    validate_summary_aggregates(summary_data, cells, terminal.stem, manifest_data.get("failure"))
    if require_passing:
        validate_cell_counts(cells)
    else:
        validate_cell_artifact_schema(cells, require_pass=False, historical=historical_protocol)
    inventory_paths = validate_root_file_inventory(root, manifest_data)
    if "summary.json" not in inventory_paths:
        raise ValueError("Manifest file_inventory must bind summary.json.")
    for cell in cells:
        generations_path = require_canonical_relative_path(cell["generations_path"], "generations_path")
        checkpoint_path = require_canonical_relative_path(cell["checkpoint_path"], "checkpoint_path")
        if generations_path not in inventory_paths:
            raise ValueError("Manifest file_inventory must bind every generations artifact.")
        if checkpoint_path not in inventory_paths:
            raise ValueError("Manifest file_inventory must bind every checkpoint artifact.")
        if not (root / generations_path).is_file():
            raise ValueError("Feasibility cell generations artifact is missing.")
        if not (root / checkpoint_path).is_file():
            raise ValueError("Feasibility cell checkpoint artifact is missing.")
        if historical_protocol:
            generation_rows = validate_historical_generation_artifact(root / generations_path, cell)
        else:
            generation_rows = validate_generation_artifact(root / generations_path, cell)
        validate_checkpoint_artifact(root / checkpoint_path, cell)
        if require_passing and not historical_protocol:
            validate_checkpoint_replays_generations(root / checkpoint_path, cell, generation_rows)


def validate_root_manifest_lineage(
    root: Path,
    manifest_data: dict[str, object],
    *,
    context: FeasibilityValidationContext,
) -> None:
    predecessor_root_bindings, predecessor_root_numbers = require_root_binding_map(manifest_data.get("predecessor_roots"))
    predecessor_selection_paths = require_selection_binding_paths(
        manifest_data.get("predecessor_selections"),
        context=context,
    )
    selected_root_numbers = [
        feasibility_root_number(Path(str(validate_selection_record(path, context=context)["selected_root"])))
        for path in predecessor_selection_paths
    ]
    root_number = feasibility_root_number(root)
    observed_numbers = sorted({*predecessor_root_numbers, *selected_root_numbers})
    expected_numbers = list(range(1, root_number))
    if observed_numbers != expected_numbers:
        raise ValueError(
            "Manifest predecessor_roots must be complete and continuous before its feasibility root; "
            f"expected {expected_numbers!r}, got {observed_numbers!r}."
        )
    if predecessor_root_numbers != observed_numbers:
        raise ValueError("Manifest predecessor_roots must explicitly bind the complete root lineage closure.")


def require_selection_binding_paths(
    bindings: object,
    *,
    context: FeasibilityValidationContext,
) -> tuple[Path, ...]:
    if not isinstance(bindings, list):
        raise ValueError("predecessor_selections must be a JSON list.")
    paths: list[Path] = []
    seen_path_keys: set[str] = set()
    selection_numbers: list[int] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("Predecessor selection binding must be a JSON object.")
        path = Path(require_canonical_path_string(binding.get("path"), "predecessor_selection.path", SELECTION_RE))
        require_artifact_location(path, "predecessor_selection.path", SELECTION_RE)
        path_key = str(path.resolve())
        if path_key in seen_path_keys:
            raise ValueError("predecessor_selections must not contain duplicate canonical paths.")
        seen_path_keys.add(path_key)
        number = selection_record_number(path)
        if selection_numbers and number <= selection_numbers[-1]:
            raise ValueError("predecessor_selections must be in strictly ascending selection-number order.")
        selection_numbers.append(number)
        expected_sha = require_exact_str(binding.get("sha256"), "predecessor_selection.sha256")
        if file_sha256(path) != expected_sha:
            raise ValueError("Predecessor selection checksum mismatch.")
        validate_selection_record(path, context=context)
        paths.append(path)
    return tuple(paths)


def validate_cell_artifact_schema(cells: object, *, require_pass: bool, historical: bool = False) -> None:
    if not isinstance(cells, list):
        raise ValueError("Feasibility cells must be a JSON list.")
    if require_pass:
        validate_cell_counts(cells)
        return
    if not historical:
        validate_current_feasibility_cells(cells, require_complete=False, require_all_pass=False)
        return
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ValueError("Feasibility cell entries must be JSON objects.")
        expected_cell_keys = HISTORICAL_CELL_KEYS if historical else CURRENT_CELL_KEYS
        if set(cell) != expected_cell_keys:
            raise ValueError("Feasibility cell entries must use the exact protocol schema.")
        require_exact_str(cell.get("family"), f"cells[{index}].family")
        require_exact_str(cell.get("model_size"), f"cells[{index}].model_size")
        require_exact_int(cell.get("seed"), f"cells[{index}].seed")
        require_exact_int(cell.get("eval_count"), f"cells[{index}].eval_count")
        require_exact_int(cell.get("exact_matches"), f"cells[{index}].exact_matches")
        require_exact_bool(cell.get("passed"), f"cells[{index}].passed")
        model_size = require_exact_str(cell.get("model_size"), f"cells[{index}].model_size")
        parameter_count = require_exact_int(cell.get("parameter_count"), f"cells[{index}].parameter_count")
        expected_counts = HISTORICAL_PARAMETER_COUNTS if historical else FROZEN_PARAMETER_COUNTS
        if model_size in expected_counts and parameter_count != expected_counts[model_size]:
            raise ValueError("Feasibility cell parameter_count does not match its protocol revision.")
        if historical:
            if "embedding_weight_tying" in cell or "model_protocol_revision" in cell:
                raise ValueError("Historical feasibility cells must not be migrated to tied revision fields.")
        else:
            if cell.get("embedding_weight_tying") is not True:
                raise ValueError("Current feasibility cells must bind embedding_weight_tying=true.")
            if cell.get("model_protocol_revision") != TIED_MODEL_PROTOCOL_REVISION:
                raise ValueError("Current feasibility cells must bind model_protocol_revision phase8_tied_io_v1.")
        require_canonical_relative_path(cell.get("generations_path"), f"cells[{index}].generations_path")
        require_canonical_relative_path(cell.get("checkpoint_path"), f"cells[{index}].checkpoint_path")


def validate_root_file_inventory(root: Path, manifest_data: dict[str, object]) -> set[str]:
    file_inventory = manifest_data.get("file_inventory")
    if not isinstance(file_inventory, list):
        raise ValueError("Manifest must contain a file_inventory list.")
    seen_paths: set[str] = set()
    root_resolved = root.resolve()
    for index, row in enumerate(file_inventory):
        if not isinstance(row, dict):
            raise ValueError("Manifest file_inventory entries must be JSON objects.")
        rel_path = require_canonical_relative_path(row.get("path"), f"file_inventory[{index}].path")
        if rel_path in seen_paths:
            raise ValueError("Manifest file_inventory must not contain duplicate paths.")
        seen_paths.add(rel_path)
        raw_candidate = root / rel_path
        if raw_candidate.is_symlink():
            raise ValueError("Manifest file_inventory must not bind symlink files.")
        candidate = raw_candidate.resolve()
        if root_resolved not in candidate.parents:
            raise ValueError("Manifest file_inventory path escapes its root.")
        if not candidate.is_file():
            raise ValueError("Manifest file_inventory path does not exist as a file.")
        expected_sha = require_exact_str(row.get("sha256"), f"file_inventory[{index}].sha256")
        expected_bytes = row.get("bytes")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ValueError(f"file_inventory[{index}].bytes must be a non-negative integer.")
        if candidate.stat().st_size != expected_bytes:
            raise ValueError("Manifest file_inventory byte count mismatch.")
        if file_sha256(candidate) != expected_sha:
            raise ValueError("Manifest file_inventory checksum mismatch.")
    actual_inventory = inventory(root)
    if sorted(file_inventory, key=lambda row: row["path"]) != actual_inventory:
        raise ValueError("Manifest file_inventory does not exactly match root files.")
    return seen_paths


def validate_summary_aggregates(
    summary_data: dict[str, object],
    cells: Sequence[dict[str, object]],
    terminal_status: str,
    failure: object,
) -> None:
    if summary_data.get("failure") != failure:
        raise ValueError("Summary failure does not match manifest failure.")
    if terminal_status == "DONE" and failure is not None:
        raise ValueError("Passing DONE roots must not record a failure string.")
    summary = summary_data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Summary aggregate must be a JSON object.")
    passed_cells = sum(1 for cell in cells if cell.get("passed") is True)
    expected = {
        "total_cells": len(cells),
        "passed_cells": passed_cells,
        "failed_cells": len(cells) - passed_cells,
        "all_cells_passed": len(cells) == len(FAMILIES) * len(MODEL_SIZES) * len(SEEDS) and passed_cells == len(cells),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"Summary aggregate {key} does not match manifest cells.")


def validate_generation_artifact(path: Path, cell: dict[str, object]) -> list[dict[str, object]]:
    family = require_exact_str(cell.get("family"), "family")
    if family not in FAMILIES:
        raise ValueError(f"Unexpected feasibility generation artifact family: {family!r}.")
    eval_records = grouped_records()[family]["eval"]

    def current_record_for_row(row_number: int) -> FeasibilityRecord:
        if row_number >= len(eval_records):
            raise ValueError("Generation artifact contains more rows than the frozen evaluation split.")
        return eval_records[row_number]

    return validate_generation_rows(
        path,
        cell,
        current_record_for_row=current_record_for_row,
    )


def validate_historical_generation_artifact(path: Path, cell: dict[str, object]) -> list[dict[str, object]]:
    family = require_exact_str(cell.get("family"), "family")
    if family not in FAMILIES:
        raise ValueError(f"Unexpected feasibility generation artifact family: {family!r}.")
    return validate_generation_rows(path, cell, current_record_for_row=None)


def validate_generation_rows(
    path: Path,
    cell: dict[str, object],
    *,
    current_record_for_row: Callable[[int], FeasibilityRecord] | None,
) -> list[dict[str, object]]:
    family = require_exact_str(cell.get("family"), "family")
    expected_count = require_exact_int(cell.get("eval_count"), "eval_count")
    expected_matches = require_exact_int(cell.get("exact_matches"), "exact_matches")
    if expected_count < 0:
        raise ValueError("Generation artifact eval_count must be non-negative.")
    if not 0 <= expected_matches <= expected_count:
        raise ValueError("Generation artifact exact_matches must satisfy 0 <= exact_matches <= eval_count.")
    rows: list[dict[str, object]] = []
    tokenizer = ByteTokenizer()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"Generation artifact contains a blank row at line {line_number}.")
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError("Generation artifact rows must be JSON objects.")
            row_family = require_exact_str(row.get("family"), "generation.family")
            row_index = require_exact_int(row.get("index"), "generation.index")
            template_id = require_exact_str(row.get("template_id"), "generation.template_id")
            operand_id = require_exact_str(row.get("operand_id"), "generation.operand_id")
            prompt = require_exact_str(row.get("prompt"), "generation.prompt")
            expected = require_exact_str(row.get("expected"), "generation.expected")
            exact_match = require_exact_bool(row.get("exact_match"), "generation.exact_match")
            raw_token_ids = row.get("raw_token_ids")
            if not isinstance(raw_token_ids, list) or not all(type(token) is int for token in raw_token_ids):
                raise ValueError("Generation artifact rows must retain raw_token_ids as JSON integers.")
            if row_family != family:
                raise ValueError("Generation artifact row family does not match its cell family.")
            if row_index != len(rows):
                raise ValueError("Generation artifact row indices must be deterministic and ordered from zero.")
            expected_prefix = list(tokenizer.encode_evaluation_prefix(prompt))
            if raw_token_ids[: len(expected_prefix)] != expected_prefix:
                raise ValueError("Generation artifact raw_token_ids prefix does not exactly encode the retained prompt.")
            if current_record_for_row is not None:
                record = current_record_for_row(len(rows))
                if row_family != record.family:
                    raise ValueError("Generation artifact row family does not match the frozen evaluation record.")
                if row_index != record.index:
                    raise ValueError("Generation artifact row index does not match the frozen evaluation record.")
                if template_id != record.template_id:
                    raise ValueError("Generation artifact row template_id does not match the frozen evaluation record.")
                if operand_id != record.operand_id:
                    raise ValueError("Generation artifact row operand_id does not match the frozen evaluation record.")
                if prompt != record.prompt:
                    raise ValueError("Generation artifact row prompt does not match the frozen evaluation record.")
                if expected != record.answer:
                    raise ValueError("Generation artifact row expected answer does not match the frozen evaluation record.")
            elif len(rows) >= expected_count:
                raise ValueError("Generation artifact contains more rows than cell eval_count.")
            decoded_error = None
            try:
                decoded = tokenizer.decode_generated_response(raw_token_ids)
            except (UnicodeDecodeError, ValueError) as exc:
                decoded = None
                decoded_error = f"{type(exc).__name__}: {exc}"
            invalid_generation = row.get("invalid_generation")
            if type(invalid_generation) is not bool:
                raise ValueError("Generation artifact invalid_generation values must be JSON booleans.")
            if invalid_generation != (decoded_error is not None):
                raise ValueError("Generation artifact invalid_generation does not match raw_token_ids decoding.")
            if decoded_error is not None:
                if row.get("generated") is not None:
                    raise ValueError("Invalid generation rows must record generated as null.")
                if row.get("generation_error") != decoded_error:
                    raise ValueError("Invalid generation rows must retain the exact generation_error string.")
            else:
                generated = require_exact_str(row.get("generated"), "generation.generated")
                if row.get("generated") != decoded:
                    raise ValueError("Generation artifact generated text does not match raw_token_ids decoding.")
                if row.get("generation_error") is not None:
                    raise ValueError("Valid generation rows must record generation_error as null.")
                if exact_match != (generated == expected):
                    raise ValueError("Generation artifact exact_match does not match generated-vs-expected semantics.")
            if decoded_error is not None and exact_match is not False:
                raise ValueError("Generation artifact exact_match does not match generated-vs-expected semantics.")
            rows.append(row)
    if len(rows) != expected_count:
        raise ValueError("Generation artifact row count does not match cell eval_count.")
    actual_matches = sum(1 for row in rows if row["exact_match"] is True)
    if actual_matches != expected_matches:
        raise ValueError("Generation artifact exact_match count does not match cell exact_matches.")
    return rows


def validate_checkpoint_artifact(path: Path, cell: dict[str, object]) -> None:
    family = require_exact_str(cell.get("family"), "family")
    model_size = require_exact_str(cell.get("model_size"), "model_size")
    seed = require_exact_int(cell.get("seed"), "seed")
    parameter_count = require_exact_int(cell.get("parameter_count"), "parameter_count")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Checkpoint artifact is not a loadable PyTorch checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint artifact must be a JSON-like mapping.")
    config = checkpoint.get("config")
    historical_checkpoint = isinstance(config, dict) and "embedding_weight_tying" not in config and "model_protocol_revision" not in config
    if historical_checkpoint:
        validate_historical_checkpoint_allowlist(path)
        expected_model = build_historical_model(model_size)
        if config != legacy_serialized_config(model_size):
            raise ValueError("Historical checkpoint config does not match the exact pre-D2 schema.")
        if checkpoint.get("parameter_count") != parameter_count or parameter_count != HISTORICAL_PARAMETER_COUNTS[model_size]:
            raise ValueError("Historical checkpoint parameter_count mismatch.")
    else:
        if cell.get("embedding_weight_tying") is not True:
            raise ValueError("Current checkpoint cell must bind embedding_weight_tying=true.")
        if cell.get("model_protocol_revision") != TIED_MODEL_PROTOCOL_REVISION:
            raise ValueError("Current checkpoint cell must bind phase8_tied_io_v1.")
        tied_config = validate_tied_checkpoint_payload(checkpoint)
        if tied_config.name != model_size:
            raise ValueError("Current checkpoint config name does not match the cell model_size.")
        expected_model = build_model(model_size)
        if checkpoint.get("config") != expected_model.config.__dict__:
            raise ValueError("Current checkpoint config does not match the frozen tied model configuration.")
        if checkpoint.get("parameter_count") != parameter_count or parameter_count != expected_model.parameter_count:
            raise ValueError("Current checkpoint parameter_count does not match the cell and tied model configuration.")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a mapping.")
    expected_metadata = {
        "family": family,
        "model_size": model_size,
        "seed": seed,
        "training_steps": TRAINING_STEPS,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"Checkpoint metadata {key} does not match the cell.")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint model_state_dict must be a mapping.")
    expected_state = expected_model.state_dict()
    if set(state) != set(expected_state):
        raise ValueError("Checkpoint state_dict keys do not match the frozen model.")
    for key, expected_tensor in expected_state.items():
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Checkpoint state_dict values must be tensors.")
        if tensor.dtype != expected_tensor.dtype or tuple(tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError("Checkpoint state_dict tensor schema does not match the frozen model.")


def validate_checkpoint_replays_generations(
    path: Path,
    cell: dict[str, object],
    generation_rows: Sequence[dict[str, object]],
) -> None:
    family = require_exact_str(cell.get("family"), "family")
    model_size = require_exact_str(cell.get("model_size"), "model_size")
    model = load_model_from_checkpoint(str(path), map_location="cpu")
    if model.config.name != model_size:
        raise ValueError("Checkpoint replay model_size mismatch.")
    device = feasibility_replay_device()
    model.to(device)
    model.eval()
    tokenizer = ByteTokenizer()
    records = grouped_records()[family]["eval"]
    with torch.no_grad():
        for record, row in zip(records, generation_rows, strict=True):
            prefix = tokenizer.encode_evaluation_prefix(record.prompt)
            prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
            generated = model.greedy_decode(prefix_tensor)
            full_ids = [int(token) for token in generated[0].detach().cpu().tolist()]
            if row.get("raw_token_ids") != full_ids:
                raise ValueError("Checkpoint replay does not reproduce retained generation raw_token_ids.")


def selection_binding(path: Path) -> dict[str, object]:
    require_canonical_path_string(str(path), "predecessor_selection.path", SELECTION_RE)
    require_artifact_location(path, "predecessor_selection.path", SELECTION_RE)
    return {"path": str(path), "sha256": file_sha256(path)}


def root_binding_key(binding: dict[str, object]) -> tuple[str, str, str, str]:
    path = Path(require_canonical_path_string(binding.get("path"), "predecessor_root.path", ROOT_RE))
    terminal_state = require_exact_str(binding.get("terminal_state"), "predecessor_root.terminal_state")
    terminal_sha = require_exact_str(binding.get("terminal_sha256"), "predecessor_root.terminal_sha256")
    manifest_sha = require_exact_str(binding.get("manifest_sha256"), "predecessor_root.manifest_sha256")
    return str(path.resolve()), terminal_state, terminal_sha, manifest_sha


def require_root_binding_map(bindings: object) -> tuple[dict[str, tuple[str, str, str]], list[int]]:
    if not isinstance(bindings, list):
        raise ValueError("predecessor_roots must be a JSON list.")
    binding_map: dict[str, tuple[str, str, str]] = {}
    root_numbers: list[int] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("Predecessor root binding must be a JSON object.")
        path_key, terminal_state, terminal_sha, manifest_sha = root_binding_key(binding)
        if path_key in binding_map:
            raise ValueError("predecessor_roots must not contain duplicate root bindings.")
        predecessor_path = Path(require_canonical_path_string(binding.get("path"), "predecessor_root.path", ROOT_RE))
        require_artifact_location(predecessor_path, "predecessor_root.path", ROOT_RE)
        root_number = feasibility_root_number(predecessor_path)
        if root_numbers and root_number <= root_numbers[-1]:
            raise ValueError("predecessor_roots must be in strictly ascending root-number order.")
        root_numbers.append(root_number)
        if terminal_state not in {"DONE", "FAILED"}:
            raise ValueError("Predecessor terminal_state must be DONE or FAILED.")
        terminal_path, terminal_data, _manifest_path, actual_manifest_sha = load_terminal_binding(predecessor_path)
        if terminal_path.stem != terminal_state:
            raise ValueError("Predecessor terminal status does not match its binding.")
        if file_sha256(terminal_path) != terminal_sha:
            raise ValueError("Predecessor terminal checksum mismatch.")
        if actual_manifest_sha != manifest_sha:
            raise ValueError("Predecessor manifest checksum mismatch.")
        if terminal_data.get("manifest_sha256") != manifest_sha:
            raise ValueError("Predecessor terminal does not bind the predecessor manifest checksum.")
        binding_map[path_key] = (terminal_state, terminal_sha, manifest_sha)
    return binding_map, root_numbers


def add_root_binding(
    binding_map: dict[str, tuple[str, str, str]],
    binding: dict[str, object],
) -> None:
    path_key, terminal_state, terminal_sha, manifest_sha = root_binding_key(binding)
    previous = binding_map.get(path_key)
    current = (terminal_state, terminal_sha, manifest_sha)
    if previous is not None and previous != current:
        raise ValueError("predecessor_roots contain inconsistent bindings for the same root.")
    binding_map[path_key] = current


def complete_predecessor_root_bindings(
    predecessor_roots: Sequence[Path],
    predecessor_selections: Sequence[Path],
) -> list[dict[str, object]]:
    context = FeasibilityValidationContext()
    completed: list[dict[str, object]] = []
    binding_map: dict[str, tuple[str, str, str]] = {}

    def add(binding: dict[str, object]) -> None:
        path_key, terminal_state, terminal_sha, manifest_sha = root_binding_key(binding)
        previous = binding_map.get(path_key)
        current = (terminal_state, terminal_sha, manifest_sha)
        if previous is not None:
            if previous != current:
                raise ValueError("predecessor_roots contain inconsistent bindings for the same root.")
            return
        binding_map[path_key] = current
        completed.append(binding)

    for root in predecessor_roots:
        add(terminal_binding(root, context=context))
    for selection_path in predecessor_selections:
        selection_data = validate_selection_record(selection_path, context=context)
        add(terminal_binding(Path(str(selection_data["selected_root"])), context=context))
        for predecessor in selection_data["predecessor_roots"]:
            add(predecessor)
    return sorted(completed, key=lambda binding: feasibility_root_number(Path(str(binding["path"]))))


def source_provenance_allowed_paths(
    manifest_data: dict[str, object],
    *,
    context: FeasibilityValidationContext | None = None,
) -> set[Path]:
    context = validation_context(context)
    predecessor_roots = manifest_data.get("predecessor_roots")
    predecessor_selections = manifest_data.get("predecessor_selections")
    if not isinstance(predecessor_roots, list) or not isinstance(predecessor_selections, list):
        raise ValueError("Manifest predecessor bindings must be JSON lists.")
    allowed = source_clean_allowed_paths(
        tuple(Path(require_canonical_path_string(binding.get("path"), "predecessor_root.path", ROOT_RE)) for binding in predecessor_roots),
        tuple(
            Path(require_canonical_path_string(binding.get("path"), "predecessor_selection.path", SELECTION_RE))
            for binding in predecessor_selections
        ),
        context=context,
    )
    decision = manifest_data.get("decision_diagnostic")
    if decision is not None:
        if decision != DECISION_DIAGNOSTIC_ROOT_BINDING:
            raise ValueError("Manifest decision_diagnostic binding mismatch.")
        allowed.update(shallow_decision_diagnostic_paths(Path(FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT)))
    return allowed


def predecessor_root_paths_from_selections(
    predecessor_selections: Sequence[Path],
    *,
    context: FeasibilityValidationContext | None = None,
) -> tuple[Path, ...]:
    context = validation_context(context)
    paths: list[Path] = []
    for selection_path in predecessor_selections:
        selection_data = validate_selection_record(selection_path, context=context)
        paths.append(Path(str(selection_data["selected_root"])))
        paths.extend(Path(str(predecessor["path"])) for predecessor in selection_data["predecessor_roots"])
    return tuple(paths)


def predecessor_selection_paths_from_selections(
    predecessor_selections: Sequence[Path],
    *,
    context: FeasibilityValidationContext | None = None,
) -> tuple[Path, ...]:
    context = validation_context(context)
    paths: list[Path] = []
    for selection_path in predecessor_selections:
        selection_data = validate_selection_record(selection_path, context=context)
        paths.append(selection_path)
        paths.extend(Path(str(predecessor["path"])) for predecessor in selection_data["predecessor_selections"])
    return tuple(paths)


def validate_selection_record(
    path: Path,
    *,
    context: FeasibilityValidationContext | None = None,
) -> dict[str, object]:
    context = validation_context(context)
    data, _root_binding_map = _validate_selection_record(path, seen=set(), context=context)
    return data


def _validate_selection_record(
    path: Path,
    *,
    seen: set[Path],
    context: FeasibilityValidationContext,
) -> tuple[dict[str, object], dict[str, tuple[str, str, str]]]:
    current_number = selection_record_number(path)
    require_canonical_path_string(str(path), "selection_record.path", SELECTION_RE)
    require_artifact_location(path, "selection_record.path", SELECTION_RE)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Selection record must be a regular file in the canonical artifact parent.")
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Selection-record lineage contains a cycle at {path}.")
    current_commit = current_source_commit()
    cache_key = (resolved, current_commit)
    cached = context.selection_records.get(cache_key)
    if cached is not None:
        return cached
    seen.add(resolved)
    data = json.loads(path.read_text())
    required = {
        "selected_root",
        "selected_manifest_sha256",
        "source_commit",
        "configuration",
        "per_cell_counts",
        "pass_decision",
        "independent_review_verdict",
        "predecessor_roots",
        "predecessor_selections",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"Selection record is missing required fields: {sorted(missing)!r}.")
    selected_root = require_canonical_path_string(data["selected_root"], "selected_root", ROOT_RE)
    require_artifact_location(Path(selected_root), "selected_root", ROOT_RE)
    selected_manifest_sha = require_exact_str(data["selected_manifest_sha256"], "selected_manifest_sha256")
    source_commit = validate_git_sha(data["source_commit"], "source_commit")
    root = Path(selected_root)
    validate_feasibility_root_artifacts(root, require_passing=True, context=context)
    terminal, terminal_data, manifest, manifest_sha = load_terminal_binding(root)
    if manifest_sha != selected_manifest_sha:
        raise ValueError("Selection record manifest checksum does not match selected root.")
    if terminal.stem != "DONE":
        raise ValueError("Selection record must bind a passing DONE root.")
    manifest_data = json.loads(manifest.read_text())
    if terminal_data.get("manifest_sha256") != selected_manifest_sha:
        raise ValueError("Selected DONE terminal does not bind the selected manifest checksum.")
    if terminal_data.get("status") != "DONE":
        raise ValueError("Selected terminal status must be DONE.")
    if manifest_data.get("source_commit") != source_commit:
        raise ValueError("Selection record source_commit does not match selected manifest.")
    if data["configuration"] != frozen_configuration():
        raise ValueError("Selection record configuration does not match the frozen feasibility schema.")
    if manifest_data.get("configuration") != data["configuration"]:
        raise ValueError("Selection record configuration does not match selected manifest.")
    cells = data["per_cell_counts"]
    validate_cell_counts(cells)
    if manifest_data.get("cells") != cells:
        raise ValueError("Selection record per_cell_counts do not match selected manifest cells.")
    if terminal_data.get("cells") != cells:
        raise ValueError("Selection record per_cell_counts do not match selected DONE cells.")
    if manifest_data.get("predecessor_roots") != data["predecessor_roots"]:
        raise ValueError("Selection record predecessor_roots do not match selected manifest.")
    if manifest_data.get("predecessor_selections") != data["predecessor_selections"]:
        raise ValueError("Selection record predecessor_selections do not match selected manifest.")
    pass_decision = data["pass_decision"]
    if type(pass_decision) is not bool or pass_decision is not True:
        raise ValueError("Selection record pass_decision must be true for a selected root.")
    verdict = require_exact_str(data["independent_review_verdict"], "independent_review_verdict")
    if verdict != ACCEPTED_INDEPENDENT_REVIEW_VERDICT:
        raise ValueError("Selection record independent_review_verdict must be exact JSON string 'ACCEPT'.")
    predecessor_root_bindings, predecessor_root_numbers = require_root_binding_map(data["predecessor_roots"])
    expected_root_bindings = dict(predecessor_root_bindings)
    predecessor_selection_numbers: list[int] = []
    predecessor_selection_bindings: list[tuple[Path, str]] = []
    predecessor_selection_path_keys: set[str] = set()
    for predecessor in data["predecessor_selections"]:
        if not isinstance(predecessor, dict):
            raise ValueError("Predecessor selection binding must be a JSON object.")
        predecessor_path = Path(require_canonical_path_string(predecessor.get("path"), "predecessor_selection.path", SELECTION_RE))
        require_artifact_location(predecessor_path, "predecessor_selection.path", SELECTION_RE)
        predecessor_path_key = str(predecessor_path.resolve())
        if predecessor_path_key in predecessor_selection_path_keys:
            raise ValueError("predecessor_selections must not contain duplicate canonical paths.")
        predecessor_selection_path_keys.add(predecessor_path_key)
        predecessor_number = selection_record_number(predecessor_path)
        if predecessor_selection_numbers and predecessor_number <= predecessor_selection_numbers[-1]:
            raise ValueError("predecessor_selections must be in strictly ascending selection-number order.")
        predecessor_selection_numbers.append(predecessor_number)
        predecessor_sha = require_exact_str(predecessor.get("sha256"), "predecessor_selection.sha256")
        predecessor_selection_bindings.append((predecessor_path, predecessor_sha))
    lineage_bindings = {str(predecessor_path.resolve()): sha for predecessor_path, sha in predecessor_selection_bindings}
    for predecessor_path, predecessor_sha in predecessor_selection_bindings:
        if file_sha256(predecessor_path) != predecessor_sha:
            raise ValueError("Predecessor selection checksum mismatch.")
        predecessor_data, predecessor_validated_roots = _validate_selection_record(
            predecessor_path,
            seen=seen,
            context=context,
        )
        predecessor_selected_root = Path(str(predecessor_data["selected_root"]))
        predecessor_root_numbers.append(feasibility_root_number(predecessor_selected_root))
        add_root_binding(expected_root_bindings, terminal_binding(predecessor_selected_root, context=context))
        for predecessor_root_key, predecessor_root_binding in predecessor_validated_roots.items():
            terminal_state, terminal_sha, manifest_sha = predecessor_root_binding
            add_root_binding(
                expected_root_bindings,
                {
                    "path": predecessor_root_key,
                    "terminal_state": terminal_state,
                    "terminal_sha256": terminal_sha,
                    "manifest_sha256": manifest_sha,
                },
            )
        for transitive in predecessor_data["predecessor_selections"]:
            transitive_path = str(Path(transitive["path"]).resolve())
            if lineage_bindings.get(transitive_path) != transitive["sha256"]:
                raise ValueError("Selection record must bind transitive predecessor selection lineage.")
    if predecessor_root_bindings != expected_root_bindings:
        raise ValueError("Selection record predecessor_roots must include the complete root lineage closure.")
    selected_root_number = feasibility_root_number(root)
    if sorted(set(predecessor_root_numbers)) != list(range(1, selected_root_number)):
        raise ValueError(
            "Selection record predecessor_roots must be complete and continuous before selected_root."
        )
    expected_selection_number = max(predecessor_selection_numbers, default=0) + 1
    if current_number != expected_selection_number:
        raise ValueError(
            f"Selection record must use next numbered selection feasibility_selection_{expected_selection_number:03d}.json; "
            f"got {path.name!r}."
        )
    result = (data, predecessor_root_bindings)
    context.selection_records[cache_key] = result
    seen.remove(resolved)
    return result


def selection_record_number(path: Path) -> int:
    match = SELECTION_RE.match(path.name)
    if match is None:
        raise ValueError("Selection record basename must be immutable numbered form feasibility_selection_NNN.json.")
    return int(match.group(1))


def require_canonical_path_string(value: object, field_name: str, basename_pattern: re.Pattern[str]) -> str:
    raw = require_exact_str(value, field_name)
    if not raw:
        raise ValueError(f"{field_name} must be a non-empty canonical path string.")
    if "\\" in raw or "//" in raw:
        raise ValueError(f"{field_name} must use canonical path spelling.")
    normalized = PurePosixPath(raw).as_posix()
    if raw != normalized:
        raise ValueError(f"{field_name} must use canonical path spelling.")
    if any(part in {".", ".."} for part in PurePosixPath(raw).parts):
        raise ValueError(f"{field_name} must use canonical path spelling.")
    path = Path(raw)
    if basename_pattern.match(path.name) is None:
        if basename_pattern is ROOT_RE:
            raise ValueError("Feasibility root basename must be immutable numbered form feasibility_NNN.")
        raise ValueError("Selection record basename must be immutable numbered form feasibility_selection_NNN.json.")
    return raw


def require_canonical_relative_path(value: object, field_name: str) -> str:
    raw = require_exact_str(value, field_name)
    if not raw:
        raise ValueError(f"{field_name} must be a non-empty canonical relative path string.")
    if "\\" in raw or "//" in raw:
        raise ValueError(f"{field_name} must use canonical relative path spelling.")
    path = PurePosixPath(raw)
    if path.is_absolute():
        raise ValueError(f"{field_name} must be relative.")
    if raw != path.as_posix() or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must use canonical relative path spelling.")
    return raw


def inventory(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Feasibility file_inventory must not include symlink paths.")
        if path.is_file() and path.name not in {"manifest.json", "DONE.json", "FAILED.json"}:
            rows.append({"path": str(path.relative_to(root)), "sha256": file_sha256(path), "bytes": path.stat().st_size})
    return rows


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp, path)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(args: Sequence[str]) -> str:
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, cwd=REPO_ROOT).stdout.strip()


def current_source_commit() -> str:
    return validate_git_sha(git_output(["git", "rev-parse", "HEAD"]), "current_source_commit")


def unchecked_source_snapshot() -> SourceSnapshot:
    commit = current_source_commit()
    return SourceSnapshot(commit=commit, status_lines=(), ignored_inputs=ignored_source_inputs())


def validate_git_sha(value: object, field_name: str) -> str:
    text = require_exact_str(value, field_name)
    if GIT_SHA_RE.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a full 40-character lowercase Git SHA.")
    return text


def validate_git_commit_exists(commit: str) -> None:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("Manifest source_commit does not exist as a commit in the CapKnow repository.")


def validate_source_provenance(value: object, source_commit: str, allowed_paths: set[Path]) -> None:
    if not isinstance(value, dict):
        raise ValueError("Manifest source_provenance must be a JSON object.")
    if value.get("commit") != source_commit:
        raise ValueError("Manifest source_provenance commit does not match source_commit.")
    status_lines = value.get("status_lines")
    if not isinstance(status_lines, list) or not all(isinstance(line, str) for line in status_lines):
        raise ValueError("Manifest source_provenance status_lines must be a JSON string list.")
    for line in status_lines:
        if not line.startswith("?? "):
            raise ValueError("Manifest source_provenance must not declare tracked or staged source changes.")
        rel = line[3:]
        rel_path = Path(rel)
        candidate = (rel_path if rel_path.is_absolute() else REPO_ROOT / rel_path).resolve()
        if candidate not in allowed_paths:
            raise ValueError("Manifest source_provenance declares an unbound untracked input.")
    ignored_inputs = value.get("ignored_inputs")
    if not isinstance(ignored_inputs, list):
        raise ValueError("Manifest source_provenance ignored_inputs must be a JSON list.")
    if ignored_inputs:
        raise ValueError("Manifest source_provenance must not contain ignored executable inputs.")
    for index, row in enumerate(ignored_inputs):
        if not isinstance(row, dict):
            raise ValueError("Manifest source_provenance ignored_inputs entries must be JSON objects.")
        require_canonical_relative_path(row.get("path"), f"source_provenance.ignored_inputs[{index}].path")
        require_exact_str(row.get("sha256"), f"source_provenance.ignored_inputs[{index}].sha256")
        bytes_value = row.get("bytes")
        if type(bytes_value) is not int or bytes_value < 0:
            raise ValueError(f"source_provenance.ignored_inputs[{index}].bytes must be a non-negative integer.")


def require_exact_mapping(value: object, expected_keys: frozenset[str], field_name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != set(expected_keys):
        raise ValueError(f"{field_name} must use the exact closed schema.")
    return value


def require_finite_json_number(value: object, field_name: str) -> int | float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite JSON number.")
    return value


def reject_postmortem_negative_zero(value: object, field_name: str) -> None:
    if isinstance(value, float):
        if value == 0.0 and math.copysign(1.0, value) < 0:
            raise ValueError(f"{field_name} must not contain negative zero.")
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            reject_postmortem_negative_zero(nested, f"{field_name}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            reject_postmortem_negative_zero(nested, f"{field_name}[{index}]")


def postmortem_json_bytes(value: object) -> bytes:
    reject_postmortem_negative_zero(value, "postmortem.json")
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def postmortem_jsonl_bytes(rows: Iterable[dict[str, object]]) -> bytes:
    lines: list[str] = []
    for index, row in enumerate(rows):
        reject_postmortem_negative_zero(row, f"postmortem.jsonl[{index}]")
        lines.append(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def require_decimal_histogram(value: object, field_name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object histogram.")
    result: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or re.fullmatch(r"(0|[1-9][0-9]*)", key) is None:
            raise ValueError(f"{field_name} keys must be canonical non-negative decimal integers.")
        if type(count) is not int or count <= 0:
            raise ValueError(f"{field_name} values must be positive JSON integers.")
        result[key] = count
    return result


def sparse_decimal_histogram(values: Iterable[object]) -> dict[str, int]:
    observed: list[int] = []
    for value in values:
        if value is None:
            continue
        if type(value) is not int or value < 0:
            raise ValueError("Postmortem sparse histogram values must be non-negative JSON integers or null.")
        observed.append(value)
    return {str(key): count for key, count in sorted(Counter(observed).items()) if count > 0}


def require_postmortem_root_path_string(value: object, field_name: str, *, temp: bool = False) -> str:
    raw = require_exact_str(value, field_name)
    if not raw:
        raise ValueError(f"{field_name} must be a non-empty canonical path string.")
    if "\\" in raw or "//" in raw:
        raise ValueError(f"{field_name} must use canonical path spelling.")
    normalized = PurePosixPath(raw).as_posix()
    if raw != normalized or any(part in {".", ".."} for part in PurePosixPath(raw).parts):
        raise ValueError(f"{field_name} must use canonical path spelling.")
    pattern = POSTMORTEM_TEMP_ROOT_RE if temp else POSTMORTEM_ROOT_RE
    match = pattern.match(Path(raw).name)
    if match is None:
        raise ValueError("Postmortem root basename must be immutable numbered form feasibility_postmortem_NNN.")
    if int(match.group(1)) != 1:
        raise ValueError("D3 postmortem forbids retry or alternate roots; only feasibility_postmortem_001 is authorized.")
    return raw


def require_postmortem_artifact_location(path: Path, field_name: str, *, temp: bool = False) -> None:
    require_postmortem_root_path_string(str(path), field_name, temp=temp)
    parent_path = ARTIFACT_PARENT if ARTIFACT_PARENT.is_absolute() else REPO_ROOT / ARTIFACT_PARENT
    artifact_path = path if path.is_absolute() else REPO_ROOT / path
    reject_existing_symlink_component(parent_path, f"{field_name} artifact parent")
    reject_existing_symlink_component(artifact_path, field_name)
    parent = parent_path.resolve(strict=False)
    resolved = artifact_path.resolve(strict=False)
    if resolved.parent != parent:
        raise ValueError(f"{field_name} must be located directly under {ARTIFACT_PARENT.as_posix()}.")


def require_postmortem_repo_cwd() -> None:
    if Path.cwd().resolve(strict=False) != REPO_ROOT.resolve(strict=True):
        raise ValueError("postmortem-failure must execute with cwd exactly equal to the CapKnow repository root.")


def postmortem_expected_predecessor_bindings() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative_path in FEASIBILITY_REQUIRED_PREDECESSOR_ROOTS:
        expected = HISTORICAL_FEASIBILITY_ROOTS[Path(relative_path).name]
        rows.append(
            {
                "path": relative_path,
                "terminal_state": expected["terminal"].removesuffix(".json"),
                "terminal_sha256": expected["terminal_sha256"],
                "manifest_sha256": expected["manifest_sha256"],
            }
        )
    return rows


def validate_postmortem_current_cells_shallow(cells: object) -> tuple[dict[str, object], ...]:
    if not isinstance(cells, list):
        raise ValueError("Postmortem input cells must be a JSON list.")
    expected_sequence = current_feasibility_cell_identities()
    if len(cells) != len(expected_sequence):
        raise ValueError("Postmortem input must bind exactly 24 feasibility_005 cells.")
    parsed: list[dict[str, object]] = []
    for index, (cell, expected_identity) in enumerate(zip(cells, expected_sequence, strict=True)):
        if not isinstance(cell, dict) or set(cell) != CURRENT_CELL_KEYS:
            raise ValueError("Postmortem input cells must use the exact tied feasibility schema.")
        family = require_exact_str(cell.get("family"), f"cells[{index}].family")
        model_size = require_exact_str(cell.get("model_size"), f"cells[{index}].model_size")
        seed = require_exact_int(cell.get("seed"), f"cells[{index}].seed")
        if (family, model_size, seed) != expected_identity:
            raise ValueError("Postmortem input cells must retain the frozen 24-cell order.")
        eval_count = require_exact_int(cell.get("eval_count"), f"cells[{index}].eval_count")
        exact_matches = require_exact_int(cell.get("exact_matches"), f"cells[{index}].exact_matches")
        passed = require_exact_bool(cell.get("passed"), f"cells[{index}].passed")
        if eval_count != EVAL_RECORDS_PER_FAMILY:
            raise ValueError("Postmortem input cells must each bind exactly 64 eval rows.")
        if not 0 <= exact_matches <= EVAL_RECORDS_PER_FAMILY:
            raise ValueError("Postmortem input cell exact_matches out of range.")
        if passed is not (exact_matches >= PASS_THRESHOLD):
            raise ValueError("Postmortem input cell pass flag does not match the fixed 52/64 threshold.")
        if cell.get("embedding_weight_tying") is not True:
            raise ValueError("Postmortem input cells must bind embedding_weight_tying=true.")
        if cell.get("model_protocol_revision") != TIED_MODEL_PROTOCOL_REVISION:
            raise ValueError("Postmortem input cells must bind phase8_tied_io_v1.")
        parameter_count = require_exact_int(cell.get("parameter_count"), f"cells[{index}].parameter_count")
        if parameter_count != FROZEN_PARAMETER_COUNTS[model_size]:
            raise ValueError("Postmortem input cell parameter_count mismatch.")
        generations_path = require_canonical_relative_path(cell.get("generations_path"), f"cells[{index}].generations_path")
        checkpoint_path = require_canonical_relative_path(cell.get("checkpoint_path"), f"cells[{index}].checkpoint_path")
        expected_prefix = f"{family}__{model_size}__seed{seed}"
        if generations_path != f"{expected_prefix}/generations.jsonl":
            raise ValueError("Postmortem input generation path does not match the frozen cell identity.")
        if checkpoint_path != f"{expected_prefix}/checkpoint_step1500.pt":
            raise ValueError("Postmortem input checkpoint path does not match the frozen cell identity.")
        parsed.append(cell)
    return tuple(parsed)


def postmortem_exact_argv(
    input_root: Path,
    output_root: Path,
    accepted_proposal_commit: str,
    accepted_implementation_commit: str,
    *,
    verifier_source: str | None = None,
) -> list[str]:
    implementation_commit = validate_git_sha(accepted_implementation_commit, "postmortem.accepted_implementation_commit")
    source = postmortem_verifier_source_text() if verifier_source is None else verifier_source
    verifier_sha = sha256(source.encode("utf-8")).hexdigest()
    if verifier_sha != POSTMORTEM_VERIFIER_SHA256:
        raise ValueError("Postmortem verifier source bytes do not match the accepted SHA-256.")
    return [
        POSTMORTEM_EXECUTABLE,
        "-I",
        "-B",
        "-S",
        "-c",
        source,
        "--verifier-sha256",
        POSTMORTEM_VERIFIER_SHA256,
        "--accepted-implementation-commit",
        implementation_commit,
        "--runner-path",
        POSTMORTEM_RUNNER_PATH,
        "--",
        "postmortem-failure",
        "--device",
        FEASIBILITY_REQUIRED_DEVICE,
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
        "--accepted-proposal-commit",
        accepted_proposal_commit,
        "--accepted-implementation-commit",
        implementation_commit,
    ]


def postmortem_runner_argv(
    input_root: Path,
    output_root: Path,
    accepted_proposal_commit: str,
    accepted_implementation_commit: str,
) -> list[str]:
    full = postmortem_exact_argv(
        input_root,
        output_root,
        accepted_proposal_commit,
        accepted_implementation_commit,
        verifier_source=postmortem_verifier_source_from_orig_argv_or_file(),
    )
    return full[13:]


def postmortem_expected_sys_argv(
    input_root: Path,
    output_root: Path,
    accepted_proposal_commit: str,
    accepted_implementation_commit: str,
    *,
    verifier_source: str,
) -> list[str]:
    return ["-c", *postmortem_exact_argv(
        input_root,
        output_root,
        accepted_proposal_commit,
        accepted_implementation_commit,
        verifier_source=verifier_source,
    )[6:]]


def postmortem_exact_command(
    input_root: Path,
    output_root: Path,
    accepted_proposal_commit: str,
    accepted_implementation_commit: str,
    *,
    argv: Sequence[str] | None = None,
) -> dict[str, object]:
    exact_argv = list(argv) if argv is not None else postmortem_exact_argv(
        input_root,
        output_root,
        accepted_proposal_commit,
        accepted_implementation_commit,
    )
    validate_postmortem_exact_argv(
        exact_argv,
        input_root=input_root,
        output_root=output_root,
        accepted_proposal_commit=accepted_proposal_commit,
        accepted_implementation_commit=accepted_implementation_commit,
    )
    return {
        "executable": POSTMORTEM_EXECUTABLE,
        "cwd": POSTMORTEM_CWD,
        "environment": dict(POSTMORTEM_EXECVE_ENVIRONMENT),
        "argv": exact_argv,
    }


def postmortem_verifier_source_text() -> str:
    path = REPO_ROOT / POSTMORTEM_VERIFIER_PATH
    raw = path.read_bytes()
    if len(raw) != POSTMORTEM_VERIFIER_BYTE_COUNT:
        raise ValueError("Postmortem verifier source byte count mismatch.")
    if sha256(raw).hexdigest() != POSTMORTEM_VERIFIER_SHA256:
        raise ValueError("Postmortem verifier source SHA-256 mismatch.")
    return raw.decode("utf-8")


def postmortem_verifier_source_from_orig_argv_or_file() -> str:
    orig = getattr(sys, "orig_argv", None)
    if isinstance(orig, list) and len(orig) == 24 and all(isinstance(arg, str) for arg in orig):
        return orig[5]
    return postmortem_verifier_source_text()


def postmortem_kernel_argv() -> list[str]:
    cmdline = Path("/proc/self/cmdline")
    try:
        raw = cmdline.read_bytes()
    except OSError as exc:
        raise ValueError("postmortem-failure cannot authorize without Linux /proc/self/cmdline.") from exc
    if not raw or not raw.endswith(b"\0"):
        raise ValueError("postmortem-failure kernel argv is unavailable or malformed.")
    parts = raw[:-1].split(b"\0")
    if not parts or any(part == b"" for part in parts):
        raise ValueError("postmortem-failure kernel argv is malformed.")
    try:
        return [os.fsdecode(part) for part in parts]
    except UnicodeDecodeError as exc:
        raise ValueError("postmortem-failure kernel argv is not decodable.") from exc


def validate_postmortem_exact_argv(
    raw_argv: Sequence[str],
    *,
    input_root: Path,
    output_root: Path,
    accepted_proposal_commit: str,
    accepted_implementation_commit: str,
) -> None:
    argv = list(raw_argv)
    implementation_commit = validate_git_sha(accepted_implementation_commit, "postmortem.accepted_implementation_commit")
    if len(argv) != 24 or any(type(arg) is not str for arg in argv):
        raise ValueError("postmortem-failure exact execve argv must contain exactly 24 strings.")
    if argv[0:5] != [POSTMORTEM_EXECUTABLE, "-I", "-B", "-S", "-c"]:
        raise ValueError("postmortem-failure executable and isolated/no-bytecode/no-site flags are not exact.")
    source = argv[5]
    if sha256(source.encode("utf-8")).hexdigest() != POSTMORTEM_VERIFIER_SHA256:
        raise ValueError("postmortem-failure inline verifier source differs from the accepted SHA-256.")
    expected = [
        POSTMORTEM_EXECUTABLE,
        "-I",
        "-B",
        "-S",
        "-c",
        source,
        "--verifier-sha256",
        POSTMORTEM_VERIFIER_SHA256,
        "--accepted-implementation-commit",
        implementation_commit,
        "--runner-path",
        POSTMORTEM_RUNNER_PATH,
        "--",
        "postmortem-failure",
        "--device",
        FEASIBILITY_REQUIRED_DEVICE,
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
        "--accepted-proposal-commit",
        accepted_proposal_commit,
        "--accepted-implementation-commit",
        implementation_commit,
    ]
    if argv != expected:
        raise ValueError("postmortem-failure execve argv differs from the frozen 24-string command.")


def validate_postmortem_process_argv(
    raw_argv: Sequence[str],
    *,
    input_root: Path,
    output_root: Path,
    accepted_proposal_commit: str,
    accepted_implementation_commit: str,
) -> None:
    validate_postmortem_exact_argv(
        raw_argv,
        input_root=input_root,
        output_root=output_root,
        accepted_proposal_commit=accepted_proposal_commit,
        accepted_implementation_commit=accepted_implementation_commit,
    )
    sys_orig_argv = getattr(sys, "orig_argv", None)
    if list(raw_argv) != sys_orig_argv:
        raise ValueError("postmortem-failure /proc/self/cmdline must equal sys.orig_argv exactly.")
    expected_runner = [str(REPO_ROOT / POSTMORTEM_RUNNER_PATH), *list(raw_argv)[13:]]
    if sys.argv != expected_runner:
        raise ValueError("postmortem-failure runner sys.argv must be the verifier-installed frozen runner suffix.")


def require_postmortem_real_main_context() -> None:
    if __name__ != "__main__":
        raise ValueError("postmortem-failure must execute from this file's real __main__ process context.")


def postmortem_injected_value(name: str) -> object:
    if name not in globals():
        raise ValueError(f"postmortem-failure requires verifier-injected global {name}.")
    return globals()[name]


def postmortem_verify_callback() -> Callable[[], tuple[tuple[str, ...], tuple[str, ...]]]:
    callback = postmortem_injected_value("__phase8_verify_repository_unchanged__")
    if not callable(callback):
        raise ValueError("postmortem-failure verifier source callback is not callable.")
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError) as exc:
        raise ValueError("postmortem-failure verifier source callback signature is unavailable.") from exc
    if signature.parameters:
        raise ValueError("postmortem-failure verifier source callback must accept no arguments.")
    if "__phase8_git_run__" in globals():
        raise ValueError("postmortem-failure forbids verifier-internal Git helper exposure.")
    return callback  # type: ignore[return-value]


def validate_postmortem_callback_path_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"postmortem-failure verifier source callback {field_name} must be an immutable tuple.")
    paths: list[str] = []
    for index, raw_path in enumerate(value):
        if type(raw_path) is not str:
            raise ValueError(f"postmortem-failure verifier source callback {field_name}[{index}] must be a string.")
        relative = PurePosixPath(raw_path)
        if not raw_path or "\\" in raw_path or "//" in raw_path or relative.is_absolute():
            raise ValueError(f"postmortem-failure verifier source callback {field_name}[{index}] is not canonical.")
        if relative.as_posix() != raw_path or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"postmortem-failure verifier source callback {field_name}[{index}] is not canonical.")
        paths.append(raw_path)
    return tuple(paths)


def call_postmortem_verify_callback() -> tuple[tuple[str, ...], tuple[str, ...]]:
    callback = postmortem_verify_callback()
    result = callback()
    if (
        not isinstance(result, tuple)
        or len(result) != 2
    ):
        raise ValueError("postmortem-failure verifier source callback must return exactly two immutable path tuples.")
    return (
        validate_postmortem_callback_path_tuple(result[0], "untracked_paths"),
        validate_postmortem_callback_path_tuple(result[1], "ignored_paths"),
    )


def postmortem_accepted_implementation_from_verifier() -> str:
    value = postmortem_injected_value("__phase8_accepted_implementation_commit__")
    return validate_git_sha(value, "postmortem.verifier.accepted_implementation_commit")


def require_postmortem_verifier_context(
    *,
    accepted_implementation_commit: str,
    input_root: Path,
    output_root: Path,
    accepted_proposal_commit: str,
    environ: dict[str, str] | None,
) -> None:
    require_postmortem_real_main_context()
    injected_sha = postmortem_injected_value("__phase8_verifier_sha256__")
    if injected_sha != POSTMORTEM_VERIFIER_SHA256:
        raise ValueError("postmortem-failure verifier SHA-256 binding mismatch.")
    if postmortem_accepted_implementation_from_verifier() != accepted_implementation_commit:
        raise ValueError("postmortem-failure accepted implementation commit must match the verifier and runner CLI.")
    postmortem_injected_value("__phase8_repository_loader__")
    postmortem_verify_callback()
    if __package__ is not None or globals().get("__cached__") is not None:
        raise ValueError("postmortem-failure requires the verifier's real __main__ execution context.")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1 or sys.flags.no_site != 1:
        raise ValueError("postmortem-failure requires isolated/no-bytecode/no-site interpreter flags.")
    actual_env = dict(os.environ if environ is None else environ)
    if actual_env != POSTMORTEM_EXECVE_ENVIRONMENT:
        raise ValueError("postmortem-failure environment must exactly equal the four-key verifier execve environment.")
    validate_postmortem_process_argv(
        postmortem_kernel_argv(),
        input_root=input_root,
        output_root=output_root,
        accepted_proposal_commit=accepted_proposal_commit,
        accepted_implementation_commit=accepted_implementation_commit,
    )


def validate_postmortem_argument_contract(
    *,
    device: str,
    input_root: Path,
    output_root: Path,
    accepted_proposal_commit: str,
    accepted_implementation_commit: str,
) -> None:
    if device != FEASIBILITY_REQUIRED_DEVICE:
        raise ValueError("postmortem-failure only supports device string cuda:0.")
    require_canonical_path_string(str(input_root), "input_root", ROOT_RE)
    require_artifact_location(input_root, "input_root", ROOT_RE)
    require_postmortem_artifact_location(output_root, "output_root")
    if str(input_root) != POSTMORTEM_REQUIRED_INPUT_ROOT:
        raise ValueError("postmortem-failure input root must be exactly artifacts/phase8_toy_lm_bridge/feasibility_005.")
    if not matches_repo_artifact_path(input_root, POSTMORTEM_REQUIRED_INPUT_ROOT):
        raise ValueError("postmortem-failure rejects copied input roots.")
    if str(output_root) != POSTMORTEM_REQUIRED_OUTPUT_ROOT:
        raise ValueError("postmortem-failure output root must be exactly artifacts/phase8_toy_lm_bridge/feasibility_postmortem_001.")
    if any(path.is_absolute() for path in (input_root, output_root)):
        raise ValueError("postmortem-failure authorized command requires canonical repository-relative artifact paths.")
    if accepted_proposal_commit != POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT:
        raise ValueError("postmortem-failure accepted proposal commit mismatch.")
    validate_git_sha(accepted_implementation_commit, "postmortem.accepted_implementation_commit")


def validate_new_postmortem_root(output_root: Path) -> None:
    require_postmortem_artifact_location(output_root, "output_root")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing postmortem root: {output_root}")
    temp_root = output_root.with_name(output_root.name + ".tmp")
    require_postmortem_artifact_location(temp_root, "temporary_output_root", temp=True)
    if temp_root.exists() or temp_root.is_symlink():
        raise FileExistsError(f"Temporary postmortem root already exists: {temp_root}")


def validate_postmortem_cli_contract(
    *,
    device: str,
    input_root: Path,
    output_root: Path,
    accepted_proposal_commit: str,
    accepted_implementation_commit: str,
    environ: dict[str, str] | None = None,
) -> None:
    validate_postmortem_argument_contract(
        device=device,
        input_root=input_root,
        output_root=output_root,
        accepted_proposal_commit=accepted_proposal_commit,
        accepted_implementation_commit=accepted_implementation_commit,
    )
    require_postmortem_repo_cwd()
    require_postmortem_verifier_context(
        accepted_implementation_commit=accepted_implementation_commit,
        input_root=input_root,
        output_root=output_root,
        accepted_proposal_commit=accepted_proposal_commit,
        environ=environ,
    )
    validate_new_postmortem_root(output_root)


def postmortem_proposal_binding(accepted_proposal_commit: str) -> dict[str, object]:
    if accepted_proposal_commit != POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT:
        raise ValueError("Postmortem proposal binding commit mismatch.")
    return {
        "commit": accepted_proposal_commit,
        "path": POSTMORTEM_PROPOSAL_PATH,
        "blob": POSTMORTEM_PROPOSAL_BLOB,
    }


def postmortem_implementation_binding(source_snapshot: SourceSnapshot) -> dict[str, object]:
    commit = validate_git_sha(source_snapshot.commit, "postmortem.implementation.commit")
    runner_path = POSTMORTEM_RUNNER_PATH
    runner_blob = validate_git_sha(source_snapshot.runner_blob, "postmortem.implementation.runner_blob")
    if not runner_blob:
        raise ValueError("Postmortem implementation runner blob is unavailable.")
    return {
        "commit": commit,
        "runner_path": runner_path,
        "runner_blob": runner_blob,
    }


def rng_states_equal(
    left: tuple[object, torch.Tensor, tuple[torch.Tensor, ...] | None],
    right: tuple[object, torch.Tensor, tuple[torch.Tensor, ...] | None],
) -> bool:
    if left[0] != right[0]:
        return False
    if not torch.equal(left[1], right[1]):
        return False
    left_cuda = left[2]
    right_cuda = right[2]
    if left_cuda is None or right_cuda is None:
        return left_cuda is None and right_cuda is None
    if len(left_cuda) != len(right_cuda):
        return False
    return all(torch.equal(a, b) for a, b in zip(left_cuda, right_cuda, strict=True))


def restore_torch_rng_states(states: tuple[object, torch.Tensor, tuple[torch.Tensor, ...] | None]) -> None:
    _python_state, torch_state, cuda_states = states
    torch.random.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(list(cuda_states))


def require_rng_states_equal(
    left: tuple[object, torch.Tensor, tuple[torch.Tensor, ...] | None],
    right: tuple[object, torch.Tensor, tuple[torch.Tensor, ...] | None],
    message: str,
) -> None:
    if not rng_states_equal(left, right):
        raise ValueError(message)


def configure_postmortem_deterministic_backend() -> None:
    before = snapshot_rng_states()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require_rng_states_equal(before, snapshot_rng_states(), "Postmortem deterministic backend configuration changed RNG state.")


def validate_postmortem_runtime_against_input(input_manifest: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    environment = current_environment_dict()
    if environment != FEASIBILITY_REQUIRED_RUNTIME_ENV or environment != input_manifest.get("environment"):
        raise ValueError("Postmortem runtime environment must exactly match the frozen feasibility_005 A800 environment.")
    deterministic_flags = current_deterministic_flags()
    validate_current_deterministic_flags(deterministic_flags)
    if deterministic_flags != input_manifest.get("deterministic_flags"):
        raise ValueError("Postmortem deterministic flags must exactly match the frozen feasibility_005 manifest.")
    return environment, deterministic_flags


def expected_tied_state_schema_without_construction(model_size: str) -> dict[str, dict[str, object]]:
    config = transformer_config(model_size)  # type: ignore[arg-type]
    schema: dict[str, dict[str, object]] = {
        "token_embedding.weight": {"dtype": "torch.float32", "shape": [config.vocab_size, config.d_model], "layout": "torch.strided"},
        "position_embedding.weight": {"dtype": "torch.float32", "shape": [config.max_seq_len, config.d_model], "layout": "torch.strided"},
        "final_norm.weight": {"dtype": "torch.float32", "shape": [config.d_model], "layout": "torch.strided"},
        "final_norm.bias": {"dtype": "torch.float32", "shape": [config.d_model], "layout": "torch.strided"},
        "lm_head.weight": {"dtype": "torch.float32", "shape": [config.vocab_size, config.d_model], "layout": "torch.strided"},
    }
    for layer_index in range(config.n_layers):
        prefix = f"blocks.{layer_index}"
        schema.update(
            {
                f"{prefix}.ln_1.weight": {"dtype": "torch.float32", "shape": [config.d_model], "layout": "torch.strided"},
                f"{prefix}.ln_1.bias": {"dtype": "torch.float32", "shape": [config.d_model], "layout": "torch.strided"},
                f"{prefix}.attn.in_proj_weight": {"dtype": "torch.float32", "shape": [3 * config.d_model, config.d_model], "layout": "torch.strided"},
                f"{prefix}.attn.in_proj_bias": {"dtype": "torch.float32", "shape": [3 * config.d_model], "layout": "torch.strided"},
                f"{prefix}.attn.out_proj.weight": {"dtype": "torch.float32", "shape": [config.d_model, config.d_model], "layout": "torch.strided"},
                f"{prefix}.attn.out_proj.bias": {"dtype": "torch.float32", "shape": [config.d_model], "layout": "torch.strided"},
                f"{prefix}.ln_2.weight": {"dtype": "torch.float32", "shape": [config.d_model], "layout": "torch.strided"},
                f"{prefix}.ln_2.bias": {"dtype": "torch.float32", "shape": [config.d_model], "layout": "torch.strided"},
                f"{prefix}.mlp.0.weight": {"dtype": "torch.float32", "shape": [config.d_ff, config.d_model], "layout": "torch.strided"},
                f"{prefix}.mlp.0.bias": {"dtype": "torch.float32", "shape": [config.d_ff], "layout": "torch.strided"},
                f"{prefix}.mlp.2.weight": {"dtype": "torch.float32", "shape": [config.d_model, config.d_ff], "layout": "torch.strided"},
                f"{prefix}.mlp.2.bias": {"dtype": "torch.float32", "shape": [config.d_model], "layout": "torch.strided"},
            }
        )
    return dict(sorted(schema.items()))


def validate_postmortem_checkpoint_payload(path: Path, cell: dict[str, object]) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Postmortem checkpoint path must be a regular inventory-bound file.")
    family = require_exact_str(cell.get("family"), "checkpoint.family")
    model_size = require_exact_str(cell.get("model_size"), "checkpoint.model_size")
    seed = require_exact_int(cell.get("seed"), "checkpoint.seed")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Postmortem checkpoint is not a loadable PyTorch checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"model_state_dict", "config", "parameter_count", "metadata"}:
        raise ValueError("Postmortem checkpoint must use the exact tied checkpoint schema.")
    config = checkpoint.get("config")
    if not isinstance(config, dict) or set(config) != set(TransformerConfig.__dataclass_fields__):
        raise ValueError("Postmortem checkpoint config must use the exact TransformerConfig schema.")
    if config != transformer_config(model_size).__dict__:  # type: ignore[arg-type]
        raise ValueError("Postmortem checkpoint config does not match the frozen cell model_size.")
    if checkpoint.get("parameter_count") != FROZEN_PARAMETER_COUNTS[model_size]:
        raise ValueError("Postmortem checkpoint parameter_count mismatch.")
    metadata = require_exact_mapping(checkpoint.get("metadata"), POSTMORTEM_CHECKPOINT_METADATA_KEYS, "checkpoint.metadata")
    expected_metadata = {
        "family": family,
        "model_size": model_size,
        "seed": seed,
        "training_steps": TRAINING_STEPS,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"Postmortem checkpoint metadata {key} mismatch.")
    require_finite_json_number(metadata.get("training_loss"), "checkpoint.metadata.training_loss")
    training_accuracy_value = require_finite_json_number(metadata.get("training_accuracy"), "checkpoint.metadata.training_accuracy")
    if not 0.0 <= float(training_accuracy_value) <= 1.0:
        raise ValueError("Postmortem checkpoint training_accuracy must lie in [0, 1].")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Postmortem checkpoint model_state_dict must be a mapping.")
    expected_schema = expected_tied_state_schema_without_construction(model_size)
    if set(state) != set(expected_schema):
        raise ValueError("Postmortem checkpoint state_dict keys do not match the frozen tied model.")
    for key in sorted(expected_schema):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Postmortem checkpoint state_dict values must be tensors.")
        expected = expected_schema[key]
        if str(tensor.dtype) != expected["dtype"]:
            raise ValueError(f"Postmortem checkpoint state_dict dtype mismatch for {key}.")
        if list(tensor.shape) != expected["shape"]:
            raise ValueError(f"Postmortem checkpoint state_dict shape mismatch for {key}.")
        if str(tensor.layout) != expected["layout"]:
            raise ValueError(f"Postmortem checkpoint state_dict layout mismatch for {key}.")
    token_weight = state["token_embedding.weight"]
    head_weight = state["lm_head.weight"]
    if token_weight.dtype != head_weight.dtype or tuple(token_weight.shape) != tuple(head_weight.shape) or token_weight.layout != head_weight.layout:
        raise ValueError("Postmortem tied checkpoint duplicate weights have mismatched schema.")
    if contiguous_uint8_bytes(token_weight) != contiguous_uint8_bytes(head_weight):
        raise ValueError("Postmortem tied checkpoint duplicate weights are not byte-equal.")
    return checkpoint


def validate_postmortem_input_shallow(input_root: Path) -> dict[str, object]:
    require_postmortem_repo_cwd()
    require_canonical_path_string(str(input_root), "input_root", ROOT_RE)
    require_artifact_location(input_root, "input_root", ROOT_RE)
    if str(input_root) != POSTMORTEM_REQUIRED_INPUT_ROOT or not matches_repo_artifact_path(input_root, POSTMORTEM_REQUIRED_INPUT_ROOT):
        raise ValueError("Postmortem input must be the exact canonical feasibility_005 root, not a copy.")
    if not input_root.is_dir() or input_root.is_symlink():
        raise ValueError("Postmortem input root must be a real directory.")
    for relative_path, expected_sha in POSTMORTEM_INPUT_CHECKSUMS.items():
        path = input_root / relative_path
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Postmortem input root lacks canonical {relative_path}.")
        if file_sha256(path) != expected_sha:
            raise ValueError(f"Postmortem input {relative_path} checksum mismatch.")
    terminal, terminal_data, manifest_path, manifest_sha = load_terminal_binding(input_root)
    if terminal.name != "FAILED.json":
        raise ValueError("Postmortem input must retain the accepted FAILED terminal.")
    if manifest_sha != POSTMORTEM_INPUT_CHECKSUMS["manifest.json"]:
        raise ValueError("Postmortem input manifest checksum mismatch.")
    manifest_data = json.loads(manifest_path.read_text())
    summary_data = json.loads((input_root / "summary.json").read_text())
    if set(manifest_data) != CURRENT_MANIFEST_KEYS:
        raise ValueError("Postmortem input manifest must use the exact tied feasibility schema.")
    if set(summary_data) != CURRENT_SUMMARY_KEYS:
        raise ValueError("Postmortem input summary must use the exact tied feasibility schema.")
    expected_terminal_keys = set(CURRENT_TERMINAL_KEYS)
    expected_terminal_keys.add("error")
    if set(terminal_data) != expected_terminal_keys:
        raise ValueError("Postmortem input terminal must use the exact failed tied feasibility schema.")
    if manifest_data.get("source_commit") != POSTMORTEM_INPUT_SOURCE_COMMIT:
        raise ValueError("Postmortem input source_commit mismatch.")
    if manifest_data.get("terminal_status") != "FAILED" or summary_data.get("terminal_status") != "FAILED":
        raise ValueError("Postmortem input must be the accepted FAILED root.")
    if manifest_data.get("configuration") != frozen_configuration() or summary_data.get("configuration") != frozen_configuration():
        raise ValueError("Postmortem input configuration mismatch.")
    if manifest_data.get("record_hashes") != FEASIBILITY_RECORD_HASHES:
        raise ValueError("Postmortem input record_hashes mismatch.")
    validate_current_deterministic_flags(manifest_data.get("deterministic_flags"))
    if manifest_data.get("deterministic_flags") != summary_data.get("deterministic_flags") or manifest_data.get("deterministic_flags") != terminal_data.get("deterministic_flags"):
        raise ValueError("Postmortem input deterministic flag bindings disagree.")
    if manifest_data.get("environment") != FEASIBILITY_REQUIRED_RUNTIME_ENV:
        raise ValueError("Postmortem input environment mismatch.")
    expected_command = feasibility_exact_command(
        input_root,
        tuple(Path(path) for path in FEASIBILITY_REQUIRED_PREDECESSOR_ROOTS),
        Path(FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT),
    )
    if manifest_data.get("exact_command") != expected_command or summary_data.get("exact_command") != expected_command or terminal_data.get("exact_command") != expected_command:
        raise ValueError("Postmortem input exact feasibility_005 command mismatch.")
    if manifest_data.get("decision_diagnostic") != DECISION_DIAGNOSTIC_ROOT_BINDING:
        raise ValueError("Postmortem input decision diagnostic binding mismatch.")
    if manifest_data.get("predecessor_selections") != []:
        raise ValueError("Postmortem input must not bind predecessor selections.")
    if manifest_data.get("predecessor_roots") != postmortem_expected_predecessor_bindings():
        raise ValueError("Postmortem input predecessor root bindings mismatch.")
    cells = validate_postmortem_current_cells_shallow(manifest_data.get("cells"))
    if summary_data.get("cells") != list(cells) or terminal_data.get("cells") != list(cells):
        raise ValueError("Postmortem input cell bindings disagree across manifest/summary/terminal.")
    validate_summary_aggregates(summary_data, list(cells), "FAILED", manifest_data.get("failure"))
    if terminal_data.get("error") != manifest_data.get("failure"):
        raise ValueError("Postmortem input failed terminal error does not match manifest failure.")
    inventory_rows = manifest_data.get("file_inventory")
    inventory_allowed = shallow_inventory_bound_paths_from_rows(input_root, inventory_rows, schema_mode="historical")
    if not isinstance(inventory_rows, list) or len(inventory_rows) != 49:
        raise ValueError("Postmortem input manifest must bind exactly 49 inventory entries.")
    inventory_paths = {require_canonical_relative_path(row.get("path"), "postmortem.input_inventory.path") for row in inventory_rows if isinstance(row, dict)}
    if "summary.json" not in inventory_paths:
        raise ValueError("Postmortem input inventory must bind summary.json.")
    for cell in cells:
        if cell["generations_path"] not in inventory_paths or cell["checkpoint_path"] not in inventory_paths:
            raise ValueError("Postmortem input inventory must bind every cell checkpoint and generation artifact.")
    allowed_paths: set[Path] = {manifest_path.resolve(), terminal.resolve(), (input_root / "summary.json").resolve(), *inventory_allowed}
    for predecessor_binding in manifest_data["predecessor_roots"]:
        predecessor = Path(require_canonical_path_string(predecessor_binding.get("path"), "postmortem.predecessor.path", ROOT_RE))
        expected = HISTORICAL_FEASIBILITY_ROOTS[predecessor.name]
        if predecessor_binding.get("terminal_state") != expected["terminal"].removesuffix(".json"):
            raise ValueError("Postmortem predecessor terminal state mismatch.")
        if predecessor_binding.get("terminal_sha256") != expected["terminal_sha256"] or predecessor_binding.get("manifest_sha256") != expected["manifest_sha256"]:
            raise ValueError("Postmortem predecessor checksum binding mismatch.")
        allowed_paths.update(shallow_inventory_bound_root_paths(predecessor))
    allowed_paths.update(shallow_decision_diagnostic_paths(Path(FEASIBILITY_REQUIRED_DECISION_DIAGNOSTIC_ROOT)))
    validate_source_provenance(manifest_data.get("source_provenance"), POSTMORTEM_INPUT_SOURCE_COMMIT, allowed_paths)
    input_binding = {
        "root": POSTMORTEM_REQUIRED_INPUT_ROOT,
        "source_commit": POSTMORTEM_INPUT_SOURCE_COMMIT,
        "manifest_path": "manifest.json",
        "manifest_sha256": POSTMORTEM_INPUT_CHECKSUMS["manifest.json"],
        "summary_path": "summary.json",
        "summary_sha256": POSTMORTEM_INPUT_CHECKSUMS["summary.json"],
        "terminal_path": "FAILED.json",
        "terminal_sha256": POSTMORTEM_INPUT_CHECKSUMS["FAILED.json"],
        "configuration": manifest_data["configuration"],
        "record_hashes": manifest_data["record_hashes"],
        "file_inventory": manifest_data["file_inventory"],
        "cells": manifest_data["cells"],
    }
    return {
        "input_binding": input_binding,
        "manifest_data": manifest_data,
        "allowed_source_paths": allowed_paths,
    }


def validate_postmortem_input_deep(
    input_root: Path,
    shallow_binding: dict[str, object],
) -> dict[str, object]:
    current = validate_postmortem_input_shallow(input_root)
    if current["input_binding"] != shallow_binding.get("input_binding"):
        raise ValueError("Postmortem input binding changed after shallow validation.")
    manifest_data = current["manifest_data"]
    if not isinstance(manifest_data, dict):
        raise ValueError("Postmortem manifest data is unavailable.")
    context = FeasibilityValidationContext()
    validate_root_manifest_lineage(input_root, manifest_data, context=context)
    cells = validate_postmortem_current_cells_shallow(manifest_data.get("cells"))
    generation_rows: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    checkpoint_payloads: dict[tuple[str, str, int], dict[str, object]] = {}
    for cell in cells:
        key = (
            require_exact_str(cell["family"], "cell.family"),
            require_exact_str(cell["model_size"], "cell.model_size"),
            require_exact_int(cell["seed"], "cell.seed"),
        )
        generations_path = input_root / require_canonical_relative_path(cell["generations_path"], "cell.generations_path")
        checkpoint_path = input_root / require_canonical_relative_path(cell["checkpoint_path"], "cell.checkpoint_path")
        generation_rows[key] = validate_generation_artifact(generations_path, cell)
        checkpoint_payloads[key] = validate_postmortem_checkpoint_payload(checkpoint_path, cell)
    return {
        "input_binding": current["input_binding"],
        "manifest_data": manifest_data,
        "generation_rows": generation_rows,
        "checkpoint_payloads": checkpoint_payloads,
        "cells": list(cells),
    }


def git_blob_oid_from_bytes(payload: bytes) -> str:
    return sha1(f"blob {len(payload)}\0".encode("ascii") + payload, usedforsecurity=False).hexdigest()


def postmortem_executed_runner_blob() -> str:
    injected_blob = globals().get("__phase8_runner_blob__")
    if injected_blob is not None:
        return validate_git_sha(injected_blob, "postmortem.injected.runner_blob")
    frame = sys._getframe()
    while frame is not None:
        runner_payload = frame.f_locals.get("runner_payload")
        runner_path = frame.f_locals.get("runner_path")
        if runner_path == POSTMORTEM_RUNNER_PATH and isinstance(runner_payload, (bytes, bytearray)):
            return git_blob_oid_from_bytes(bytes(runner_payload))
        frame = frame.f_back
    raise ValueError("postmortem-failure cannot bind the executed runner blob without verifier-captured runner bytes.")


def postmortem_untracked_status_lines(untracked_paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"?? {path}" for path in untracked_paths)


def postmortem_reject_verifier_returned_dangerous_ignored_paths(ignored_paths: tuple[str, ...]) -> None:
    for path in ignored_paths:
        relative = Path(path)
        if ignored_import_surface_candidate(relative, is_symlink=False, link_target=None):
            raise RuntimeError("Verifier returned a dangerous ignored import surface; postmortem fails closed.")


def path_is_under_active_postmortem_root(path: str, active_output_root: Path | None) -> bool:
    if active_output_root is None:
        return False
    active = PurePosixPath(active_output_root.as_posix())
    relative = PurePosixPath(path)
    return relative == active or active in relative.parents


def capture_postmortem_source_provenance(
    input_root: Path,
    output_root: Path,
    *,
    allowed_paths: set[Path],
) -> SourceSnapshot:
    require_canonical_path_string(str(input_root), "input_root", ROOT_RE)
    require_postmortem_artifact_location(output_root, "output_root")
    untracked_paths, ignored_paths = call_postmortem_verify_callback()
    for rel in untracked_paths:
        candidate = (Path(rel) if Path(rel).is_absolute() else REPO_ROOT / rel).resolve()
        if candidate not in allowed_paths:
            raise RuntimeError(f"Untracked file is not an exact supplied feasibility/postmortem binding: ?? {rel}")
    postmortem_reject_verifier_returned_dangerous_ignored_paths(ignored_paths)
    accepted_commit = postmortem_accepted_implementation_from_verifier()
    return SourceSnapshot(
        commit=accepted_commit,
        status_lines=postmortem_untracked_status_lines(untracked_paths),
        ignored_inputs=ignored_paths,
        runner_blob=postmortem_executed_runner_blob(),
    )


def verify_postmortem_source_unchanged(snapshot: SourceSnapshot, *, active_output_root: Path | None = None) -> None:
    if postmortem_accepted_implementation_from_verifier() != snapshot.commit:
        raise SourceChangedError("Postmortem implementation commit binding changed before terminal publication.")
    untracked_paths, ignored_paths = call_postmortem_verify_callback()
    filtered_untracked = tuple(path for path in untracked_paths if not path_is_under_active_postmortem_root(path, active_output_root))
    if postmortem_untracked_status_lines(filtered_untracked) != snapshot.status_lines:
        raise SourceChangedError("Postmortem source/untracked path set changed before terminal publication.")
    postmortem_reject_verifier_returned_dangerous_ignored_paths(ignored_paths)
    if ignored_paths != snapshot.ignored_inputs:
        raise SourceChangedError("Postmortem ignored executable source inputs changed before terminal publication.")
    if postmortem_executed_runner_blob() != snapshot.runner_blob:
        raise SourceChangedError("Postmortem executed runner blob changed before terminal publication.")


def verify_postmortem_preflight_bindings(bindings: dict[str, object], input_root: Path) -> None:
    if validate_postmortem_input_shallow(input_root)["input_binding"] != bindings.get("input_binding"):
        raise ValueError("Postmortem input binding changed after preflight.")


class PostmortemForbiddenOperationGuard:
    def __init__(self) -> None:
        self._restore_callbacks: list[Callable[[], None]] = []
        self._allow_rng_restore = 0
        self._allow_constructor_grad_restore = 0

    def _patch(self, owner: object, name: str, replacement: object) -> None:
        original = getattr(owner, name)
        setattr(owner, name, replacement)
        self._restore_callbacks.append(lambda owner=owner, name=name, original=original: setattr(owner, name, original))

    def __enter__(self) -> "PostmortemForbiddenOperationGuard":
        def forbidden(*_args: object, **_kwargs: object) -> object:
            raise ValueError("Postmortem read-only guard rejected a forbidden training/generation/mutation/RNG operation.")

        def guarded_set_grad_enabled(mode: object) -> object:
            if mode is True and self._allow_constructor_grad_restore <= 0:
                raise ValueError("Postmortem read-only guard rejected grad enablement.")
            return original_set_grad_enabled(mode)  # type: ignore[misc]

        def guarded_rng_restore(original: Callable[..., object]) -> Callable[..., object]:
            def inner(*args: object, **kwargs: object) -> object:
                if self._allow_rng_restore <= 0:
                    raise ValueError("Postmortem read-only guard rejected RNG-state mutation outside the constructor restore scope.")
                return original(*args, **kwargs)
            return inner

        original_set_grad_enabled = torch.set_grad_enabled
        try:
            self._patch(random, "seed", forbidden)
            self._patch(random, "setstate", forbidden)
            self._patch(torch, "manual_seed", forbidden)
            self._patch(torch.random, "set_rng_state", guarded_rng_restore(torch.random.set_rng_state))
            self._patch(torch.optim, "AdamW", forbidden)
            self._patch(torch, "save", forbidden)
            self._patch(torch.Tensor, "backward", forbidden)
            self._patch(torch.autograd, "backward", forbidden)
            self._patch(torch, "enable_grad", forbidden)
            self._patch(torch, "set_grad_enabled", guarded_set_grad_enabled)
            if hasattr(torch.cuda, "manual_seed_all"):
                self._patch(torch.cuda, "manual_seed_all", forbidden)
            if hasattr(torch.cuda, "manual_seed"):
                self._patch(torch.cuda, "manual_seed", forbidden)
            if hasattr(torch.cuda, "set_rng_state"):
                self._patch(torch.cuda, "set_rng_state", guarded_rng_restore(torch.cuda.set_rng_state))
            if hasattr(torch.cuda, "set_rng_state_all"):
                self._patch(torch.cuda, "set_rng_state_all", guarded_rng_restore(torch.cuda.set_rng_state_all))
            self._patch(sys.modules[__name__], "make_optimizer", forbidden)
            self._patch(sys.modules[__name__], "train_text_records", forbidden)
            self._patch(sys.modules[__name__], "save_checkpoint", forbidden)
            self._patch(sys.modules[__name__], "evaluate_model", forbidden)
            self._patch(sys.modules[__name__], "generate_named_diagnostic_rows", forbidden)
            self._patch(sys.modules[__name__], "generate_array_rows", forbidden)
            self._patch(ToyCausalTransformer, "greedy_decode", forbidden)
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    @contextmanager
    def constructor_rng_restore_scope(self) -> Iterator[None]:
        self._allow_rng_restore += 1
        self._allow_constructor_grad_restore += 1
        try:
            yield
        finally:
            self._allow_rng_restore -= 1
            self._allow_constructor_grad_restore -= 1

    @contextmanager
    def constructor_grad_restore_scope(self) -> Iterator[None]:
        self._allow_constructor_grad_restore += 1
        try:
            yield
        finally:
            self._allow_constructor_grad_restore -= 1

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        while self._restore_callbacks:
            self._restore_callbacks.pop()()


def construct_postmortem_model_from_checkpoint(
    checkpoint: dict[str, object],
    *,
    model_size: str,
    device: torch.device,
    guard: PostmortemForbiddenOperationGuard,
) -> torch.nn.Module:
    before = snapshot_rng_states()
    before_python = before[0]
    with guard.constructor_rng_restore_scope():
        model = build_model(model_size)  # type: ignore[arg-type]
        restore_torch_rng_states(before)
    if random.getstate() != before_python:
        raise ValueError("Postmortem CPU model constructor changed Python RNG state.")
    require_rng_states_equal(before, snapshot_rng_states(), "Postmortem CPU model constructor restore did not exactly restore Torch RNG state.")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Postmortem checkpoint state_dict missing before load.")
    with guard.constructor_grad_restore_scope():
        model.load_state_dict(state)
        if model.lm_head.weight is not model.token_embedding.weight:
            raise ValueError("Postmortem loaded tied checkpoint did not preserve parameter identity.")
        model.to(device)
        model.eval()
    require_rng_states_equal(before, snapshot_rng_states(), "Postmortem checkpoint load or device transfer changed RNG state.")
    return model


def postmortem_model_snapshot(model: torch.nn.Module) -> dict[str, tuple[str, tuple[int, ...], str, bytes]]:
    snapshot: dict[str, tuple[str, tuple[int, ...], str, bytes]] = {}
    for key, tensor in sorted(model.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Postmortem model snapshot can contain only tensors.")
        snapshot[key] = (str(tensor.dtype), tuple(int(dim) for dim in tensor.shape), str(tensor.layout), tensor_bytes(tensor))
    return snapshot


def require_postmortem_model_unchanged(
    model: torch.nn.Module,
    snapshot: dict[str, tuple[str, tuple[int, ...], str, bytes]],
    stage: str,
) -> None:
    if postmortem_model_snapshot(model) != snapshot:
        raise ValueError(f"Postmortem model parameters or buffers changed after {stage}.")


def postmortem_record_sets_rng_neutral() -> tuple[dict[str, dict[str, tuple[FeasibilityRecord, ...]]], dict[str, bool]]:
    before = snapshot_rng_states()
    records = grouped_records()
    actual_hashes = {name: canonical_record_set_sha256(rows) for name, rows in feasibility_record_sets().items()}
    if actual_hashes != FEASIBILITY_RECORD_HASHES:
        raise ValueError("Postmortem reconstructed record hashes do not match feasibility_005.")
    after = snapshot_rng_states()
    require_rng_states_equal(before, after, "Postmortem record reconstruction changed global Python/Torch RNG state.")
    return records, {"record_reconstruction_global_state_unchanged": True}


def autocast_is_enabled() -> bool:
    enabled = torch.is_autocast_enabled()
    try:
        enabled = enabled or torch.is_autocast_enabled("cuda")
    except TypeError:
        pass
    return enabled


def postmortem_teacher_aggregate(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    for row in rows:
        require_exact_mapping(row, POSTMORTEM_TEACHER_ROW_KEYS, "postmortem.teacher_forced_row")
    selected_total = sum(require_exact_int(row["selected_token_count"], "teacher.selected_token_count") for row in rows)
    correct_total = sum(require_exact_int(row["correct_token_count"], "teacher.correct_token_count") for row in rows)
    nll_total = math.fsum(float.fromhex(require_exact_str(row["nll_numerator_hex"], "teacher.nll_numerator_hex")) for row in rows)
    return {
        "record_count": len(rows),
        "sequence_exact_numerator": sum(1 for row in rows if row["sequence_exact"] is True),
        "selected_token_count": selected_total,
        "correct_token_count": correct_total,
        "nll_numerator_hex": nll_total.hex(),
        "nll_per_token_hex": (nll_total / selected_total).hex() if selected_total else float("nan").hex(),
    }


def postmortem_teacher_forced_rows(
    model: torch.nn.Module,
    records: Sequence[FeasibilityRecord],
    tokenizer: ByteTokenizer,
    device: torch.device,
    *,
    split: str,
    family: str,
    model_size: str,
    seed: int,
    checkpoint_path: str,
    checkpoint_sha256: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    expected_count = TRAIN_RECORDS_PER_FAMILY if split == "train" else EVAL_RECORDS_PER_FAMILY
    if split not in {"train", "eval"} or len(records) != expected_count or len(records) % BATCH_SIZE != 0:
        raise ValueError("Postmortem teacher-forced rows must use exact train/eval cardinalities and full 64-row batches.")
    if model.training:
        raise ValueError("Postmortem teacher-forced evaluation requires model.eval().")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Postmortem teacher-forced evaluation requires float32 model parameters.")
    if autocast_is_enabled():
        raise ValueError("Postmortem teacher-forced evaluation forbids autocast.")
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch_start in range(0, len(records), BATCH_SIZE):
            batch_records = records[batch_start : batch_start + BATCH_SIZE]
            input_ids = encode_record_batch(
                tuple(TextRecord(record.prompt, record.answer) for record in batch_records),
                tokenizer,
                max_length=ByteTokenizer.max_sequence_length,
            )
            if tuple(input_ids.shape) != (BATCH_SIZE, ByteTokenizer.max_sequence_length) or input_ids.dtype != torch.int64:
                raise ValueError("Postmortem teacher-forced inputs must have int64 shape [64,256].")
            input_ids = input_ids.to(device)
            labels = response_only_labels(input_ids)
            if tuple(labels.shape) != (BATCH_SIZE, ByteTokenizer.max_sequence_length) or labels.dtype != torch.int64:
                raise ValueError("Postmortem teacher-forced labels must have int64 shape [64,256].")
            logits = model(input_ids)
            if tuple(logits.shape) != (BATCH_SIZE, ByteTokenizer.max_sequence_length, ByteTokenizer.vocab_size):
                raise ValueError("Postmortem teacher-forced logits must have shape [64,256,260].")
            if logits.dtype != torch.float32:
                raise ValueError("Postmortem teacher-forced logits must be float32.")
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(BATCH_SIZE, ByteTokenizer.max_sequence_length)
            if tuple(losses.shape) != (BATCH_SIZE, ByteTokenizer.max_sequence_length) or losses.dtype != torch.float32:
                raise ValueError("Postmortem teacher-forced unreduced CE must be float32 with shape [64,256].")
            predictions = logits.argmax(dim=-1)
            batch_index = batch_start // BATCH_SIZE
            for row_within_batch, record in enumerate(batch_records):
                selected_positions = [position for position, label in enumerate(labels[row_within_batch].detach().cpu().tolist()) if label != -100]
                selected_count = len(selected_positions)
                if selected_count <= 0:
                    raise ValueError("Postmortem teacher-forced rows require at least one response token including EOS.")
                correct_count = sum(
                    int(predictions[row_within_batch, position].detach().cpu().item())
                    == int(labels[row_within_batch, position].detach().cpu().item())
                    for position in selected_positions
                )
                nll_values = [float(losses[row_within_batch, position].detach().cpu()) for position in selected_positions]
                rows.append(
                    {
                        "schema_version": POSTMORTEM_SCHEMA_VERSION,
                        "family": family,
                        "model_size": model_size,
                        "seed": seed,
                        "split": split,
                        "record_index": record.index,
                        "template_id": record.template_id,
                        "operand_id": record.operand_id,
                        "batch_index": batch_index,
                        "row_within_batch": row_within_batch,
                        "selected_token_count": selected_count,
                        "correct_token_count": correct_count,
                        "sequence_exact": correct_count == selected_count,
                        "nll_numerator_hex": math.fsum(nll_values).hex(),
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_sha256": checkpoint_sha256,
                    }
                )
    aggregate = postmortem_teacher_aggregate(rows)
    return rows, aggregate


def postmortem_generation_common(record: FeasibilityRecord, retained_row: dict[str, object]) -> dict[str, object]:
    tokenizer = ByteTokenizer()
    raw_token_ids = retained_row.get("raw_token_ids")
    if not isinstance(raw_token_ids, list) or not all(type(token) is int for token in raw_token_ids):
        raise ValueError("Postmortem retained generation raw_token_ids must be JSON integers.")
    prefix = list(tokenizer.encode_evaluation_prefix(record.prompt))
    if raw_token_ids[: len(prefix)] != prefix:
        raise ValueError("Postmortem retained generation row does not encode the frozen prompt prefix.")
    generation_slice = raw_token_ids[len(prefix) :]
    decoded_error = None
    try:
        decoded = tokenizer.decode_generated_response(raw_token_ids)
    except (UnicodeDecodeError, ValueError) as exc:
        decoded = None
        decoded_error = f"{type(exc).__name__}: {exc}"
    if retained_row.get("generation_error") != decoded_error:
        raise ValueError("Postmortem retained generation_error does not match raw token decoding.")
    if retained_row.get("generated") != decoded:
        raise ValueError("Postmortem retained generated response does not match raw token decoding.")
    greedy_exact = require_exact_bool(retained_row.get("exact_match"), "postmortem.retained.exact_match")
    if greedy_exact is not (decoded == record.answer):
        raise ValueError("Postmortem retained exact_match does not match decoded response.")
    has_eos = decoded_error is None and generation_slice.count(EOS_ID) == 1 and generation_slice[-1:] == [EOS_ID]
    hits_generation_cap = not has_eos and len(generation_slice) == ByteTokenizer.max_generated_tokens
    hits_context_cap = not has_eos and len(raw_token_ids) >= ByteTokenizer.max_sequence_length
    generated_bytes = None if decoded is None else len(decoded.encode("utf-8"))
    target_bytes = len(record.answer.encode("utf-8"))
    return {
        "prompt": record.prompt,
        "expected_response": record.answer,
        "generated_response": decoded,
        "raw_token_ids": list(raw_token_ids),
        "generation_error": decoded_error,
        "greedy_exact": greedy_exact,
        "decode_valid": decoded_error is None,
        "has_eos": has_eos,
        "hits_generation_cap": hits_generation_cap,
        "hits_context_cap": hits_context_cap,
        "generation_token_count": len(generation_slice),
        "generation_utf8_bytes": generated_bytes,
        "target_utf8_bytes": target_bytes,
        "length_matches": generated_bytes == target_bytes if generated_bytes is not None else False,
        "hamming_distance": None if decoded is None else byte_hamming_distance(decoded, record.answer),
        "edit_distance": None if decoded is None else byte_edit_distance(decoded, record.answer),
        "first_error": None if decoded is None else byte_first_error(decoded, record.answer),
    }


def postmortem_named_taxonomy(record: FeasibilityRecord, common: dict[str, object]) -> dict[str, object]:
    generated = common["generated_response"]
    parsed: object = None
    valid_json_string = False
    valid_named_grammar = False
    if isinstance(generated, str):
        try:
            parsed = json.loads(generated)
            valid_json_string = isinstance(parsed, str)
            valid_named_grammar = valid_json_string and NAMED_VALUE_RE.fullmatch(parsed) is not None
        except json.JSONDecodeError:
            parsed = None
    target_key = named_value_target_key(record)
    expected_value = json.loads(record.answer)
    roster_values = {value for _key, value in named_roster(record)}
    target_prefix = isinstance(parsed, str) and parsed.startswith(f"{target_key}-")
    suffix_positional_correct = None
    suffix_positional_total = None
    suffix_hamming_distance = None
    suffix_edit_distance = None
    suffix_first_error = None
    if target_prefix:
        suffix = parsed[len(target_key) + 1 :]
        expected_suffix = expected_value[len(target_key) + 1 :]
        suffix_bytes = suffix.encode("utf-8")
        expected_bytes = expected_suffix.encode("utf-8")
        suffix_positional_correct = sum(
            index < len(suffix_bytes) and suffix_bytes[index] == expected_bytes[index]
            for index in range(4)
        )
        suffix_positional_total = 4
        suffix_hamming_distance = byte_hamming_distance(suffix, expected_suffix) if len(suffix_bytes) == 4 else None
        suffix_edit_distance = byte_edit_distance(suffix, expected_suffix)
        suffix_first_error = byte_first_error(suffix, expected_suffix)
    return {
        "surface_source": "held_surface",
        "operand_source": "held_operand",
        "valid_json_string": valid_json_string,
        "valid_named_grammar": valid_named_grammar,
        "target_prefix": target_prefix,
        "occurs_in_prompt": isinstance(parsed, str) and parsed in record.prompt,
        "exact_target_value": parsed == expected_value,
        "exact_distractor_value": isinstance(parsed, str) and parsed in (roster_values - {expected_value}),
        "suffix_positional_correct": suffix_positional_correct,
        "suffix_positional_total": suffix_positional_total,
        "suffix_hamming_distance": suffix_hamming_distance,
        "suffix_edit_distance": suffix_edit_distance,
        "suffix_first_error": suffix_first_error,
    }


def first_wrong_item_position(generated_items: Sequence[str], expected_items: Sequence[str]) -> int | None:
    if list(generated_items) == list(expected_items):
        return None
    for index, (got, expected) in enumerate(zip(generated_items, expected_items)):
        if got != expected:
            return index
    return min(len(generated_items), len(expected_items))


def postmortem_array_taxonomy(record: FeasibilityRecord, common: dict[str, object]) -> dict[str, object]:
    generated = common["generated_response"]
    expected_items = json.loads(record.answer)
    if not isinstance(expected_items, list) or not all(isinstance(item, str) for item in expected_items):
        raise ValueError("Postmortem array expected response must be a JSON string array.")
    target_count = array_item_count(record)
    if len(expected_items) != target_count or target_count not in {1, 2, 3, 4}:
        raise ValueError("Postmortem array target item count mismatch.")
    valid_json_syntax = False
    valid_array_schema = False
    correct_item_count = False
    positional_exact_count = None
    positional_denominator = None
    missing_items = None
    extra_items = None
    all_items_copied = False
    all_items_exact = False
    first_wrong = None
    if isinstance(generated, str):
        try:
            parsed = json.loads(generated)
            valid_json_syntax = True
        except json.JSONDecodeError:
            parsed = None
        valid_array_schema = isinstance(parsed, list) and all(isinstance(item, str) for item in parsed)
        if valid_array_schema:
            generated_items = list(parsed)  # type: ignore[arg-type]
            correct_item_count = len(generated_items) == target_count
            positional_exact_count = sum(1 for got, expected in zip(generated_items, expected_items) if got == expected)
            positional_denominator = target_count
            missing_items = sorted((Counter(expected_items) - Counter(generated_items)).elements())
            extra_items = sorted((Counter(generated_items) - Counter(expected_items)).elements())
            prompt_item_multiset = Counter(ARRAY_ITEM_RE.findall(record.prompt))
            all_items_copied = not bool(Counter(generated_items) - prompt_item_multiset)
            all_items_exact = correct_item_count and positional_exact_count == target_count
            first_wrong = first_wrong_item_position(generated_items, expected_items)
    return {
        "comparison_source": "feasibility_005_retained_eval",
        "training_steps": TRAINING_STEPS,
        "target_item_count": target_count,
        "valid_json_syntax": valid_json_syntax,
        "valid_array_schema": valid_array_schema,
        "correct_item_count": correct_item_count,
        "positional_exact_count": positional_exact_count,
        "positional_denominator": positional_denominator,
        "missing_items": missing_items,
        "extra_items": extra_items,
        "all_items_copied": all_items_copied,
        "all_items_exact": all_items_exact,
        "first_wrong_item_position": first_wrong,
    }


def postmortem_taxonomy_row(
    record: FeasibilityRecord,
    retained_row: dict[str, object],
    *,
    family: str,
    model_size: str,
    seed: int,
    generation_path: str,
    generation_sha256: str,
) -> dict[str, object]:
    common = postmortem_generation_common(record, retained_row)
    named = postmortem_named_taxonomy(record, common) if family == "named_value_json" else None
    array = postmortem_array_taxonomy(record, common) if family == "array_json" else None
    row = {
        "schema_version": POSTMORTEM_SCHEMA_VERSION,
        "family": family,
        "model_size": model_size,
        "seed": seed,
        "record_index": record.index,
        "generation_path": generation_path,
        "generation_sha256": generation_sha256,
        "common": common,
        "named": named,
        "array": array,
    }
    validate_postmortem_taxonomy_row(row)
    return row


def validate_postmortem_common(common: object) -> dict[str, object]:
    data = require_exact_mapping(common, POSTMORTEM_COMMON_KEYS, "postmortem.common")
    require_exact_str(data["prompt"], "postmortem.common.prompt")
    require_exact_str(data["expected_response"], "postmortem.common.expected_response")
    if data["generated_response"] is not None:
        require_exact_str(data["generated_response"], "postmortem.common.generated_response")
    if data["generation_error"] is not None:
        require_exact_str(data["generation_error"], "postmortem.common.generation_error")
    raw = data["raw_token_ids"]
    if not isinstance(raw, list) or not all(type(token) is int for token in raw):
        raise ValueError("Postmortem common raw_token_ids must be a JSON integer list.")
    for key in ("greedy_exact", "decode_valid", "has_eos", "hits_generation_cap", "hits_context_cap", "length_matches"):
        require_exact_bool(data[key], f"postmortem.common.{key}")
    for key in ("generation_token_count", "target_utf8_bytes"):
        value = require_exact_int(data[key], f"postmortem.common.{key}")
        if value < 0:
            raise ValueError(f"postmortem.common.{key} must be non-negative.")
    for key in ("generation_utf8_bytes", "hamming_distance", "edit_distance", "first_error"):
        value = data[key]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"postmortem.common.{key} must be null or a non-negative integer.")
    return data


def validate_postmortem_named(named: object) -> dict[str, object]:
    data = require_exact_mapping(named, POSTMORTEM_NAMED_KEYS, "postmortem.named")
    if data["surface_source"] != "held_surface" or data["operand_source"] != "held_operand":
        raise ValueError("Postmortem Named source labels must be held_surface/held_operand.")
    for key in (
        "valid_json_string",
        "valid_named_grammar",
        "target_prefix",
        "occurs_in_prompt",
        "exact_target_value",
        "exact_distractor_value",
    ):
        require_exact_bool(data[key], f"postmortem.named.{key}")
    target_prefix = data["target_prefix"] is True
    suffix_keys = (
        "suffix_positional_correct",
        "suffix_positional_total",
        "suffix_hamming_distance",
        "suffix_edit_distance",
        "suffix_first_error",
    )
    if not target_prefix and any(data[key] is not None for key in suffix_keys):
        raise ValueError("Postmortem Named suffix fields must be null without a target-prefix parse.")
    if target_prefix:
        total = require_exact_int(data["suffix_positional_total"], "postmortem.named.suffix_positional_total")
        correct = require_exact_int(data["suffix_positional_correct"], "postmortem.named.suffix_positional_correct")
        if total != 4 or not 0 <= correct <= 4:
            raise ValueError("Postmortem Named suffix positional counts are invalid.")
        for key in ("suffix_hamming_distance", "suffix_edit_distance", "suffix_first_error"):
            value = data[key]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"postmortem.named.{key} must be null or a non-negative integer.")
    return data


def validate_postmortem_array(array: object) -> dict[str, object]:
    data = require_exact_mapping(array, POSTMORTEM_ARRAY_KEYS, "postmortem.array")
    if data["comparison_source"] != "feasibility_005_retained_eval" or data["training_steps"] != TRAINING_STEPS:
        raise ValueError("Postmortem Array comparison source/training_steps mismatch.")
    target_count = require_exact_int(data["target_item_count"], "postmortem.array.target_item_count")
    if target_count not in {1, 2, 3, 4}:
        raise ValueError("Postmortem Array target_item_count must be 1..4.")
    for key in (
        "valid_json_syntax",
        "valid_array_schema",
        "correct_item_count",
        "all_items_copied",
        "all_items_exact",
    ):
        require_exact_bool(data[key], f"postmortem.array.{key}")
    valid_schema = data["valid_array_schema"] is True
    nullable_keys = ("positional_exact_count", "positional_denominator", "missing_items", "extra_items", "first_wrong_item_position")
    if not valid_schema and any(data[key] is not None for key in nullable_keys):
        raise ValueError("Invalid Postmortem Array schema rows must use null item-detail fields.")
    if valid_schema:
        denominator = require_exact_int(data["positional_denominator"], "postmortem.array.positional_denominator")
        exact_count = require_exact_int(data["positional_exact_count"], "postmortem.array.positional_exact_count")
        if denominator != target_count or not 0 <= exact_count <= denominator:
            raise ValueError("Postmortem Array positional counts are invalid.")
        for key in ("missing_items", "extra_items"):
            value = data[key]
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value) or value != sorted(value):
                raise ValueError(f"postmortem.array.{key} must be a lexicographically sorted string list.")
        first_wrong = data["first_wrong_item_position"]
        if first_wrong is not None and (type(first_wrong) is not int or first_wrong < 0):
            raise ValueError("postmortem.array.first_wrong_item_position must be null or a non-negative integer.")
    return data


def validate_postmortem_taxonomy_row(row: object) -> dict[str, object]:
    data = require_exact_mapping(row, POSTMORTEM_TAXONOMY_KEYS, "postmortem.error_taxonomy_row")
    if data["schema_version"] != POSTMORTEM_SCHEMA_VERSION:
        raise ValueError("Postmortem taxonomy schema_version mismatch.")
    family = require_exact_str(data["family"], "postmortem.taxonomy.family")
    if family not in FAMILIES:
        raise ValueError("Postmortem taxonomy family mismatch.")
    require_exact_str(data["model_size"], "postmortem.taxonomy.model_size")
    require_exact_int(data["seed"], "postmortem.taxonomy.seed")
    require_exact_int(data["record_index"], "postmortem.taxonomy.record_index")
    require_canonical_relative_path(data["generation_path"], "postmortem.taxonomy.generation_path")
    require_sha256_hex(data["generation_sha256"], "postmortem.taxonomy.generation_sha256")
    validate_postmortem_common(data["common"])
    if family == "named_value_json":
        validate_postmortem_named(data["named"])
        if data["array"] is not None:
            raise ValueError("Postmortem Named taxonomy rows must have array=null.")
    elif family == "array_json":
        validate_postmortem_array(data["array"])
        if data["named"] is not None:
            raise ValueError("Postmortem Array taxonomy rows must have named=null.")
    else:
        if data["named"] is not None or data["array"] is not None:
            raise ValueError("Postmortem non-Named/non-Array taxonomy rows must have named=array=null.")
    return data


def postmortem_named_aggregates(taxonomy_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_size in MODEL_SIZES:
        for seed in SEEDS:
            group = [
                row
                for row in taxonomy_rows
                if row["family"] == "named_value_json" and row["model_size"] == model_size and row["seed"] == seed
            ]
            if len(group) != EVAL_RECORDS_PER_FAMILY:
                raise ValueError("Postmortem Named aggregates require exactly 64 rows per model_size/seed.")
            named_rows = [validate_postmortem_named(row["named"]) for row in group]
            common_rows = [validate_postmortem_common(row["common"]) for row in group]
            target_prefix_count = sum(1 for row in named_rows if row["target_prefix"] is True)
            rows.append(
                {
                    "model_size": model_size,
                    "seed": seed,
                    "denominator": EVAL_RECORDS_PER_FAMILY,
                    "valid_json_string_count": sum(1 for row in named_rows if row["valid_json_string"] is True),
                    "valid_named_grammar_count": sum(1 for row in named_rows if row["valid_named_grammar"] is True),
                    "target_prefix_count": target_prefix_count,
                    "occurs_in_prompt_count": sum(1 for row in named_rows if row["occurs_in_prompt"] is True),
                    "exact_target_value_count": sum(1 for row in named_rows if row["exact_target_value"] is True),
                    "exact_distractor_value_count": sum(1 for row in named_rows if row["exact_distractor_value"] is True),
                    "suffix_positional_correct": sum(int(row["suffix_positional_correct"]) for row in named_rows if row["suffix_positional_correct"] is not None),
                    "suffix_positional_denominator": 4 * target_prefix_count,
                    "suffix_first_error_histogram": sparse_decimal_histogram(row["suffix_first_error"] for row in named_rows),
                    "suffix_first_error_observation_count": sum(1 for row in named_rows if row["suffix_first_error"] is not None),
                    "eos_present_count": sum(1 for row in common_rows if row["has_eos"] is True),
                    "generation_cap_count": sum(1 for row in common_rows if row["hits_generation_cap"] is True),
                    "target_length_match_count": sum(1 for row in common_rows if row["length_matches"] is True),
                    "first_error_histogram": sparse_decimal_histogram(row["first_error"] for row in common_rows),
                    "first_error_observation_count": sum(1 for row in common_rows if row["first_error"] is not None),
                    "generation_token_count_histogram": sparse_decimal_histogram(row["generation_token_count"] for row in common_rows),
                    "generation_utf8_bytes_histogram": sparse_decimal_histogram(row["generation_utf8_bytes"] for row in common_rows),
                }
            )
    return rows


def postmortem_array_aggregates(taxonomy_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_size in MODEL_SIZES:
        for seed in SEEDS:
            for item_count in (1, 2, 3, 4):
                group = [
                    row
                    for row in taxonomy_rows
                    if row["family"] == "array_json"
                    and row["model_size"] == model_size
                    and row["seed"] == seed
                    and row["array"]["target_item_count"] == item_count
                ]
                if len(group) != EVAL_RECORDS_PER_FAMILY // 4:
                    raise ValueError("Postmortem Array aggregates require exactly 16 rows per item-count stratum.")
                array_rows = [validate_postmortem_array(row["array"]) for row in group]
                common_rows = [validate_postmortem_common(row["common"]) for row in group]
                rows.append(
                    {
                        "model_size": model_size,
                        "seed": seed,
                        "target_item_count": item_count,
                        "denominator": EVAL_RECORDS_PER_FAMILY // 4,
                        "valid_json_syntax_count": sum(1 for row in array_rows if row["valid_json_syntax"] is True),
                        "valid_array_schema_count": sum(1 for row in array_rows if row["valid_array_schema"] is True),
                        "correct_item_count_count": sum(1 for row in array_rows if row["correct_item_count"] is True),
                        "positional_exact_count": sum(int(row["positional_exact_count"]) for row in array_rows if row["positional_exact_count"] is not None),
                        "positional_denominator": sum(int(row["positional_denominator"]) for row in array_rows if row["positional_denominator"] is not None),
                        "all_items_copied_count": sum(1 for row in array_rows if row["all_items_copied"] is True),
                        "all_items_exact_count": sum(1 for row in array_rows if row["all_items_exact"] is True),
                        "missing_item_count": sum(len(row["missing_items"]) for row in array_rows if row["missing_items"] is not None),
                        "extra_item_count": sum(len(row["extra_items"]) for row in array_rows if row["extra_items"] is not None),
                        "first_wrong_histogram": sparse_decimal_histogram(row["first_wrong_item_position"] for row in array_rows),
                        "first_wrong_observation_count": sum(1 for row in array_rows if row["first_wrong_item_position"] is not None),
                        "greedy_exact_count": sum(1 for row in common_rows if row["greedy_exact"] is True),
                        "eos_present_count": sum(1 for row in common_rows if row["has_eos"] is True),
                        "generation_cap_count": sum(1 for row in common_rows if row["hits_generation_cap"] is True),
                        "target_length_match_count": sum(1 for row in common_rows if row["length_matches"] is True),
                        "first_error_histogram": sparse_decimal_histogram(row["first_error"] for row in common_rows),
                        "first_error_observation_count": sum(1 for row in common_rows if row["first_error"] is not None),
                        "generation_token_count_histogram": sparse_decimal_histogram(row["generation_token_count"] for row in common_rows),
                        "generation_utf8_bytes_histogram": sparse_decimal_histogram(row["generation_utf8_bytes"] for row in common_rows),
                    }
                )
    return rows


def postmortem_file_sha_from_input_binding(input_binding: dict[str, object], relative_path: str) -> str:
    inventory_rows = input_binding.get("file_inventory")
    if not isinstance(inventory_rows, list):
        raise ValueError("Postmortem input binding file_inventory must be a list.")
    matches = [row for row in inventory_rows if isinstance(row, dict) and row.get("path") == relative_path]
    if len(matches) != 1:
        raise ValueError(f"Postmortem input binding does not contain exactly one inventory row for {relative_path!r}.")
    return require_sha256_hex(matches[0].get("sha256"), "postmortem.input_inventory.sha256")


def postmortem_checkpoint_metadata(checkpoint: dict[str, object]) -> dict[str, object]:
    metadata = require_exact_mapping(checkpoint.get("metadata"), POSTMORTEM_CHECKPOINT_METADATA_KEYS, "postmortem.checkpoint_metadata")
    return dict(metadata)


def postmortem_paired_eval_exact(
    eval_teacher_rows: Sequence[dict[str, object]],
    generation_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    if len(eval_teacher_rows) != EVAL_RECORDS_PER_FAMILY or len(generation_rows) != EVAL_RECORDS_PER_FAMILY:
        raise ValueError("Postmortem paired eval exact table requires 64 teacher and 64 retained greedy rows.")
    both_exact = tf_only = greedy_only = neither_exact = 0
    for teacher_row, generation_row in zip(eval_teacher_rows, generation_rows, strict=True):
        tf_exact = require_exact_bool(teacher_row.get("sequence_exact"), "paired.teacher.sequence_exact")
        greedy_exact = require_exact_bool(generation_row.get("exact_match"), "paired.greedy.exact_match")
        if tf_exact and greedy_exact:
            both_exact += 1
        elif tf_exact and not greedy_exact:
            tf_only += 1
        elif not tf_exact and greedy_exact:
            greedy_only += 1
        else:
            neither_exact += 1
    denominator = both_exact + tf_only + greedy_only + neither_exact
    return {
        "both_exact": both_exact,
        "tf_only": tf_only,
        "greedy_only": greedy_only,
        "neither_exact": neither_exact,
        "denominator": denominator,
        "count_difference": tf_only - greedy_only,
    }


def postmortem_cell_metrics_row(
    *,
    cell: dict[str, object],
    input_binding: dict[str, object],
    checkpoint: dict[str, object],
    train_teacher_rows: Sequence[dict[str, object]],
    eval_teacher_rows: Sequence[dict[str, object]],
    generation_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    family = require_exact_str(cell["family"], "postmortem.cell.family")
    model_size = require_exact_str(cell["model_size"], "postmortem.cell.model_size")
    seed = require_exact_int(cell["seed"], "postmortem.cell.seed")
    checkpoint_path = require_canonical_relative_path(cell["checkpoint_path"], "postmortem.cell.checkpoint_path")
    generation_path = require_canonical_relative_path(cell["generations_path"], "postmortem.cell.generations_path")
    retained_exact = sum(1 for row in generation_rows if row.get("exact_match") is True)
    row = {
        "schema_version": POSTMORTEM_SCHEMA_VERSION,
        "family": family,
        "model_size": model_size,
        "seed": seed,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": postmortem_file_sha_from_input_binding(input_binding, checkpoint_path),
        "generation_path": generation_path,
        "generation_sha256": postmortem_file_sha_from_input_binding(input_binding, generation_path),
        "checkpoint_metadata": postmortem_checkpoint_metadata(checkpoint),
        "train_teacher_forced": postmortem_teacher_aggregate(train_teacher_rows),
        "eval_teacher_forced": postmortem_teacher_aggregate(eval_teacher_rows),
        "retained_greedy_exact": {"numerator": retained_exact, "denominator": EVAL_RECORDS_PER_FAMILY},
        "paired_eval_exact": postmortem_paired_eval_exact(eval_teacher_rows, generation_rows),
    }
    validate_postmortem_cell_metrics_row(row)
    return row


def validate_postmortem_teacher_aggregate(value: object, field_name: str) -> dict[str, object]:
    data = require_exact_mapping(value, POSTMORTEM_TEACHER_AGGREGATE_KEYS, field_name)
    record_count = require_exact_int(data["record_count"], f"{field_name}.record_count")
    sequence_exact_numerator = require_exact_int(data["sequence_exact_numerator"], f"{field_name}.sequence_exact_numerator")
    selected = require_exact_int(data["selected_token_count"], f"{field_name}.selected_token_count")
    correct = require_exact_int(data["correct_token_count"], f"{field_name}.correct_token_count")
    if record_count <= 0 or not 0 <= sequence_exact_numerator <= record_count or selected <= 0 or not 0 <= correct <= selected:
        raise ValueError(f"{field_name} integer counts are inconsistent.")
    for key in ("nll_numerator_hex", "nll_per_token_hex"):
        text = require_exact_str(data[key], f"{field_name}.{key}")
        value_float = float.fromhex(text)
        if not math.isfinite(value_float) or value_float < 0:
            raise ValueError(f"{field_name}.{key} must be a finite non-negative float.hex string.")
    return data


def validate_postmortem_cell_metrics_row(row: object) -> dict[str, object]:
    data = require_exact_mapping(row, POSTMORTEM_CELL_KEYS, "postmortem.cell_metrics_row")
    if data["schema_version"] != POSTMORTEM_SCHEMA_VERSION:
        raise ValueError("Postmortem cell_metrics schema_version mismatch.")
    family = require_exact_str(data["family"], "postmortem.cell.family")
    model_size = require_exact_str(data["model_size"], "postmortem.cell.model_size")
    seed = require_exact_int(data["seed"], "postmortem.cell.seed")
    if family not in FAMILIES or model_size not in MODEL_SIZES or seed not in SEEDS:
        raise ValueError("Postmortem cell identity outside frozen grid.")
    require_canonical_relative_path(data["checkpoint_path"], "postmortem.cell.checkpoint_path")
    require_canonical_relative_path(data["generation_path"], "postmortem.cell.generation_path")
    require_sha256_hex(data["checkpoint_sha256"], "postmortem.cell.checkpoint_sha256")
    require_sha256_hex(data["generation_sha256"], "postmortem.cell.generation_sha256")
    metadata = require_exact_mapping(data["checkpoint_metadata"], POSTMORTEM_CHECKPOINT_METADATA_KEYS, "postmortem.cell.checkpoint_metadata")
    if metadata.get("family") != family or metadata.get("model_size") != model_size or metadata.get("seed") != seed or metadata.get("training_steps") != TRAINING_STEPS:
        raise ValueError("Postmortem checkpoint_metadata does not match its cell identity.")
    require_finite_json_number(metadata.get("training_loss"), "postmortem.cell.training_loss")
    accuracy = require_finite_json_number(metadata.get("training_accuracy"), "postmortem.cell.training_accuracy")
    if not 0.0 <= float(accuracy) <= 1.0:
        raise ValueError("Postmortem checkpoint_metadata training_accuracy out of range.")
    validate_postmortem_teacher_aggregate(data["train_teacher_forced"], "postmortem.cell.train_teacher_forced")
    validate_postmortem_teacher_aggregate(data["eval_teacher_forced"], "postmortem.cell.eval_teacher_forced")
    greedy = require_exact_mapping(data["retained_greedy_exact"], frozenset({"numerator", "denominator"}), "postmortem.cell.retained_greedy_exact")
    numerator = require_exact_int(greedy["numerator"], "postmortem.cell.retained_greedy_exact.numerator")
    denominator = require_exact_int(greedy["denominator"], "postmortem.cell.retained_greedy_exact.denominator")
    if denominator != EVAL_RECORDS_PER_FAMILY or not 0 <= numerator <= denominator:
        raise ValueError("Postmortem retained_greedy_exact counts are invalid.")
    paired = require_exact_mapping(
        data["paired_eval_exact"],
        frozenset({"both_exact", "tf_only", "greedy_only", "neither_exact", "denominator", "count_difference"}),
        "postmortem.cell.paired_eval_exact",
    )
    paired_counts = {key: require_exact_int(paired[key], f"postmortem.cell.paired_eval_exact.{key}") for key in ("both_exact", "tf_only", "greedy_only", "neither_exact")}
    if any(value < 0 for value in paired_counts.values()):
        raise ValueError("Postmortem paired eval counts must be non-negative.")
    paired_denominator = require_exact_int(paired["denominator"], "postmortem.cell.paired_eval_exact.denominator")
    if paired_denominator != sum(paired_counts.values()) or paired_denominator != EVAL_RECORDS_PER_FAMILY:
        raise ValueError("Postmortem paired eval denominator mismatch.")
    if paired["count_difference"] != paired_counts["tf_only"] - paired_counts["greedy_only"]:
        raise ValueError("Postmortem paired eval count_difference mismatch.")
    return data


def validate_postmortem_teacher_rows(
    rows: Sequence[dict[str, object]],
    *,
    input_binding: dict[str, object],
    records: dict[str, dict[str, tuple[FeasibilityRecord, ...]]],
) -> None:
    if len(rows) != POSTMORTEM_ROW_COUNTS["teacher_forced_rows"]:
        raise ValueError("Postmortem teacher_forced_rows cardinality mismatch.")
    index = 0
    for cell in validate_postmortem_current_cells_shallow(input_binding.get("cells")):
        family = require_exact_str(cell["family"], "teacher.cell.family")
        model_size = require_exact_str(cell["model_size"], "teacher.cell.model_size")
        seed = require_exact_int(cell["seed"], "teacher.cell.seed")
        checkpoint_path = require_canonical_relative_path(cell["checkpoint_path"], "teacher.cell.checkpoint_path")
        checkpoint_sha = postmortem_file_sha_from_input_binding(input_binding, checkpoint_path)
        for split in ("train", "eval"):
            split_records = records[family][split]
            for record_index, record in enumerate(split_records):
                row = rows[index]
                require_exact_mapping(row, POSTMORTEM_TEACHER_ROW_KEYS, "postmortem.teacher_forced_row")
                if row["schema_version"] != POSTMORTEM_SCHEMA_VERSION:
                    raise ValueError("Postmortem teacher row schema_version mismatch.")
                expected_identity = {
                    "family": family,
                    "model_size": model_size,
                    "seed": seed,
                    "split": split,
                    "record_index": record.index,
                    "template_id": record.template_id,
                    "operand_id": record.operand_id,
                    "batch_index": record_index // BATCH_SIZE,
                    "row_within_batch": record_index % BATCH_SIZE,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_sha256": checkpoint_sha,
                }
                for key, expected in expected_identity.items():
                    if row.get(key) != expected:
                        raise ValueError(f"Postmortem teacher row {key} mismatch.")
                selected = require_exact_int(row["selected_token_count"], "teacher.selected_token_count")
                correct = require_exact_int(row["correct_token_count"], "teacher.correct_token_count")
                if selected <= 0 or not 0 <= correct <= selected:
                    raise ValueError("Postmortem teacher selected/correct counts mismatch.")
                if require_exact_bool(row["sequence_exact"], "teacher.sequence_exact") is not (correct == selected):
                    raise ValueError("Postmortem teacher sequence_exact mismatch.")
                nll = float.fromhex(require_exact_str(row["nll_numerator_hex"], "teacher.nll_numerator_hex"))
                if not math.isfinite(nll) or nll < 0:
                    raise ValueError("Postmortem teacher nll_numerator_hex must be finite and non-negative.")
                index += 1


def validate_postmortem_taxonomy_rows(
    rows: Sequence[dict[str, object]],
    *,
    input_binding: dict[str, object],
    records: dict[str, dict[str, tuple[FeasibilityRecord, ...]]],
) -> None:
    if len(rows) != POSTMORTEM_ROW_COUNTS["error_taxonomy"]:
        raise ValueError("Postmortem error_taxonomy cardinality mismatch.")
    index = 0
    input_root = Path(require_exact_str(input_binding.get("root"), "postmortem.input_binding.root"))
    for cell in validate_postmortem_current_cells_shallow(input_binding.get("cells")):
        family = require_exact_str(cell["family"], "taxonomy.cell.family")
        model_size = require_exact_str(cell["model_size"], "taxonomy.cell.model_size")
        seed = require_exact_int(cell["seed"], "taxonomy.cell.seed")
        generation_path = require_canonical_relative_path(cell["generations_path"], "taxonomy.cell.generation_path")
        generation_sha = postmortem_file_sha_from_input_binding(input_binding, generation_path)
        retained_rows = validate_generation_artifact(input_root / generation_path, cell)
        for record, retained_row in zip(records[family]["eval"], retained_rows, strict=True):
            expected = postmortem_taxonomy_row(
                record,
                retained_row,
                family=family,
                model_size=model_size,
                seed=seed,
                generation_path=generation_path,
                generation_sha256=generation_sha,
            )
            if rows[index] != expected:
                raise ValueError("Postmortem taxonomy row does not rebuild from retained generation evidence.")
            index += 1


def postmortem_cell_rows_from_outputs(
    cell_rows: Sequence[dict[str, object]],
    teacher_rows: Sequence[dict[str, object]],
    taxonomy_rows: Sequence[dict[str, object]],
    *,
    input_binding: dict[str, object],
) -> list[dict[str, object]]:
    expected: list[dict[str, object]] = []
    teacher_by_key_split: dict[tuple[str, str, int, str], list[dict[str, object]]] = {}
    for row in teacher_rows:
        key = (
            require_exact_str(row["family"], "teacher.family"),
            require_exact_str(row["model_size"], "teacher.model_size"),
            require_exact_int(row["seed"], "teacher.seed"),
            require_exact_str(row["split"], "teacher.split"),
        )
        teacher_by_key_split.setdefault(key, []).append(row)
    taxonomy_by_key: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in taxonomy_rows:
        key = (
            require_exact_str(row["family"], "taxonomy.family"),
            require_exact_str(row["model_size"], "taxonomy.model_size"),
            require_exact_int(row["seed"], "taxonomy.seed"),
        )
        taxonomy_by_key.setdefault(key, []).append(row)
    for observed, cell in zip(cell_rows, validate_postmortem_current_cells_shallow(input_binding.get("cells")), strict=True):
        observed_row = validate_postmortem_cell_metrics_row(observed)
        key = (
            require_exact_str(cell["family"], "cell.family"),
            require_exact_str(cell["model_size"], "cell.model_size"),
            require_exact_int(cell["seed"], "cell.seed"),
        )
        generation_rows = taxonomy_by_key.get(key, [])
        train_rows = teacher_by_key_split.get((*key, "train"), [])
        eval_rows = teacher_by_key_split.get((*key, "eval"), [])
        retained_exact = sum(1 for row in generation_rows if row["common"]["greedy_exact"] is True)
        expected_row = {
            **observed_row,
            "train_teacher_forced": postmortem_teacher_aggregate(train_rows),
            "eval_teacher_forced": postmortem_teacher_aggregate(eval_rows),
            "retained_greedy_exact": {"numerator": retained_exact, "denominator": EVAL_RECORDS_PER_FAMILY},
            "paired_eval_exact": postmortem_paired_eval_exact(eval_rows, [{"exact_match": row["common"]["greedy_exact"]} for row in generation_rows]),
        }
        if observed_row != expected_row:
            raise ValueError("Postmortem cell_metrics row does not rebuild from retained teacher/taxonomy rows.")
        expected.append(observed_row)
    if len(cell_rows) != len(expected):
        raise ValueError("Postmortem cell_metrics cardinality mismatch.")
    return expected


def postmortem_output_inventory(root: Path) -> list[dict[str, object]]:
    rows = inventory(root)
    expected_paths = {"cell_metrics.jsonl", "error_taxonomy.jsonl", "summary.json", "teacher_forced_rows.jsonl"}
    observed_paths = {require_canonical_relative_path(row.get("path"), "postmortem.file_inventory.path") for row in rows}
    if observed_paths != expected_paths:
        raise ValueError("Postmortem manifest inventory must bind exactly the four non-manifest, non-terminal files.")
    return rows


def read_all_from_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def postmortem_regular_file_bytes_at(root_fd: int, name: str, field_name: str) -> tuple[bytes, os.stat_result]:
    try:
        metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"{field_name} must exist.") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{field_name} must be a regular non-symlink file.")
    file_fd = os.open(name, postmortem_file_open_flags(), dir_fd=root_fd)
    try:
        opened = os.fstat(file_fd)
        if (
            opened.st_dev,
            opened.st_ino,
            stat.S_IFMT(opened.st_mode),
            opened.st_size,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            stat.S_IFMT(metadata.st_mode),
            metadata.st_size,
        ):
            raise ValueError(f"{field_name} identity changed while opening descriptor.")
        raw = read_all_from_fd(file_fd)
        observed = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (
            observed.st_dev,
            observed.st_ino,
            stat.S_IFMT(observed.st_mode),
            observed.st_size,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            stat.S_IFMT(metadata.st_mode),
            metadata.st_size,
        ):
            raise ValueError(f"{field_name} identity changed while reading descriptor.")
        return raw, metadata
    finally:
        os.close(file_fd)


def postmortem_file_sha256_at(root_fd: int, name: str) -> str:
    raw, _metadata = postmortem_regular_file_bytes_at(root_fd, name, f"postmortem.{name}")
    return sha256(raw).hexdigest()


def postmortem_output_inventory_at(root_fd: int) -> list[dict[str, object]]:
    expected_paths = ("cell_metrics.jsonl", "error_taxonomy.jsonl", "summary.json", "teacher_forced_rows.jsonl")
    rows: list[dict[str, object]] = []
    for relative_path in expected_paths:
        raw, metadata = postmortem_regular_file_bytes_at(root_fd, relative_path, f"postmortem.file_inventory.{relative_path}")
        rows.append({"path": relative_path, "sha256": sha256(raw).hexdigest(), "bytes": int(metadata.st_size)})
    return rows


def write_postmortem_bytes_at(root_fd: int, name: str, payload: bytes) -> None:
    if name not in POSTMORTEM_TERMINAL_FILE_NAMES:
        raise ValueError("Postmortem fd-relative writer only accepts declared terminal file names.")
    temp_name = f".{name}.tmp.{os.getpid()}.{time.time_ns()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    file_fd = os.open(temp_name, flags, 0o644, dir_fd=root_fd)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(file_fd, view)
            view = view[written:]
        os.fsync(file_fd)
    except Exception:
        try:
            os.unlink(temp_name, dir_fd=root_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(file_fd)
    try:
        atomic_rename_noreplace_at(root_fd, temp_name, root_fd, name, name)
    except Exception:
        try:
            os.unlink(temp_name, dir_fd=root_fd)
        except FileNotFoundError:
            pass
        raise


def write_postmortem_json_at(guard: PostmortemPublicationGuard, name: str, value: object) -> None:
    require_postmortem_temp_fd_identity(guard, f"before writing {name}")
    write_postmortem_bytes_at(guard.temp_fd, name, postmortem_json_bytes(value))
    require_postmortem_temp_fd_identity(guard, f"after writing {name}")


def write_postmortem_jsonl_at(guard: PostmortemPublicationGuard, name: str, rows: Iterable[dict[str, object]]) -> None:
    require_postmortem_temp_fd_identity(guard, f"before writing {name}")
    write_postmortem_bytes_at(guard.temp_fd, name, postmortem_jsonl_bytes(rows))
    require_postmortem_temp_fd_identity(guard, f"after writing {name}")


def write_postmortem_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(postmortem_json_bytes(value))
    os.replace(temp, path)


def write_postmortem_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(postmortem_jsonl_bytes(rows))
    os.replace(temp, path)


def read_canonical_postmortem_json_bytes(raw: bytes, field_name: str) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field_name} must be UTF-8 JSON.") from exc
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    if raw != postmortem_json_bytes(value):
        raise ValueError(f"{field_name} canonical JSON bytes mismatch.")
    return value


def read_canonical_postmortem_json(path: Path, field_name: str) -> dict[str, object]:
    return read_canonical_postmortem_json_bytes(path.read_bytes(), field_name)


def read_canonical_postmortem_json_at(root_fd: int, name: str, field_name: str) -> dict[str, object]:
    raw, _metadata = postmortem_regular_file_bytes_at(root_fd, name, field_name)
    return read_canonical_postmortem_json_bytes(raw, field_name)


def read_canonical_postmortem_jsonl_bytes(raw: bytes, field_name: str) -> list[dict[str, object]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field_name} must be UTF-8 JSONL.") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} line {line_number} must be a JSON object.")
        rows.append(value)
    if raw != postmortem_jsonl_bytes(rows):
        raise ValueError(f"{field_name} canonical JSONL bytes mismatch.")
    return rows


def read_canonical_postmortem_jsonl(path: Path, field_name: str) -> list[dict[str, object]]:
    return read_canonical_postmortem_jsonl_bytes(path.read_bytes(), field_name)


def read_canonical_postmortem_jsonl_at(root_fd: int, name: str, field_name: str) -> list[dict[str, object]]:
    raw, _metadata = postmortem_regular_file_bytes_at(root_fd, name, field_name)
    return read_canonical_postmortem_jsonl_bytes(raw, field_name)


def build_postmortem_summary(
    *,
    input_binding: dict[str, object],
    proposal_binding: dict[str, object],
    implementation_binding: dict[str, object],
    cell_rows: Sequence[dict[str, object]],
    taxonomy_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": POSTMORTEM_SCHEMA_VERSION,
        "artifact_class": POSTMORTEM_ARTIFACT_CLASS,
        "input_binding": input_binding,
        "proposal_binding": proposal_binding,
        "implementation_binding": implementation_binding,
        "configuration": input_binding["configuration"],
        "row_counts": dict(POSTMORTEM_ROW_COUNTS),
        "cells": list(cell_rows),
        "named_aggregates": postmortem_named_aggregates(taxonomy_rows),
        "array_aggregates": postmortem_array_aggregates(taxonomy_rows),
        "interpretation_limits": {
            "non_evidence": True,
            "no_causal_weight_tying_claim": True,
            "no_verdict_change": True,
            "no_010d_authority": True,
        },
        "terminal_status": "DONE",
    }


def build_postmortem_manifest(
    root: Path,
    *,
    input_binding: dict[str, object],
    proposal_binding: dict[str, object],
    implementation_binding: dict[str, object],
    exact_command: dict[str, object],
    environment: dict[str, object],
    deterministic_flags: dict[str, object],
    rng_state_contract: dict[str, bool],
    file_inventory: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": POSTMORTEM_SCHEMA_VERSION,
        "artifact_class": POSTMORTEM_ARTIFACT_CLASS,
        "input_binding": input_binding,
        "proposal_binding": proposal_binding,
        "implementation_binding": implementation_binding,
        "runner_path": implementation_binding["runner_path"],
        "exact_command": exact_command,
        "environment": environment,
        "deterministic_flags": deterministic_flags,
        "rng_state_contract": rng_state_contract,
        "configuration": input_binding["configuration"],
        "row_counts": dict(POSTMORTEM_ROW_COUNTS),
        "file_inventory": postmortem_output_inventory(root) if file_inventory is None else file_inventory,
        "terminal_status": "DONE",
    }


def write_postmortem_done(root: Path, *, input_binding: dict[str, object]) -> None:
    manifest_path = root / "manifest.json"
    write_postmortem_json(
        root / "DONE.json",
        {
            "schema_version": POSTMORTEM_SCHEMA_VERSION,
            "status": "DONE",
            "manifest_path": "manifest.json",
            "manifest_sha256": file_sha256(manifest_path),
            "input_manifest_sha256": input_binding["manifest_sha256"],
            "row_counts": dict(POSTMORTEM_ROW_COUNTS),
        },
    )


def write_postmortem_done_at(guard: PostmortemPublicationGuard, *, input_binding: dict[str, object]) -> None:
    write_postmortem_json_at(
        guard,
        "DONE.json",
        {
            "schema_version": POSTMORTEM_SCHEMA_VERSION,
            "status": "DONE",
            "manifest_path": "manifest.json",
            "manifest_sha256": postmortem_file_sha256_at(guard.temp_fd, "manifest.json"),
            "input_manifest_sha256": input_binding["manifest_sha256"],
            "row_counts": dict(POSTMORTEM_ROW_COUNTS),
        },
    )


def validate_postmortem_bindings(
    *,
    input_binding: object,
    proposal_binding: object,
    implementation_binding: object,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    input_data = require_exact_mapping(input_binding, POSTMORTEM_INPUT_BINDING_KEYS, "postmortem.input_binding")
    proposal_data = require_exact_mapping(proposal_binding, POSTMORTEM_PROPOSAL_BINDING_KEYS, "postmortem.proposal_binding")
    implementation_data = require_exact_mapping(implementation_binding, POSTMORTEM_IMPLEMENTATION_BINDING_KEYS, "postmortem.implementation_binding")
    if proposal_data != {
        "commit": POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
        "path": POSTMORTEM_PROPOSAL_PATH,
        "blob": POSTMORTEM_PROPOSAL_BLOB,
    }:
        raise ValueError("Postmortem proposal_binding mismatch.")
    implementation_commit = validate_git_sha(implementation_data["commit"], "postmortem.implementation.commit")
    if implementation_data["runner_path"] != POSTMORTEM_RUNNER_PATH:
        raise ValueError("Postmortem implementation runner_path mismatch.")
    validate_git_sha(implementation_data["runner_blob"], "postmortem.implementation.runner_blob")
    if input_data["root"] != POSTMORTEM_REQUIRED_INPUT_ROOT:
        raise ValueError("Postmortem input_binding root mismatch.")
    for key, expected in POSTMORTEM_INPUT_CHECKSUMS.items():
        binding_key = {"manifest.json": "manifest_sha256", "summary.json": "summary_sha256", "FAILED.json": "terminal_sha256"}[key]
        if input_data[binding_key] != expected:
            raise ValueError("Postmortem input top-level checksum binding mismatch.")
    if input_data["source_commit"] != POSTMORTEM_INPUT_SOURCE_COMMIT:
        raise ValueError("Postmortem input source_commit binding mismatch.")
    if input_data["configuration"] != frozen_configuration():
        raise ValueError("Postmortem input configuration binding mismatch.")
    if input_data["record_hashes"] != FEASIBILITY_RECORD_HASHES:
        raise ValueError("Postmortem input record_hashes binding mismatch.")
    inventory_rows = input_data["file_inventory"]
    if not isinstance(inventory_rows, list) or len(inventory_rows) != 49:
        raise ValueError("Postmortem input file_inventory binding must retain exactly 49 entries.")
    for index, row in enumerate(inventory_rows):
        require_exact_mapping(row, frozenset({"path", "sha256", "bytes"}), f"postmortem.input.file_inventory[{index}]")
        require_canonical_relative_path(row["path"], f"postmortem.input.file_inventory[{index}].path")
        require_sha256_hex(row["sha256"], f"postmortem.input.file_inventory[{index}].sha256")
        byte_count = require_exact_int(row["bytes"], f"postmortem.input.file_inventory[{index}].bytes")
        if byte_count < 0:
            raise ValueError("Postmortem input file_inventory byte counts must be non-negative.")
    validate_postmortem_current_cells_shallow(input_data["cells"])
    return input_data, proposal_data, implementation_data


def validate_postmortem_exact_command_object(
    value: object,
    *,
    input_root: Path,
    output_root: Path,
    accepted_proposal_commit: str,
    accepted_implementation_commit: str,
    runtime_environment: dict[str, object],
) -> dict[str, object]:
    command = require_exact_mapping(value, POSTMORTEM_EXACT_COMMAND_KEYS, "postmortem.exact_command")
    if command["executable"] != POSTMORTEM_EXECUTABLE or command["cwd"] != POSTMORTEM_CWD:
        raise ValueError("Postmortem exact_command executable/cwd mismatch.")
    environment = require_exact_mapping(command["environment"], frozenset(POSTMORTEM_EXECVE_ENVIRONMENT), "postmortem.exact_command.environment")
    if environment != POSTMORTEM_EXECVE_ENVIRONMENT:
        raise ValueError("Postmortem exact_command environment must be the four-key execve environment.")
    if environment == runtime_environment:
        raise ValueError("Postmortem exact_command environment must remain distinct from the copied feasibility_005 runtime environment.")
    argv = command["argv"]
    if not isinstance(argv, list):
        raise ValueError("Postmortem exact_command.argv must be a JSON array.")
    validate_postmortem_exact_argv(
        argv,
        input_root=input_root,
        output_root=output_root,
        accepted_proposal_commit=accepted_proposal_commit,
        accepted_implementation_commit=accepted_implementation_commit,
    )
    if argv[9] != argv[23] or argv[9] != accepted_implementation_commit:
        raise ValueError("Postmortem exact_command must duplicate the accepted implementation commit at argv indexes 9 and 23.")
    if argv[21] != accepted_proposal_commit or argv[7] != POSTMORTEM_VERIFIER_SHA256:
        raise ValueError("Postmortem exact_command proposal/verifier binding mismatch.")
    return command


def expected_postmortem_input_binding_for_posthoc_validation() -> dict[str, object]:
    binding = validate_postmortem_input_shallow(Path(POSTMORTEM_REQUIRED_INPUT_ROOT))["input_binding"]
    return require_exact_mapping(binding, POSTMORTEM_INPUT_BINDING_KEYS, "postmortem.expected_input_binding")


def postmortem_path_identity(path: Path, *, field_name: str, directory: bool) -> tuple[int, int, int]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{field_name} must exist for postmortem publication.") from exc
    file_type = stat.S_IFMT(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{field_name} must be a real non-symlink path.")
    if directory:
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{field_name} must be a real non-symlink directory.")
    elif not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{field_name} must be a regular non-symlink file.")
    return (int(metadata.st_dev), int(metadata.st_ino), int(file_type))


def postmortem_metadata_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (int(metadata.st_dev), int(metadata.st_ino), int(stat.S_IFMT(metadata.st_mode)))


def postmortem_directory_open_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def postmortem_file_open_flags() -> int:
    return os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def postmortem_fd_identity(fd: int, *, field_name: str, directory: bool) -> tuple[int, int, int]:
    metadata = os.fstat(fd)
    file_type = stat.S_IFMT(metadata.st_mode)
    if directory:
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{field_name} descriptor must be a real directory.")
    elif not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{field_name} descriptor must be a regular file.")
    return (int(metadata.st_dev), int(metadata.st_ino), int(file_type))


def postmortem_dir_entry_exists(parent_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False


def postmortem_dir_entry_identity(parent_fd: int, name: str, *, field_name: str, directory: bool) -> tuple[int, int, int]:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"{field_name} must exist for postmortem publication.") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{field_name} must be a real non-symlink path.")
    if directory:
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{field_name} must be a real non-symlink directory.")
    elif not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{field_name} must be a regular non-symlink file.")
    return postmortem_metadata_identity(metadata)


def postmortem_file_sha256_from_fd(fd: int) -> str:
    digest = sha256()
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def require_postmortem_publication_guard_paths(temp_root: Path, output_root: Path) -> None:
    if temp_root.parent != output_root.parent or temp_root.name != output_root.name + ".tmp":
        raise ValueError("Postmortem temp root must be the exact same-parent .tmp path for the output root.")


def open_postmortem_publication_parent(output_root: Path) -> tuple[int, tuple[int, int, int]]:
    parent_path = output_root.parent
    path_identity = postmortem_path_identity(parent_path, field_name="postmortem artifact parent", directory=True)
    parent_fd = os.open(parent_path, postmortem_directory_open_flags())
    try:
        fd_identity = postmortem_fd_identity(parent_fd, field_name="postmortem artifact parent", directory=True)
        if fd_identity != path_identity:
            raise ValueError("Postmortem artifact parent identity changed while opening descriptor.")
        return parent_fd, fd_identity
    except Exception:
        os.close(parent_fd)
        raise


def open_postmortem_publication_guard(temp_root: Path, output_root: Path) -> PostmortemPublicationGuard:
    require_postmortem_publication_guard_paths(temp_root, output_root)
    parent_fd, parent_identity = open_postmortem_publication_parent(output_root)
    temp_fd = -1
    try:
        temp_identity = postmortem_dir_entry_identity(
            parent_fd,
            temp_root.name,
            field_name="postmortem temp root",
            directory=True,
        )
        temp_fd = os.open(temp_root.name, postmortem_directory_open_flags(), dir_fd=parent_fd)
        if postmortem_fd_identity(temp_fd, field_name="postmortem temp root", directory=True) != temp_identity:
            raise ValueError("Postmortem temp root identity changed while opening descriptor.")
        return PostmortemPublicationGuard(
            parent_fd=parent_fd,
            temp_fd=temp_fd,
            parent_path=output_root.parent,
            temp_root=temp_root,
            output_root=output_root,
            parent_identity=parent_identity,
            temp_identity=temp_identity,
        )
    except Exception:
        if temp_fd >= 0:
            os.close(temp_fd)
        os.close(parent_fd)
        raise


def create_postmortem_publication_guard(output_root: Path) -> PostmortemPublicationGuard:
    temp_root = output_root.with_name(output_root.name + ".tmp")
    require_postmortem_publication_guard_paths(temp_root, output_root)
    parent_fd, parent_identity = open_postmortem_publication_parent(output_root)
    temp_fd = -1
    try:
        if postmortem_dir_entry_exists(parent_fd, output_root.name):
            raise FileExistsError(f"Refusing to overwrite existing postmortem root: {output_root}")
        if postmortem_dir_entry_exists(parent_fd, temp_root.name):
            raise FileExistsError(f"Temporary postmortem root already exists: {temp_root}")
        os.mkdir(temp_root.name, mode=0o755, dir_fd=parent_fd)
        temp_identity = postmortem_dir_entry_identity(
            parent_fd,
            temp_root.name,
            field_name="postmortem temp root",
            directory=True,
        )
        temp_fd = os.open(temp_root.name, postmortem_directory_open_flags(), dir_fd=parent_fd)
        if postmortem_fd_identity(temp_fd, field_name="postmortem temp root", directory=True) != temp_identity:
            raise ValueError("Postmortem temp root identity changed while opening descriptor.")
        return PostmortemPublicationGuard(
            parent_fd=parent_fd,
            temp_fd=temp_fd,
            parent_path=output_root.parent,
            temp_root=temp_root,
            output_root=output_root,
            parent_identity=parent_identity,
            temp_identity=temp_identity,
        )
    except Exception:
        if temp_fd >= 0:
            os.close(temp_fd)
        os.close(parent_fd)
        raise


def require_postmortem_guard_open(guard: PostmortemPublicationGuard) -> None:
    if guard.closed:
        raise ValueError("Postmortem publication guard is closed.")


def require_postmortem_parent_identity(guard: PostmortemPublicationGuard, stage: str) -> None:
    require_postmortem_guard_open(guard)
    if postmortem_fd_identity(guard.parent_fd, field_name="postmortem artifact parent", directory=True) != guard.parent_identity:
        raise ValueError(f"Postmortem artifact parent identity changed {stage}.")


def require_postmortem_temp_fd_identity(guard: PostmortemPublicationGuard, stage: str) -> None:
    require_postmortem_guard_open(guard)
    if postmortem_fd_identity(guard.temp_fd, field_name="postmortem temp root", directory=True) != guard.temp_identity:
        raise ValueError(f"Postmortem temp descriptor identity changed {stage}.")


def require_postmortem_temp_path_identity(guard: PostmortemPublicationGuard, stage: str) -> None:
    require_postmortem_parent_identity(guard, stage)
    if postmortem_dir_entry_identity(
        guard.parent_fd,
        guard.temp_name,
        field_name="postmortem temp root",
        directory=True,
    ) != guard.temp_identity:
        raise ValueError(f"Postmortem temp root identity changed {stage}.")


def require_postmortem_output_path_identity(guard: PostmortemPublicationGuard, stage: str) -> None:
    require_postmortem_parent_identity(guard, stage)
    if postmortem_dir_entry_identity(
        guard.parent_fd,
        guard.output_name,
        field_name="postmortem output root",
        directory=True,
    ) != guard.temp_identity:
        raise ValueError(f"Postmortem output root identity changed {stage}.")


def postmortem_terminal_root_identity(root: Path) -> tuple[int, int, int]:
    return postmortem_path_identity(root, field_name="postmortem terminal root", directory=True)


def require_postmortem_terminal_root_identity(root: Path, expected: tuple[int, int, int], stage: str) -> None:
    observed = postmortem_terminal_root_identity(root)
    if observed != expected:
        raise ValueError(f"Postmortem terminal root identity changed {stage}.")


def postmortem_terminal_file_identity(path: Path) -> tuple[int, int, int, str, int]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError("Postmortem terminal root must contain exactly the six required files.") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Postmortem terminal root entries must be regular non-symlink files.")
    digest = sha256_regular_file_no_follow(path, metadata)
    observed = path.lstat()
    if (observed.st_dev, observed.st_ino, stat.S_IFMT(observed.st_mode), observed.st_size) != (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
    ):
        raise ValueError("Postmortem terminal file identity changed while fingerprinting.")
    return (int(metadata.st_dev), int(metadata.st_ino), int(stat.S_IFMT(metadata.st_mode)), digest, int(metadata.st_size))


def postmortem_terminal_file_identity_at(root_fd: int, name: str) -> tuple[int, int, int, str, int]:
    try:
        metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError("Postmortem terminal root must contain exactly the six required files.") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Postmortem terminal root entries must be regular non-symlink files.")
    file_fd = os.open(name, postmortem_file_open_flags(), dir_fd=root_fd)
    try:
        opened = os.fstat(file_fd)
        if (
            opened.st_dev,
            opened.st_ino,
            stat.S_IFMT(opened.st_mode),
            opened.st_size,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            stat.S_IFMT(metadata.st_mode),
            metadata.st_size,
        ):
            raise ValueError("Postmortem terminal file identity changed while opening descriptor.")
        digest = postmortem_file_sha256_from_fd(file_fd)
        observed = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (
            observed.st_dev,
            observed.st_ino,
            stat.S_IFMT(observed.st_mode),
            observed.st_size,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            stat.S_IFMT(metadata.st_mode),
            metadata.st_size,
        ):
            raise ValueError("Postmortem terminal file identity changed while fingerprinting.")
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(stat.S_IFMT(metadata.st_mode)),
            digest,
            int(metadata.st_size),
        )
    finally:
        os.close(file_fd)


def validate_postmortem_terminal_root_entries(root: Path) -> tuple[tuple[str, int, int, int, str, int], ...]:
    postmortem_terminal_root_identity(root)
    try:
        entry_names = sorted(entry.name for entry in os.scandir(root))
    except FileNotFoundError as exc:
        raise ValueError("Postmortem terminal root must exist before publication.") from exc
    if entry_names != sorted(POSTMORTEM_TERMINAL_FILE_NAMES):
        raise ValueError("Postmortem terminal root must contain exactly six root-level entries with the required names.")
    rows: list[tuple[str, int, int, int, str, int]] = []
    for relative_path in POSTMORTEM_TERMINAL_FILE_NAMES:
        device, inode, file_type, digest, byte_count = postmortem_terminal_file_identity(root / relative_path)
        rows.append((relative_path, device, inode, file_type, digest, byte_count))
    return tuple(rows)


def validate_postmortem_terminal_root_entries_at(root_fd: int) -> tuple[tuple[str, int, int, int, str, int], ...]:
    try:
        entry_names = sorted(os.listdir(root_fd))
    except FileNotFoundError as exc:
        raise ValueError("Postmortem terminal root must exist before publication.") from exc
    if entry_names != sorted(POSTMORTEM_TERMINAL_FILE_NAMES):
        raise ValueError("Postmortem terminal root must contain exactly six root-level entries with the required names.")
    rows: list[tuple[str, int, int, int, str, int]] = []
    for relative_path in POSTMORTEM_TERMINAL_FILE_NAMES:
        device, inode, file_type, digest, byte_count = postmortem_terminal_file_identity_at(root_fd, relative_path)
        rows.append((relative_path, device, inode, file_type, digest, byte_count))
    return tuple(rows)


def postmortem_terminal_fingerprint(root: Path) -> tuple[tuple[object, ...], ...]:
    root_device, root_inode, root_type = postmortem_terminal_root_identity(root)
    file_rows = validate_postmortem_terminal_root_entries(root)
    return (("__root__", root_device, root_inode, root_type), *file_rows)


def postmortem_terminal_fingerprint_at(guard: PostmortemPublicationGuard) -> tuple[tuple[object, ...], ...]:
    require_postmortem_temp_fd_identity(guard, "while fingerprinting")
    root_device, root_inode, root_type = postmortem_fd_identity(
        guard.temp_fd,
        field_name="postmortem terminal root",
        directory=True,
    )
    file_rows = validate_postmortem_terminal_root_entries_at(guard.temp_fd)
    return (("__root__", root_device, root_inode, root_type), *file_rows)


def validate_postmortem_terminal_root(
    root: Path,
    output_root: Path,
    *,
    expected_input_binding: dict[str, object] | None = None,
    expected_proposal_binding: dict[str, object] | None = None,
    expected_implementation_binding: dict[str, object] | None = None,
    expected_source_snapshot: SourceSnapshot | None = None,
    expected_teacher_rows: Sequence[dict[str, object]] | None = None,
    expected_cell_rows: Sequence[dict[str, object]] | None = None,
    expected_taxonomy_rows: Sequence[dict[str, object]] | None = None,
) -> None:
    validate_postmortem_terminal_root_entries(root)
    terminals = [path.name for path in (root / "DONE.json", root / "FAILED.json") if path.exists()]
    if terminals != ["DONE.json"]:
        raise ValueError("Postmortem root must contain DONE.json only; finalized FAILED roots are forbidden.")
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise ValueError("Postmortem root must contain manifest.json and summary.json before publication.")
    manifest = read_canonical_postmortem_json(manifest_path, "postmortem.manifest")
    summary = read_canonical_postmortem_json(summary_path, "postmortem.summary")
    done = read_canonical_postmortem_json(root / "DONE.json", "postmortem.done")
    require_exact_mapping(manifest, POSTMORTEM_MANIFEST_KEYS, "postmortem.manifest")
    require_exact_mapping(summary, POSTMORTEM_SUMMARY_KEYS, "postmortem.summary")
    require_exact_mapping(done, POSTMORTEM_DONE_KEYS, "postmortem.done")
    for record_name, record in (("manifest", manifest), ("summary", summary), ("done", done)):
        if record.get("schema_version") != POSTMORTEM_SCHEMA_VERSION:
            raise ValueError(f"Postmortem {record_name} schema_version mismatch.")
    if manifest["artifact_class"] != POSTMORTEM_ARTIFACT_CLASS or summary["artifact_class"] != POSTMORTEM_ARTIFACT_CLASS:
        raise ValueError("Postmortem artifact_class mismatch.")
    if manifest["terminal_status"] != "DONE" or summary["terminal_status"] != "DONE" or done["status"] != "DONE":
        raise ValueError("Postmortem terminal status mismatch.")
    input_binding, proposal_binding, implementation_binding = validate_postmortem_bindings(
        input_binding=manifest["input_binding"],
        proposal_binding=manifest["proposal_binding"],
        implementation_binding=manifest["implementation_binding"],
    )
    if expected_input_binding is not None and input_binding != expected_input_binding:
        raise ValueError("Postmortem manifest input_binding does not match the frozen in-memory preflight binding.")
    if expected_proposal_binding is not None and proposal_binding != expected_proposal_binding:
        raise ValueError("Postmortem manifest proposal_binding does not match the frozen in-memory proposal binding.")
    if expected_implementation_binding is not None and implementation_binding != expected_implementation_binding:
        differing_keys = sorted(
            key
            for key in POSTMORTEM_IMPLEMENTATION_BINDING_KEYS
            if implementation_binding.get(key) != expected_implementation_binding.get(key)
        )
        raise ValueError(
            "Postmortem manifest implementation_binding does not match the frozen in-memory implementation binding"
            f" for {','.join(differing_keys)}."
        )
    if expected_source_snapshot is not None and postmortem_implementation_binding(expected_source_snapshot) != implementation_binding:
        raise ValueError("Postmortem implementation binding does not match the frozen source snapshot.")
    if expected_input_binding is None:
        expected_input_binding = expected_postmortem_input_binding_for_posthoc_validation()
    if input_binding != expected_input_binding:
        raise ValueError("Postmortem manifest input_binding does not match the canonical input binding.")
    if summary["input_binding"] != input_binding or summary["proposal_binding"] != proposal_binding or summary["implementation_binding"] != implementation_binding:
        raise ValueError("Postmortem summary/manifest binding mismatch.")
    if manifest["runner_path"] != implementation_binding["runner_path"]:
        raise ValueError("Postmortem runner_path must equal implementation_binding.runner_path.")
    if manifest["configuration"] != input_binding["configuration"] or summary["configuration"] != input_binding["configuration"]:
        raise ValueError("Postmortem configuration must be copied from input_binding.")
    if manifest["row_counts"] != POSTMORTEM_ROW_COUNTS or summary["row_counts"] != POSTMORTEM_ROW_COUNTS or done["row_counts"] != POSTMORTEM_ROW_COUNTS:
        raise ValueError("Postmortem row_counts mismatch.")
    if manifest["environment"] != FEASIBILITY_REQUIRED_RUNTIME_ENV:
        raise ValueError("Postmortem manifest environment mismatch.")
    validate_postmortem_exact_command_object(
        manifest["exact_command"],
        input_root=Path(POSTMORTEM_REQUIRED_INPUT_ROOT),
        output_root=output_root,
        accepted_proposal_commit=POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
        accepted_implementation_commit=require_exact_str(implementation_binding["commit"], "postmortem.implementation.commit"),
        runtime_environment=manifest["environment"],
    )
    validate_current_deterministic_flags(manifest["deterministic_flags"])
    rng_contract = require_exact_mapping(manifest["rng_state_contract"], POSTMORTEM_RNG_CONTRACT_KEYS, "postmortem.rng_state_contract")
    if any(value is not True for value in rng_contract.values()):
        raise ValueError("Postmortem RNG state contract must contain only true values.")
    if summary["interpretation_limits"] != {
        "non_evidence": True,
        "no_causal_weight_tying_claim": True,
        "no_verdict_change": True,
        "no_010d_authority": True,
    }:
        raise ValueError("Postmortem interpretation_limits mismatch.")
    if done["manifest_path"] != "manifest.json" or done["manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("Postmortem DONE manifest checksum mismatch.")
    if done["input_manifest_sha256"] != input_binding["manifest_sha256"]:
        raise ValueError("Postmortem DONE input manifest checksum mismatch.")
    if manifest["file_inventory"] != postmortem_output_inventory(root):
        raise ValueError("Postmortem manifest file_inventory mismatch.")
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    expected_files = {"teacher_forced_rows.jsonl", "cell_metrics.jsonl", "error_taxonomy.jsonl", "summary.json", "manifest.json", "DONE.json"}
    if actual_files != expected_files:
        raise ValueError("Postmortem root contains incomplete or extra files.")
    teacher_rows = read_canonical_postmortem_jsonl(root / "teacher_forced_rows.jsonl", "postmortem.teacher_forced_rows")
    cell_rows = read_canonical_postmortem_jsonl(root / "cell_metrics.jsonl", "postmortem.cell_metrics")
    taxonomy_rows = read_canonical_postmortem_jsonl(root / "error_taxonomy.jsonl", "postmortem.error_taxonomy")
    records = grouped_records()
    validate_postmortem_teacher_rows(teacher_rows, input_binding=input_binding, records=records)
    validate_postmortem_taxonomy_rows(taxonomy_rows, input_binding=input_binding, records=records)
    rebuilt_cells = postmortem_cell_rows_from_outputs(cell_rows, teacher_rows, taxonomy_rows, input_binding=input_binding)
    if expected_teacher_rows is not None and teacher_rows != list(expected_teacher_rows):
        raise ValueError("Postmortem teacher_forced_rows changed after construction.")
    if expected_cell_rows is not None and rebuilt_cells != list(expected_cell_rows):
        raise ValueError("Postmortem cell metrics changed after construction.")
    if expected_taxonomy_rows is not None and taxonomy_rows != list(expected_taxonomy_rows):
        raise ValueError("Postmortem error_taxonomy rows changed after construction.")
    if summary["cells"] != rebuilt_cells:
        raise ValueError("Postmortem summary cells do not match cell_metrics rows.")
    if summary["named_aggregates"] != postmortem_named_aggregates(taxonomy_rows):
        raise ValueError("Postmortem Named aggregates do not rebuild from taxonomy rows.")
    if summary["array_aggregates"] != postmortem_array_aggregates(taxonomy_rows):
        raise ValueError("Postmortem Array aggregates do not rebuild from taxonomy rows.")


def validate_postmortem_terminal_root_at(
    guard: PostmortemPublicationGuard,
    output_root: Path,
    *,
    expected_input_binding: dict[str, object] | None = None,
    expected_proposal_binding: dict[str, object] | None = None,
    expected_implementation_binding: dict[str, object] | None = None,
    expected_source_snapshot: SourceSnapshot | None = None,
    expected_teacher_rows: Sequence[dict[str, object]] | None = None,
    expected_cell_rows: Sequence[dict[str, object]] | None = None,
    expected_taxonomy_rows: Sequence[dict[str, object]] | None = None,
) -> None:
    require_postmortem_temp_fd_identity(guard, "during fd-relative terminal validation")
    entry_rows = validate_postmortem_terminal_root_entries_at(guard.temp_fd)
    if {row[0] for row in entry_rows} != set(POSTMORTEM_TERMINAL_FILE_NAMES):
        raise ValueError("Postmortem fd terminal root must contain exactly the six required files.")
    manifest = read_canonical_postmortem_json_at(guard.temp_fd, "manifest.json", "postmortem.manifest")
    summary = read_canonical_postmortem_json_at(guard.temp_fd, "summary.json", "postmortem.summary")
    done = read_canonical_postmortem_json_at(guard.temp_fd, "DONE.json", "postmortem.done")
    require_exact_mapping(manifest, POSTMORTEM_MANIFEST_KEYS, "postmortem.manifest")
    require_exact_mapping(summary, POSTMORTEM_SUMMARY_KEYS, "postmortem.summary")
    require_exact_mapping(done, POSTMORTEM_DONE_KEYS, "postmortem.done")
    for record_name, record in (("manifest", manifest), ("summary", summary), ("done", done)):
        if record.get("schema_version") != POSTMORTEM_SCHEMA_VERSION:
            raise ValueError(f"Postmortem {record_name} schema_version mismatch.")
    if manifest["artifact_class"] != POSTMORTEM_ARTIFACT_CLASS or summary["artifact_class"] != POSTMORTEM_ARTIFACT_CLASS:
        raise ValueError("Postmortem artifact_class mismatch.")
    if manifest["terminal_status"] != "DONE" or summary["terminal_status"] != "DONE" or done["status"] != "DONE":
        raise ValueError("Postmortem terminal status mismatch.")
    input_binding, proposal_binding, implementation_binding = validate_postmortem_bindings(
        input_binding=manifest["input_binding"],
        proposal_binding=manifest["proposal_binding"],
        implementation_binding=manifest["implementation_binding"],
    )
    if expected_input_binding is not None and input_binding != expected_input_binding:
        raise ValueError("Postmortem manifest input_binding does not match the frozen in-memory preflight binding.")
    if expected_proposal_binding is not None and proposal_binding != expected_proposal_binding:
        raise ValueError("Postmortem manifest proposal_binding does not match the frozen in-memory proposal binding.")
    if expected_implementation_binding is not None and implementation_binding != expected_implementation_binding:
        differing_keys = sorted(
            key
            for key in POSTMORTEM_IMPLEMENTATION_BINDING_KEYS
            if implementation_binding.get(key) != expected_implementation_binding.get(key)
        )
        raise ValueError(
            "Postmortem manifest implementation_binding does not match the frozen in-memory implementation binding"
            f" for {','.join(differing_keys)}."
        )
    if expected_source_snapshot is not None and postmortem_implementation_binding(expected_source_snapshot) != implementation_binding:
        raise ValueError("Postmortem implementation binding does not match the frozen source snapshot.")
    if expected_input_binding is None:
        expected_input_binding = expected_postmortem_input_binding_for_posthoc_validation()
    if input_binding != expected_input_binding:
        raise ValueError("Postmortem manifest input_binding does not match the canonical input binding.")
    if summary["input_binding"] != input_binding or summary["proposal_binding"] != proposal_binding or summary["implementation_binding"] != implementation_binding:
        raise ValueError("Postmortem summary/manifest binding mismatch.")
    if manifest["runner_path"] != implementation_binding["runner_path"]:
        raise ValueError("Postmortem runner_path must equal implementation_binding.runner_path.")
    if manifest["configuration"] != input_binding["configuration"] or summary["configuration"] != input_binding["configuration"]:
        raise ValueError("Postmortem configuration must be copied from input_binding.")
    if manifest["row_counts"] != POSTMORTEM_ROW_COUNTS or summary["row_counts"] != POSTMORTEM_ROW_COUNTS or done["row_counts"] != POSTMORTEM_ROW_COUNTS:
        raise ValueError("Postmortem row_counts mismatch.")
    if manifest["environment"] != FEASIBILITY_REQUIRED_RUNTIME_ENV:
        raise ValueError("Postmortem manifest environment mismatch.")
    validate_postmortem_exact_command_object(
        manifest["exact_command"],
        input_root=Path(POSTMORTEM_REQUIRED_INPUT_ROOT),
        output_root=output_root,
        accepted_proposal_commit=POSTMORTEM_ACCEPTED_PROPOSAL_COMMIT,
        accepted_implementation_commit=require_exact_str(implementation_binding["commit"], "postmortem.implementation.commit"),
        runtime_environment=manifest["environment"],
    )
    validate_current_deterministic_flags(manifest["deterministic_flags"])
    rng_contract = require_exact_mapping(manifest["rng_state_contract"], POSTMORTEM_RNG_CONTRACT_KEYS, "postmortem.rng_state_contract")
    if any(value is not True for value in rng_contract.values()):
        raise ValueError("Postmortem RNG state contract must contain only true values.")
    if summary["interpretation_limits"] != {
        "non_evidence": True,
        "no_causal_weight_tying_claim": True,
        "no_verdict_change": True,
        "no_010d_authority": True,
    }:
        raise ValueError("Postmortem interpretation_limits mismatch.")
    if done["manifest_path"] != "manifest.json" or done["manifest_sha256"] != postmortem_file_sha256_at(guard.temp_fd, "manifest.json"):
        raise ValueError("Postmortem DONE manifest checksum mismatch.")
    if done["input_manifest_sha256"] != input_binding["manifest_sha256"]:
        raise ValueError("Postmortem DONE input manifest checksum mismatch.")
    if manifest["file_inventory"] != postmortem_output_inventory_at(guard.temp_fd):
        raise ValueError("Postmortem manifest file_inventory mismatch.")
    teacher_rows = read_canonical_postmortem_jsonl_at(guard.temp_fd, "teacher_forced_rows.jsonl", "postmortem.teacher_forced_rows")
    cell_rows = read_canonical_postmortem_jsonl_at(guard.temp_fd, "cell_metrics.jsonl", "postmortem.cell_metrics")
    taxonomy_rows = read_canonical_postmortem_jsonl_at(guard.temp_fd, "error_taxonomy.jsonl", "postmortem.error_taxonomy")
    records = grouped_records()
    validate_postmortem_teacher_rows(teacher_rows, input_binding=input_binding, records=records)
    validate_postmortem_taxonomy_rows(taxonomy_rows, input_binding=input_binding, records=records)
    rebuilt_cells = postmortem_cell_rows_from_outputs(cell_rows, teacher_rows, taxonomy_rows, input_binding=input_binding)
    if expected_teacher_rows is not None and teacher_rows != list(expected_teacher_rows):
        raise ValueError("Postmortem teacher_forced_rows changed after construction.")
    if expected_cell_rows is not None and rebuilt_cells != list(expected_cell_rows):
        raise ValueError("Postmortem cell metrics changed after construction.")
    if expected_taxonomy_rows is not None and taxonomy_rows != list(expected_taxonomy_rows):
        raise ValueError("Postmortem error_taxonomy rows changed after construction.")
    if summary["cells"] != rebuilt_cells:
        raise ValueError("Postmortem summary cells do not match cell_metrics rows.")
    if summary["named_aggregates"] != postmortem_named_aggregates(taxonomy_rows):
        raise ValueError("Postmortem Named aggregates do not rebuild from taxonomy rows.")
    if summary["array_aggregates"] != postmortem_array_aggregates(taxonomy_rows):
        raise ValueError("Postmortem Array aggregates do not rebuild from taxonomy rows.")


def validate_postmortem_terminal_inventory_snapshot(root: Path) -> None:
    validate_postmortem_terminal_root_entries(root)
    manifest_path = root / "manifest.json"
    done_path = root / "DONE.json"
    if not manifest_path.is_file() or not done_path.is_file():
        raise ValueError("Postmortem root lacks terminal inventory files.")
    manifest = read_canonical_postmortem_json(manifest_path, "postmortem.final_manifest")
    done = read_canonical_postmortem_json(done_path, "postmortem.final_done")
    expected_inventory = postmortem_output_inventory(root)
    if manifest.get("file_inventory") != expected_inventory:
        raise ValueError("Postmortem final manifest inventory changed.")
    if done.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Postmortem final DONE manifest checksum changed.")
    for row in expected_inventory:
        path = root / require_canonical_relative_path(row.get("path"), "postmortem.final_inventory.path")
        if row.get("sha256") != file_sha256(path) or row.get("bytes") != path.stat().st_size:
            raise ValueError("Postmortem final inventory checksum or byte count mismatch.")


def validate_postmortem_terminal_inventory_snapshot_at(guard: PostmortemPublicationGuard) -> None:
    require_postmortem_temp_fd_identity(guard, "during fd-relative terminal inventory validation")
    validate_postmortem_terminal_root_entries_at(guard.temp_fd)
    manifest = read_canonical_postmortem_json_at(guard.temp_fd, "manifest.json", "postmortem.final_manifest")
    done = read_canonical_postmortem_json_at(guard.temp_fd, "DONE.json", "postmortem.final_done")
    expected_inventory = postmortem_output_inventory_at(guard.temp_fd)
    if manifest.get("file_inventory") != expected_inventory:
        raise ValueError("Postmortem final manifest inventory changed.")
    if done.get("manifest_sha256") != postmortem_file_sha256_at(guard.temp_fd, "manifest.json"):
        raise ValueError("Postmortem final DONE manifest checksum changed.")
    for row in expected_inventory:
        relative_path = require_canonical_relative_path(row.get("path"), "postmortem.final_inventory.path")
        raw, metadata = postmortem_regular_file_bytes_at(guard.temp_fd, relative_path, f"postmortem.final_inventory.{relative_path}")
        if row.get("sha256") != sha256(raw).hexdigest() or row.get("bytes") != metadata.st_size:
            raise ValueError("Postmortem final inventory checksum or byte count mismatch.")


def demote_postmortem_terminal_markers_at_fd(root_fd: int) -> None:
    for name in ("DONE.json", "FAILED.json"):
        try:
            os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        for attempt in range(1000):
            quarantine_name = f".{name}.demoted.{os.getpid()}.{time.time_ns()}.{attempt}"
            if postmortem_dir_entry_exists(root_fd, quarantine_name):
                continue
            try:
                atomic_rename_noreplace_at(root_fd, name, root_fd, quarantine_name, quarantine_name)
                break
            except FileExistsError:
                continue
        else:
            raise FeasibilityPublicationError(f"Unable to demote postmortem terminal marker {name}.")


def demote_postmortem_terminal_markers_at_path_if_safe(
    root: Path,
    *,
    trusted_identity: tuple[int, int, int] | None = None,
) -> None:
    try:
        root_metadata = root.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        return
    if trusted_identity is not None and postmortem_metadata_identity(root_metadata) == trusted_identity:
        return
    root_fd = os.open(root, postmortem_directory_open_flags())
    try:
        if postmortem_fd_identity(root_fd, field_name="postmortem terminal root", directory=True) != postmortem_metadata_identity(root_metadata):
            raise ValueError("Postmortem terminal root identity changed while opening demotion descriptor.")
        demote_postmortem_terminal_markers_at_fd(root_fd)
    finally:
        os.close(root_fd)


def demote_postmortem_terminal_markers(root: Path | PostmortemPublicationGuard) -> None:
    if isinstance(root, PostmortemPublicationGuard):
        demote_postmortem_terminal_markers_at_fd(root.temp_fd)
        demote_postmortem_terminal_markers_at_path_if_safe(root.temp_root, trusted_identity=root.temp_identity)
        return
    demote_postmortem_terminal_markers_at_path_if_safe(root)


def demote_postmortem_collision_quarantine_if_directory(parent_fd: int, quarantine_name: str) -> None:
    try:
        metadata = os.stat(quarantine_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        return
    quarantine_fd = os.open(quarantine_name, postmortem_directory_open_flags(), dir_fd=parent_fd)
    try:
        if postmortem_fd_identity(quarantine_fd, field_name="postmortem temp collision quarantine", directory=True) != postmortem_metadata_identity(metadata):
            raise ValueError("Postmortem collision quarantine identity changed while opening descriptor.")
        demote_postmortem_terminal_markers_at_fd(quarantine_fd)
    finally:
        os.close(quarantine_fd)


def preserve_postmortem_temp_collision(temp_root: Path | PostmortemPublicationGuard) -> Path | None:
    if isinstance(temp_root, PostmortemPublicationGuard):
        guard = temp_root
        if not postmortem_dir_entry_exists(guard.parent_fd, guard.temp_name):
            return None
        for attempt in range(1000):
            quarantine_name = f".{guard.temp_name}.collision.{os.getpid()}.{time.time_ns()}.{attempt}"
            quarantine_path = guard.temp_root.with_name(quarantine_name)
            try:
                atomic_rename_noreplace_at(
                    guard.parent_fd,
                    guard.temp_name,
                    guard.parent_fd,
                    quarantine_name,
                    quarantine_path,
                )
                demote_postmortem_collision_quarantine_if_directory(guard.parent_fd, quarantine_name)
                return quarantine_path
            except FileExistsError:
                continue
        raise FeasibilityPublicationError("Unable to preserve occupied postmortem temp collision.")
    if not temp_root.exists() and not temp_root.is_symlink():
        return None
    for attempt in range(1000):
        quarantine = temp_root.with_name(f".{temp_root.name}.collision.{os.getpid()}.{time.time_ns()}.{attempt}")
        try:
            atomic_rename_noreplace(temp_root, quarantine)
            demote_postmortem_terminal_markers_at_path_if_safe(quarantine)
            return quarantine
        except FileExistsError:
            continue
    raise FeasibilityPublicationError("Unable to preserve occupied postmortem temp collision.")


def publish_postmortem_root(
    temp_root: Path,
    output_root: Path,
    *,
    expected_input_binding: dict[str, object],
    expected_proposal_binding: dict[str, object],
    expected_implementation_binding: dict[str, object],
    expected_source_snapshot: SourceSnapshot,
    expected_teacher_rows: Sequence[dict[str, object]],
    expected_cell_rows: Sequence[dict[str, object]],
    expected_taxonomy_rows: Sequence[dict[str, object]],
    final_callback: Callable[[], None] | None = None,
    publication_guard: PostmortemPublicationGuard | None = None,
) -> None:
    guard = publication_guard
    owned_guard = False
    if guard is None:
        guard = open_postmortem_publication_guard(temp_root, output_root)
        owned_guard = True
    try:
        if guard.temp_root != temp_root or guard.output_root != output_root:
            raise ValueError("Postmortem publication guard paths do not match the requested roots.")
        if postmortem_dir_entry_exists(guard.parent_fd, guard.output_name):
            raise FileExistsError(f"Refusing to overwrite existing postmortem root: {output_root}")
        require_postmortem_temp_fd_identity(guard, "before validation")
        require_postmortem_temp_path_identity(guard, "before validation")
        validate_postmortem_terminal_root_at(
            guard,
            output_root,
            expected_input_binding=expected_input_binding,
            expected_proposal_binding=expected_proposal_binding,
            expected_implementation_binding=expected_implementation_binding,
            expected_source_snapshot=expected_source_snapshot,
            expected_teacher_rows=expected_teacher_rows,
            expected_cell_rows=expected_cell_rows,
            expected_taxonomy_rows=expected_taxonomy_rows,
        )
        require_postmortem_temp_fd_identity(guard, "after initial validation")
        require_postmortem_temp_path_identity(guard, "after initial validation")
        fingerprint = postmortem_terminal_fingerprint_at(guard)
        if final_callback is not None:
            final_callback()
        require_postmortem_temp_fd_identity(guard, "after validation callback")
        require_postmortem_temp_path_identity(guard, "after validation callback")
        validate_postmortem_terminal_root_at(
            guard,
            output_root,
            expected_input_binding=expected_input_binding,
            expected_proposal_binding=expected_proposal_binding,
            expected_implementation_binding=expected_implementation_binding,
            expected_source_snapshot=expected_source_snapshot,
            expected_teacher_rows=expected_teacher_rows,
            expected_cell_rows=expected_cell_rows,
            expected_taxonomy_rows=expected_taxonomy_rows,
        )
        require_postmortem_temp_fd_identity(guard, "after callback validation")
        require_postmortem_temp_path_identity(guard, "after callback validation")
        if postmortem_terminal_fingerprint_at(guard) != fingerprint:
            raise ValueError("Postmortem terminal files changed after validation.")
        validate_postmortem_terminal_inventory_snapshot_at(guard)
        require_postmortem_temp_fd_identity(guard, "after final inventory validation")
        require_postmortem_temp_path_identity(guard, "after final inventory validation")
        if final_callback is not None:
            final_callback()
        require_postmortem_temp_fd_identity(guard, "after final callback")
        require_postmortem_temp_path_identity(guard, "after final callback")
        validate_postmortem_terminal_root_at(
            guard,
            output_root,
            expected_input_binding=expected_input_binding,
            expected_proposal_binding=expected_proposal_binding,
            expected_implementation_binding=expected_implementation_binding,
            expected_source_snapshot=expected_source_snapshot,
            expected_teacher_rows=expected_teacher_rows,
            expected_cell_rows=expected_cell_rows,
            expected_taxonomy_rows=expected_taxonomy_rows,
        )
        require_postmortem_temp_fd_identity(guard, "after final callback validation")
        require_postmortem_temp_path_identity(guard, "after final callback validation")
        if postmortem_terminal_fingerprint_at(guard) != fingerprint:
            raise ValueError("Postmortem terminal files changed after final callback.")
        require_postmortem_temp_fd_identity(guard, "immediately before atomic rename")
        require_postmortem_temp_path_identity(guard, "immediately before atomic rename")
        if postmortem_terminal_fingerprint_at(guard) != fingerprint:
            raise ValueError("Postmortem terminal files changed immediately before atomic rename.")
        atomic_rename_noreplace_at(
            guard.parent_fd,
            guard.temp_name,
            guard.parent_fd,
            guard.output_name,
            output_root,
        )
        try:
            require_postmortem_output_path_identity(guard, "after atomic rename")
            validate_postmortem_terminal_root_at(
                guard,
                output_root,
                expected_input_binding=expected_input_binding,
                expected_proposal_binding=expected_proposal_binding,
                expected_implementation_binding=expected_implementation_binding,
                expected_source_snapshot=expected_source_snapshot,
                expected_teacher_rows=expected_teacher_rows,
                expected_cell_rows=expected_cell_rows,
                expected_taxonomy_rows=expected_taxonomy_rows,
            )
            if postmortem_terminal_fingerprint_at(guard) != fingerprint:
                raise ValueError("Postmortem terminal fingerprint changed after atomic rename.")
            validate_postmortem_terminal_inventory_snapshot_at(guard)
            require_postmortem_output_path_identity(guard, "after final output validation")
        except Exception:
            preserve_postmortem_temp_collision(guard)
            try:
                atomic_rename_noreplace_at(
                    guard.parent_fd,
                    guard.output_name,
                    guard.parent_fd,
                    guard.temp_name,
                    temp_root,
                )
            except Exception as rollback_exc:
                raise FeasibilityPublicationError("Postmortem final root identity mismatch and rollback failed.") from rollback_exc
            demote_postmortem_terminal_markers(guard)
            if postmortem_dir_entry_exists(guard.parent_fd, guard.output_name):
                raise FeasibilityPublicationError("Postmortem final root remained present after rollback.")
            raise
    except Exception:
        demote_postmortem_terminal_markers(guard)
        raise
    finally:
        if owned_guard:
            guard.close()


def publish_postmortem_root_or_leave_incomplete(
    temp_root: Path,
    output_root: Path,
    *,
    expected_input_binding: dict[str, object],
    expected_proposal_binding: dict[str, object],
    expected_implementation_binding: dict[str, object],
    expected_source_snapshot: SourceSnapshot,
    expected_teacher_rows: Sequence[dict[str, object]],
    expected_cell_rows: Sequence[dict[str, object]],
    expected_taxonomy_rows: Sequence[dict[str, object]],
    final_callback: Callable[[], None] | None = None,
    publication_guard: PostmortemPublicationGuard | None = None,
) -> None:
    guard = publication_guard
    owned_guard = False
    try:
        if guard is None:
            guard = open_postmortem_publication_guard(temp_root, output_root)
            owned_guard = True
        publish_postmortem_root(
            temp_root,
            output_root,
            expected_input_binding=expected_input_binding,
            expected_proposal_binding=expected_proposal_binding,
            expected_implementation_binding=expected_implementation_binding,
            expected_source_snapshot=expected_source_snapshot,
            expected_teacher_rows=expected_teacher_rows,
            expected_cell_rows=expected_cell_rows,
            expected_taxonomy_rows=expected_taxonomy_rows,
            final_callback=final_callback,
            publication_guard=guard,
        )
    except Exception as exc:
        if guard is None:
            demote_postmortem_terminal_markers(temp_root)
        else:
            demote_postmortem_terminal_markers(guard)
        raise FeasibilityPublicationError(
            f"Postmortem publication failed without overwriting {output_root}; the temporary root is incomplete."
        ) from exc
    finally:
        if owned_guard and guard is not None:
            guard.close()


def finalize_postmortem_outputs_under_guard(
    *,
    input_root: Path,
    output_root: Path,
    temp_root: Path,
    shallow_binding: dict[str, object],
    input_binding: dict[str, object],
    source_snapshot: SourceSnapshot,
    input_manifest: dict[str, object],
    proposal_binding: dict[str, object],
    implementation_binding: dict[str, object],
    environment: dict[str, object],
    deterministic_flags: dict[str, object],
    rng_contract: dict[str, bool],
    rng_baseline: tuple[object, torch.Tensor, tuple[torch.Tensor, ...] | None],
    teacher_rows: Sequence[dict[str, object]],
    cell_rows: Sequence[dict[str, object]],
    taxonomy_rows: Sequence[dict[str, object]],
    accepted_proposal_commit: str,
    final_callback: Callable[[], None],
    publication_guard: PostmortemPublicationGuard,
) -> None:
    rng_contract.update(
        {
            "constructor_scope_restored": True,
            "post_load_state_unchanged": True,
            "post_inference_state_unchanged": True,
        }
    )
    if rng_contract.keys() != POSTMORTEM_RNG_CONTRACT_KEYS:
        raise ValueError("Postmortem RNG contract schema mismatch.")
    if len(teacher_rows) != POSTMORTEM_ROW_COUNTS["teacher_forced_rows"]:
        raise ValueError("Postmortem teacher row cardinality mismatch before publication.")
    if len(cell_rows) != POSTMORTEM_ROW_COUNTS["cell_metrics"]:
        raise ValueError("Postmortem cell row cardinality mismatch before publication.")
    if len(taxonomy_rows) != POSTMORTEM_ROW_COUNTS["error_taxonomy"]:
        raise ValueError("Postmortem taxonomy row cardinality mismatch before publication.")
    verify_postmortem_preflight_bindings(shallow_binding, input_root)
    verify_postmortem_source_unchanged(source_snapshot, active_output_root=temp_root)
    validate_postmortem_runtime_against_input(input_manifest)
    require_rng_states_equal(rng_baseline, snapshot_rng_states(), "Postmortem RNG state changed before output write.")
    write_postmortem_jsonl_at(publication_guard, "teacher_forced_rows.jsonl", teacher_rows)
    write_postmortem_jsonl_at(publication_guard, "cell_metrics.jsonl", cell_rows)
    write_postmortem_jsonl_at(publication_guard, "error_taxonomy.jsonl", taxonomy_rows)
    write_postmortem_json_at(
        publication_guard,
        "summary.json",
        build_postmortem_summary(
            input_binding=input_binding,
            proposal_binding=proposal_binding,
            implementation_binding=implementation_binding,
            cell_rows=cell_rows,
            taxonomy_rows=taxonomy_rows,
        ),
    )
    write_postmortem_json_at(
        publication_guard,
        "manifest.json",
        build_postmortem_manifest(
            temp_root,
            input_binding=input_binding,
            proposal_binding=proposal_binding,
            implementation_binding=implementation_binding,
            exact_command=postmortem_exact_command(
                input_root,
                output_root,
                accepted_proposal_commit,
                require_exact_str(implementation_binding["commit"], "postmortem.implementation.commit"),
                argv=getattr(sys, "orig_argv", None) if "__phase8_verifier_sha256__" in globals() else None,
            ),
            environment=environment,
            deterministic_flags=deterministic_flags,
            rng_state_contract=rng_contract,
            file_inventory=postmortem_output_inventory_at(publication_guard.temp_fd),
        ),
    )
    write_postmortem_done_at(publication_guard, input_binding=input_binding)
    publish_postmortem_root_or_leave_incomplete(
        temp_root,
        output_root,
        expected_input_binding=input_binding,
        expected_proposal_binding=proposal_binding,
        expected_implementation_binding=implementation_binding,
        expected_source_snapshot=source_snapshot,
        expected_teacher_rows=teacher_rows,
        expected_cell_rows=cell_rows,
        expected_taxonomy_rows=taxonomy_rows,
        final_callback=final_callback,
        publication_guard=publication_guard,
    )


def run_postmortem_failure(
    *,
    device: str,
    input_root: Path,
    output_root: Path,
    accepted_proposal_commit: str,
    accepted_implementation_commit: str,
    environ: dict[str, str] | None = None,
) -> None:
    validate_postmortem_argument_contract(
        device=device,
        input_root=input_root,
        output_root=output_root,
        accepted_proposal_commit=accepted_proposal_commit,
        accepted_implementation_commit=accepted_implementation_commit,
    )
    validate_postmortem_cli_contract(
        device=device,
        input_root=input_root,
        output_root=output_root,
        accepted_proposal_commit=accepted_proposal_commit,
        accepted_implementation_commit=accepted_implementation_commit,
        environ=environ,
    )
    shallow_binding = validate_postmortem_input_shallow(input_root)
    allowed_paths = shallow_binding["allowed_source_paths"]
    if not isinstance(allowed_paths, set):
        raise ValueError("Postmortem shallow preflight did not return source-clean exclusions.")
    source_snapshot = capture_postmortem_source_provenance(input_root, output_root, allowed_paths=allowed_paths)
    configure_postmortem_deterministic_backend()
    input_manifest = shallow_binding["manifest_data"]
    if not isinstance(input_manifest, dict):
        raise ValueError("Postmortem shallow preflight did not return the input manifest.")
    environment, deterministic_flags = validate_postmortem_runtime_against_input(input_manifest)
    proposal_binding = postmortem_proposal_binding(accepted_proposal_commit)
    implementation_binding = postmortem_implementation_binding(source_snapshot)
    deep_binding = validate_postmortem_input_deep(input_root, shallow_binding)
    input_binding = require_exact_mapping(deep_binding["input_binding"], POSTMORTEM_INPUT_BINDING_KEYS, "postmortem.deep.input_binding")
    records, rng_contract = postmortem_record_sets_rng_neutral()
    rng_baseline = snapshot_rng_states()
    temp_root = output_root.with_name(output_root.name + ".tmp")
    validate_new_postmortem_root(output_root)
    publication_guard = create_postmortem_publication_guard(output_root)
    teacher_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    taxonomy_rows: list[dict[str, object]] = []

    def final_publication_callback() -> None:
        verify_postmortem_preflight_bindings(shallow_binding, input_root)
        verify_postmortem_source_unchanged(source_snapshot, active_output_root=temp_root)
        validate_postmortem_runtime_against_input(input_manifest)
        require_rng_states_equal(rng_baseline, snapshot_rng_states(), "Postmortem RNG state changed before publication.")

    try:
        target_device = torch.device(device)
        tokenizer = ByteTokenizer()
        generation_rows_by_key = deep_binding["generation_rows"]
        checkpoint_payloads = deep_binding["checkpoint_payloads"]
        if not isinstance(generation_rows_by_key, dict) or not isinstance(checkpoint_payloads, dict):
            raise ValueError("Postmortem deep preflight did not return checkpoint/generation bindings.")
        with PostmortemForbiddenOperationGuard() as guard:
            for cell in validate_postmortem_current_cells_shallow(deep_binding["cells"]):
                family = require_exact_str(cell["family"], "postmortem.cell.family")
                model_size = require_exact_str(cell["model_size"], "postmortem.cell.model_size")
                seed = require_exact_int(cell["seed"], "postmortem.cell.seed")
                key = (family, model_size, seed)
                checkpoint = checkpoint_payloads[key]
                generation_rows = generation_rows_by_key[key]
                checkpoint_path = require_canonical_relative_path(cell["checkpoint_path"], "postmortem.cell.checkpoint_path")
                generation_path = require_canonical_relative_path(cell["generations_path"], "postmortem.cell.generation_path")
                checkpoint_sha = postmortem_file_sha_from_input_binding(input_binding, checkpoint_path)
                generation_sha = postmortem_file_sha_from_input_binding(input_binding, generation_path)
                model = construct_postmortem_model_from_checkpoint(
                    checkpoint,
                    model_size=model_size,
                    device=target_device,
                    guard=guard,
                )
                require_rng_states_equal(rng_baseline, snapshot_rng_states(), "Postmortem checkpoint load changed global RNG state.")
                model_snapshot = postmortem_model_snapshot(model)
                train_tf_rows, _train_tf_aggregate = postmortem_teacher_forced_rows(
                    model,
                    records[family]["train"],
                    tokenizer,
                    target_device,
                    split="train",
                    family=family,
                    model_size=model_size,
                    seed=seed,
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha,
                )
                require_postmortem_model_unchanged(model, model_snapshot, "train teacher forcing")
                require_rng_states_equal(rng_baseline, snapshot_rng_states(), "Postmortem train teacher forcing changed global RNG state.")
                eval_tf_rows, _eval_tf_aggregate = postmortem_teacher_forced_rows(
                    model,
                    records[family]["eval"],
                    tokenizer,
                    target_device,
                    split="eval",
                    family=family,
                    model_size=model_size,
                    seed=seed,
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha,
                )
                require_postmortem_model_unchanged(model, model_snapshot, "eval teacher forcing")
                require_rng_states_equal(rng_baseline, snapshot_rng_states(), "Postmortem eval teacher forcing changed global RNG state.")
                teacher_rows.extend(train_tf_rows)
                teacher_rows.extend(eval_tf_rows)
                for record, retained_row in zip(records[family]["eval"], generation_rows, strict=True):
                    taxonomy_rows.append(
                        postmortem_taxonomy_row(
                            record,
                            retained_row,
                            family=family,
                            model_size=model_size,
                            seed=seed,
                            generation_path=generation_path,
                            generation_sha256=generation_sha,
                        )
                    )
                cell_rows.append(
                    postmortem_cell_metrics_row(
                        cell=cell,
                        input_binding=input_binding,
                        checkpoint=checkpoint,
                        train_teacher_rows=train_tf_rows,
                        eval_teacher_rows=eval_tf_rows,
                        generation_rows=generation_rows,
                    )
                )
                require_postmortem_model_unchanged(model, model_snapshot, "cell completion")
                require_rng_states_equal(rng_baseline, snapshot_rng_states(), "Postmortem cell completion changed global RNG state.")
            finalize_postmortem_outputs_under_guard(
                input_root=input_root,
                output_root=output_root,
                temp_root=temp_root,
                shallow_binding=shallow_binding,
                input_binding=input_binding,
                source_snapshot=source_snapshot,
                input_manifest=input_manifest,
                proposal_binding=proposal_binding,
                implementation_binding=implementation_binding,
                environment=environment,
                deterministic_flags=deterministic_flags,
                rng_contract=rng_contract,
                rng_baseline=rng_baseline,
                teacher_rows=teacher_rows,
                cell_rows=cell_rows,
                taxonomy_rows=taxonomy_rows,
                accepted_proposal_commit=accepted_proposal_commit,
                final_callback=final_publication_callback,
                publication_guard=publication_guard,
            )
    except Exception:
        demote_postmortem_terminal_markers(publication_guard)
        raise
    finally:
        publication_guard.close()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 non-scientific sequence-transduction feasibility runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--device", required=True)
    run.add_argument("--root", required=True)
    run.add_argument("--predecessor-root", action="append", default=[])
    run.add_argument("--predecessor-selection", action="append", default=[])
    run.add_argument("--decision-diagnostic-root", required=True)
    validate = subparsers.add_parser("validate-selection")
    validate.add_argument("path")
    inspect = subparsers.add_parser("inspect-records")
    inspect.add_argument("--family", choices=FAMILIES)
    diagnose = subparsers.add_parser("diagnose-failure")
    diagnose.add_argument("--device", required=True)
    diagnose.add_argument("--input-root", required=True)
    diagnose.add_argument("--output-root", required=True)
    diagnose.add_argument("--predecessor-diagnostic-root", action="append", default=[])
    postmortem = subparsers.add_parser("postmortem-failure")
    postmortem.add_argument("--device", required=True)
    postmortem.add_argument("--input-root", required=True)
    postmortem.add_argument("--output-root", required=True)
    postmortem.add_argument("--accepted-proposal-commit", required=True)
    postmortem.add_argument("--accepted-implementation-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "run":
        if argv is not None:
            raise ValueError("run must be launched as a real process command, not via main(argv=...).")
        root = Path(require_canonical_path_string(args.root, "root", ROOT_RE))
        predecessor_roots = tuple(
            Path(require_canonical_path_string(path, "predecessor_root.path", ROOT_RE))
            for path in args.predecessor_root
        )
        predecessor_selections = tuple(
            Path(require_canonical_path_string(path, "predecessor_selection.path", SELECTION_RE))
            for path in args.predecessor_selection
        )
        decision_diagnostic_root = Path(require_canonical_path_string(args.decision_diagnostic_root, "decision_diagnostic_root", DIAGNOSTIC_ROOT_RE))
        validate_feasibility_cli_contract(
            device=args.device,
            root=root,
            predecessor_roots=predecessor_roots,
            predecessor_selections=predecessor_selections,
            decision_diagnostic_root=decision_diagnostic_root,
        )
        run_current_suite(
            root,
            predecessor_roots,
            predecessor_selections,
            device=args.device,
            decision_diagnostic_root=decision_diagnostic_root,
        )
        return 0
    if args.command == "validate-selection":
        validate_selection_record(Path(require_canonical_path_string(args.path, "selection_record.path", SELECTION_RE)))
        return 0
    if args.command == "inspect-records":
        families = (args.family,) if args.family else FAMILIES
        data = {family: [asdict(record) for record in build_family_records(family)] for family in families}
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    if args.command == "diagnose-failure":
        if argv is not None:
            raise ValueError("diagnose-failure must be launched as a real process command, not via main(argv=...).")
        input_root = Path(require_canonical_path_string(args.input_root, "input_root", ROOT_RE))
        output_root = Path(require_canonical_path_string(args.output_root, "output_root", DIAGNOSTIC_ROOT_RE))
        predecessor_diagnostic_roots = tuple(
            Path(require_canonical_path_string(path, "predecessor_diagnostic_root", DIAGNOSTIC_ROOT_RE))
            for path in args.predecessor_diagnostic_root
        )
        run_diagnostic_failure(
            device=args.device,
            input_root=input_root,
            output_root=output_root,
            predecessor_diagnostic_roots=predecessor_diagnostic_roots,
            environ=os.environ,
        )
        return 0
    if args.command == "postmortem-failure":
        if argv is not None:
            raise ValueError("postmortem-failure must be launched as a real process command, not via main(argv=...).")
        input_root = Path(require_canonical_path_string(args.input_root, "input_root", ROOT_RE))
        output_root = Path(require_postmortem_root_path_string(args.output_root, "output_root"))
        run_postmortem_failure(
            device=args.device,
            input_root=input_root,
            output_root=output_root,
            accepted_proposal_commit=args.accepted_proposal_commit,
            accepted_implementation_commit=args.accepted_implementation_commit,
            environ=os.environ,
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
