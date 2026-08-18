#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import errno
from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import platform
import random
import re
import subprocess
import sys
import time
from typing import Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from capability_certificate_lab.lm_bridge import corpus_generator as cg
from capability_certificate_lab.lm_bridge.model import build_model
from capability_certificate_lab.lm_bridge.model import transformer_config
from capability_certificate_lab.lm_bridge.tokenizer import ByteTokenizer
from capability_certificate_lab.lm_bridge.tokenizer import EOS_ID
from capability_certificate_lab.lm_bridge.train import (
    GRADIENT_CLIP_NORM,
    TextRecord,
    deterministic_batch_indices,
    encode_record_batch,
    make_optimizer,
    response_only_labels,
    response_only_loss,
    save_checkpoint,
    set_deterministic_backend,
    train_text_records,
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
DIAGNOSTIC_INPUT_SOURCE_COMMIT = "3cb75ad550c4357562c0d4d9a9b098bfb2cf66ea"
DIAGNOSTIC_HANDOFF_PATH = "phase8/Task_010C_D1_Feasibility_Failure_Diagnostic.md"
DIAGNOSTIC_REQUIRED_DEVICE = "cuda:0"
DIAGNOSTIC_REQUIRED_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONPATH": ".",
}
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
    "small": 149760,
    "medium": 892672,
}
DIAGNOSTIC_RECORD_HASHES = {
    "named_train512": "9b55084d281a9420e12a1b6a35c3eb199abd74f4b766a14a833e6241c59564b8",
    "named_train_first64": "6806025fa541c4f71d84a2dfce29777e0ac4e056c1bad762aeb10d309f5503ad",
    "named_eval64": "6cc73c98616430a12a357de99702e8cb4db95258225adab80fd11212aa203d20",
    "array_train512": "6bbcae6203dfcf443f9871f30b48c29580fa721317387ff26b757b4066a98e94",
    "array_eval64": "8d504dd63ad2538aedc8195a4f3c0729ed38ec95a078b9c9895fbf93cf8c173a",
}
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
    ignored_inputs: tuple[dict[str, object], ...]


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


def validate_cell_counts(cells: Sequence[dict[str, object]]) -> None:
    expected_keys = {(family, model_size, seed) for family in FAMILIES for model_size in MODEL_SIZES for seed in SEEDS}
    seen: set[tuple[str, str, int]] = set()
    for cell in cells:
        seed = require_exact_int(cell["seed"], "seed")
        family = require_exact_str(cell["family"], "family")
        model_size = require_exact_str(cell["model_size"], "model_size")
        key = (family, model_size, seed)
        if key not in expected_keys:
            raise ValueError(f"Unexpected feasibility cell: {key!r}.")
        if key in seen:
            raise ValueError(f"Duplicate feasibility cell: {key!r}.")
        seen.add(key)
        eval_count = require_exact_int(cell["eval_count"], "eval_count")
        exact_matches = require_exact_int(cell["exact_matches"], "exact_matches")
        passed_value = require_exact_bool(cell["passed"], "passed")
        if eval_count != EVAL_RECORDS_PER_FAMILY:
            raise ValueError("Every feasibility cell must evaluate 64 records.")
        if not 0 <= exact_matches <= eval_count:
            raise ValueError("Feasibility exact_matches must satisfy 0 <= exact_matches <= eval_count.")
        if exact_matches < PASS_THRESHOLD or passed_value is not True:
            raise ValueError("Every feasibility cell must pass the independent 52/64 requirement.")
        parameter_count = require_exact_int(cell.get("parameter_count"), "parameter_count")
        if parameter_count != expected_parameter_count(model_size):
            raise ValueError("Feasibility cell parameter_count does not match the frozen model configuration.")
        require_canonical_relative_path(cell.get("generations_path"), "generations_path")
        require_canonical_relative_path(cell.get("checkpoint_path"), "checkpoint_path")
    missing = expected_keys - seen
    if missing:
        raise ValueError(f"Missing feasibility cells: {sorted(missing)!r}.")


@lru_cache(maxsize=None)
def expected_parameter_count(model_size: str) -> int:
    if model_size not in FROZEN_PARAMETER_COUNTS:
        raise ValueError(f"Unknown model_size for parameter_count validation: {model_size!r}.")
    return FROZEN_PARAMETER_COUNTS[model_size]


def frozen_configuration() -> dict[str, object]:
    return {
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
    missing_items = Counter(item for row in rows for item in (row["missing_items"] or ()))
    extra_items = Counter(item for row in rows for item in (row["extra_items"] or ()))
    positional_den = sum(int(row["item_count"]) for row in rows if row["positional_exact_items"] is not None)
    positional_num = sum(int(row["positional_exact_items"]) for row in rows if row["positional_exact_items"] is not None)
    return {
        "row_count": len(rows),
        "greedy_exact_matches": count_rate(sum(1 for row in rows if row["exact_match"] is True), len(rows)),
        "valid_json_syntax": count_rate(sum(1 for row in rows if row["valid_json_syntax"] is True), len(rows)),
        "valid_array_schema": count_rate(sum(1 for row in rows if row["valid_array_schema"] is True), len(rows)),
        "correct_item_count": count_rate(sum(1 for row in rows if row["correct_item_count"] is True), len(rows)),
        "item_count_exact_matches": item_count_exact,
        "template_item_count_exact_matches": template_item_exact,
        "positional_exact_items": count_rate(positional_num, positional_den),
        "expected_item_counts": dict(sorted(expected_items.items())),
        "missing_item_counts": dict(sorted(missing_items.items())),
        "extra_item_counts": dict(sorted(extra_items.items())),
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(model_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def generate_named_diagnostic_rows(
    model: torch.nn.Module,
    matrix_rows: Sequence[dict[str, object]],
    device: torch.device,
) -> list[dict[str, object]]:
    tokenizer = ByteTokenizer()
    output_rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for row in matrix_rows:
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
) -> list[dict[str, object]]:
    tokenizer = ByteTokenizer()
    output_rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for row_index, record in enumerate(records):
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
        current_record_for_row=None,
    )
    for row_index, (raw_row, record) in enumerate(zip(raw_rows, records, strict=True)):
        if raw_row["prompt"] != record.prompt or raw_row["expected"] != record.answer:
            raise ValueError("Reused array generation row no longer matches current frozen array eval record.")
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


def array_item_template_counts(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[f"{row['template_id']}__items{row['item_count']}"] += 1
    return dict(sorted(counts.items()))


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
    validate_diagnostic_core_blobs()
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
        "path": str(root),
        "terminal_state": terminal.stem,
        "terminal_sha256": file_sha256(terminal),
        "manifest_sha256": manifest_sha,
        "source_commit": terminal_data.get("source_commit"),
        "configuration": terminal_data.get("configuration"),
        "failure_classification": terminal_data.get("failure_classification"),
        "diagnostic_lineage": terminal_data.get("diagnostic_lineage", []),
    }


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
    expected_inventory = diagnostic_inventory(root)
    file_inventory = manifest_data.get("file_inventory")
    if file_inventory != expected_inventory:
        raise ValueError("Diagnostic manifest file_inventory does not exactly match current root inventory.")
    if terminal_data.get("file_inventory") != file_inventory:
        raise ValueError("Diagnostic terminal does not bind the exact manifest inventory.")
    shared_fields = (
        "source_commit",
        "source_provenance",
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


def diagnostic_preflight_bindings(input_root: Path, predecessor_diagnostic_roots: Sequence[Path]) -> dict[str, object]:
    input_binding = {
        relative_path: file_sha256(input_root / relative_path)
        for relative_path in sorted(DIAGNOSTIC_INPUT_CHECKSUMS)
    }
    predecessor_bindings = [diagnostic_terminal_binding(root) for root in predecessor_diagnostic_roots]
    return {
        "input_root": str(input_root),
        "input": input_binding,
        "predecessors": predecessor_bindings,
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
    frozen_configuration_value = {
        "artifact_class": DIAGNOSTIC_ARTIFACT_CLASS,
        "feasibility_selection_eligible": False,
        "task_010d_authorized": False,
        "named_cells": list(NAMED_DIAGNOSTIC_CELLS),
        "array_new_training": diagnostic_array_training_plan(),
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "formal_training_steps": TRAINING_STEPS,
        "pass_threshold": PASS_THRESHOLD,
    }
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


def validate_diagnostic_cli_contract(
    *,
    device: str,
    input_root: Path,
    output_root: Path,
    predecessor_diagnostic_roots: Sequence[Path],
    raw_argv: Sequence[str] | None = None,
    environ: dict[str, str] | None = None,
) -> None:
    if device != DIAGNOSTIC_REQUIRED_DEVICE:
        raise ValueError("diagnose-failure only supports device string cuda:0.")
    if raw_argv is not None and any(path.is_absolute() for path in (input_root, output_root, *predecessor_diagnostic_roots)):
        raise ValueError("diagnose-failure authorized command requires canonical repository-relative artifact paths.")
    expected_argv = [
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
    if raw_argv is not None:
        if list(raw_argv) != expected_argv:
            raise ValueError(f"diagnose-failure raw argv must exactly match the authorized command: {expected_argv!r}.")
        actual_env = os.environ if environ is None else environ
        for key, expected_value in DIAGNOSTIC_REQUIRED_ENV.items():
            if actual_env.get(key) != expected_value:
                raise ValueError(f"diagnose-failure environment {key} must exactly equal {expected_value!r}.")
    validate_new_diagnostic_root(input_root, output_root, predecessor_diagnostic_roots)
    if output_root.name == "feasibility_diagnostic_001" and predecessor_diagnostic_roots:
        raise ValueError("Initial diagnostic command must not include predecessor roots.")


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


def compare_model_to_checkpoint_step1500(model: torch.nn.Module, checkpoint_path: Path, model_size: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("config") != transformer_config(model_size).__dict__:
        raise ValueError("Step-1500 equality gate config mismatch.")
    if checkpoint.get("parameter_count") != expected_parameter_count(model_size):
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


def train_array_small_3000_diagnostic(
    *,
    seed: int,
    train_records: Sequence[FeasibilityRecord],
    tokenizer: ByteTokenizer,
    device: torch.device,
    frozen_step1500_checkpoint: Path,
    checkpoint_path: Path,
) -> dict[str, object]:
    if device.type != "cuda" or device.index != 0:
        raise ValueError("Array diagnostic training must run on cuda:0.")
    set_deterministic_backend(seed)
    model = build_model("small")
    model.to(device)
    model.train()
    optimizer = make_optimizer(model)
    batches = deterministic_batch_indices(
        record_count=len(train_records),
        seed=seed,
        state_mask=0,
        batch_size=BATCH_SIZE,
        steps=DIAGNOSTIC_STEPS,
    )
    if batches[:TRAINING_STEPS] != deterministic_batch_indices(
        record_count=len(train_records),
        seed=seed,
        state_mask=0,
        batch_size=BATCH_SIZE,
        steps=TRAINING_STEPS,
    ):
        raise ValueError("Diagnostic 3000-step batch stream does not extend the frozen 1500-step stream.")
    final_loss = float("nan")
    step1501_executed = False
    for step_index, batch in enumerate(batches, start=1):
        if step_index == TRAINING_STEPS + 1:
            step1501_executed = True
        batch_records = [TextRecord(train_records[index].prompt, train_records[index].answer) for index in batch]
        input_ids = encode_record_batch(batch_records, tokenizer).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = response_only_loss(logits, input_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step_index == TRAINING_STEPS:
            rng_state = random.getstate()
            torch_rng_state = torch.random.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state(device)
            optimizer_state = optimizer_state_fingerprint(optimizer)
            compare_model_to_checkpoint_step1500(model, frozen_step1500_checkpoint, "small")
            if random.getstate() != rng_state:
                raise ValueError("Step-1500 equality snapshot changed Python RNG state.")
            if not torch.equal(torch.random.get_rng_state(), torch_rng_state):
                raise ValueError("Step-1500 equality snapshot changed Torch RNG state.")
            if not torch.equal(torch.cuda.get_rng_state(device), cuda_rng_state):
                raise ValueError("Step-1500 equality snapshot changed CUDA RNG state.")
            if optimizer_state_fingerprint(optimizer) != optimizer_state:
                raise ValueError("Step-1500 equality snapshot changed optimizer state.")
    save_checkpoint(
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
        },
    )
    if not step1501_executed:
        raise ValueError("Diagnostic 3000-step training did not prove execution of step 1501.")
    return {"final_loss": final_loss, "checkpoint_path": str(checkpoint_path), "step1501_executed": True}


def diagnostic_inventory(root: Path) -> list[dict[str, object]]:
    rows = []
    for row in inventory(root):
        role = "summary" if row["path"] == "summary.json" else "diagnostic_artifact"
        if str(row["path"]).endswith(".pt"):
            role = "checkpoint"
        elif str(row["path"]).endswith(".jsonl"):
            role = "retained_rows"
        rows.append({**row, "role": role})
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
    shared_fields = (
        "source_commit",
        "source_provenance",
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


def publish_diagnostic_root(temp_root: Path, output_root: Path, *, terminal_status: str) -> None:
    validate_diagnostic_terminal_root(temp_root, terminal_status=terminal_status)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic root: {output_root}")
    atomic_rename_noreplace(temp_root, output_root)


def publish_diagnostic_root_or_leave_incomplete(temp_root: Path, output_root: Path, *, terminal_status: str) -> None:
    try:
        publish_diagnostic_root(temp_root, output_root, terminal_status=terminal_status)
    except Exception as exc:
        terminal = temp_root / f"{terminal_status}.json"
        if terminal.is_file() and not terminal.is_symlink():
            terminal.unlink()
        raise DiagnosticPublicationError(
            f"Diagnostic publication failed without overwriting {output_root}; the temporary root is incomplete."
        ) from exc


def atomic_rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic no-clobber diagnostic publication requires Linux renameat2.")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, "Refusing to overwrite existing diagnostic root", destination)
    raise OSError(error_number, os.strerror(error_number), destination)


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
    input_manifest = json.loads((input_root / "manifest.json").read_text())
    lineage = flattened_diagnostic_lineage(predecessor_diagnostic_roots)
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
        "handoff": {
            "path": DIAGNOSTIC_HANDOFF_PATH,
            "sha256": file_sha256(REPO_ROOT / DIAGNOSTIC_HANDOFF_PATH),
        },
        "input_root": {
            "path": str(input_root),
            "manifest_sha256": file_sha256(input_root / "manifest.json"),
            "summary_sha256": file_sha256(input_root / "summary.json"),
            "terminal_sha256": file_sha256(input_root / "FAILED.json"),
            "source_commit": input_manifest.get("source_commit"),
            "predecessor_roots": input_manifest.get("predecessor_roots"),
        },
        "diagnostic_lineage": lineage,
        "configuration": {
            "artifact_class": DIAGNOSTIC_ARTIFACT_CLASS,
            "feasibility_selection_eligible": False,
            "task_010d_authorized": False,
            "named_cells": list(NAMED_DIAGNOSTIC_CELLS),
            "array_new_training": diagnostic_array_training_plan(),
            "diagnostic_steps": DIAGNOSTIC_STEPS,
            "formal_training_steps": TRAINING_STEPS,
            "pass_threshold": PASS_THRESHOLD,
        },
        "record_hashes": validate_diagnostic_record_hashes(),
        "core_blobs": validate_diagnostic_core_blobs(),
        "environment": current_environment_dict(),
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
    if isinstance(error, (OSError, subprocess.SubprocessError, torch.OutOfMemoryError)):
        return "transient_infrastructure"
    return "diagnostic_implementation_defect"


def run_diagnostic_failure(
    *,
    device: str,
    input_root: Path,
    output_root: Path,
    predecessor_diagnostic_roots: Sequence[Path] = (),
    raw_argv: Sequence[str] | None = None,
    environ: dict[str, str] | None = None,
) -> None:
    validate_diagnostic_cli_contract(
        device=device,
        input_root=input_root,
        output_root=output_root,
        predecessor_diagnostic_roots=predecessor_diagnostic_roots,
        raw_argv=raw_argv,
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
                named_rows = generate_named_diagnostic_rows(model, named_matrix[cell], target_device)
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
                )
                eval_tf_rows, eval_tf_aggregate = teacher_forced_rows(
                    model,
                    records["array_json"]["eval"],
                    tokenizer,
                    target_device,
                    split="eval",
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
            )
            model = load_checkpoint_model(run_dir / "checkpoint_step3000.pt", "small", target_device)
            array_rows = generate_array_rows(
                model,
                records["array_json"]["eval"],
                target_device,
                comparison_source="diagnostic_small_3000",
                model_size="small",
                steps=DIAGNOSTIC_STEPS,
                seed=seed,
            )
            write_jsonl(run_dir / "generations.jsonl", array_rows)
            train_tf_rows, train_tf_aggregate = teacher_forced_rows(
                model,
                records["array_json"]["train"],
                tokenizer,
                target_device,
                split="train",
            )
            eval_tf_rows, eval_tf_aggregate = teacher_forced_rows(
                model,
                records["array_json"]["eval"],
                tokenizer,
                target_device,
                split="eval",
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
            wall_time_seconds=time.monotonic() - start_time,
        )
        publish_diagnostic_root_or_leave_incomplete(temp_root, output_root, terminal_status="DONE")
    except (SourceChangedError, DiagnosticPublicationError):
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
            wall_time_seconds=time.monotonic() - start_time,
        )
        publish_diagnostic_root_or_leave_incomplete(temp_root, output_root, terminal_status="FAILED")
        raise


def run_suite(root: Path, predecessor_roots: Sequence[Path], predecessor_selections: Sequence[Path]) -> None:
    validate_new_root(root, predecessor_roots, predecessor_selections)
    source_snapshot = capture_source_provenance(root, predecessor_roots, predecessor_selections)
    temp_root = root.with_name(root.name + ".tmp")
    if temp_root.exists():
        raise FileExistsError(f"Temporary feasibility root already exists: {temp_root}")
    temp_root.mkdir(parents=True)
    cells: list[dict[str, object]] = []
    try:
        records = grouped_records()
        tokenizer = ByteTokenizer()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for family in FAMILIES:
            train_records = tuple(TextRecord(r.prompt, r.answer) for r in records[family]["train"])
            eval_records = records[family]["eval"]
            for model_size in MODEL_SIZES:
                for seed in SEEDS:
                    set_deterministic_backend(seed)
                    model = build_model(model_size)
                    result = train_text_records(
                        model,
                        train_records,
                        seed=seed,
                        state_mask=0,
                        steps=TRAINING_STEPS,
                        batch_size=BATCH_SIZE,
                        tokenizer=tokenizer,
                        device=device,
                    )
                    exact_matches, generations = evaluate_model(model, eval_records, tokenizer, device)
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
                        }
                    )
        validate_cell_counts(cells)
        terminal_status = "DONE" if all(cell["passed"] for cell in cells) else "FAILED"
        verify_source_unchanged(source_snapshot, active_output_root=temp_root)
        write_terminal(temp_root, terminal_status, cells, predecessor_roots, predecessor_selections, source_snapshot=source_snapshot)
        os.replace(temp_root, root)
    except SourceChangedError:
        raise
    except Exception as exc:
        verify_source_unchanged(source_snapshot, active_output_root=temp_root)
        write_terminal(
            temp_root,
            "FAILED",
            cells,
            predecessor_roots,
            predecessor_selections,
            failure=repr(exc),
            source_snapshot=source_snapshot,
        )
        os.replace(temp_root, root)
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
) -> SourceSnapshot:
    require_artifact_location(root, "root", ROOT_RE)
    allowed = source_clean_allowed_paths(predecessor_roots, predecessor_selections)
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


def ignored_source_inputs() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    completed = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--ignored",
            "--untracked-files=all",
            "capability_certificate_lab",
            "scripts",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        cwd=REPO_ROOT,
    )
    for line in completed.stdout.splitlines():
        if not line.startswith("!! "):
            continue
        rel = line[3:]
        path = Path(rel)
        if "__pycache__" not in path.parts and path.suffix not in {".py", ".pyc", ".pyo", ".so"}:
            continue
        resolved = (path if path.is_absolute() else REPO_ROOT / path).resolve()
        if not resolved.is_file():
            continue
        rows.append({"path": path.as_posix(), "sha256": file_sha256(resolved), "bytes": resolved.stat().st_size})
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
    )
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    manifest_sha = file_sha256(manifest_path)
    terminal = {
        "status": terminal_status,
        "manifest_path": "manifest.json",
        "manifest_sha256": manifest_sha,
        "pass_threshold": PASS_THRESHOLD,
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
    current_protocol = source_commit == current_commit
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
    if manifest_data.get("configuration") != frozen_configuration():
        raise ValueError("Manifest configuration does not match the frozen feasibility schema.")
    if summary_data.get("configuration") != frozen_configuration():
        raise ValueError("Summary configuration does not match the frozen feasibility schema.")
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
        validate_cell_artifact_schema(cells, require_pass=False)
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
        if current_protocol:
            generation_rows = validate_generation_artifact(root / generations_path, cell)
        else:
            generation_rows = validate_historical_generation_artifact(root / generations_path, cell)
        validate_checkpoint_artifact(root / checkpoint_path, cell)
        if require_passing and current_protocol:
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


def validate_cell_artifact_schema(cells: object, *, require_pass: bool) -> None:
    if not isinstance(cells, list):
        raise ValueError("Feasibility cells must be a JSON list.")
    if require_pass:
        validate_cell_counts(cells)
        return
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ValueError("Feasibility cell entries must be JSON objects.")
        require_exact_str(cell.get("family"), f"cells[{index}].family")
        require_exact_str(cell.get("model_size"), f"cells[{index}].model_size")
        require_exact_int(cell.get("seed"), f"cells[{index}].seed")
        require_exact_int(cell.get("eval_count"), f"cells[{index}].eval_count")
        require_exact_int(cell.get("exact_matches"), f"cells[{index}].exact_matches")
        require_exact_bool(cell.get("passed"), f"cells[{index}].passed")
        require_exact_int(cell.get("parameter_count"), f"cells[{index}].parameter_count")
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
    expected_model = build_model(model_size)
    if checkpoint.get("config") != expected_model.config.__dict__:
        raise ValueError("Checkpoint config does not match the frozen model configuration.")
    if checkpoint.get("parameter_count") != parameter_count or parameter_count != expected_model.parameter_count:
        raise ValueError("Checkpoint parameter_count does not match the cell and frozen model configuration.")
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
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError("Checkpoint state_dict tensor shapes do not match the frozen model.")


def validate_checkpoint_replays_generations(
    path: Path,
    cell: dict[str, object],
    generation_rows: Sequence[dict[str, object]],
) -> None:
    family = require_exact_str(cell.get("family"), "family")
    model_size = require_exact_str(cell.get("model_size"), "model_size")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(model_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    return source_clean_allowed_paths(
        tuple(Path(require_canonical_path_string(binding.get("path"), "predecessor_root.path", ROOT_RE)) for binding in predecessor_roots),
        tuple(
            Path(require_canonical_path_string(binding.get("path"), "predecessor_selection.path", SELECTION_RE))
            for binding in predecessor_selections
        ),
        context=context,
    )


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


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 non-scientific sequence-transduction feasibility runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--root", required=True)
    run.add_argument("--predecessor-root", action="append", default=[])
    run.add_argument("--predecessor-selection", action="append", default=[])
    validate = subparsers.add_parser("validate-selection")
    validate.add_argument("path")
    inspect = subparsers.add_parser("inspect-records")
    inspect.add_argument("--family", choices=FAMILIES)
    diagnose = subparsers.add_parser("diagnose-failure")
    diagnose.add_argument("--device", required=True)
    diagnose.add_argument("--input-root", required=True)
    diagnose.add_argument("--output-root", required=True)
    diagnose.add_argument("--predecessor-diagnostic-root", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "run":
        root = Path(require_canonical_path_string(args.root, "root", ROOT_RE))
        predecessor_roots = tuple(
            Path(require_canonical_path_string(path, "predecessor_root.path", ROOT_RE))
            for path in args.predecessor_root
        )
        predecessor_selections = tuple(
            Path(require_canonical_path_string(path, "predecessor_selection.path", SELECTION_RE))
            for path in args.predecessor_selection
        )
        run_suite(root, predecessor_roots, predecessor_selections)
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
            raw_argv=sys.argv if argv is None else None,
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
