#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import platform
import random
import re
import subprocess
import sys
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from capability_certificate_lab.lm_bridge import corpus_generator as cg
from capability_certificate_lab.lm_bridge.model import build_model
from capability_certificate_lab.lm_bridge.tokenizer import ByteTokenizer
from capability_certificate_lab.lm_bridge.train import (
    TextRecord,
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
SELECTION_RE = re.compile(r"^feasibility_selection_(\d{3})\.json$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ARTIFACT_PARENT = REPO_ROOT / "artifacts" / "phase8_toy_lm_bridge"
HEX_OPERAND_RE = re.compile(r"(?<![0-9A-Fa-f])([0-9a-f]{16})(?![0-9A-Fa-f])", re.IGNORECASE)
NAMED_VALUE_KEYS_BY_SPLIT = {
    "train": ("red", "blue", "green", "silver"),
    "eval": ("amber", "violet", "teal", "bronze"),
}
NAMED_VALUE_KEY_RE = re.compile(r"\b(?:red|blue|green|silver|amber|violet|teal|bronze)\b", re.IGNORECASE)
NAMED_VALUE_RE = re.compile(
    r"\b(?:red|blue|green|silver|amber|violet|teal|bronze)-[te][0-9a-f]{8}\b",
    re.IGNORECASE,
)
ARRAY_ITEM_RE = re.compile(r"\bs[te][0-9a-f]{8}\b", re.IGNORECASE)
ACCEPTED_INDEPENDENT_REVIEW_VERDICT = "ACCEPT"

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
            "Repeat exactly these sixteen hex digits labeled batch {a}: {b}",
            "Copy only this hex code from note {a}: {b}",
            "Answer with the following hex string after cue {a}: {b}",
            "Transcribe the sixteen-character hex token numbered {a}: {b}",
        ),
        "eval": (
            "Produce the 16-digit hex code shown after sign {a}: {b}",
            "Write just the hex sequence beside marker {a}: {b}",
            "Return only the shown lowercase hex entry {a}: {b}",
            "Give the exact hex characters listed at slot {a}: {b}",
        ),
    },
    "named_value_json": {
        "train": (
            "Respond with compact JSON string for name {a}; entries are {b}",
            "Using the listed pairs, emit compact JSON string for {a}: {b}",
            "Choose {a} and output its compact JSON string from pairs {b}",
            "From these pairs answer as compact JSON string for {a}: {b}",
        ),
        "eval": (
            "Write a compact JSON string for label {a} after reading pairs {b}",
            "Give the JSON string associated with {a}; pairs {b}",
            "Output compact JSON string matching {a} in this roster {b}",
            "Find {a} among pairs {b} and reply as compact JSON string",
        ),
    },
    "boolean_json": {
        "train": (
            "Reply JSON bool for this comparison: {a} {b} {c}.",
            "Convert the comparison to JSON bool: {a} {b} {c}.",
            "For {a} {b} {c}, write only JSON bool.",
            "Give JSON bool after checking: {a} {b} {c}.",
        ),
        "eval": (
            "Write canonical JSON bool for: {a} {b} {c}.",
            "Evaluate {a} {b} {c}; answer with JSON bool.",
            "For the statement {a} {b} {c}, return JSON bool.",
            "Decide {a} {b} {c} and emit JSON bool.",
        ),
    },
    "array_json": {
        "train": (
            "Return compact JSON array from chunks: {a}",
            "Convert these chunks into a compact JSON array: {a}",
            "Write a compact JSON array containing these chunks: {a}",
            "Emit JSON array only for chunks: {a}",
        ),
        "eval": (
            "Produce compact JSON array from pieces: {a}",
            "Turn these pieces into compact JSON array: {a}",
            "Answer with a compact JSON array holding these pieces: {a}",
            "Give JSON array only for pieces: {a}",
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
class SourceSnapshot:
    commit: str
    status_lines: tuple[str, ...]
    ignored_inputs: tuple[dict[str, object], ...]


class SourceChangedError(RuntimeError):
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
        if pattern.fullmatch(text):
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


def semantic_values_for_record(record: FeasibilityRecord) -> frozenset[str]:
    prompt_and_answer = f"{record.prompt}\n{record.answer}"
    values: set[str] = set()
    if record.family == "hex_copy":
        values.update(value.lower() for value in HEX_OPERAND_RE.findall(prompt_and_answer))
    elif record.family == "named_value_json":
        decoded = json.loads(record.answer)
        if not isinstance(decoded, str):
            raise ValueError("named_value_json answers must be JSON strings.")
        values.add(decoded.casefold())
        values.update(value.casefold() for value in NAMED_VALUE_KEY_RE.findall(record.prompt))
        values.update(value.casefold() for value in NAMED_VALUE_RE.findall(prompt_and_answer))
    elif record.family == "array_json":
        decoded = json.loads(record.answer)
        if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
            raise ValueError("array_json answers must be JSON arrays of strings.")
        normalized = [item.casefold() for item in decoded]
        values.update(normalized)
        values.add(compact_json(normalized))
        values.update(value.casefold() for value in ARRAY_ITEM_RE.findall(prompt_and_answer))
    elif record.family == "boolean_json":
        values.update(re.findall(r"\b\d+\b", record.prompt))
    else:
        raise ValueError(f"Unknown feasibility family: {record.family!r}.")
    return frozenset(values)


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
    for surface in (*train_surfaces, *eval_surfaces):
        reject_scientific_markers(surface)
        if "template" in surface.lower():
            raise ValueError("Feasibility prompt surfaces must not expose template markers.")


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
    patterns: list[re.Pattern[str]] = []
    patterns.extend(re.compile(pattern, re.IGNORECASE) for pattern in sorted(pattern_texts))
    for prompts in grouped.values():
        unique_prompts = sorted(set(prompts))
        if len(unique_prompts) < 2:
            continue
        pattern = generalized_prompt_pattern(unique_prompts[0], unique_prompts[-1])
        if pattern is not None:
            patterns.append(pattern)
    return tuple(patterns)


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
        escaped = escaped.replace(re.escape(value), ".+?")
    if ".+?" not in escaped:
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
            parts.append(".+?")
        if block.size:
            literal = "".join(left_tokens[block.a : block.a + block.size])
            literal_chars += len(literal.strip())
            parts.append(re.escape(literal))
        last_left = block.a + block.size
        last_right = block.b + block.size
    if literal_chars < 24 or not parts:
        return None
    return re.compile("".join(parts), re.IGNORECASE)


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
    return (*_family_split(family, "train", TRAIN_RECORDS_PER_FAMILY), *_family_split(family, "eval", EVAL_RECORDS_PER_FAMILY))


def _family_split(family: str, split: str, count: int) -> tuple[FeasibilityRecord, ...]:
    base = 10_000 if split == "eval" else 0
    return tuple(_make_record(family, split, base + index, index) for index in range(count))


def _make_record(family: str, split: str, operand_number: int, index: int) -> FeasibilityRecord:
    rng = random.Random(730000 + 10000 * FAMILIES.index(family) + operand_number)
    split_code = "t" if split == "train" else "e"
    template_id = f"seq_{family}_{split}_{index % 4}"
    operand_id = f"seq_operand_{family}_{operand_number:05d}"
    surface = PROMPT_SURFACES[family][split][index % 4]
    if family == "hex_copy":
        high_bit = 0 if split == "train" else 1 << 63
        value = f"{high_bit | rng.getrandbits(63):016x}"
        prompt = surface.format(a=index % 17, b=value)
        answer = value
        semantic_values = (value,)
    elif family == "named_value_json":
        keys = NAMED_VALUE_KEYS_BY_SPLIT[split]
        target = keys[index % len(keys)]
        fields = {key: f"{key}-{split_code}{rng.getrandbits(32):08x}" for key in keys}
        field_text = "; ".join(f"{key}={fields[key]}" for key in keys)
        prompt = surface.format(a=target, b=field_text)
        answer = compact_json(fields[target])
        semantic_values = tuple(fields[key] for key in keys)
    elif family == "boolean_json":
        split_offset = 0 if split == "train" else 2000
        left = split_offset + rng.randrange(1, 200)
        right = split_offset + rng.randrange(1, 200)
        truth = left <= right if index % 2 == 0 else left > right
        relation = "is at most" if index % 2 == 0 else "is greater than"
        prompt = surface.format(a=left, b=relation, c=right)
        answer = "true" if truth else "false"
        semantic_values = (str(left), str(right))
    elif family == "array_json":
        items = [f"s{split_code}{rng.getrandbits(32):08x}" for _ in range(1 + index % 4)]
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
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unknown model_size for parameter_count validation: {model_size!r}.")
    return build_model(model_size).parameter_count


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
    require_canonical_path_string(str(root), "root", ROOT_RE)
    require_artifact_location(root, "root", ROOT_RE)
    root_number = feasibility_root_number(root)
    direct_numbers: list[int] = []
    for predecessor_root in predecessor_roots:
        predecessor_number = feasibility_root_number(predecessor_root)
        if direct_numbers and predecessor_number <= direct_numbers[-1]:
            raise ValueError("Feasibility predecessor roots must be in strictly ascending root-number order.")
        direct_numbers.append(predecessor_number)
        terminal_binding(predecessor_root)
    selected_numbers: list[int] = []
    for selection in predecessor_selections:
        data = validate_selection_record(selection)
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


def source_clean_allowed_paths(predecessor_roots: Sequence[Path], predecessor_selections: Sequence[Path]) -> set[Path]:
    allowed: set[Path] = set()
    for selection in predecessor_selection_paths_from_selections(predecessor_selections):
        allowed.add(selection.resolve())
    for predecessor_root in (*predecessor_roots, *predecessor_root_paths_from_selections(predecessor_selections)):
        allowed.update(inventory_bound_root_paths(predecessor_root))
    return allowed


def inventory_bound_root_paths(root: Path) -> set[Path]:
    require_canonical_path_string(str(root), "predecessor_root.path", ROOT_RE)
    require_artifact_location(root, "predecessor_root.path", ROOT_RE)
    validate_feasibility_root_artifacts(root, require_passing=False)
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


def terminal_binding(root: Path) -> dict[str, object]:
    require_canonical_path_string(str(root), "predecessor_root.path", ROOT_RE)
    require_artifact_location(root, "predecessor_root.path", ROOT_RE)
    validate_feasibility_root_artifacts(root, require_passing=False)
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


def validate_feasibility_root_artifacts(root: Path, *, require_passing: bool) -> None:
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
    validate_source_provenance(
        manifest_data.get("source_provenance"),
        source_commit,
        source_provenance_allowed_paths(manifest_data),
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
        generation_rows = validate_generation_artifact(root / generations_path, cell)
        validate_checkpoint_artifact(root / checkpoint_path, cell)
        if require_passing:
            validate_checkpoint_replays_generations(root / checkpoint_path, cell, generation_rows)


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
    expected_count = require_exact_int(cell.get("eval_count"), "eval_count")
    expected_matches = require_exact_int(cell.get("exact_matches"), "exact_matches")
    rows: list[dict[str, object]] = []
    eval_records = grouped_records()[family]["eval"]
    tokenizer = ByteTokenizer()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"Generation artifact contains a blank row at line {line_number}.")
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError("Generation artifact rows must be JSON objects.")
            exact_match = row.get("exact_match")
            if type(exact_match) is not bool:
                raise ValueError("Generation artifact exact_match values must be JSON booleans.")
            raw_token_ids = row.get("raw_token_ids")
            if not isinstance(raw_token_ids, list) or not all(type(token) is int for token in raw_token_ids):
                raise ValueError("Generation artifact rows must retain raw_token_ids as JSON integers.")
            if len(rows) >= len(eval_records):
                raise ValueError("Generation artifact contains more rows than the frozen evaluation split.")
            record = eval_records[len(rows)]
            if row.get("family") != record.family:
                raise ValueError("Generation artifact row family does not match the frozen evaluation record.")
            if row.get("index") != record.index:
                raise ValueError("Generation artifact row index does not match the frozen evaluation record.")
            if row.get("template_id") != record.template_id:
                raise ValueError("Generation artifact row template_id does not match the frozen evaluation record.")
            if row.get("operand_id") != record.operand_id:
                raise ValueError("Generation artifact row operand_id does not match the frozen evaluation record.")
            if row.get("prompt") != record.prompt:
                raise ValueError("Generation artifact row prompt does not match the frozen evaluation record.")
            if row.get("expected") != record.answer:
                raise ValueError("Generation artifact row expected answer does not match the frozen evaluation record.")
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
                if not isinstance(row.get("generation_error"), str):
                    raise ValueError("Invalid generation rows must retain a generation_error string.")
            else:
                if row.get("generated") != decoded:
                    raise ValueError("Generation artifact generated text does not match raw_token_ids decoding.")
                if row.get("generation_error") is not None:
                    raise ValueError("Valid generation rows must record generation_error as null.")
            if exact_match != (decoded == record.answer):
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
        add(terminal_binding(root))
    for selection_path in predecessor_selections:
        selection_data = validate_selection_record(selection_path)
        add(terminal_binding(Path(str(selection_data["selected_root"]))))
        for predecessor in selection_data["predecessor_roots"]:
            add(predecessor)
    return sorted(completed, key=lambda binding: feasibility_root_number(Path(str(binding["path"]))))


def source_provenance_allowed_paths(manifest_data: dict[str, object]) -> set[Path]:
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
    )


def predecessor_root_paths_from_selections(predecessor_selections: Sequence[Path]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for selection_path in predecessor_selections:
        selection_data = validate_selection_record(selection_path)
        paths.append(Path(str(selection_data["selected_root"])))
        paths.extend(Path(str(predecessor["path"])) for predecessor in selection_data["predecessor_roots"])
    return tuple(paths)


def predecessor_selection_paths_from_selections(predecessor_selections: Sequence[Path]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for selection_path in predecessor_selections:
        selection_data = validate_selection_record(selection_path)
        paths.append(selection_path)
        paths.extend(Path(str(predecessor["path"])) for predecessor in selection_data["predecessor_selections"])
    return tuple(paths)


def validate_selection_record(path: Path) -> dict[str, object]:
    data, _root_binding_map = _validate_selection_record(path, seen=set())
    return data


def _validate_selection_record(
    path: Path,
    *,
    seen: set[Path],
) -> tuple[dict[str, object], dict[str, tuple[str, str, str]]]:
    current_number = selection_record_number(path)
    require_canonical_path_string(str(path), "selection_record.path", SELECTION_RE)
    require_artifact_location(path, "selection_record.path", SELECTION_RE)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Selection record must be a regular file in the canonical artifact parent.")
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Selection-record lineage contains a cycle at {path}.")
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
    validate_feasibility_root_artifacts(root, require_passing=True)
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
        predecessor_data, predecessor_validated_roots = _validate_selection_record(predecessor_path, seen=seen)
        predecessor_selected_root = Path(str(predecessor_data["selected_root"]))
        predecessor_root_numbers.append(feasibility_root_number(predecessor_selected_root))
        add_root_binding(expected_root_bindings, terminal_binding(predecessor_selected_root))
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
    seen.remove(resolved)
    return data, predecessor_root_bindings


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


def unchecked_source_snapshot() -> SourceSnapshot:
    commit = git_output(["git", "rev-parse", "HEAD"])
    validate_git_sha(commit, "source_commit")
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
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
