#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
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

GENERIC_JSON_MARKERS = frozenset({"true", "false", "null", "unable", "[]", "{}"})


@dataclass(frozen=True)
class FeasibilityRecord:
    family: str
    split: str
    index: int
    template_id: str
    operand_id: str
    prompt: str
    answer: str


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def reject_scientific_markers(text: str) -> None:
    upper_text = text.upper()
    for marker in SCIENTIFIC_MARKERS:
        if marker.upper() in upper_text:
            raise ValueError(f"Feasibility text contains forbidden Phase 8 scientific marker: {marker!r}.")
    for marker in phase8_scientific_identity_markers():
        if marker in text:
            raise ValueError(f"Feasibility text contains forbidden Phase 8 scientific identity marker: {marker!r}.")


def validate_feasibility_records(records: Sequence[FeasibilityRecord]) -> None:
    if not records:
        raise ValueError("records must be non-empty.")
    for record in records:
        reject_scientific_markers(record.prompt)
        reject_scientific_markers(record.answer)
        reject_scientific_markers(record.template_id)
        reject_scientific_markers(record.operand_id)
    train = {(r.template_id, r.operand_id, r.prompt, r.answer) for r in records if r.split == "train"}
    eval_ = {(r.template_id, r.operand_id, r.prompt, r.answer) for r in records if r.split == "eval"}
    if train & eval_:
        raise ValueError("Feasibility train/evaluation records must be disjoint.")


@lru_cache(maxsize=1)
def phase8_scientific_identity_markers() -> frozenset[str]:
    markers: set[str] = set()
    for task_id in cg.TRAINING_TASK_ORDER:
        for record in cg.build_split_records("training", task_id, cg.CORPUS_RECORDS_PER_FAMILY["large"]):
            _add_phase8_record_markers(markers, record)
    for probe in cg.build_evaluation_probe_pack():
        _add_phase8_record_markers(markers, probe)
    return frozenset(markers)


def _add_phase8_record_markers(markers: set[str], record: object) -> None:
    for field_name in ("task_id", "template_id", "payload_id"):
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
    template_id = f"seq_{family}_{split}_{index % 4}"
    operand_id = f"seq_operand_{family}_{operand_number:05d}"
    if family == "hex_copy":
        value = f"{rng.getrandbits(64):016x}"
        prompt = f"Echo the 16 hex characters after tag {index % 17}: {value}"
        answer = value
    elif family == "named_value_json":
        keys = ("red", "blue", "green", "silver")
        target = keys[index % len(keys)]
        fields = {key: f"{key}-{rng.randrange(1000, 9999)}" for key in keys}
        prompt = "Return compact JSON for " + target + " from " + "; ".join(f"{key}={fields[key]}" for key in keys)
        answer = compact_json(fields[target])
    elif family == "boolean_json":
        left = rng.randrange(1, 200)
        right = rng.randrange(1, 200)
        truth = left <= right if index % 2 == 0 else left > right
        relation = "is at most" if index % 2 == 0 else "is greater than"
        prompt = f"Return JSON truth value: {left} {relation} {right}."
        answer = "true" if truth else "false"
    elif family == "array_json":
        items = [f"s{rng.randrange(100, 999)}" for _ in range(1 + index % 4)]
        prompt = "Return compact JSON list from operands: " + " | ".join(items)
        answer = compact_json(items)
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
        decoded = tokenizer.decode_generated_response(full_ids)
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
                "exact_match": match,
            }
        )
    return correct, rows


def validate_cell_counts(cells: Sequence[dict[str, object]]) -> None:
    expected_keys = {(family, model_size, seed) for family in FAMILIES for model_size in MODEL_SIZES for seed in SEEDS}
    seen: set[tuple[str, str, int]] = set()
    for cell in cells:
        key = (str(cell["family"]), str(cell["model_size"]), int(cell["seed"]))
        if key not in expected_keys:
            raise ValueError(f"Unexpected feasibility cell: {key!r}.")
        if key in seen:
            raise ValueError(f"Duplicate feasibility cell: {key!r}.")
        seen.add(key)
        if int(cell["eval_count"]) != EVAL_RECORDS_PER_FAMILY:
            raise ValueError("Every feasibility cell must evaluate 64 records.")
        passed = int(cell["exact_matches"]) >= PASS_THRESHOLD
        if bool(cell["passed"]) != passed:
            raise ValueError("Feasibility cell pass/fail does not match the 52/64 requirement.")
    missing = expected_keys - seen
    if missing:
        raise ValueError(f"Missing feasibility cells: {sorted(missing)!r}.")


def run_suite(root: Path, predecessor_roots: Sequence[Path], predecessor_selections: Sequence[Path]) -> None:
    validate_new_root(root, predecessor_roots, predecessor_selections)
    validate_source_clean(root, predecessor_roots, predecessor_selections)
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
        write_terminal(temp_root, terminal_status, cells, predecessor_roots, predecessor_selections)
        os.replace(temp_root, root)
    except Exception as exc:
        write_terminal(
            temp_root,
            "FAILED",
            cells,
            predecessor_roots,
            predecessor_selections,
            failure=repr(exc),
        )
        os.replace(temp_root, root)
        raise


def validate_new_root(
    root: Path,
    predecessor_roots: Sequence[Path] = (),
    predecessor_selections: Sequence[Path] = (),
) -> None:
    root_number = feasibility_root_number(root)
    previous_numbers = [feasibility_root_number(path) for path in predecessor_roots]
    for predecessor_root in predecessor_roots:
        terminal_binding(predecessor_root)
    for selection in predecessor_selections:
        data = validate_selection_record(selection)
        previous_numbers.append(feasibility_root_number(Path(str(data["selected_root"]))))
    expected_number = max(previous_numbers, default=0) + 1
    if root_number != expected_number:
        raise ValueError(
            f"Feasibility root must use the next numbered root feasibility_{expected_number:03d}; "
            f"got {root.name!r}."
        )
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing feasibility root: {root}")


def feasibility_root_number(root: Path) -> int:
    match = ROOT_RE.match(root.name)
    if match is None:
        raise ValueError("Feasibility root basename must be immutable numbered form feasibility_NNN.")
    return int(match.group(1))


def validate_source_clean(root: Path, predecessor_roots: Sequence[Path], predecessor_selections: Sequence[Path]) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    allowed = {path.resolve() for path in (*predecessor_roots, *predecessor_selections)}
    allowed.add(root.resolve())
    for line in status:
        code = line[:2]
        rel = line[3:]
        candidate = Path(rel).resolve()
        if code != "??":
            raise RuntimeError(f"Tracked or staged source change blocks feasibility run: {line}")
        if not any(candidate == item or item in candidate.parents for item in allowed):
            raise RuntimeError(f"Untracked file is not an exact supplied predecessor/root binding: {line}")


def build_manifest(
    root: Path,
    cells: Sequence[dict[str, object]],
    predecessor_roots: Sequence[Path],
    predecessor_selections: Sequence[Path],
    terminal_status: str,
    failure: str | None = None,
) -> dict[str, object]:
    return {
        "protocol": "phase8_sequence_feasibility",
        "terminal_status": terminal_status,
        "failure": failure,
        "source_commit": git_output(["git", "rev-parse", "HEAD"]),
        "configuration": {
            "families": FAMILIES,
            "model_sizes": MODEL_SIZES,
            "seeds": SEEDS,
            "train_records_per_family": TRAIN_RECORDS_PER_FAMILY,
            "eval_records_per_family": EVAL_RECORDS_PER_FAMILY,
            "training_steps": TRAINING_STEPS,
            "batch_size": BATCH_SIZE,
            "pass_threshold": PASS_THRESHOLD,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "cells": list(cells),
        "predecessor_roots": [terminal_binding(path) for path in predecessor_roots],
        "predecessor_selections": [selection_binding(path) for path in predecessor_selections],
        "file_inventory": inventory(root),
    }


def write_terminal(
    root: Path,
    terminal_status: str,
    cells: Sequence[dict[str, object]],
    predecessor_roots: Sequence[Path],
    predecessor_selections: Sequence[Path],
    *,
    failure: str | None = None,
) -> None:
    if terminal_status not in {"DONE", "FAILED"}:
        raise ValueError(f"Unknown terminal status: {terminal_status!r}.")
    manifest = build_manifest(root, cells, predecessor_roots, predecessor_selections, terminal_status, failure)
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
    terminal = root / "DONE.json"
    if not terminal.exists():
        terminal = root / "FAILED.json"
    if not terminal.exists():
        raise ValueError(f"Predecessor root lacks DONE.json or FAILED.json: {root}")
    manifest = root / "manifest.json"
    if not manifest.exists():
        raise ValueError(f"Predecessor root lacks manifest.json: {root}")
    return {
        "path": str(root),
        "terminal_state": terminal.stem,
        "terminal_sha256": file_sha256(terminal),
        "manifest_sha256": file_sha256(manifest),
    }


def selection_binding(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": file_sha256(path)}


def validate_selection_record(path: Path) -> dict[str, object]:
    return _validate_selection_record(path, seen=set())


def _validate_selection_record(path: Path, *, seen: set[Path]) -> dict[str, object]:
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
    root = Path(data["selected_root"])
    manifest = root / "manifest.json"
    if file_sha256(manifest) != data["selected_manifest_sha256"]:
        raise ValueError("Selection record manifest checksum does not match selected root.")
    terminal = root / "DONE.json"
    if not terminal.exists():
        raise ValueError("Selection record must bind a passing DONE root.")
    terminal_data = json.loads(terminal.read_text())
    if terminal_data.get("manifest_sha256") != data["selected_manifest_sha256"]:
        raise ValueError("Selected DONE terminal does not bind the selected manifest checksum.")
    cells = data["per_cell_counts"]
    validate_cell_counts(cells)
    if not bool(data["pass_decision"]):
        raise ValueError("Selection record pass_decision must be true for a selected root.")
    for predecessor in data["predecessor_roots"]:
        terminal_path = Path(predecessor["path"]) / f"{predecessor['terminal_state']}.json"
        manifest_path = Path(predecessor["path"]) / "manifest.json"
        if file_sha256(terminal_path) != predecessor["terminal_sha256"]:
            raise ValueError("Predecessor terminal checksum mismatch.")
        if file_sha256(manifest_path) != predecessor["manifest_sha256"]:
            raise ValueError("Predecessor manifest checksum mismatch.")
    lineage_bindings = {str(Path(predecessor["path"]).resolve()): predecessor["sha256"] for predecessor in data["predecessor_selections"]}
    for predecessor in data["predecessor_selections"]:
        predecessor_path = Path(predecessor["path"])
        if file_sha256(predecessor_path) != predecessor["sha256"]:
            raise ValueError("Predecessor selection checksum mismatch.")
        predecessor_data = _validate_selection_record(predecessor_path, seen=seen)
        for transitive in predecessor_data["predecessor_selections"]:
            transitive_path = str(Path(transitive["path"]).resolve())
            if lineage_bindings.get(transitive_path) != transitive["sha256"]:
                raise ValueError("Selection record must bind transitive predecessor selection lineage.")
    seen.remove(resolved)
    return data


def inventory(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
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
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 non-scientific sequence-transduction feasibility runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--root", required=True, type=Path)
    run.add_argument("--predecessor-root", action="append", type=Path, default=[])
    run.add_argument("--predecessor-selection", action="append", type=Path, default=[])
    validate = subparsers.add_parser("validate-selection")
    validate.add_argument("path", type=Path)
    inspect = subparsers.add_parser("inspect-records")
    inspect.add_argument("--family", choices=FAMILIES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "run":
        run_suite(args.root, args.predecessor_root, args.predecessor_selection)
        return 0
    if args.command == "validate-selection":
        validate_selection_record(args.path)
        return 0
    if args.command == "inspect-records":
        families = (args.family,) if args.family else FAMILIES
        data = {family: [asdict(record) for record in build_family_records(family)] for family in families}
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
