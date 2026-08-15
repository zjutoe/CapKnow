from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
import re
from types import MappingProxyType
from typing import Any

from capability_certificate_lab.certificate import solve_exact_certificate, validate_certificate
from capability_certificate_lab.dsl.executor import (
    MissingCapabilityError,
    execute,
)
from capability_certificate_lab.dsl.primitives import get_primitive
from capability_certificate_lab.dsl.program import PrimitiveNode, Program, SequenceNode
from capability_certificate_lab.knowledge_space.space import KnowledgeSpace
from capability_certificate_lab.knowledge_space.state import KnowledgeState
from capability_certificate_lab.knowledge_space.tasks import TaskUniverse
from capability_certificate_lab.validation.identifiability import check_identifiability


PRIMITIVE_ORDER: tuple[str, ...] = ("MEMORY", "SEARCH", "FILTER", "CONDITION")
STATE_MASKS: tuple[int, ...] = tuple(range(16))
TASK_ORDER: tuple[str, ...] = (
    "MEMORY",
    "SEARCH",
    "FILTER",
    "CONDITION",
    "MEMORY_FILTER",
    "FILTER_CONDITION",
    "SEARCH_CONDITION",
    "MEMORY_SEARCH",
)

_TASK_REQUIRED: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "MEMORY": ("MEMORY",),
        "SEARCH": ("SEARCH",),
        "FILTER": ("FILTER",),
        "CONDITION": ("CONDITION",),
        "MEMORY_FILTER": ("MEMORY", "FILTER"),
        "FILTER_CONDITION": ("FILTER", "CONDITION"),
        "SEARCH_CONDITION": ("SEARCH", "CONDITION"),
        "MEMORY_SEARCH": ("MEMORY", "SEARCH"),
    }
)

_PRIMITIVE_TEMPLATES: tuple[str, ...] = (
    "neutral_a",
    "neutral_b",
    "neutral_c",
    "neutral_d",
)
_COMPOSITION_EXPLICIT_TEMPLATES: tuple[str, ...] = (
    "explicit_a",
    "explicit_b",
    "explicit_c",
    "explicit_d",
)
_COMPOSITION_INDIRECT_TEMPLATES: tuple[str, ...] = (
    "indirect_a",
    "indirect_b",
    "indirect_c",
    "indirect_d",
)
_TRAIN_TEMPLATES: tuple[str, ...] = ("train_a", "train_b", "train_c", "train_d")
_EVALUATION_SPLIT = "evaluation"
_TRAINING_SPLIT = "training"

_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:MEMORY|SEARCH|FILTER|CONDITION|MEMORY_FILTER|FILTER_CONDITION|SEARCH_CONDITION|MEMORY_SEARCH)\b"
    ),
    *(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\b(?:primitive|capability|certificate|knowledge state|graph|prerequisite)\b",
            r"\brule\s+tables?\b",
            r"\b(?:state|model|seed)\s*(?:id|ids|[0-9])\b",
            r"\bseed\s+[0-9]+\b",
            r"\b[01]{4}\b",
            r"[A-Z]\s*\+\s*[A-Z]\s*->\s*[A-Z]",
            r"->",
        )
    ),
)


@dataclass(frozen=True)
class SplitRecord:
    split: str
    task_id: str
    task_index: int
    record_index: int
    template_id: str
    payload_id: str
    program_dict: Mapping[str, Any]
    canonical_context: str
    normalized_payload: tuple[Any, ...]
    prompt: str
    answer: str
    probe_key: str | None = None
    style: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "program_dict", _deep_freeze_json(self.program_dict))
        object.__setattr__(self, "normalized_payload", _deep_freeze_json(self.normalized_payload))


@dataclass(frozen=True)
class EvaluationProbe:
    probe_key: str
    task_id: str
    task_index: int
    record_index: int
    template_id: str
    payload_id: str
    program_dict: Mapping[str, Any]
    canonical_context: str
    normalized_payload: tuple[Any, ...]
    prompt: str
    answer: str
    style: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "program_dict", _deep_freeze_json(self.program_dict))
        object.__setattr__(self, "normalized_payload", _deep_freeze_json(self.normalized_payload))


def _canonical_json(value: object) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, MappingProxyType | Mapping):
        return MappingProxyType({key: _deep_freeze_json(nested) for key, nested in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze_json(nested) for nested in value)
    return value


def _json_ready(value: object) -> object:
    if isinstance(value, MappingProxyType | Mapping):
        return {key: _json_ready(nested) for key, nested in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(nested) for nested in value]
    return value


def _programs() -> dict[str, Program]:
    memory = PrimitiveNode(get_primitive("MEMORY"))
    search = PrimitiveNode(get_primitive("SEARCH"))
    filter_ = PrimitiveNode(get_primitive("FILTER"))
    condition = PrimitiveNode(get_primitive("CONDITION"))
    return {
        "MEMORY": memory,
        "SEARCH": search,
        "FILTER": filter_,
        "CONDITION": condition,
        "MEMORY_FILTER": SequenceNode((memory, filter_)),
        "FILTER_CONDITION": SequenceNode((filter_, condition)),
        "SEARCH_CONDITION": SequenceNode((search, condition)),
        "MEMORY_SEARCH": SequenceNode((memory, search)),
    }


def program_for_task(task_id: str) -> Program:
    _validate_task_id(task_id)
    return _programs()[task_id]


def program_dict(task_id: str) -> Mapping[str, Any]:
    return _deep_freeze_json(program_for_task(task_id).to_dict())


def _validate_task_id(task_id: str) -> None:
    if task_id not in TASK_ORDER:
        raise ValueError(f"Unknown Phase 8 task id: {task_id!r}")


def required_primitives(task_id: str) -> tuple[str, ...]:
    _validate_task_id(task_id)
    return _TASK_REQUIRED[task_id]


def state_from_mask(mask: int) -> Mapping[str, bool]:
    if mask not in STATE_MASKS:
        raise ValueError(f"State mask must be in 0..15, got {mask!r}.")
    return MappingProxyType(
        {primitive: bool(mask & (1 << idx)) for idx, primitive in enumerate(PRIMITIVE_ORDER)}
    )


def canonical_context_bytes(context: Mapping[str, Any]) -> bytes:
    return _canonical_json(context).encode("utf-8")


def _expect_string(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")


def _expect_string_items(items: object) -> None:
    if not isinstance(items, list) or not items:
        raise ValueError("items must be a non-empty list.")
    for idx, item in enumerate(items):
        _expect_string(item, f"items[{idx}]")


def validate_context_schema(task_id: str, context: Mapping[str, Any]) -> None:
    _validate_task_id(task_id)
    if not isinstance(context, Mapping):
        raise ValueError("context must be a mapping.")
    keys = set(context)
    if task_id == "MEMORY":
        if keys != {"memory", "key"}:
            raise ValueError("MEMORY context must have exactly memory and key.")
        _validate_memory_context(context)
        return
    if task_id in {"SEARCH", "FILTER", "FILTER_CONDITION", "SEARCH_CONDITION"}:
        if keys != {"items", "target"}:
            raise ValueError(f"{task_id} context must have exactly items and target.")
        _expect_string_items(context["items"])
        _expect_string(context["target"], "target")
        return
    if task_id == "CONDITION":
        if keys != {"condition"}:
            raise ValueError("CONDITION context must have exactly condition.")
        if type(context["condition"]) is not bool:
            raise ValueError("condition must be a Boolean.")
        return
    if task_id in {"MEMORY_FILTER", "MEMORY_SEARCH"}:
        if keys != {"memory", "key", "items"}:
            raise ValueError(f"{task_id} context must have exactly memory, key, and items.")
        _validate_memory_context(context)
        _expect_string_items(context["items"])
        return
    raise AssertionError("unreachable")


def _validate_memory_context(context: Mapping[str, Any]) -> None:
    memory = context["memory"]
    key = context["key"]
    _expect_string(key, "key")
    if not isinstance(memory, Mapping) or set(memory) != {key}:
        raise ValueError("memory must contain exactly the shown key.")
    _expect_string(memory[key], "memory[key]")


def execute_task(task_id: str, context: Mapping[str, Any], state_has: Mapping[str, bool]) -> object:
    validate_context_schema(task_id, context)
    result = execute(program_for_task(task_id), state_has, context)
    if task_id == "MEMORY":
        value = result.value
        if not isinstance(value, Mapping) or "value" not in value:
            raise ValueError("MEMORY execution did not produce a value field.")
        return value["value"]
    return result.value


def canonical_answer(task_id: str, context: Mapping[str, Any]) -> str:
    return _canonical_json(execute_task(task_id, context, _full_state()))


def normalized_payload_tuple(task_id: str, context: Mapping[str, Any]) -> tuple[Any, ...]:
    validate_context_schema(task_id, context)
    if task_id == "MEMORY":
        key = context["key"]
        return (key, context["memory"][key])
    if task_id in {"SEARCH", "FILTER"}:
        return (tuple(context["items"]), context["target"])
    if task_id == "CONDITION":
        return (context["condition"],)
    if task_id in {"MEMORY_FILTER", "MEMORY_SEARCH"}:
        key = context["key"]
        return (key, context["memory"][key], tuple(context["items"]))
    if task_id in {"FILTER_CONDITION", "SEARCH_CONDITION"}:
        return (tuple(context["items"]), context["target"])
    raise AssertionError("unreachable")


def _full_state() -> Mapping[str, bool]:
    return MappingProxyType({primitive: True for primitive in PRIMITIVE_ORDER})


def _operand(task_id: str, payload_seed: int, field_name: str) -> str:
    text = f"phase8-payload|{task_id}|{payload_seed}|{field_name}"
    return sha256(text.encode("ascii")).hexdigest()[:16]


def _payload_seed(split: str, task_index: int, record_index: int) -> int:
    if record_index < 0:
        raise ValueError("record_index must be non-negative.")
    if split == _TRAINING_SPLIT:
        return 100000 + 1000 * task_index + record_index
    if split == _EVALUATION_SPLIT:
        return 300000 + 1000 * task_index + record_index
    raise ValueError(f"Unknown split: {split!r}")


def _boolean_value(task_id: str, record_index: int) -> bool:
    if task_id == "CONDITION":
        return record_index % 2 == 0
    if task_id in {"FILTER_CONDITION", "SEARCH_CONDITION", "MEMORY_SEARCH"}:
        return record_index % 32 < 16
    return record_index % 64 < 32


def context_for_payload(task_id: str, split: str, record_index: int) -> Mapping[str, Any]:
    _validate_task_id(task_id)
    task_index = TASK_ORDER.index(task_id)
    seed = _payload_seed(split, task_index, record_index)
    key = _operand(task_id, seed, "key")
    value = _operand(task_id, seed, "value")
    target = _operand(task_id, seed, "target")
    distractor_a = _operand(task_id, seed, "item_a")
    distractor_b = _operand(task_id, seed, "item_b")

    if task_id == "MEMORY":
        return {"memory": {key: value}, "key": key}
    if task_id == "SEARCH":
        present = _boolean_value(task_id, record_index)
        items = [distractor_a, target, distractor_b] if present else [distractor_a, value, distractor_b]
        return {"items": items, "target": target}
    if task_id == "FILTER":
        return {"items": [distractor_a, target, distractor_b], "target": target}
    if task_id == "CONDITION":
        return {"condition": _boolean_value(task_id, record_index)}
    if task_id == "MEMORY_FILTER":
        return {"memory": {key: value}, "key": key, "items": [distractor_a, value, distractor_b]}
    if task_id == "FILTER_CONDITION":
        present = _boolean_value(task_id, record_index)
        items = [distractor_a, target, distractor_b] if present else [distractor_a, value, distractor_b]
        return {"items": items, "target": target}
    if task_id == "SEARCH_CONDITION":
        present = _boolean_value(task_id, record_index)
        items = [distractor_a, target, distractor_b] if present else [distractor_a, value, distractor_b]
        return {"items": items, "target": target}
    if task_id == "MEMORY_SEARCH":
        present = _boolean_value(task_id, record_index)
        items = [distractor_a, value, distractor_b] if present else [distractor_a, target, distractor_b]
        return {"memory": {key: value}, "key": key, "items": items}
    raise AssertionError("unreachable")


def _template_id(task_id: str, split: str, record_index: int, *, evaluation_pack: bool) -> tuple[str, str]:
    if split == _TRAINING_SPLIT:
        return f"{task_id.lower()}__{_TRAIN_TEMPLATES[record_index % len(_TRAIN_TEMPLATES)]}", "train"
    if split != _EVALUATION_SPLIT:
        raise ValueError(f"Unknown split: {split!r}")
    if task_id in {"MEMORY", "SEARCH", "FILTER", "CONDITION"}:
        return f"neutral_eval__{_PRIMITIVE_TEMPLATES[record_index % len(_PRIMITIVE_TEMPLATES)]}", "neutral"
    if record_index < 32:
        return (
            f"{task_id.lower()}__{_COMPOSITION_EXPLICIT_TEMPLATES[record_index % len(_COMPOSITION_EXPLICIT_TEMPLATES)]}",
            "explicit",
        )
    return (
        f"{task_id.lower()}__{_COMPOSITION_INDIRECT_TEMPLATES[record_index % len(_COMPOSITION_INDIRECT_TEMPLATES)]}",
        "indirect",
    )


def _items_text(items: Sequence[str]) -> str:
    return ", ".join(items)


def _template_variant(template_id: str) -> int:
    suffix = template_id.rsplit("_", maxsplit=1)[-1]
    variants = {"a": 0, "b": 1, "c": 2, "d": 3}
    if suffix not in variants:
        raise ValueError(f"Unknown template variant in {template_id!r}.")
    return variants[suffix]


def render_prompt(task_id: str, context: Mapping[str, Any], template_id: str, style: str) -> str:
    validate_context_schema(task_id, context)
    variant = _template_variant(template_id)
    if task_id == "MEMORY":
        key = context["key"]
        value = context["memory"][key]
        if style == "train":
            prompt = (
                f"A table entry says {key} has stored value {value}. Write the value for {key} in compact JSON.",
                f"Entry {key}:{value} is available. Return compact JSON containing the value paired with {key}.",
                f"For key {key}, the table gives {value}. Output the associated value as compact JSON.",
                f"Use pair {key}:{value}; answer with the paired value for {key} in compact JSON.",
            )[variant]
        else:
            prompt = (
                f"In the table, {key} maps to {value}. Return the stored value for {key} as compact JSON.",
                f"Given entry {key}:{value}, write the value paired with {key} as compact JSON.",
                f"The lookup table contains {key} = {value}. Answer with the value for {key} in compact JSON.",
                f"Use the entry {key} paired with {value}; return only the value for {key} as compact JSON.",
            )[variant]
    elif task_id == "SEARCH":
        items = _items_text(context["items"])
        target = context["target"]
        if style == "train":
            prompt = (
                f"Given entries {items}, write compact JSON true or false for whether {target} is present.",
                f"Return whether {target} occurs in this entry list as compact JSON true or false: {items}.",
                f"Use compact JSON true or false to answer if {items} contains {target}.",
                f"Check the list {items} for {target} and output the Boolean in compact JSON.",
            )[variant]
        else:
            prompt = (
                f"Decide whether {target} appears in this list: {items}. Return compact JSON true or false.",
                f"Check if {target} is one of these entries: {items}. Return compact JSON true or false.",
                f"For entries {items}, answer whether {target} is included using compact JSON true or false.",
                f"Look through {items}; return compact JSON true or false for the presence of {target}.",
            )[variant]
    elif task_id == "FILTER":
        items = _items_text(context["items"])
        target = context["target"]
        if style == "train":
            prompt = (
                f"Given entries {items}, write a compact JSON list containing only values equal to {target}.",
                f"Output entries from {items} that match {target}, using a compact JSON list.",
                f"Keep the members of {items} equal to {target} and return them in compact JSON.",
                f"Use {target} as the match value for {items}; answer with the matching entries as compact JSON.",
            )[variant]
        else:
            prompt = (
                f"From this list, keep only entries equal to {target}: {items}. Return the compact JSON list.",
                f"For entries {items}, output the entries matching {target} as a compact JSON list.",
                f"Select every item equal to {target} from {items}; return compact JSON.",
                f"Return a compact JSON list containing only members of {items} that equal {target}.",
            )[variant]
    elif task_id == "CONDITION":
        word = "true" if context["condition"] else "false"
        if style == "train":
            prompt = (
                f"A recorded claim has value {word}. Write that Boolean in compact JSON.",
                f"The supplied yes-or-no answer is {word}; return the same compact JSON Boolean.",
                f"Mirror the given truth value {word} using compact JSON.",
                f"Output compact JSON true or false matching this value: {word}.",
            )[variant]
        else:
            prompt = (
                f"The statement value is {word}. Return the same Boolean as compact JSON.",
                f"Given the truth value {word}, output that Boolean as compact JSON.",
                f"A yes-or-no value is {word}; return compact JSON true or false with the same value.",
                f"Use {word} as the truth value and return it as compact JSON.",
            )[variant]
    elif task_id == "MEMORY_FILTER":
        key = context["key"]
        value = context["memory"][key]
        items = _items_text(context["items"])
        if style == "explicit":
            prompt = (
                f"First read that {key} maps to {value}. Then keep list entries equal to that read value from {items}. Return compact JSON.",
                f"Read the table pair {key}:{value}; next, from {items}, keep entries equal to the read value. Return compact JSON.",
                f"Use {key} to read {value}, then select list entries equal to {value} from {items}. Return compact JSON.",
                f"Step one gives {value} from key {key}. Step two keeps matching entries from {items}. Return compact JSON.",
            )[variant]
        elif style == "indirect":
            prompt = (
                f"Using the table entry {key}:{value}, return the list items matching the entry value from {items} as compact JSON.",
                f"From table pair {key}:{value} and list {items}, return compact JSON entries equal to the pair value.",
                f"Return the compact JSON list of entries in {items} that match the value paired with {key}, which is {value}.",
                f"With {key}:{value} and entries {items}, output only the entries equal to the paired value as compact JSON.",
            )[variant]
        else:
            prompt = (
                f"Given table pair {key}:{value}, keep entries from {items} equal to {value}. Return compact JSON.",
                f"Use {key}:{value} with list {items}; output matching entries as compact JSON.",
                f"Find the value paired with {key}, then return matching entries from {items} as compact JSON.",
                f"For pair {key}:{value} and entries {items}, return compact JSON entries equal to {value}.",
            )[variant]
    elif task_id == "FILTER_CONDITION":
        items = _items_text(context["items"])
        target = context["target"]
        if style == "explicit":
            prompt = (
                f"First keep entries equal to {target} from {items}. Then report whether anything remains as compact JSON true or false.",
                f"Select entries matching {target} in {items}; after that, return whether the selected list is non-empty as compact JSON.",
                f"Keep only {target} from {items}. Then answer compact JSON true or false for whether any entry remains.",
                f"Step one keeps matches for {target} in {items}. Step two returns whether the kept list has an entry as compact JSON.",
            )[variant]
        elif style == "indirect":
            prompt = (
                f"Do any listed entries match {target} in {items}? Return compact JSON true or false.",
                f"For entries {items}, return compact JSON true or false for whether at least one equals {target}.",
                f"Is there an entry equal to {target} among {items}? Answer in compact JSON true or false.",
                f"Return whether {items} contains a member equal to {target}, using compact JSON true or false.",
            )[variant]
        else:
            prompt = (
                f"From {items}, determine whether any entry equals {target}. Return compact JSON true or false.",
                f"Check entries {items} for at least one match to {target}; answer as compact JSON true or false.",
                f"Return compact JSON true or false for whether matching {target} leaves a non-empty list from {items}.",
                f"Using entries {items}, say whether {target} appears at least once as compact JSON true or false.",
            )[variant]
    elif task_id == "SEARCH_CONDITION":
        items = _items_text(context["items"])
        target = context["target"]
        if style == "explicit":
            prompt = (
                f"First decide whether {target} is in {items}. Then return that decision as compact JSON true or false.",
                f"Check membership of {target} in {items}; after deciding, output the same Boolean as compact JSON.",
                f"Determine if {target} appears in {items}. Then return the resulting truth value as compact JSON.",
                f"Step one asks whether {target} is present in {items}. Step two returns that answer as compact JSON.",
            )[variant]
        elif style == "indirect":
            prompt = (
                f"Is {target} included among {items}? Return compact JSON true or false.",
                f"Return compact JSON true or false for whether {target} occurs in {items}.",
                f"Among {items}, is {target} present? Answer as compact JSON true or false.",
                f"Using entries {items}, output whether {target} is included as compact JSON true or false.",
            )[variant]
        else:
            prompt = (
                f"Decide if {target} is present in {items}; return compact JSON true or false.",
                f"For {items}, answer whether it includes {target} using compact JSON true or false.",
                f"Check whether {items} contains {target}. Return compact JSON true or false.",
                f"Return a compact JSON Boolean for whether {target} appears in {items}.",
            )[variant]
    elif task_id == "MEMORY_SEARCH":
        key = context["key"]
        value = context["memory"][key]
        items = _items_text(context["items"])
        if style == "explicit":
            prompt = (
                f"First read that {key} maps to {value}. Then decide whether that read value appears in {items}. Return compact JSON true or false.",
                f"Read {value} from key {key}; next, check whether {value} is in {items}. Return compact JSON true or false.",
                f"Step one gives {value} for {key}. Step two asks whether that value occurs in {items}. Return compact JSON.",
                f"Use {key} to read {value}, then answer whether the read value appears in {items} as compact JSON true or false.",
            )[variant]
        elif style == "indirect":
            prompt = (
                f"Using table entry {key}:{value}, is the entry value present in {items}? Return compact JSON true or false.",
                f"Given {key}:{value} and entries {items}, return whether the paired value is included as compact JSON true or false.",
                f"Does the value paired with {key}, namely {value}, appear in {items}? Return compact JSON true or false.",
                f"With pair {key}:{value}, answer whether {items} contains the pair value using compact JSON true or false.",
            )[variant]
        else:
            prompt = (
                f"Given table pair {key}:{value}, return whether {value} appears in {items} as compact JSON true or false.",
                f"Use {key}:{value} with entries {items}; answer whether the paired value is present as compact JSON.",
                f"Find the value paired with {key}, then say whether it is in {items} using compact JSON true or false.",
                f"For pair {key}:{value} and list {items}, return compact JSON true or false for whether the value appears.",
            )[variant]
    else:
        raise AssertionError("unreachable")
    leak_check_model_text(prompt)
    return prompt


def leak_check_model_text(text: str) -> None:
    for pattern in _LEAK_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"Model-facing text leaks forbidden Phase 8 metadata: {text!r}")


def _record(split: str, task_id: str, record_index: int) -> SplitRecord:
    task_index = TASK_ORDER.index(task_id)
    context = context_for_payload(task_id, split, record_index)
    template_id, style = _template_id(task_id, split, record_index, evaluation_pack=split == _EVALUATION_SPLIT)
    payload_id = f"{split}-payload-{task_index:02d}-{record_index:03d}"
    prompt = render_prompt(task_id, context, template_id, style)
    answer = canonical_answer(task_id, context)
    leak_check_model_text(answer)
    return SplitRecord(
        split=split,
        task_id=task_id,
        task_index=task_index,
        record_index=record_index,
        template_id=template_id,
        payload_id=payload_id,
        program_dict=program_dict(task_id),
        canonical_context=canonical_context_bytes(context).decode("utf-8"),
        normalized_payload=normalized_payload_tuple(task_id, context),
        prompt=prompt,
        answer=answer,
        style=style,
    )


def build_split_records(split: str, task_id: str, count: int) -> tuple[SplitRecord, ...]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    if split == _TRAINING_SPLIT and task_id == "MEMORY_SEARCH":
        raise ValueError("MEMORY_SEARCH is the held-out composition and cannot appear in training.")
    records = tuple(_record(split, task_id, idx) for idx in range(count))
    _reject_internal_collisions(records)
    return records


def _reject_internal_collisions(records: Sequence[SplitRecord]) -> None:
    seen_keys: set[tuple[str, str | tuple[Any, ...]]] = set()
    for record in records:
        keys: list[tuple[str, str | tuple[Any, ...]]] = [
            ("payload_id", record.payload_id),
        ]
        if record.task_id != "CONDITION":
            keys.append(("prompt", record.prompt))
        if record.task_id != "CONDITION":
            keys.extend(
                [
                    ("context", record.canonical_context),
                    ("payload", record.normalized_payload),
                ]
            )
        for key in keys:
            if key in seen_keys:
                raise ValueError(f"Duplicate generated {key[0]} detected for {record.task_id}.")
            seen_keys.add(key)


def validate_split_disjointness(
    training_records: Sequence[SplitRecord],
    evaluation_records: Sequence[SplitRecord],
) -> None:
    by_task: dict[str, dict[str, list[SplitRecord]]] = defaultdict(lambda: {"training": [], "evaluation": []})
    for record in training_records:
        by_task[record.task_id]["training"].append(record)
    for record in evaluation_records:
        by_task[record.task_id]["evaluation"].append(record)
    for task_id, groups in by_task.items():
        train = groups["training"]
        evaluation = groups["evaluation"]
        _reject_overlap(task_id, "prompt", (r.prompt for r in train), (r.prompt for r in evaluation))
        _reject_overlap(task_id, "template_id", (r.template_id for r in train), (r.template_id for r in evaluation))
        _reject_overlap(task_id, "payload_id", (r.payload_id for r in train), (r.payload_id for r in evaluation))
        _reject_overlap(
            task_id,
            "program/template/payload",
            ((_canonical_json(r.program_dict), r.template_id, r.payload_id) for r in train),
            ((_canonical_json(r.program_dict), r.template_id, r.payload_id) for r in evaluation),
        )
        if task_id == "CONDITION":
            train_semantic = {r.normalized_payload for r in train}
            eval_semantic = {r.normalized_payload for r in evaluation}
            if train_semantic != {(False,), (True,)} or eval_semantic != {(False,), (True,)}:
                raise ValueError("CONDITION must have exactly the semantic set {false,true} in both splits.")
            continue
        _reject_overlap(task_id, "canonical_context", (r.canonical_context for r in train), (r.canonical_context for r in evaluation))
        _reject_overlap(task_id, "normalized_payload", (r.normalized_payload for r in train), (r.normalized_payload for r in evaluation))


def _reject_overlap(task_id: str, label: str, left: Iterable[Any], right: Iterable[Any]) -> None:
    overlap = set(left) & set(right)
    if overlap:
        raise ValueError(f"{task_id} split overlap for {label}: {sorted(map(repr, overlap))[:3]}")


def _expected_probe(task_index: int, task_id: str, record_index: int) -> EvaluationProbe:
    record = _record(_EVALUATION_SPLIT, task_id, record_index)
    probe_key = sha256(f"phase8-probe|{task_index}|{record_index}".encode("ascii")).hexdigest()[:24]
    return EvaluationProbe(
        probe_key=probe_key,
        task_id=record.task_id,
        task_index=record.task_index,
        record_index=record.record_index,
        template_id=record.template_id,
        payload_id=record.payload_id,
        program_dict=record.program_dict,
        canonical_context=record.canonical_context,
        normalized_payload=record.normalized_payload,
        prompt=record.prompt,
        answer=record.answer,
        style=record.style or "neutral",
    )


def _build_evaluation_probe_pack_unvalidated() -> tuple[EvaluationProbe, ...]:
    probes: list[EvaluationProbe] = []
    for task_index, task_id in enumerate(TASK_ORDER):
        for record_index in range(64):
            probes.append(_expected_probe(task_index, task_id, record_index))
    return tuple(probes)


def build_evaluation_probe_pack() -> tuple[EvaluationProbe, ...]:
    pack = _build_evaluation_probe_pack_unvalidated()
    validate_evaluation_pack(pack)
    return pack


def evaluation_pack_checksum(pack: Sequence[EvaluationProbe]) -> str:
    payload = [
        {
            "probe_key": p.probe_key,
            "task_id": p.task_id,
            "task_index": p.task_index,
            "record_index": p.record_index,
            "template_id": p.template_id,
            "payload_id": p.payload_id,
            "program_dict": p.program_dict,
            "context": p.canonical_context,
            "normalized_payload": p.normalized_payload,
            "prompt": p.prompt,
            "answer": p.answer,
            "style": p.style,
        }
        for p in pack
    ]
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def validate_evaluation_pack(pack: Sequence[EvaluationProbe]) -> None:
    if len(pack) != 512:
        raise ValueError("Evaluation pack must contain exactly 512 probes.")
    if len({probe.probe_key for probe in pack}) != len(pack):
        raise ValueError("Evaluation probe keys must be unique.")
    for position, probe in enumerate(pack):
        task_index = position // 64
        record_index = position % 64
        task_id = TASK_ORDER[task_index]
        expected_probe_key = sha256(f"phase8-probe|{task_index}|{record_index}".encode("ascii")).hexdigest()[:24]
        if probe.task_index != task_index or probe.record_index != record_index:
            raise ValueError("Evaluation pack indices must match frozen task-major position.")
        if probe.task_id != task_id:
            raise ValueError("Evaluation pack task_id must match frozen task-major order.")
        if probe.probe_key != expected_probe_key:
            raise ValueError("Evaluation probe_key must be the frozen opaque key for its position.")
        leak_check_model_text(probe.prompt)
        leak_check_model_text(probe.answer)
        context = _parse_canonical_context(probe.canonical_context)
        if probe.answer != canonical_answer(task_id, context):
            raise ValueError("Evaluation answer must match canonical full-capability execution.")
        expected = _expected_probe(task_index, task_id, record_index)
        for field_name in (
            "template_id",
            "payload_id",
            "program_dict",
            "canonical_context",
            "normalized_payload",
            "prompt",
            "answer",
            "style",
        ):
            if getattr(probe, field_name) != getattr(expected, field_name):
                raise ValueError(f"Evaluation probe {field_name} does not match the canonical generated record.")
    checksum = evaluation_pack_checksum(pack)
    if checksum != EVALUATION_PACK_CHECKSUM:
        raise ValueError(
            "Evaluation pack checksum does not match the frozen checksum: "
            f"{checksum} != {EVALUATION_PACK_CHECKSUM}"
        )
    by_task = {task_id: [probe for probe in pack if probe.task_id == task_id] for task_id in TASK_ORDER}
    bool_tasks = {"SEARCH", "CONDITION", "FILTER_CONDITION", "SEARCH_CONDITION", "MEMORY_SEARCH"}
    for task_id, probes in by_task.items():
        answers = [probe.answer for probe in probes]
        if task_id in bool_tasks:
            if answers.count("true") != 32 or answers.count("false") != 32:
                raise ValueError(f"{task_id} evaluation answers must be 32 true and 32 false.")
        else:
            if len(set(answers)) != 64:
                raise ValueError(f"{task_id} evaluation answers must be 64 distinct values.")
        if task_id in {"MEMORY_FILTER", "FILTER_CONDITION", "SEARCH_CONDITION", "MEMORY_SEARCH"}:
            explicit = [probe for probe in probes[:32] if probe.style == "explicit"]
            indirect = [probe for probe in probes[32:] if probe.style == "indirect"]
            if len(explicit) != 32 or len(indirect) != 32:
                raise ValueError(f"{task_id} must allocate 32 explicit then 32 indirect probes.")
            if task_id in bool_tasks:
                for block in (probes[:32], probes[32:]):
                    block_answers = [probe.answer for probe in block]
                    if block_answers.count("true") != 16 or block_answers.count("false") != 16:
                        raise ValueError(f"{task_id} style block must be 16 true and 16 false.")
            elif task_id == "MEMORY_FILTER":
                if len({probe.answer for probe in probes[:32]}) != 32 or len({probe.answer for probe in probes[32:]}) != 32:
                    raise ValueError("MEMORY_FILTER must have 32 distinct answers per style block.")


def _parse_canonical_context(canonical_context: str) -> Mapping[str, Any]:
    context = json.loads(canonical_context)
    if not isinstance(context, Mapping):
        raise ValueError("canonical_context must encode a JSON object.")
    if _canonical_json(context) != canonical_context:
        raise ValueError("canonical_context must use canonical compact JSON serialization.")
    return context


def _representative_context(task_id: str) -> Mapping[str, Any]:
    return context_for_payload(task_id, _EVALUATION_SPLIT, 0)


def ground_truth_matrix() -> tuple[tuple[int, ...], ...]:
    matrix: list[tuple[int, ...]] = []
    for mask in STATE_MASKS:
        state = state_from_mask(mask)
        row: list[int] = []
        for task_id in TASK_ORDER:
            try:
                execute_task(task_id, _representative_context(task_id), state)
            except MissingCapabilityError:
                row.append(0)
            else:
                row.append(1)
        matrix.append(tuple(row))
    return tuple(matrix)


def ground_truth_knowledge_space() -> KnowledgeSpace:
    states = tuple(
        KnowledgeState(task_id for task_id, bit in zip(TASK_ORDER, row) if bit)
        for row in ground_truth_matrix()
    )
    return KnowledgeSpace(
        tasks=TaskUniverse(TASK_ORDER),
        valid_states=states,
        metadata={"phase": "8", "world": "toy_lm_bridge_ground_truth"},
    )


def _signatures_for_tasks(selected_tasks: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    indices = tuple(TASK_ORDER.index(task_id) for task_id in selected_tasks)
    matrix = ground_truth_matrix()
    return tuple(tuple(row[idx] for idx in indices) for row in matrix)


def _is_identifying(selected_tasks: Sequence[str]) -> bool:
    signatures = _signatures_for_tasks(selected_tasks)
    return len(set(signatures)) == len(signatures)


def enumerate_minimum_certificates() -> tuple[tuple[str, ...], ...]:
    for size in range(len(TASK_ORDER) + 1):
        found = tuple(candidate for candidate in combinations(TASK_ORDER, size) if _is_identifying(candidate))
        if found:
            return found
    raise AssertionError("full task universe should identify all states")


def audit_ground_truth_oracle() -> Mapping[str, Any]:
    matrix = ground_truth_matrix()
    if len(matrix) != 16 or any(len(row) != 8 for row in matrix):
        raise ValueError("Ground-truth matrix must be 16 x 8.")
    if len(set(matrix)) != 16:
        raise ValueError("Ground-truth matrix must be identifiable.")
    knowledge_space = ground_truth_knowledge_space()
    identifiability = check_identifiability(knowledge_space)
    if not identifiability.identifiable:
        raise ValueError("Accepted identifiability audit rejected the ground-truth matrix.")
    exact = solve_exact_certificate(knowledge_space)
    if not exact.valid:
        raise ValueError("Accepted exact solver did not produce a valid certificate.")
    if exact.certificate_size != 4 or tuple(exact.selected_tasks) != PRIMITIVE_ORDER:
        raise ValueError(
            "Accepted exact solver did not recover the canonical primitive certificate: "
            f"{exact.selected_tasks!r}"
        )
    if not validate_certificate(knowledge_space, PRIMITIVE_ORDER):
        raise ValueError("Accepted certificate validator rejected the primitive certificate.")
    minimum = enumerate_minimum_certificates()
    if minimum != (PRIMITIVE_ORDER,):
        raise ValueError(f"Unique size-four primitive certificate mismatch: {minimum!r}")
    for primitive in PRIMITIVE_ORDER:
        primitive_index = TASK_ORDER.index(primitive)
        singleton_mask = 1 << PRIMITIVE_ORDER.index(primitive)
        if matrix[0][primitive_index] != 0 or matrix[singleton_mask][primitive_index] != 1:
            raise ValueError(f"Missing empty/singleton witness for {primitive}.")
        for task_id in TASK_ORDER[4:]:
            if primitive not in required_primitives(task_id):
                continue
            task_index = TASK_ORDER.index(task_id)
            if matrix[singleton_mask][task_index] != 0:
                raise ValueError("Composite task appears in singleton missing-primitive witness.")
    return MappingProxyType(
        {
            "matrix": matrix,
            "identifiable": True,
            "accepted_identifiability": identifiability.to_dict(),
            "accepted_solver_selected_certificate": tuple(exact.selected_tasks),
            "accepted_solver_certificate_size": exact.certificate_size,
            "minimum_certificates": minimum,
            "certificate_size": 4,
            "selected_certificate": PRIMITIVE_ORDER,
        }
    )


def validate_memory_dependency(task_id: str, context: Mapping[str, Any]) -> tuple[str, str]:
    if task_id not in {"MEMORY_FILTER", "MEMORY_SEARCH"}:
        raise ValueError("Memory dependency validation applies only to memory sequences.")
    validate_context_schema(task_id, context)
    key = context["key"]
    original_value = context["memory"][key]
    present_value = original_value
    absent_value = "0" * 16
    if absent_value in context["items"] or absent_value == present_value:
        absent_value = "1" * 16
    present_context = {"memory": {key: present_value}, "key": key, "items": list(context["items"])}
    absent_context = {"memory": {key: absent_value}, "key": key, "items": list(context["items"])}
    present_answer = canonical_answer(task_id, present_context)
    absent_answer = canonical_answer(task_id, absent_context)
    if present_answer == absent_answer:
        raise ValueError(f"{task_id} does not depend on the memory result.")
    return present_answer, absent_answer


EVALUATION_PACK_CHECKSUM = evaluation_pack_checksum(_build_evaluation_probe_pack_unvalidated())
