from __future__ import annotations

from collections.abc import Sequence

from ..knowledge_space.space import KnowledgeSpace
from ..knowledge_space.state import KnowledgeState
from ..knowledge_space.tasks import TaskUniverse


def _normalize_blocks(blocks: Sequence[Sequence[str]]) -> list[list[str]]:
    normalized = [[str(task_id) for task_id in block] for block in blocks]
    if not normalized:
        raise ValueError("Block collection must not be empty.")

    seen: set[str] = set()
    for block_index, block in enumerate(normalized):
        if not block:
            raise ValueError(f"Block {block_index} must not be empty.")
        for task_id in block:
            if task_id in seen:
                raise ValueError(f"Duplicate task id: {task_id}")
            seen.add(task_id)
    return normalized


def _block_prerequisites(blocks: Sequence[Sequence[str]], *, prefix: bool) -> dict[str, list[str]]:
    prerequisites: dict[str, list[str]] = {}
    earlier_tasks: list[str] = []
    for block in blocks:
        block_tasks = list(block)
        for task_id in block_tasks:
            required = [other for other in block_tasks if other != task_id]
            if prefix:
                required = [*required, *earlier_tasks]
            if required:
                prerequisites[task_id] = required
        earlier_tasks.extend(block_tasks)
    return prerequisites


def _task_ids(blocks: Sequence[Sequence[str]]) -> list[str]:
    return [task_id for block in blocks for task_id in block]


def _block_metadata(blocks: Sequence[Sequence[str]], family: str) -> dict[str, object]:
    return {
        "family": family,
        "blocks": [list(block) for block in blocks],
        "block_count": len(blocks),
        "block_sizes": [len(block) for block in blocks],
    }


def generate_independent_block_world(blocks: Sequence[Sequence[str]]) -> KnowledgeSpace:
    """Generate all unions of complete, independently toggled task blocks."""

    normalized = _normalize_blocks(blocks)
    task_ids = _task_ids(normalized)
    valid_states: list[KnowledgeState] = []

    for mask in range(1 << len(normalized)):
        selected: list[str] = []
        for block_index, block in enumerate(normalized):
            if (mask >> block_index) & 1:
                selected.extend(block)
        valid_states.append(KnowledgeState(selected))

    return KnowledgeSpace(
        tasks=TaskUniverse(task_ids),
        valid_states=valid_states,
        generator_rules={
            "type": "independent_block",
            "prerequisites": _block_prerequisites(normalized, prefix=False),
        },
        metadata=_block_metadata(normalized, "independent_block"),
    )


def generate_prefix_block_world(blocks: Sequence[Sequence[str]]) -> KnowledgeSpace:
    """Generate the empty state and all successive ordered block prefixes."""

    normalized = _normalize_blocks(blocks)
    task_ids = _task_ids(normalized)
    valid_states: list[KnowledgeState] = []

    prefix_tasks: list[str] = []
    valid_states.append(KnowledgeState(prefix_tasks))
    for block in normalized:
        prefix_tasks = [*prefix_tasks, *block]
        valid_states.append(KnowledgeState(prefix_tasks))

    return KnowledgeSpace(
        tasks=TaskUniverse(task_ids),
        valid_states=valid_states,
        generator_rules={
            "type": "prefix_block",
            "prerequisites": _block_prerequisites(normalized, prefix=True),
        },
        metadata=_block_metadata(normalized, "prefix_block"),
    )
