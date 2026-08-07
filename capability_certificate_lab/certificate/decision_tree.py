from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DecisionNode:
    question: str | None = None
    yes_child: "DecisionNode | None" = None
    no_child: "DecisionNode | None" = None
    candidate_state_ids: list[str] | None = None

    def is_leaf(self) -> bool:
        return self.question is None

    def to_dict(self) -> dict[str, Any]:
        if self.question is None:
            return {
                "question": None,
                "candidate_state_ids": sorted(self.candidate_state_ids or ()),
            }
        return {
            "question": self.question,
            "yes_child": self.yes_child.to_dict() if self.yes_child else None,
            "no_child": self.no_child.to_dict() if self.no_child else None,
        }


def tree_signature(node: DecisionNode | None) -> str:
    if node is None:
        return "#"
    if node.question is None:
        payload = ",".join(sorted(node.candidate_state_ids or ()))
        return f"L[{payload}]"
    return f"Q:{node.question}|Y:{tree_signature(node.yes_child)}|N:{tree_signature(node.no_child)}"
