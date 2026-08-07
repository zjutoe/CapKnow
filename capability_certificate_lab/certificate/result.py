from __future__ import annotations
from dataclasses import dataclass

from .decision_tree import DecisionNode

@dataclass(frozen=True)
class CertificateResult:
    task_count: int
    state_count: int
    certificate_size: int
    selected_tasks: list[str]
    valid: bool
    separated_pairs: int
    total_pairs: int
    runtime_ms: float
    method: str = "exact"

    def to_dict(self) -> dict[str, object]:
        return {
            "task_count": self.task_count,
            "state_count": self.state_count,
            "certificate_size": self.certificate_size,
            "selected_tasks": self.selected_tasks,
            "valid": self.valid,
            "separated_pairs": self.separated_pairs,
            "total_pairs": self.total_pairs,
            "runtime_ms": self.runtime_ms,
            "method": self.method,
        }


@dataclass(frozen=True)
class AdaptiveCertificate:
    root: DecisionNode
    worst_case_depth: int
    average_depth: float
    node_count: int
    valid: bool
    method: str = "adaptive"
    policy: str | None = None
    seed: int | None = None
    runtime_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "worst_case_depth": self.worst_case_depth,
            "average_depth": self.average_depth,
            "node_count": self.node_count,
            "valid": self.valid,
            "method": self.method,
            "policy": self.policy,
            "seed": self.seed,
            "runtime_ms": self.runtime_ms,
        }
