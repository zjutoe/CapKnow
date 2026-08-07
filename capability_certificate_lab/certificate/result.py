from __future__ import annotations

from dataclasses import dataclass


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
