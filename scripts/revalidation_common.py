from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "phase2_6_revalidation"
BASELINE_COMMIT = "1648c8f201936e56f7eb0544b39c45cf0f431c9b"
HANDOFFS = (
    "Capability_Certificate_Task_Handoff_007_Correctness_Repair_and_Revalidation.md",
    "Capability_Certificate_Task_Handoff_008_Second_Review_Repair.md",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _source_status() -> str:
    artifact_path = ARTIFACT_DIR.relative_to(ROOT).as_posix()
    return _git(
        [
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            ".",
            f":(exclude){artifact_path}/**",
        ]
    )


def source_binding_before_run() -> dict[str, Any]:
    status = _source_status()
    if status:
        raise RuntimeError(
            "Formal revalidation requires a clean committed source before computation. "
            f"Dirty status:\n{status}"
        )

    return {
        "baseline_commit": BASELINE_COMMIT,
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_status_porcelain_before_run": status,
        "clean_worktree_before_run": True,
        "source_binding": "git_commit",
        "tracked_source_note": "Git commit is the source identity; checksums bind outputs only.",
        "handoff_paths": HANDOFFS,
    }


def _verify_source_binding(source: dict[str, Any]) -> None:
    current_head = _git(["rev-parse", "HEAD"])
    if current_head != source["git_head"]:
        raise RuntimeError("Source commit changed during formal revalidation.")

    status = _source_status()
    if status:
        raise RuntimeError(
            "Versioned source changed during formal revalidation. "
            f"Dirty status:\n{status}"
        )


def write_result_and_manifest(
    *,
    phase: str,
    script_name: str,
    command: str,
    config: dict[str, Any],
    result: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, str]:
    _verify_source_binding(source)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = ARTIFACT_DIR / f"{phase}.json"
    result_path.write_bytes(canonical_json_bytes(result))
    result_hash = file_sha256(result_path)

    manifest = {
        "phase": phase,
        "script": script_name,
        "command": command,
        "python_version": platform.python_version(),
        "config": config,
        "source": source,
        "outputs": {
            result_path.relative_to(ROOT).as_posix(): result_hash,
        },
    }
    manifest_path = ARTIFACT_DIR / f"manifest_{phase}.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    manifest_hash = file_sha256(manifest_path)
    return {
        "result_path": result_path.relative_to(ROOT).as_posix(),
        "result_sha256": result_hash,
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": manifest_hash,
    }
