from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import revalidation_common


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_source_binding_allows_only_fixed_artifact_outputs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "source.txt").write_text("source\n", encoding="utf-8")
    _git(repo, "add", "source.txt")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-q",
        "-m",
        "source",
    )

    artifact_dir = repo / "artifacts" / "phase2_6_revalidation"
    monkeypatch.setattr(revalidation_common, "ROOT", repo)
    monkeypatch.setattr(revalidation_common, "ARTIFACT_DIR", artifact_dir)

    source = revalidation_common.source_binding_before_run()
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "phase2.json").write_text("{}\n", encoding="utf-8")
    revalidation_common._verify_source_binding(source)
    assert revalidation_common.source_binding_before_run()["git_head"] == source["git_head"]

    (repo / "source.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Versioned source changed"):
        revalidation_common._verify_source_binding(source)
