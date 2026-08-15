# Task 010H — Resource Benchmark and Formal-Run Readiness

## Launch contract

```text
executor role: Spark
model override: gpt-5.5
fork_turns: none
working directory: /home/mye/src/llm/CapKnow
```

`main` supplies the exact accepted 010G prerequisite commit. Read master Sections
9.5 and 12–16. The executor prepares non-scientific resource evidence; only `main`
may accept ceilings and authorize formal execution.

## Objective

Implement and test the two fixed dummy-data benchmark fixtures and produce the
measurements and parameterized projection needed for `main` to decide whether the
formal eighteen-shard grid is operationally authorized.

## Allowed paths

```text
scripts/phase8_resource_benchmark.py
tests/test_lm_bridge.py
```

Accepted benchmark artifacts are generated later under the fixed external-evidence
root and are not an implementation edit lease.

## Required implementation

- Implement only `benchmark_small_base` and `benchmark_medium_large` with dummy data
  that contain no Phase 8 scientific task, template, operand, payload, state, or
  outcome.
- Exercise the representative formal shard shape: sixteen trained checkpoints, the
  same batch size and sequence-length distribution, 1500 optimizer steps each, five
  512-prompt evaluation points, checkpoint/status/raw publication, and manifest
  creation through the accepted runtime paths.
- Record training/evaluation wall time, CPU time, peak RSS, peak VRAM, checkpoint
  bytes, raw-generation bytes, final artifact bytes, and per-stage breakdown.
- Emit measured per-shard costs and a whole-grid projection whose concurrency and
  safety factor are explicit parameters. Do not silently fix or approve a `1.25`
  multiplier.
- Refuse overwrite; retain atomic status and complete manifest/checksum evidence at
  the output root supplied by `main`.
- Implement verification that a later `authorization.json` binds explicit ceilings
  for concurrency, wall time, GPU-hours, peak VRAM, peak RSS, temporary disk, and
  final disk. A missing field or projected/observed breach must block formal
  `prepare` as an operational failure.

## Acceptance checks

Add master Section 13 item 24 tests using reduced test-local fixture sizes and mocked
resource readings. Test missing metrics, projection arithmetic, explicit safety
factor, missing authorization fields, limit breaches, overwrite refusal, and the
absence of scientific markers.

Run:

```text
python -m pytest -q tests/test_lm_bridge.py
git diff --check
git status --short
```

After implementation is committed and independently reviewed, `main` may separately
launch the full accepted benchmark from a clean worktree. The executor must not run
the benchmark as accepted evidence during this implementation task.

## Stop boundary

Do not choose ceilings, write `authorization.json`, launch `formal_001`, change the
grid, reduce completeness requirements, or interpret scientific outcomes. Return the
benchmark interface and projected fields to `main`; only `main` decides formal-run
authorization. Do not commit unless explicitly authorized. Return using the common
format.
