# Task 010H — Resource Benchmark and Formal-Run Readiness

## Launch contract

```text
executor: fresh-context Codex subagent
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
- Map small/base directly to the nine core shards and medium/large directly to its
  three scale shards. Bound both unmeasured cross cells by the componentwise maximum
  of the two fixtures; do not interpolate. Use sixteen model slots, respectively
  896/3584 dummy records per slot, encoded 256-token training sequences and encoded
  192-token evaluation prefixes including special tokens plus 64-token generation,
  batch size 64, 1500 steps, five 512-prompt evaluation points, and the accepted
  checkpoint/status/raw/manifest paths.
- Record training/evaluation wall time, CPU time, peak RSS, peak VRAM, checkpoint
  bytes, raw-generation bytes, peak artifact bytes, final artifact bytes, and
  per-stage breakdown in the exact units of master Section 9.5.
- Emit measured costs and the exact Section 9.5 componentwise-maximum projection for
  `N=18`, including waves, stage/total wall seconds, CPU/GPU hours, checkpoint/raw
  bytes, combined concurrent RSS/VRAM, temporary disk, and final disk. Parse the
  canonical decimal safety factor with `decimal.Decimal`, serialize ceilings as
  integers and hour projections as decimal strings. Concurrency and safety factor
  `>=1` are explicit inputs; do not interpolate cells or silently approve `1.25`.
- Refuse overwrite; retain atomic status and complete manifest/checksum evidence at
  the output root supplied by `main`. The accepted benchmark manifest must bind the
  exact source commit intended for formal execution.
- Require no tracked/staged change and allow untracked source-cleanliness exclusions
  only for exact finalized feasibility roots/selection records supplied as
  checksum-bound inputs; reject broad artifact-directory exclusions.
- Implement verification that a later `authorization.json` binds explicit ceilings
  and units/scopes for concurrency, wall time, GPU-hours, peak VRAM on `cuda:0`, total
  peak RSS, root temporary disk, and root final disk; both benchmark manifest/summary
  checksums and measurements; exact source commit; fixture configurations; formula
  version; shard-registry checksum; projection inputs/outputs; accepted safety
  factor; and selected feasibility path/checksum. A missing/mismatched field or
  projected/observed breach must block formal `prepare` as an operational failure.
  Later numbered benchmarks must also bind every predecessor root/manifest/terminal
  checksum so preserved failures remain an exact allowed evidence lineage.

## Acceptance checks

Add master Section 13 item 24 tests using reduced test-local fixture sizes and mocked
resource readings. Test missing metrics, projection arithmetic, explicit safety
factor and decimal serialization, fixture/cell mapping, source/evidence/checksum
mismatches, units/scopes, missing authorization fields, limit breaches, overwrite
refusal, and the absence of scientific markers.

Run:

```text
python -m pytest -q tests/test_lm_bridge.py
git diff --check
git status --short
```

After implementation is committed and independently reviewed, `main` may separately
obtain the selected feasibility pass from that exact commit and then launch the full
accepted benchmark from the same clean commit. The executor must not run either as
accepted evidence during this implementation task.

## Stop boundary

Do not choose ceilings, write `authorization.json`, launch `formal_001`, change the
grid, reduce completeness requirements, or interpret scientific outcomes. Return the
benchmark interface and projected fields to `main`; only `main` decides formal-run
authorization. Do not commit unless explicitly authorized. Return using the common
format.
