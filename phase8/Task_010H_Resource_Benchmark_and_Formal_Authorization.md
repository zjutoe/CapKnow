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
  batch size 64, 1500 steps, and five 512-prompt evaluation points.
- Exercise the exact accepted writers with role-for-role dummy equivalents for every
  formal artifact class: corpus/split/evaluation packs, checkpoints, logs/status/
  progress/terminal files, generations, primary/sensitivity matrices, collision/
  certificate/tree results, summaries, attempt manifest, concurrency lock, run
  contract, eighteen-entry shards manifest, aggregate outputs, and top manifest.
  Use exact formal cardinalities and maximum protocol-permitted serialized lengths.
  Reject an omitted, zero-assumed, or unbounded formal role, including logs.
- Record training/evaluation wall time; shard and root-overhead wall/CPU/GPU-device
  time; peak RSS/VRAM; checkpoint and raw-generation bytes; peak/final shard bytes;
  peak/final root-overhead bytes; and per-stage breakdown in the exact units of
  master Section 9.5.
- Emit measured costs and the exact Section 9.5 componentwise-maximum projection for
  `N=18` and frozen `concurrency=1`, including stage/total wall seconds, CPU/GPU
  hours, checkpoint/raw bytes, single-process RSS/VRAM, all shard/root artifact roles,
  temporary disk, and final disk. Parse the canonical decimal safety factor with
  `decimal.Decimal`, serialize ceilings as integers and hour projections as decimal
  strings. Safety factor `>=1` is explicit; do not interpolate cells or silently
  approve `1.25`.
- Run fixtures sequentially under the same one-process `cuda:0` policy as formal
  execution and record other-process/device-utilization checks in each manifest.
- Refuse overwrite; retain atomic status and complete manifest/checksum evidence at
  the output root supplied by `main`. The accepted benchmark manifest must bind the
  exact source commit intended for formal execution.
- Require no tracked/staged change and allow untracked source-cleanliness exclusions
  only for exact finalized feasibility roots/selection records and predecessor
  benchmark roots supplied as checksum-bound inputs; reject broad artifact-directory
  exclusions.
- Implement verification that a later `authorization.json` binds fixed
  `concurrency=1` and explicit ceilings/units/scopes for wall time, GPU-hours, peak
  VRAM on `cuda:0`, total peak RSS, root temporary disk, and root final disk; both
  benchmark manifest/summary checksums and measurements; exact source commit; fixture
  configurations; formula version; shard-registry checksum; fixed serial concurrency;
  projection inputs/outputs; accepted safety factor; selected feasibility path and
  checksum; and its transitive predecessor-root/selection-record lineage. A
  missing/mismatched field or
  projected/observed breach must block formal `prepare` as an operational failure.
  Later numbered benchmarks must also bind every predecessor root/manifest/terminal
  checksum so preserved failures remain an exact allowed evidence lineage.

## Acceptance checks

Add master Section 13 item 24 tests using reduced test-local fixture sizes and mocked
resource readings. Test missing metrics, projection arithmetic, explicit safety
factor and decimal serialization, fixture/cell and complete artifact-role coverage,
source/evidence/checksum mismatches, fixed concurrency, units/scopes, missing
authorization fields, limit breaches, overwrite refusal, and the absence of
scientific markers.

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
