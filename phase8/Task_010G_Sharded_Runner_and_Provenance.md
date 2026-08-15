# Task 010G — Sharded Runner and Provenance

## Launch contract

```text
executor role: Spark
model override: gpt-5.5
fork_turns: none
working directory: /home/mye/src/llm/CapKnow
```

`main` supplies the exact accepted 010F prerequisite commit. Read master Sections
9.1–9.2, 12–16, and the repository provenance rules. This task implements the
execution boundary but is not authorized to start a formal run.

## Objective

Implement the fixed eighteen-shard formal command surface, atomic attempt state,
strict aggregation, and complete provenance binding for the 288-run grid.

## Allowed paths

```text
scripts/phase8_toy_lm_bridge.py
tests/test_lm_bridge.py
```

Do not modify scientific modules to accommodate orchestration. Return prerequisite
defects to `main`.

## Required implementation

- Expose only `prepare`, `run-shard`, and `aggregate` with the master handoff's exact
  arguments and fixed eighteen shard IDs.
- Make `prepare` require a clean committed source and accepted feasibility/resource
  inputs, refuse an existing root, and atomically bind source commit, configuration,
  device/environment, shared evaluation pack, shard registry, resource authorization,
  and output root.
- Make each shard contain exactly one condition/model/corpus/seed cell, all sixteen
  state runs, and all five checkpoint evaluations. Reuse
  `core_A_small_base_seed*` for the small/base scale cell.
- Publish logs plus atomic `status.json`, `progress.json`, and exactly one terminal
  `DONE.json` or `FAILED.json` per attempt. Never overwrite an attempt.
- Permit a later attempt only for an identical source/configuration binding and an
  explicit authorization record supplied by `main`. Preserve failed attempts.
- Make `aggregate` require exactly one declared successful attempt for all eighteen
  registered shards, reject extras/missing/failures and any binding mismatch, and
  reconstruct all aggregate results from shard raw records.
- Inventory every output with role, byte size, and SHA-256 while excluding only the
  self-hashing manifest. Bind corpora, split/evaluation packs, weights, generations,
  matrices, trees, results, and logs as required by the master protocol.
- Distinguish transient attempt retry from source/configuration/data/evaluation or
  semantic invalidation. The latter must instruct `main` to preserve the root and use
  a new top-level root after repair; the runner must never mix roots or bindings.

## Acceptance checks

Use test-local temporary roots and stubbed training callables; tests must not launch
formal training. Add master Section 13 item 23 coverage for clean-tree gating,
overwrite refusal, atomic transitions, interruption/failure, explicit retry
authorization, all eighteen IDs, shared-A accounting, mixed bindings, inventory, and
aggregation completeness.

Run:

```text
python -m pytest -q tests/test_lm_bridge.py
git diff --check
git status --short
```

## Stop boundary

Do not run `prepare`, a shard, or aggregation against `formal_001`. Do not create or
authorize a resource ceiling, repair scientific code, edit `.gitignore`, or commit
unless explicitly authorized. Return using the common format.
