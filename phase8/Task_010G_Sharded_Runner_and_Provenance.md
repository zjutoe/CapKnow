# Task 010G — Sharded Runner and Provenance

## Launch contract

```text
executor: fresh-context Codex subagent
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
  arguments, explicit feasibility-selection/resource-authorization inputs, and fixed
  eighteen shard IDs.
- Make `prepare` require a clean committed source and accepted feasibility/resource
  inputs, refuse an existing root, and atomically bind source commit, configuration,
  device/environment, shared evaluation pack, shard registry, exact selected
  feasibility manifest, benchmark manifest/summary, resource authorization,
  projection/ceilings, and output root in immutable `run_contract.json`. Both
  non-scientific evidence manifests must bind the exact current source commit. Freeze
  concurrency to one, create the persistent formal `concurrency.lock` first, and
  publish the run contract last as the readiness marker; bind the lock checksum and
  refuse `run-shard` before readiness.
- Reject every tracked/staged change and every untracked path except the exact
  selected feasibility root/record, accepted benchmark root, and finalized
  predecessor roots and selection records enumerated and checksum-bound by those
  inputs. After preparation, allow only those paths plus the exact formal root; never
  exclude an artifact directory broadly.
- Make each shard contain exactly one condition/model/corpus/seed cell, all sixteen
  state runs, and all five checkpoint evaluations. Reuse
  `core_A_small_base_seed*` for the small/base scale cell.
- Before creating an attempt, take a non-blocking exclusive `fcntl.flock` on the
  formal lock file and hold it through manifest/terminal publication. Refuse a
  concurrent process without allocating an attempt or writing shard evidence. Treat
  an unterminated attempt after process/host death as root-blocking, not retryable
  evidence.
- Publish logs plus atomic `status.json` and `progress.json`. Atomically finalize an
  immutable attempt-local `manifest.json` with the run-contract checksum, every
  source/configuration/evidence/environment binding, output inventory/checksum, and
  failure classification before publishing exactly one terminal `DONE.json` or
  `FAILED.json` that binds the manifest checksum. Never overwrite an attempt.
- Permit a later attempt only for an identical source/configuration binding and an
  immutable retry-authorization record supplied by `main` that binds the failed
  manifest and unchanged run contract. It must recompute cumulative wall/GPU/disk
  use and remaining work including retained failed bytes under the accepted formula
  and prove every original ceiling still holds. Preserve failed attempts.
- Make `aggregate` require exactly one declared successful attempt for all eighteen
  registered shards, permit retained registered failed histories, reject unregistered
  shards, missing or multiple successes and any binding/checksum mismatch, and
  reconstruct all aggregate results from selected shard raw records. Write the full
  attempt/selection history to `shards_manifest.json` before the final manifest.
- Inventory every output with role, byte size, and SHA-256 while excluding only the
  self-hashing manifest. Bind corpora, split/evaluation packs, weights, generations,
  matrices, trees, results, and logs as required by the master protocol.
- Distinguish transient attempt retry from source/configuration/data/evaluation or
  semantic invalidation. The latter must instruct `main` to preserve the root and use
  a new top-level root after repair; a transient terminal failure stays in the same
  root with the same evidence inputs and next attempt number. The runner must never
  mix roots or bindings.

## Acceptance checks

Use test-local temporary roots and stubbed training callables; tests must not launch
formal training. Add master Section 13 item 23 coverage for clean-tree gating,
overwrite refusal, run-contract contents, attempt-manifest-before-terminal ordering,
exclusive-lock contention and crash release, interruption/failure, incomplete-attempt
blocking, retry resource reauthorization, retained failed attempts, all eighteen IDs,
shared-A accounting, selected-success uniqueness, mixed bindings, inventory/checksums,
resource enforcement, and aggregation completeness.

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
