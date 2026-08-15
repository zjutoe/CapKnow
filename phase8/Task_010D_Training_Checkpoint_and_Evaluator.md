# Task 010D — Training, Checkpoints, and Isolated Evaluator

## Launch contract

```text
executor role: Spark
model override: gpt-5.5
fork_turns: none
working directory: /home/mye/src/llm/CapKnow
```

`main` supplies the exact accepted 010C prerequisite commit and proof that the
accepted feasibility run passed. Read master Sections 6.2, 8.3, 9.1–9.2, 10.3, 12–15.

## Objective

Implement the formal per-state training/checkpoint path and an evaluation path that
scores only the immutable shared probe pack. Keep these components independent of
certificate construction and experiment orchestration.

## Allowed paths

```text
capability_certificate_lab/lm_bridge/train.py
capability_certificate_lab/lm_bridge/evaluator.py
tests/test_lm_bridge.py
```

Change model or corpus code only by stopping and returning a prerequisite defect to
`main`; do not expand the lease.

## Required implementation

- Implement the exact 1500-step, batch-64 cycling schedule, corpus-order RNG, paired
  initialization RNG, deterministic backend requirements, and failure behavior.
- Evaluate after steps `0,100,300,750,1500`; retain training loss and every raw
  generation; save exact weights only at steps 0 and 1500.
- Verify save/load tensor identity and logits before accepting a checkpoint path.
- Require every run to reference the one 512-prompt evaluation-pack checksum and
  reject per-condition or per-state probe rendering.
- Score exact canonical success and exact `unable` separately. Refusal never counts
  as success for a positive cell, and an answer never counts as correct refusal for a
  negative cell.
- Record all 64 outputs per state/task cell and all fields needed for later
  sensitivity matrices and length diagnostics. Do not threshold or build
  certificates in this task.
- Keep callable training and evaluation primitives parameterized enough for the
  later fixed shard runner, but add no generic framework or registry hierarchy.

## Acceptance checks

Add targeted tests for master Section 13 items 13–16, shared-pack checksum isolation,
the exact checkpoint schedule, response-only labels, and raw-output completeness.

Run:

```text
python -m pytest -q tests/test_lm_bridge.py
git diff --check
git status --short
```

Do not launch a 16-state population or any formal scientific training.

## Stop boundary

Do not implement response matrices, certificates, aggregate metrics, shard/status
orchestration, benchmark projections, or reports. Do not commit unless explicitly
authorized. Return using the common format.
