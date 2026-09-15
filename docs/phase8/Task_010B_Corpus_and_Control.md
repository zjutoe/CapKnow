# Task 010B — Corpus Conditions and Randomized Control

## Launch contract

```text
executor: fresh-context Codex subagent
model override: gpt-5.5
fork_turns: none
working directory: /home/mye/src/llm/CapKnow
```

`main` must supply the exact accepted 010A prerequisite commit plus the master and
this handoff. Verify that commit before editing. Read master Sections 6–7, 8.3 corpus
counts, 10.3 target diagnostics, 13, and 15.

## Objective

Implement deterministic A/B/C corpus construction and every corpus-side validity
gate. Extend the accepted 010A symbolic layer without changing its frozen order,
oracle, split semantics, or evaluation pack.

## Allowed paths

```text
capability_certificate_lab/lm_bridge/corpus_generator.py
tests/test_lm_bridge.py
```

No other path may change.

## Required implementation

- Generate exactly 128 records per trained task family for base corpora and 512 per
  family for large corpora, across the four primitive and three seen-composition
  families.
- Implement Condition A explicit-stepwise records and Condition B indirect end-to-end
  records with the exact shared oracle outcomes and byte-identical primitive records.
- Exclude `MEMORY_SEARCH`, its program dictionary, and all assigned templates from
  every training corpus.
- Implement Condition C from the exact A prompts using the single declared RNG and
  degree-preserving `2 x 2` switches, including accepted-switch and proposal limits.
- Gate primitive byte identity, per-state and per-record degrees, aggregate UTF-8 byte
  and tokenizer-token histograms, family changes, the `15%` change floor, and
  determinism.
- Gate all train/evaluation ID, text, template, canonical-context, and normalized
  payload separation contracts inherited from 010A.
- Produce deterministic record ordering inputs and the A/C response-target length
  diagnostic inputs required by master Section 10.3. Do not alter corpora to make
  those diagnostics match.

## Acceptance checks

Add targeted tests for master Section 13 items 4, 7, and 9–11, including deliberate
malformed controls, insufficient randomization, held-out leakage, and split
collisions.

Run:

```text
python -m pytest -q tests/test_lm_bridge.py
git diff --check
git status --short
```

## Stop boundary

Do not implement the tokenizer, model, training loop, evaluator, certificate
analysis, shard runner, or formal artifacts. Do not change scientific constants when
a generator gate fails; return the exact failure to `main`. Do not commit unless
explicitly authorized. Return using the common format.
