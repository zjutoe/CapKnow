# CapKnow Agent Guidelines

## Core Principles

- KISS and YAGNI: prefer minimal, explicit implementations; avoid over-engineering and defensive programming unless required by explicit protocol.

## Experiment Execution

- Run experiments in the risk order frozen by `main`, using narrow parameterized scripts under `scripts/`.
- Bind the launcher path and commit or hash, config hash, and fixed output roots in the operator handoff.
- Do not continuously read stdout. Send long-task output to job logs and inspect small `status.json`, `progress.json`, `DONE.json`, `FAILED.json`, and experiment `summary.json` files at bounded intervals.
- Preserve manifests, config hashes, fixed splits, seed provenance, and artifact checksums.
- Return failed or scientifically suspicious runs to `main`; `operator` must not patch code or silently alter protocol.

## Research Review and Acceptance

- High-risk changes include simulators, observation semantics, task distributions, train/calibration/evaluation splits, metrics, statistical aggregation, acceptance logic, and result interpretation.
- Any high-risk change must trigger an independent review agent using the same model family as the main thread, with no shared context between the review agent and main thread.
- Before committing non-trivial changes, spawn an independent strict read-only review agent by default with `fork_context=false`.
- The review handoff should include only: intended diff, relevant artifacts, acceptance criteria, protocol constraints, and verification already run.
- For each high-risk change handoff, include: concise change summary, reproducible command(s), key touched files, verification outputs, and bound hashes/artifact IDs.
- Review protocol/implementation consistency, leakage, objective/inference consistency, statistical independence, clustered aggregation, metric semantics, acceptance logic, artifact provenance, and theoretical overclaiming.
- Bind exact artifact hashes before reviewing or accepting experiment outputs that may later support a scientific claim.
- `main` accepts or rejects each frozen implementation or research package only after independent agent review.
- After code changes, run relevant targeted tests first; escalate to repository-wide tests and static checks based on risk, failures, or scope changes.
- Never use ignored artifacts from superseded or methodologically invalid runs in final conclusions.

## Verification and Hash Policy

- Verification must be proportional to risk. Prefer targeted tests and direct semantic checks for ordinary code changes.
- Use hash binding for experiment inputs, frozen data splits, seeds, configs, launchers, accepted artifacts, and outputs that must be exactly reproducible.
- Do not make source-package hashes a routine trigger for full revalidation after comments, formatting, EOF whitespace, or other non-semantic edits. If a live gate currently binds exact source bytes, either avoid non-semantic edits inside the bound package or update only the minimal affected binding with a clear note that no scientific semantics changed.
- Hashes are evidence of byte identity, not evidence of scientific validity. They should support reproducibility and tamper detection, not replace review, tests, or judgment.
- A mechanical cleanup should not cascade into benchmark permission changes, scientific reinterpretation, or SOL audit unless it changes the milestone boundary or the accepted scientific evidence package.
