# CapKnow Agent Guidelines

## Core Principles

- KISS and YAGNI: prefer minimal, explicit implementations; avoid over-engineering and defensive programming unless required by explicit protocol.

## Version Control and Provenance

- Git is the source of truth for versioned source, documentation, tests, scripts, and configurations. Do not maintain parallel source-package hashes.
- Create one branch per milestone from a clean `main`. Commit each task stage, review repair, and accepted report on that branch.
- Freeze reviews to an exact commit SHA or commit range, never a branch name. Do not rewrite reviewed history; record fixes as new commits.
- Run formal experiments only from a clean, committed worktree. Record the exact source commit in every experiment manifest.
- After final acceptance, merge the milestone branch into `main` while preserving its review and repair history.
- Use checksums only for evidence outside Git, such as external datasets, uncommitted fixed splits, model weights, and experiment artifacts. A checksum proves byte identity, not validity.
- Any untracked or ignored input that can affect an experiment must be committed or explicitly bound in its manifest by path and checksum.

## Experiment Execution

- Run experiments in the risk order frozen by `main`, using narrow parameterized scripts under `scripts/`.
- Bind the exact source commit, launcher path, configuration, and fixed output roots in the operator handoff.
- Do not continuously read stdout. Send long-task output to job logs and inspect small `status.json`, `progress.json`, `DONE.json`, `FAILED.json`, and experiment `summary.json` files at bounded intervals.
- Preserve manifests, fixed splits, seed provenance, and checksums for evidence outside Git.
- Return failed or scientifically suspicious runs to `main`; `operator` must not patch code or silently alter protocol.

## Agent Orchestration and Model Routing

- Use `gpt-5.6-sol` with `reasoning_effort=medium` for task execution, including implementation, debugging, repairs, experiment operation, and documentation work.
- Use a fresh-context `gpt-5.6-sol` agent with `reasoning_effort=high` for every independent review. Bind the review to an exact commit or commit range and keep the reviewer strictly read-only.
- Start delegated executors and reviewers with `fork_turns=none` and an exact handoff. Do not inherit the main conversation as an implicit task specification.
- Reserve `gpt-5.6-sol` with `reasoning_effort=xhigh` only for top-level design work such as architecture design. Do not use xhigh for ordinary implementation, diagnosis, repair, experiment execution, or independent review.
- When a request reaches that top-level design boundary, stop before performing the design and ask the user to switch the current session manually to `gpt-5.6-sol` with `reasoning_effort=xhigh`. Do not prompt for a manual model switch in other cases.

## Research Review and Acceptance

- High-risk changes include simulators, observation semantics, task distributions, train/calibration/evaluation splits, metrics, statistical aggregation, acceptance logic, and result interpretation.
- Any high-risk change must trigger an independent `gpt-5.6-sol` review agent with `reasoning_effort=high` and no shared context with the main thread.
- Before accepting non-trivial changes, freeze them in a commit and spawn an independent strict read-only review agent with `fork_turns=none`.
- The review handoff should include only: intended diff, relevant artifacts, acceptance criteria, protocol constraints, and verification already run.
- For each high-risk change handoff, include: exact commit or range, concise change summary, reproducible commands, key touched files, verification outputs, and relevant artifact checksums.
- Review protocol/implementation consistency, leakage, objective/inference consistency, statistical independence, clustered aggregation, metric semantics, acceptance logic, artifact provenance, and theoretical overclaiming.
- Bind exact artifact checksums before reviewing or accepting experiment outputs that may later support a scientific claim.
- `main` accepts or rejects each frozen implementation or research package only after independent agent review.
- After code changes, run relevant targeted tests first; escalate to repository-wide tests and static checks based on risk, failures, or scope changes.
- Never use ignored artifacts from superseded or methodologically invalid runs in final conclusions.

## Verification and Checksum Policy

- Verification must be proportional to risk. Prefer targeted tests and direct semantic checks for ordinary code changes.
- Use the Git commit to bind versioned inputs. Use checksums only where Git does not bind the evidence.
- Do not trigger scientific revalidation for comments, formatting, EOF whitespace, or other non-semantic edits. Revalidate when experiment semantics or accepted evidence change.
- Checksums support reproducibility and tamper detection; they never replace review, tests, or scientific judgment.
- A mechanical cleanup should not cascade into benchmark permission changes, scientific reinterpretation, or SOL audit unless it changes the milestone boundary or the accepted scientific evidence package.
