# Future-Only Postmortem Process-Isolation Simplification Design

## 1. Status and authority

```text
status: future-only process design
current D3 effect: none
implementation authority: none
execution authority: none
experiment authority: none
training authority: none
retry authority: none
selection authority: none
Task 010D authority: false
artifact mutation authority: none
```

This design is not current Phase 8 scientific authority. Accepted
`feasibility_005` remains `FAILED` with no selection, `feasibility_006`, retry, or
Task 010D authority. This document grants no implementation, execution, experiment,
training, retry, selection, artifact-mutation, or Task 010D authority.

```text
Shared protocol path: phase8/README.md
Shared protocol commit: 0aff481d49010c0a84bfaa65dd89de5d73d5f45c
Inherited sections: Sequence and mutation lease, Common return format
Task-specific overrides: Sequence and mutation lease — work is limited to the two documentation paths in this design and grants no implementation or execution authority; Common return format — New artifacts means no experiment or evidence artifacts.
```

## 2. Historical D3 non-retroactivity

D3 is DONE only as a frozen, non-evidence postmortem. It remains bound to accepted
implementation commit `e3f99fcabbce812408793b62a1cdad536b8b303b` and its
independently reviewed direct-`execve` verifier/supervisor trust root. Its proposal,
implementation, verifier, schemas, artifacts, hashes, interpretation limits, and
archive-in-place status remain byte-for-byte historical records. This future design
does not rebind, replace, repair, rename, or reinterpret them and cannot change the
`feasibility_005` verdict or create selection, retry, `feasibility_006`, or Task 010D
authority.

Any separately authorized future implementation must likewise bind its exact
accepted implementation commit and tree. This design intentionally chooses no such
commit.

## 3. Future trust boundary

The independently reviewed, direct-`execve` verifier/supervisor parent is the trust
root. It alone retains repository, Git, launch, process, input, output, validation,
and publication authority. Before the fork it verifies the accepted commit and
captures all runner and module bytes from that commit, never from mutable worktree
imports.

The worker is an untrusted domain-computation child. Repository and worktree paths
never enter its import resolution. Its closed authenticated source loader may
execute only the inherited captured bytes and must reject every uncaptured module.
The supervisor owns fixed input/output bindings, non-evidence classification, and
no-overwrite publication. The domain runner receives no authority token, repository
callback, loader object through its globals, or other Python capability object;
the child bootstrap installs the closed loader before executing the runner.

## 4. Minimal supervisor/worker architecture

There is exactly one supervisor parent and one worker child:

```text
main launch authorization
  -> direct execve external verifier/supervisor parent
  -> parent verifies launch and accepted commit, captures accepted sources, retains repository/process authority
  -> parent forks once
  -> worker closes authority/Git descriptors, installs closed loader over inherited captured bytes, runs domain computation, writes fixed temp root
  -> parent checks worker status and fixed output schema, then alone performs one atomic no-replace promotion
```

Fork/COW separation replaces Python object- and code-graph sealing. The worker must
close inherited authority and Git descriptors before computation and cannot create
another authority-bearing process. Only the parent validates terminal state,
schema, and required hashes and promotes the fixed temp root. An abnormal worker
exit or validation failure leaves a clearly non-terminal temp root and stops; it
does not create quarantine trees or invoke an automatic recovery protocol.

This docs task chooses no concrete supervisor path, source, or launch command. A
future accepted implementation handoff must bind all three exactly.

## 5. Runner simplification requirements

A future implementation must not inherit D3's duplicate defenses:

- Do not security-bind versioned code to both byte count and digest. Bind the exact
  Git commit/tree, use one digest for the verifier trust root, and identify each
  captured inventory entry by canonical path plus hash. Historical schemas remain
  read-only and unchanged.
- Do not add a runner-side authority token, repository callback, loader-object
  injection, or Python capability graph.
- Do not recursively snapshot injected globals, descriptors, functions, code
  objects, helper globals, slots, or loader/authority object graphs.
- Do not repeat supervisor checks for `/proc/self/cmdline`, `sys.orig_argv`, Git
  plumbing, worktree cleanliness, or loader-identity authentication inside the
  runner.
- Do not implement multi-level collision quarantine or rollback repair in the
  domain runner. The parent alone performs one atomic no-replace promotion.
- Do not defend against an exact, accepted, independently reviewed implementation
  maliciously attacking itself.

## 6. Threat model and assumptions

In scope are executable, CWD, environment, or argv drift; commit, HEAD, index, or
worktree mismatch; startup or import shadowing; post-capture repository mutation;
input/output symlink or path replacement; unexpected module import; worker abnormal
exit; and overwrite, terminal, schema, or hash mismatch.

Out of scope are a malicious exact accepted implementation; a compromised kernel,
root account, or supervisor; a hostile local process that defeats the single-writer
lease; cryptographic breaks; and hardware faults. A stronger hostile-local-process
requirement requires a new OS sandbox, mount, and process-isolation proposal, not
runner object-graph defenses.

The design assumes main enforces the single-writer lease, the kernel and direct
`execve` supervisor boundary behave as specified, authenticated hashes remain
sound, and fixed input/output bindings can be established before worker execution.

## 7. Stop conditions

Step 6 stops without further edits if it would require any path outside this design
and its one README index row, an implementation choice, an artifact or schema
change, execution, or language reopening retry, selection, or Task 010D.

Any future supervisor must stop on launch or accepted-commit drift, worktree/startup
shadow, mutable or late source capture, input/output replacement, uncaptured import,
worker abnormal exit, validation failure, an existing final root, or terminal,
schema, or hash mismatch. Failure leaves only clearly non-terminal temp state; it
must not overwrite, publish, quarantine, repair, retry, select, or proceed to Task
010D.

## 8. Required tests for any separately authorized implementation

Only after separate implementation authorization, tests must prove:

- the parent rejects verifier, executable, CWD, environment, argv, and commit drift;
- worktree and startup shadow code never executes;
- post-capture repository mutation cannot alter worker bytes, and source-identity
  drift blocks publication;
- the worker has no repository callback, token, authority, or loader injected
  globals;
- the closed loader rejects uncaptured repository modules;
- fork/COW prevents the worker from mutating parent authority state;
- new inventory identity is canonical path plus hash, with no redundant byte-count
  security check;
- the fixed temp root, no-overwrite rule, and one atomic no-replace promotion hold;
- non-zero exit or validation failure leaves only non-terminal temp state; and
- current D3 files, hashes, interpretation, and no-010D status remain unchanged.

## 9. Review and authorization gates

Main exclusively owns implementation acceptance, commits, review resolution, and
launch authorization. The Step 6 docs commit requires a fresh-context strict
read-only protocol review bound to the exact commit. It requires no scientific
artifact review, experiment rerun, or artifact revalidation.

Any future implementation requires separate explicit authority and an accepted
handoff binding the exact implementation commit/tree, supervisor path and source,
direct launch command, single verifier trust-root digest, captured-source inventory,
fixed input/output roots, and verification commands. Main must freeze that change,
obtain independent exact-commit implementation review, accept it, and separately
authorize any launch. None of those gates is satisfied by this design.

## 10. Return format

```text
Scope completed: future-only postmortem process-isolation simplification design and README index row
Prerequisite commit used: 0aff481d49010c0a84bfaa65dd89de5d73d5f45c
Changed paths: phase8/Future_Only_Postmortem_Process_Isolation_Design.md; phase8/README.md
Scientific contract changes: none
Engineering-only changes: future-only process-isolation simplification design and one index row
Tests run and exact results: docs-only diff, scope, protected-path identity, and required-text checks
New artifacts: none
Unresolved decisions: none for Step 6; future implementation path, source, command, and commit remain intentionally unchosen
Protocol deviations: none
Resource observations: none
Independent review required: yes — exact-commit fresh-context strict read-only docs protocol review
Recommended next authorized step: main commit/review only
```
