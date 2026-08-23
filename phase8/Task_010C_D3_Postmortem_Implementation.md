# Task 010C-D3-I — Read-Only Postmortem Implementation

## Status and authority

```text
status: revised implementation handoff pending independent review
accepted proposal commit: 06ee71d958eb1c4cb446ede3d120e99eb64bba97
accepted proposal blob: 5ea32f67d078408a6764adbb2d66518f713245d7
implementation authority: none until this handoff is accepted at an exact commit
execution authority: none
training authority: none
selection authority: none
Task 010D authority: false
```

This revised handoff is subordinate to the independently accepted proposal at
`phase8/Task_010C_D3_Feasibility_005_Postmortem_Proposal.md`. The proposal is the
scientific contract. This document freezes the bounded implementation package,
command surface, tests, and return format. It does not authorize a postmortem run.
The accepted repair corrects only the descriptive logits width and pre-import/
publication trust boundaries; it does not change the accepted 005 evidence, metric
semantics, model, data, optimizer, step budget, threshold, or authority boundary.
Implementation commits `365cd33aad276ac9d77255b6a75d86b23e668b5c`,
`5c078b2c375c373228196e09f244e5e16fe6e983`, and
`ca0626b12e198eefb0d3d0f9ac4254818705c97c` remain rejected and have no launch
authority; the next executor repairs the current code under this revised handoff and
the accepted proposal above. Handoff commit
`3f99add9a512986a9fc6f7b16d45f5b01f5aa1ce` is likewise rejected and supplies no
delegation authority. Repair commit
`58f6466e2ffc11b39f07da7ec95b7bc46228306f` is also rejected and supplies no
delegation authority; this revision closes its four review findings.
Implementation commit `c0f4a9f3f6350e520a0339266a02dcd3e999fcf8` is rejected and
supplies no launch authority. Its independent review found replaceable verifier
objects, unbound output-file staging identities/modes, an incomplete rollback
absence guarantee, missing runtime proposal-object authentication, a resolved-path
source-clean alias, and test-oracle/trust-boundary gaps. This revision closes those
findings without changing the accepted scientific contract.
Protocol repair commit `4c47adcc3c0f4a928cc029631057626d40769516` is also rejected and
supplies no implementation or launch authority. Its review confirmed the proposal,
lexical source, retained-file, and test-oracle repairs, but found that the loader
attribute and raw function callback behavior were not sealed, ignored executable
mode was omitted, and the final evacuation promise did not define persistent
filesystem failure. This revision closes those remaining findings.
Protocol repair commit `2650d8821dcb649393333cd603fea62d0cf1b61d` is likewise rejected
and supplies no implementation or launch authority. Its review confirmed the sealed
repository authority, executable-mode, and two-level evacuation repairs, but found
that inherited import ABCs still supplied the loader with an instance dictionary and
that terminal-marker demotion/absence-check failures were not themselves fail-stop.
This revision removes those last two paths.

After this handoff is accepted, `main` may delegate implementation to one
fresh-context Codex subagent with model identifier exactly `gpt-5.5`,
`fork_turns="none"`, working directory `/home/mye/src/llm/CapKnow`, and one mutation
lease. The executor returns an unstaged diff. `main` freezes and independently reviews
the implementation before separately deciding whether to run the postmortem.

## Immutable input

The implementation binds only this accepted artifact package:

```text
input root: artifacts/phase8_toy_lm_bridge/feasibility_005
input source commit: 2dc8c50fab9ece27a07eb0dc04021b78de52895c
manifest SHA-256: 51b43e3f3eba98b31c3574a9720c0df3f8e2a72c94d22730d70eb8682e8f516b
summary SHA-256: bfea56b52dfd1ffe094feb0774ce7895f4dd703770451aeda4d899aafdfef785
FAILED SHA-256: 6ac97f081586ca7fb8d7d8297dae4e0c7a3d0314a0567bbc493fce1b5dcc4c08
artifact review: ACCEPT, no findings
manifest inventory entries: 49
cells: 24
generation rows: 1536
terminal: FAILED
```

The implementation must validate the exact canonical path, source commit, top-level
hashes, terminal schema/hash chain, configuration, record hashes, ordered 24 cells,
complete inventory, every generation row, every checkpoint payload/metadata/state,
and every checkpoint/generation path/size/hash before CUDA construction or output
creation. A copied or equivalent-looking root is invalid. Do not reinterpret the
input terminal as a postmortem failure or a feasibility selection.

The four 005 predecessor roots and its D1 decision diagnostic are transitive
provenance and source-cleanliness inputs only. They may never supply a model, record,
metric, or result to D3. Their canonical paths and top-level hashes come only from the
accepted 005 manifest; shallow cleanliness authorization additionally requires each
root's exact terminal/manifest and complete path/size/hash inventory. Do not authorize
a copied root or the artifact parent directory.

## Allowed paths

The executor may modify only:

```text
scripts/phase8_sequence_feasibility.py
tests/test_lm_bridge.py
```

Do not modify this handoff, the accepted D3 proposal, the exact verifier source at
`phase8/Task_010C_D3_Postmortem_Verifier.py.txt`, model/tokenizer/train modules,
corpus generators, earlier handoffs, any artifact, or any later Phase 8 task. If the
implementation cannot fit these paths without changing an accepted scientific
contract, stop and return the smallest ambiguity to `main`.

## Required command surface

Add exactly one CLI subcommand named `postmortem-failure`. Its future canonical
launch is not a shell command. A separately reviewed external supervisor, bound only
in a later launch packet, must call `execve()` with this exact template:

```text
executable: /opt/anaconda3/bin/python
cwd: /home/mye/src/llm/CapKnow
environment: {"LC_ALL":"C","PYTHONDONTWRITEBYTECODE":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8","PYTHONPATH":"."}
argv:
  /opt/anaconda3/bin/python
  -I
  -B
  -S
  -c
  <exact UTF-8 bytes of phase8/Task_010C_D3_Postmortem_Verifier.py.txt>
  --verifier-sha256
  55c1e76086da332c9b7827980912f6ffe2bc1f38486c410be10c71e647b4e522
  --accepted-implementation-commit
  <later independently accepted implementation commit>
  --runner-path
  scripts/phase8_sequence_feasibility.py
  --
  postmortem-failure
  --device
  cuda:0
  --input-root
  artifacts/phase8_toy_lm_bridge/feasibility_005
  --output-root
  artifacts/phase8_toy_lm_bridge/feasibility_postmortem_001
  --accepted-proposal-commit
  06ee71d958eb1c4cb446ede3d120e99eb64bba97
  --accepted-implementation-commit
  <same later independently accepted implementation commit>
```

The verifier source is exactly `39,455` bytes, has Git blob
`8d1a305f9a28d995f53b26c22b24241da1f0d5fe`, and has SHA-256
`55c1e76086da332c9b7827980912f6ffe2bc1f38486c410be10c71e647b4e522`. The
implementation must embed and test that SHA and reject any byte difference. It must
not execute the launch. This handoff supplies no supervisor identity and no accepted
implementation commit, so it supplies no complete launch command and no execution
authority.

`manifest.json.exact_command` is one closed JSON object, never a shell string or token
display list. It has exactly four keys: `executable,cwd,environment,argv`. Its values
are constructed only from the already authenticated real process state:

```text
executable: "/opt/anaconda3/bin/python"
cwd: "/home/mye/src/llm/CapKnow"
environment: exactly the four-key object printed above
argv: exactly sys.orig_argv, as a JSON array of 24 strings
```

Within `argv`, index 5 is the exact verifier source decoded as UTF-8 with no newline
normalization or escaping transformation before JSON serialization; its UTF-8
re-encoding must have the verifier SHA-256 above. Index 7 is that SHA-256. Indexes 9
and 23 are the same later accepted implementation commit; index 21 is the accepted
proposal commit. The other indexes equal the command template above. The manifest's
separate top-level `environment` object is intentionally different: as required by
the accepted proposal, it equals the feasibility-005 manifest's complete
CUDA/Python/Torch/platform runtime environment field-for-field. Only
`exact_command.environment` is the four-key `execve` environment.
`implementation_binding.commit` must equal both implementation-commit argv positions.
Canonical JSON serialization may escape the source string but decoding the JSON must
reproduce the exact original string and bytes.

Two pre-existing ignored `__pycache__` files under `scripts/` and `tests/` currently
make verifier source authentication fail closed. The executor must not delete or
modify them to make tests or a launch pass. Their disposition belongs to a later
main-thread launch-precondition decision; this implementation task remains valid and
non-executing while they are present.

Refuse programmatic `main(argv)`, direct script execution of `postmortem-failure`,
`runpy`, and direct-import invocation of the normative writer. The writer must require
the verifier-injected globals, exact verifier SHA, exact accepted implementation
commit repeated in CLI, real `__main__` process context, exact `/proc/self/cmdline`,
exact environment, isolated/no-bytecode/no-site flags, `--device cuda:0`, the two
exact roots, and the accepted proposal commit before model construction or output.

The proposal binding written later is:

```text
commit: 06ee71d958eb1c4cb446ede3d120e99eb64bba97
path: phase8/Task_010C_D3_Feasibility_005_Postmortem_Proposal.md
blob: 5ea32f67d078408a6764adbb2d66518f713245d7
```

The implementation binding is the clean runtime HEAD plus the Git blob of
`scripts/phase8_sequence_feasibility.py`. Do not embed the future implementation
commit into source; accept it only from the verifier and the duplicate exact CLI
argument. `main` supplies and reviews that commit plus the external supervisor in a
later launch packet.

Before runner execution, the verifier must authenticate through its retained Git
descriptors that `06ee71d958eb1c4cb446ede3d120e99eb64bba97` is a commit, that its
full tree maps
`phase8/Task_010C_D3_Feasibility_005_Postmortem_Proposal.md` to blob
`5ea32f67d078408a6764adbb2d66518f713245d7`, and that the blob bytes hash back to
that object id. It repeats this authentication inside every later repository
callback. A constant-only comparison is not authentication.

The verifier injects the accepted commit, executed runner blob, authenticated
immutable proposal tuple, verifier SHA, a unique authority token, sealed closed
loader, and one sealed zero-argument callable repository-authority object into the
runner globals. The exact injected names are
`__phase8_accepted_implementation_commit__`, `__phase8_runner_blob__`,
`__phase8_proposal_binding__`, `__phase8_verifier_sha256__`,
`__phase8_authority_token__`, `__phase8_repository_loader__`, and
`__phase8_verify_repository_unchanged__`; the proposal tuple is exactly
`(commit,path,blob)` in that order. The loader is a slots-only duck-typed finder/
loader and must not inherit an ABC that supplies an instance dictionary. It exposes its
captured source map only through a read-only property backed by a sealed slot, and
rejects attribute rebinding. The repository-authority object likewise has no instance
dictionary, stores immutable captured source/mode maps and exact helper/code
bindings in sealed slots, verifies those bindings before and after every call,
reauthenticates the full repository, and returns exactly an immutable pair of
`(untracked_paths, ignored_paths)` tuples for the evidence-aware cleanliness check.
It exposes no argument-taking Git callable. At initial verifier-context validation,
the normative runner must retain the exact injected authority, loader, proposal tuple,
authority token, accepted commit, runner blob, and verifier SHA objects/values in its
internal runtime binding. It must additionally snapshot the loader source-map object
and contents; the loader type/property/finder/loader method objects and code; and the
authority type/call method/code, sealed-slot object identities/values, and helper/code
bindings. Every later source check invokes only that retained authority and first and
afterward requires the entire snapshot plus every injected global to remain
identical. Replacing an injected object, rebinding a loader source map, or mutating
the authority's code, globals/helper lookup, or sealed state while preserving object
identity must fail. Calling ambient `git`, reaching
verifier-internal Git plumbing, reparsing local configuration, reopening repository
source, or replacing an injected object is forbidden and covered by sentinels.

Verifier Git plumbing has an exact complete argv allowlist, so variants such as
`cat-file --filters`, `--textconv`, option injection, and new subcommands are
unrepresentable. Before and after every allowed Git call it revalidates retained
no-follow descriptors plus device/inode/full-mode/size/content snapshots for the
repository root, `.git`, object/ref/info stores, HEAD, index, local config,
`packed-refs`, and `info/exclude`; Git receives only `/proc/self/fd` paths through
`pass_fds`. The complete `.git` control namespace/content is snapshotted around every
call, `GIT_COMMON_DIR` is pinned to the retained `.git` descriptor, and `commondir`,
`config.worktree`, and `info/attributes` must remain absent. The local config is
parsed structurally and must equal the frozen inert `core` and `remote "origin"` map;
an extra section or key, including a filter/diff/include/helper setting under any
case or whitespace spelling, fails closed.

## Preflight order

The exact fail-closed order is:

1. before the runner or any repository/third-party module executes, the external
   supervisor supplies the exact clean `execve()` boundary and the inline verifier
   validates its source/flags/CWD/environment plus the exact 24-string kernel argv,
   exact 19-string Python `sys.argv`, complete frozen runner suffix, and absence of
   reordered, substituted, or extra arguments;
2. the verifier authenticates the canonical non-redirected Git store with hermetic
   exact-argv plumbing over retained descriptors, the accepted implementation HEAD,
   complete tree/index/worktree bytes and full modes, untracked/ignored import
   surface, runner bytes, every repository Python module buffer, and the accepted
   proposal commit/type/tree/path/blob bytes; it installs the closed in-memory
   repository loader, adds only the frozen site-packages path, and executes the
   captured runner;
3. validate runner argument types and canonical root basenames;
4. require exact real main/verifier context, kernel argv, environment strings, and absent final
   and same-parent `.tmp` output roots;
5. shallow-validate the input path, terminal, three top-level hashes, 49-entry
   inventory paths/sizes/hashes, plus the exact 005-bound paths/top-level hashes and
   complete inventories of predecessors 001–004 and D1; derive only those exact files
   as source-clean exclusions;
6. repeat the full evidence-aware clean-source check and reject every ignored
   executable input using only the retained hermetic Git/source authority; compare
   verifier-returned untracked names to a canonical repository-relative lexical
   allowlist without `resolve()` or target equivalence, and reject every untracked
   symlink/non-regular entry before runner execution; reject every ignored symlink,
   ignored import surface, and ignored regular file with any executable mode bit;
7. verify current HEAD, the verifier-authenticated proposal commit/path/blob tuple,
   executed runner blob, accepted
   implementation commit, full A800 runtime object,
   and deterministic flags;
8. deep-validate input manifest/summary/terminal equality, 24 ordered cells,
   generation rows and scores, tied checkpoint schemas/duplicate byte equality, and
   complete input lineage;
9. reconstruct the eight record sets with local PRNGs, verify all eight frozen hashes,
   and prove global Python/Torch CPU/all-CUDA RNG states unchanged before any restore;
10. only then create the `.tmp` root and enter the checkpoint-forward loop.

No worktree repository byte may execute: the runner and every
`capability_certificate_lab` package/module execute only from verifier-captured,
accepted-commit buffers through the closed highest-priority loader. The repository,
`scripts`, and artifact paths never enter `sys.path`; site-packages shadowing and
post-capture source mutation cannot change executed bytes. No tensor load, model
construction, CUDA forward, record schedule, or output path may occur before full
source cleanliness. Shallow validation may hash files but must not load
checkpoints or construct models. Tests must use call-order sentinels for both failure
and passing paths.

## Read-only checkpoint execution

Implement the accepted proposal literally for all 24 cells. Use the existing tied
checkpoint validator; construct each model on CPU only inside the proposal's single
Torch RNG snapshot/restore scope, load and validate exact state, then move it to
`torch.device("cuda:0")`. Python RNG must not change. After load, snapshot every named
parameter's dtype, shape, layout, and contiguous bytes; require equality after train
teacher forcing, after eval teacher forcing, and before publication.

Use `model.eval()` and `torch.inference_mode()`. Do not instantiate an optimizer,
enable gradients, call backward, mutate parameters/buffers, save a checkpoint, or call
any generation/greedy-decode function. A sentinel around every forbidden operation is
mandatory in production and tests. The only generation evidence is the retained
input JSONL.

For each cell, evaluate exactly its own family records: 512 train records in eight
contiguous 64-row batches followed by 64 eval records in one batch. Preserve exact
record order, `[64,256]` int64 inputs/labels, `[64,256,260]` float32 logits,
`[64,256]` float32 unreduced CE, response-only shifted labels including EOS, argmax,
`math.fsum` row/cell order, `float.hex()` NLL serialization, integer-first metrics,
and every Named/Array/common rule in the accepted proposal and its bound D1 metric
oracle. Error messages and tests must use width `260`; `259` is forbidden.

Snapshot process-global Python, Torch CPU, and all-CUDA RNG states at every boundary
required by the proposal. Record reconstruction must compare before any restore.
Only CPU model construction may restore Torch RNG in its local scope. No restore is
allowed afterward. Verify runtime environment/deterministic flags and RNG-state
equality again immediately before publication.

## Closed outputs and publication

Implement exactly the proposal schemas and cardinalities:

```text
teacher_forced_rows.jsonl: 13824 rows
cell_metrics.jsonl: 24 rows
error_taxonomy.jsonl: 1536 rows
summary.json
manifest.json
DONE.json only after complete validation
```

Do not add a file, field, compatibility alias, display float, checkpoint, regenerated
output, record pack, log, status file, or selection. Use the proposal's canonical
JSON/JSONL serialization, exact nested schemas, sparse histogram rules, conditional
denominators, `all_items_copied` predicate, paired four-cell counts, input/proposal/
implementation bindings, runtime objects, row order, and complete four-file manifest
inventory. Independently reconstruct every aggregate from retained rows before
publication; a second forward for aggregation is forbidden.

The temp root is exactly:

```text
artifacts/phase8_toy_lm_bridge/feasibility_postmortem_001.tmp
```

Create it exclusively only after preflight. A finalized FAILED root is forbidden. On
creation retain no-follow descriptors for the real artifact parent and temp root and
bind their device/inode/full-mode/type. Create each of the six declared root-level
regular non-symlink files first under a unique hidden name with exclusive no-follow
`openat`; retain that file descriptor until publication completes, bind its full
`(device,inode,st_mode,size,digest)`, verify the hidden name still denotes the same
object, rename it no-replace through the temp descriptor while it remains open, and
verify the declared name denotes that same object. Retain the six descriptor-backed
bindings across every callback, validation, root rename, and post-rename validation.
All JSON/JSONL/hash/inventory validation in the normative path must read those
retained descriptors; a same-byte different-inode substitution or any chmod drift is
fatal. Do not accept a first whole-root fingerprint taken only after all writers have
closed their descriptors.
On any exception, remove or atomically demote both terminal names through the trusted
directory descriptor without following a root or marker symlink, then verify both
names are absent. Attempt both demotions even if the first fails. If either demotion
or the absence verification cannot complete, enter immediate `os._exit(74)` and do
not raise or return through ordinary Python control flow. Otherwise preserve a
non-terminal incomplete temp root and re-raise. On success, write summary, then
manifest, then DONE; validate every schema, cardinality, path, hash, aggregate,
binding, parameter snapshot, RNG state, source/runtime snapshot, identity, and
inventory; repeat the final source/runtime/RNG check; then use Linux atomic
rename-no-replace and verify the same directory identity at the final path.

If post-rename validation fails, first atomically demote both terminal names through
the retained directory descriptor, before any collision or evacuation operation, so
no subsequent filesystem error can leave a terminal-looking canonical root. Attempt
both demotions and verify both terminal names are absent; if either operation or that
verification fails, enter immediate `os._exit(74)` without an ordinary raise or
return. Only then preserve an unexpected occupant of the exact temp name under a unique noncanonical,
non-terminal same-parent collision quarantine, then move the invalid final entry back
to the exact temp path without overwrite,
demote terminal names through its no-follow descriptor, and require the final path to
be absent before returning an error. Collision-preservation failure must not skip
final-root evacuation. If the exact final-to-temp rollback fails, atomically move the
invalid final entry to a unique noncanonical, non-terminal same-parent quarantine,
demote its terminal names through the retained descriptor, and still require the
canonical final name to be absent before control returns. Tests must inject failures
in collision preservation and the first rollback rename. If both the exact rollback
and final-to-quarantine evacuation fail, revalidate through the retained descriptors
that both terminal names are absent and enter immediate `os._exit(74)` fail-stop;
this catastrophic path must not return or raise through ordinary Python
control flow and permanently blocks acceptance or retry pending manual inspection.
Thus every ordinary post-rename error return still has an absent canonical final as
required by the accepted proposal, while an irrecoverable directory-operation failure cannot
return a terminal-looking root as evidence. A quarantine is never an output, retry,
fallback, or evidence root. Never follow/delete an external symlink target, overwrite, retry,
or create `feasibility_postmortem_002`.

## Required tests

Add focused tests covering at least:

1. exact verifier source bytes/blob/SHA-256, authenticated proposal commit/type/tree/
   path/blob bytes, supervisor `execve` executable/CWD/
   environment/argv, isolated/no-bytecode/no-site flags, `/proc/self/cmdline`, and
   duplicate implementation-commit binding; exact four-key `exact_command` object,
   including round-tripped verifier bytes, the deliberate distinction from the
   top-level 005 runtime `environment`, and agreement with the separate implementation
   binding; mutate every runner-suffix token and add/reorder a token, and reject each
   before runner/repository/Torch/output/model work; likewise reject shell,
   `/usr/bin/env`, direct script, `runpy`, import, changed verifier, and
   programmatic-main routes;
2. execute the exact verifier bytes themselves in an isolated non-main harness (the
   source has a guarded normative `main()` entry for this purpose), then exercise its
   real `PinnedGitContext`, allowlisted `git_run`, repository authenticator, proposal
   authenticator, and closed loader against temporary Git repositories. Copying or
   reimplementing verifier helper logic in runner tests is not coverage. Include
   anti-execution tests for inherited loader/Python/Git variables, local/global
   FSMonitor and hooks, pager, external diff/textconv/filter,
   assume-unchanged/skip-worktree, replacements, alternates/grafts, `commondir`,
   partial-clone/promisor packs and lazy fetch; directly attempt `cat-file --filters`,
   `--textconv`, option injection, extra config sections/keys, `info/attributes`,
   same-size `.git` namespace mutation, and `commondir`/common-info insertion during
   plumbing, plus config spelling/case variants. Every sentinel remains absent;
3. complete accepted commit/HEAD/tree/index/mode/raw-worktree/blob equality, regular
   no-follow reads, full-mode/chmod drift, Git metadata/object/ref/config/info entry
   identity/content swaps before/during/after plumbing, unauthorized untracked and
   repo-wide ignored import surfaces, actual runner bytes, all captured repository
   Python modules, canonical lexical untracked authorization, rejection of an
   untracked symlink alias to an allowed artifact and an extensionless ignored
   regular `0755` file, the sole immutable no-argument source authority, immutable
   loader/proposal/authority-token bindings, repeated
   source verification, and rejection of every ambient, raw, argument-taking, or
   same-signature replaced Git/source helper; mutate the retained authority's type
   call code, helper globals/code, and sealed slots plus the retained loader's source
   property/slot and method code without changing injected object identity, and
   require rejection before any subsequent repository import or publication; assert
   the loader has no `__dict__` and cannot shadow `find_spec`, `create_module`, or
   `exec_module` through an instance attribute;
4. a closed highest-priority in-memory loader test for package initializers, relative
   and transitive imports, missing-module rejection, no worktree path on `sys.path`,
   site-packages namespace shadowing, and post-capture worktree mutation. Only the
   captured accepted buffers may execute;
5. exact CLI argument/environment/main-context acceptance plus refusal before
   output/model work;
6. exact input root/top-level hashes/configuration/record hashes/cells/inventory and
   rejection of copies, substitutions, symlinks, missing/extra fields/files, size/hash
   drift, terminal drift, generation mutation, and tied-checkpoint mutation;
7. verifier → shallow input → full clean source → deep input → records/RNG → `.tmp`
   → model/forward call
   order, with no early tensor/model/CUDA/output work;
8. local record PRNG success, global RNG mutation rejection before restore, the single
   constructor restore scope, and byte-identical Python/Torch CPU/all-CUDA states
   after load/inference/publication;
9. exact 24-cell order, 13,824 teacher rows, 24 cell rows, 1,536 taxonomy rows,
   `[64,256]` inputs/labels/loss, `[64,256,260]` logits, labels including EOS,
   float32/no-autocast CE, `math.fsum`, `float.hex()`, and no second aggregate forward;
10. an independent test oracle that never calls production taxonomy, row builder, or
   aggregate helpers to construct its expected values; use explicit edge-case
   fixtures and literal expected common/Named/Array row fields, null conditions,
   sparse histograms, item-count strata, conditional denominators, item multisets,
   paired exact table, and all summary aggregates;
11. checkpoint metadata copied descriptively without comparing it to fresh
   teacher-forced statistics;
12. sentinels rejecting optimizer construction, grad enablement, backward, post-load
   parameter/buffer mutation, checkpoint save, generation, extra records, or new data;
13. exact JSON/JSONL keys, types, enums, nullability, ordering, canonical bytes,
   cardinalities, bindings, four-file inventory, and one-way DONE→manifest checksum;
14. descriptor-bound publication mutation tests for every output/aggregate/binding/
    callback, hidden staging-name same-byte/different-inode replacement, persistent
    chmod/full-mode drift, root and marker symlinks, same-byte directory swaps,
    extra/non-regular entries, marker FIFO/socket/directory/device forms, no-clobber,
    post-rename mismatch, occupied-temp collision preservation failure, first
    rollback-rename failure, quarantine fallback, and final quarantine-evacuation
    failure in a child-process fail-stop test; inject failure at each pre-rename and
    post-rename DONE/FAILED demotion plus terminal-absence verification and require
    the same fail-stop. Every ordinary failure leaves no
    terminal-looking marker and no canonical final root without following external
    targets;
15. permanent rejection of selection use, feasibility verdict change, retry,
    `feasibility_006`, alternate postmortem root, or `010D` authorization;
16. regression coverage for existing D1/D2 validators, current feasibility artifacts,
    and selection rejection.

Run in this order:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q tests/test_lm_bridge.py -k 'd3 or postmortem'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q tests/test_lm_bridge.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -c 'from pathlib import Path; [compile(path.read_bytes(), str(path), "exec") for path in (Path("scripts/phase8_sequence_feasibility.py"), Path("tests/test_lm_bridge.py"))]'
git diff --check
```

Tests must use temporary roots and synthetic fixtures. They must not read CUDA unless
explicitly marked as a non-artifact device smoke, and must never create the normative
postmortem root.

## Documentation boundary

Do not modify documentation in the implementation package. `main` records accepted
implementation status and any later launch boundary only after exact-commit review.
Do not record a postmortem outcome because this task cannot run it.

## Executor return

Return exactly:

```text
Scope completed:
Prerequisite handoff commit used:
Accepted proposal commit/blob used:
Changed paths:
Scientific contract changes:
Engineering-only changes:
Tests run and exact results:
New artifacts:
Forbidden-operation sentinels:
Unresolved decisions:
Protocol deviations:
Resource observations:
Independent review required:
Recommended next authorized step:
```

For acceptance, `Scientific contract changes`, `New artifacts`, `Unresolved
decisions`, and `Protocol deviations` are `none`. The recommended next step is only
`main` inspection, commit, and fresh independent strict read-only review of the exact
implementation commit. It is never the postmortem run.

## Independent handoff-review gate

Before delegation, freeze this handoff and review it at an exact commit with a
fresh-context `gpt-5.6-sol`, `xhigh`, strict read-only agent. The review must verify:

- exact proposal and 005 evidence binding;
- exact verifier source/blob/SHA, external supervisor `execve` template, unavailable-
  supervisor no-execution boundary, exact full command suffix/cardinalities, duplicate
  accepted implementation commit, closed four-key `exact_command` value, separation
  of execve and 005 runtime environment objects, and command/main-context contract;
- exact-argv hermetic Git plumbing, descriptor/content-bound Git inputs, structural
  local-config allowlist, complete tree/index/full-mode worktree/import-surface
  authentication, no-helper/no-filter/no-lazy-fetch behavior, sealed no-argument
  repository authority, and sealed closed captured-buffer repository loader;
- shallow/source-clean/deep/RNG/output ordering;
- teacher-forced and taxonomy semantic fidelity;
- RNG-neutral construction and post-load parameter immutability;
- closed schemas, cardinalities, provenance, runtime, descriptor-bound no-follow
  cleanup, occupied-temp quarantine/rollback, and DONE-only atomic publication;
- forbidden-operation and tamper test completeness; and
- no training, generation, selection, retry, verdict reversal, or `010D` authority.

Return `ACCEPT` or `REJECT` with findings ordered by severity, exact file/line
references, impact, and the smallest repair. A rejected handoff has no delegation
authority.
