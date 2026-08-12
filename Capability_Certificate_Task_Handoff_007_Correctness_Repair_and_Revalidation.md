# Capability Certificate Task Handoff 007

## Phase 2-6 Correctness Repair and Scientific Revalidation

## 1. Objective

Repair the confirmed correctness and protocol defects in the Phase 2-6 implementation, rerun the experiments whose conclusions may be affected, and produce evidence that can be independently reviewed.

This is a repair milestone, not a feature milestone. Make the smallest changes that restore the frozen mathematical and execution contracts. Do not introduce frameworks, plugin systems, generalized configuration layers, compatibility fallbacks, or speculative abstractions.

Baseline under review:

```text
commit: 1648c8f201936e56f7eb0544b39c45cf0f431c9b
baseline functional test command: python -m pytest -q
baseline result: 43 passed
```

Passing the existing tests is not acceptance: several confirmed defects are not covered by them.

## Independent Review Result (2026-08-08)

An independent strict read-only review was performed after implementation and revalidation. The reviewer used the same default model as the main thread, started with `fork_context=false`, and received only this handoff, the frozen working-tree change, relevant artifacts and hashes, acceptance criteria, protocol constraints, and verification already run. The main thread remained read-only during review.

Gate verdict: **REJECT**. The current implementation and evidence package must not be accepted or committed as a completed repair milestone until the confirmed findings below are resolved and the affected experiments are rerun from a provenance-bound source state.

### High: provenance failure

- The five revalidation manifests record baseline commit `1648c8f201936e56f7eb0544b39c45cf0f431c9b`, while the experiments actually imported uncommitted repaired code from the working tree. The manifests also omit the `PYTHONPATH=.` prefix present in the executed commands. The artifacts are therefore not reproducible from their declared source commit, and their hashes bind output bytes but not the source that generated them.
- Required repair: freeze the repaired source as a commit or an explicit normalized diff/tree hash, record the actual command and dirty/clean state, and rerun all five revalidation experiments from that bound state.

### Medium: correctness and protocol findings

1. `validate_adaptive_certificate` can accept an incomplete, non-progressing decision tree. Recursively validate the candidate subset: every internal node must have both branches, split the subset into two non-empty partitions, avoid repeated questions, and terminate in a leaf containing exactly the single remaining state.
2. `LoopNode` declares `LOOP` as a required capability, but the executor does not check it. Execution must fail when `LOOP` is absent.
3. DSL signature generation catches every `InvalidProgramError` and converts it to response `0`, masking malformed programs as capability failures. Distinguish capability absence from invalid program structure and propagate the latter.
4. Phase 5 contains 12 conditions where sequential/batch MAP consistency is below 1.0, with a minimum of 0.86, while the report states that all cases satisfy consistency. Emit posterior differences and MAP tie information, define a shared tie rule, include consistency in the failure gate, and rerun the experiment.
5. Phase 4 labels an aggregate of `query_count` as average query depth even though `query_count` is the worst-case depth. Aggregate `average_depth` separately from `worst_case_depth` and state the weighting and cross-seed aggregation semantics.
6. Phase 6 uses `MEMORY -> SEARCH` as a held-out composition even though the same structure already exists in the default `RETRIEVAL` rule. Replace it with a primitive composition absent from the default rules and freeze its input and expected output.

### Low: required test gaps

- Add the explicitly required too-long response-signature regression test.
- Replace the adaptive serialization self-comparison with assertions for concrete `yes_child`, `no_child`, and leaf state IDs so that symmetric omissions cannot pass.

### Confirmed unaffected areas

- Static review found no repeated use of historical observations in the repaired Bayesian update.
- Log-space posterior normalization and explicit all-zero-likelihood behavior match this handoff.
- JSON-array state IDs have no identified collision path.
- Composite result-ID conflict detection has no identified semantic defect.
- The reviewer confirmed the supplied SHA256 values for all ten result and manifest files, but did not rerun tests or experiments.

## 2. Binding Principles

- KISS and YAGNI. Prefer a local, explicit repair over a reusable framework.
- Fail early and loud on malformed worlds, signatures, policies, and composition rules.
- Do not silently filter invalid scientific inputs or silently repair inconsistent metadata.
- Tests must check mathematical invariants and exact expected values, not only output types or smoke-level execution.
- Do not change a probability definition merely to make a test pass.
- Run experiments only after all P0/P1 correctness tests pass.
- Do not use results generated before the correctness repairs as scientific evidence.
- Record seeds, configs, source commit, commands, and output hashes for new experimental evidence.

## 3. Explicit Non-goals and Accepted Exceptions

The following are already accepted and must not be expanded in this repair:

- Phase 3 may continue using exhaustive subset search. Do not add ILP, SAT, OR-Tools, or another solver in this handoff.
- Do not repair the tree-mapping loss of a parent with an empty child list. Retain the documented assumption that such input is not supplied.
- Do not add LLM execution, learned semantics, distributed execution, databases, dashboards, or a generic experiment framework.
- Do not preserve invalid historical behavior through defensive compatibility fallbacks.
- Do not redesign public APIs unless a change below explicitly requires it.

## 4. Risk Order

Execute in this order:

1. P0 probabilistic posterior correctness.
2. P1 identifiability and adaptive-certificate correctness.
3. P1 DSL deterministic execution and composition contracts.
4. Targeted regression tests.
5. Repository-wide tests and static checks already available in the repository.
6. Phase 2-6 revalidation experiments.
7. Reports, manifests, hashes, and independent review.

Do not start experiments while any earlier correctness item is unresolved.

## 5. P0: Repair Adaptive Bayesian Updating

### 5.1 Confirmed defect

`solve_noisy_adaptive_certificate` currently aggregates every historical observation on every round and also passes the previous posterior as the next prior. After two rounds it computes:

\[
p_0(K)L_1(K)^2L_2(K)
\]

instead of:

\[
p_0(K)L_1(K)L_2(K)
\]

This creates false confidence and can stop assessment too early.

Relevant code:

```text
capability_certificate_lab/probabilistic/noisy_certificate.py
capability_certificate_lab/probabilistic/posterior.py
tests/test_probabilistic.py
```

### 5.2 Required repair

Use exactly one coherent update method:

- Recommended: pass only the newly collected responses to `infer_state_posterior`, using the previous posterior as the prior.
- Also valid, but do not combine with the first method: recompute from the original prior using the complete observation history.

Keep complete query history for reporting, but do not use it twice in inference.

### 5.3 Required tests

Add exact invariant tests:

1. Sequential Bayesian updates equal one batch update over the same observations within tight floating-point tolerance.
2. For an unstructured world with tasks `A, B`, uniform prior, `slip=0.1`, `guess=0.1`, and observations `A=1, B=0`, the posterior must be:

```text
{}      0.09
{A}     0.81
{A,B}   0.09
{B}     0.01
```

3. Query-history length and response counts remain correct when `attempts_per_query > 1`.
4. A stopping-threshold test must show that a run does not cross `1-delta` solely because an old observation was counted again.

## 6. P0: Repair Posterior Numerical Stability

### 6.1 Confirmed defect

`infer_state_posterior` multiplies likelihoods directly in probability space. Sufficiently many possible observations can underflow every state weight to zero and be misreported as an impossible observation.

### 6.2 Required repair

Compute posterior weights in log space and normalize with a minimal log-sum-exp calculation. Handle mathematically impossible events as `-inf`; do not introduce epsilon smoothing or a fallback uniform posterior.

If every state is genuinely impossible under the declared model, retain explicit failure semantics. Distinguish this case from numerical underflow.

### 6.3 Required tests

- A long, mathematically possible observation sequence must return finite normalized probabilities whose sum is approximately one.
- The result must have a non-empty MAP state.
- A genuinely impossible observation under a zero-probability model must remain explicit and must not be silently converted to a uniform posterior.

## 7. P1: Repair Identifiability Audit Contracts

Relevant code:

```text
capability_certificate_lab/validation/identifiability/core.py
capability_certificate_lab/knowledge_space/space.py
tests/test_identifiability.py
```

### 7.1 Invalid declared states

Current behavior silently removes invalid entries from `KnowledgeSpace.valid_states`. An all-invalid population is then reported as `num_states=0` and `identifiable=True`.

Required behavior:

- Treat `valid_states` as a declared scientific population.
- Before auditing, reject the world with `ValueError` if any declared state violates its world rules.
- Reject an empty declared population because identifiability is not meaningful there.
- Do not silently discard states.

### 7.2 Duplicate declared states

Required behavior:

- Reject duplicate `KnowledgeState` entries with `ValueError` before collision analysis.
- Collision groups are only for distinct states with equal valid response signatures.

### 7.3 Response-signature contract

For every audited state:

- signature length must equal the number of queried tasks;
- every coordinate must be binary `0` or `1`;
- malformed signatures must fail early with a clear exception.

Replace the existing artificial collision test with a full-length binary signature. Add tests for too-short, too-long, and non-binary signatures.

### 7.4 Required semantic tests

- A valid artificial world containing two distinct states with the same full response vector is reported as non-identifiable.
- An invalid-state population raises instead of producing vacuous success.
- A duplicate-state population raises instead of producing a collision group.
- Existing generated chain, tree, and unstructured worlds retain their expected audit results.

## 8. P1: Make State Identity Unambiguous

### 8.1 Confirmed defect

`stable_state_id` joins raw task IDs with commas. The state containing one task named `A,B` collides with the state containing two tasks named `A` and `B`.

### 8.2 Required repair

Use an unambiguous deterministic representation. A compact JSON array of task IDs in universe order is sufficient. Do not invent a custom escaping grammar.

Use the canonical identity consistently in identifiability reports, posterior maps, decision-tree leaves, validation, and serialization.

### 8.3 Required tests

- `{"A,B"}` and `{"A", "B"}` have distinct identities.
- IDs containing commas, braces, quotes, and backslashes remain distinct and deterministic.
- Adaptive validation cannot accept a zero-question leaf for two different states because their display IDs collide.
- Existing reproducibility tests are updated for the new canonical representation.

## 9. P1: Repair Adaptive Random Policy

Relevant code:

```text
capability_certificate_lab/certificate/policies.py
capability_certificate_lab/certificate/adaptive_solver.py
tests/test_adaptive_certificate.py
```

### 9.1 Confirmed defect

The random policy can select a task that is constant over the current candidate subset. The tree builder then creates a multi-state leaf even when another unasked task can split the states.

### 9.2 Required repair

- A built-in policy may choose only among unasked tasks that produce two non-empty partitions of the current candidate set.
- The random baseline chooses uniformly from those splitting tasks using the supplied RNG.
- Return `None` only when no unasked splitting task exists.
- If a custom policy returns a task that violates this contract, fail loudly instead of silently terminating with a misleading leaf.

### 9.3 Required tests

- The existing identifiable tree world returns a valid random certificate for seeds `0..99`.
- Random-policy output remains reproducible for a fixed seed.
- A genuinely non-identifiable candidate set terminates as invalid.
- A custom policy returning a non-splitting task raises a clear contract error.

## 10. P1: Preserve Adaptive Certificate Trees in Serialization

`AdaptiveCertificate.to_dict()` currently omits the root tree, so persisted results cannot be independently replayed or validated.

Required repair:

- Add a minimal recursive `DecisionNode.to_dict()` representation.
- Include `root` in `AdaptiveCertificate.to_dict()`.
- Preserve task ID, candidate state IDs, and both branches.
- Do not add a general serialization framework.

Required test:

- Serialized adaptive output contains the complete tree and enough information for structural comparison with the in-memory tree.

## 11. P1: Tighten DSL Deterministic and Composition Contracts

Relevant code:

```text
capability_certificate_lab/dsl/simulator.py
capability_certificate_lab/dsl/executor.py
capability_certificate_lab/dsl/primitives.py
capability_certificate_lab/dsl/task_generator.py
tests/test_dsl.py
```

### 11.1 Exact-solver signature must be deterministic

`make_dsl_response_signature` currently permits mutable RNG-backed noisy responses, while exact solvers may evaluate the callback more than once. Construction and validation can therefore observe different signatures.

Required repair:

- Make the signature callback used by exact and identifiability solvers deterministic.
- Recommended minimal change: remove `noise` and `rng` from `make_dsl_response_signature` and keep noisy single-task sampling in the probabilistic response path.
- Do not add callback metadata, runtime introspection, or a new stochastic-solver abstraction.

Required tests:

- Repeated deterministic signature calls return identical values.
- Exact certificate solving is tested only with deterministic signatures.
- Probabilistic DSL sampling remains available through the explicit probabilistic API, not through the exact-solver callback.

### 11.2 Composition result collisions

Custom rules can currently reuse a primitive or an earlier rule result. Program construction may skip a rule while metadata and prerequisites still include or overwrite it.

Required behavior:

- Every composition result ID must be unique.
- A result ID must not collide with a primitive ID.
- Duplicate or conflicting result definitions raise before world construction.
- Unresolvable left/right dependencies continue to fail loudly.
- Program map, prerequisites, active rules, constraints, and metadata must describe the same rule set.

Required tests:

- duplicate result IDs raise;
- primitive/result collision raises;
- chained unique rules resolve correctly;
- `include_composite=False` retains empty composition metadata.

## 12. P1: Complete the Minimal Phase 6 Execution Contract

### 12.1 Acceptance gap

The Phase 6 handoff specifies `execute(program, input) -> correct answer`. The current executor ignores `input_context` and returns only whether a capability is present. This is a capability-membership AST, not yet an executable task world.

### 12.2 Required minimal design

Do not build a general interpreter. Implement one small, deterministic typed execution domain sufficient to test primitive and composite execution:

- Primitive metadata must use concrete input/output type names rather than `any -> any`.
- Primitive nodes must consume explicit input from `input_context` and produce deterministic values.
- Program execution must fail if the required primitive capability is unavailable.
- Sequence/composition must pass explicit intermediate values according to one documented rule.
- Condition and bounded loop semantics must consume and return explicit values.
- Separate “the state has the required capability” from “the program's computed answer.” Do not encode both as the same integer without an explicit result model.

Keep the primitive set already present. Define the smallest concrete semantics for each primitive in the code and report. Do not add user-defined primitives or dynamic registration.

### 12.3 Required tests

- Primitive execution checks exact outputs, not only `0/1` membership.
- Composite execution checks exact intermediate and final outputs.
- Missing capabilities fail explicitly.
- Invalid input types and invalid programs fail explicitly.
- Same program, state, and input produce the same answer.
- A composite task succeeds from its required primitive capabilities without reading a composite capability label from the state.

If this contract cannot be implemented without changing the frozen Phase 6 model, stop and report the exact semantic ambiguity before running Phase 6 experiments. Do not silently keep the Boolean scaffold and call the milestone complete.

## 13. Targeted Test Gate

Run after implementing the repairs:

```bash
python -m pytest -q tests/test_identifiability.py
python -m pytest -q tests/test_certificate.py tests/test_adaptive_certificate.py
python -m pytest -q tests/test_probabilistic.py
python -m pytest -q tests/test_dsl.py
python -m pytest -q
```

Record the exact command, Python version, pytest version, test count, elapsed time, and result. Do not copy historical counts into the report.

The canonical repository command for this milestone is `python -m pytest -q`. Do not add packaging infrastructure solely to make the `pytest` console entry point behave differently.

## 14. Required Revalidation Experiments

Create narrow deterministic scripts under `scripts/`. Do not add an experiment framework. Write outputs below `artifacts/phase2_6_revalidation/` and include a manifest with source commit, config, seeds, commands, and SHA-256 hashes of accepted outputs.

### 14.1 Phase 2 identifiability revalidation

Run:

- generated chain world;
- generated tree world under the accepted non-empty-child input assumption;
- generated unstructured world;
- valid artificial full-vector collision world.

Report state count, task count, unique signatures, collision groups, and identifiable status. Malformed-world tests belong in pytest and must not be counted as scientific experiment conditions.

### 14.2 Phase 3 fixed-certificate regression

Rerun exact, greedy, and random fixed certificates on the same chain/tree/unstructured worlds used in the earlier report.

Report:

- exact certificate size and validity;
- greedy and random certificate sizes and validity;
- seed for random baseline;
- runtime;
- confirmation that exhaustive exact search remains the accepted temporary method.

Do not claim new scaling results unless they are actually run.

### 14.3 Phase 4 adaptive regression and comparison

For each structured and unstructured world:

- run entropy, balanced-split, and random policies;
- run random seeds `0..99`;
- validate every returned tree independently;
- report valid-run rate, average depth, worst-case depth, node count, and fixed-vs-adaptive certificate cost.

Any invalid result on an identifiable world is a failed experiment condition and must be investigated before acceptance.

### 14.4 Phase 5 robustness experiments

Use fixed world definitions and deterministic seeds. At minimum use one chain, one tree, and one unstructured world with the exact definitions recorded in the config.

Noise conditions:

```text
(slip, guess)
(0.00, 0.00)
(0.05, 0.05)
(0.10, 0.10)
(0.20, 0.20)
(0.30, 0.30)
(0.30, 0.05)
(0.05, 0.30)
```

For every condition run seeds `0..99` and compare:

- fixed certificate with repeated attempts `1, 3, 5`;
- adaptive certificate with repeated attempts `1, 3, 5`;
- MAP accuracy;
- mean posterior confidence;
- threshold reach rate;
- average and worst-case query count;
- average total response count.

Required sanity checks:

- zero-noise results agree with deterministic Phase 3/4 behavior;
- increasing symmetric noise must not be reported as improving evidence without an explicit statistical explanation;
- sequential and batch posterior calculations agree on sampled traces;
- fixed/adaptive comparisons use the same worlds, true-state sampling, noise conditions, and seed set.

Generate machine-readable CSV or JSON summaries and concise Markdown tables. Plotting is optional; the numeric curve data is mandatory.

### 14.5 Phase 6 executable DSL experiments

Run and report:

1. Primitive execution with concrete inputs and exact expected outputs.
2. Composite execution requiring multiple primitives.
3. Held-out composition: construct and execute a new composition from already defined primitives without adding a dedicated composite capability label. Describe this as deterministic structural composition, not learned generalization.
4. Certificate transfer: compare a matched direct-task world and DSL-generated world using the same underlying capability states. Report validity and certificate cost, and explain any non-equivalence.
5. Reproducibility: rerun the same DSL conditions and compare canonical outputs.

Do not use noisy callbacks with exact certificate solvers.

## 15. Reports and Corrections

Update affected phase reports where earlier statements are no longer accurate:

```text
phase2_report.md
phase4_report.md
phase5_report.md
phase6_report.md
```

Create:

```text
phase2_6_revalidation_report.md
```

The revalidation report must contain:

- baseline commit and repaired commit/diff identifier;
- confirmed defects and their scientific impact;
- exact repair for each defect;
- tests actually run and real results;
- experiment definitions, seeds, and artifact paths/hashes;
- corrected results;
- remaining limitations;
- explicit statement that superseded Phase 5 adaptive results are not evidence.

Do not rewrite old results as though they had originally been correct. Distinguish historical result, invalidation reason, and replacement result.

## 16. Acceptance Criteria

This handoff is complete only when all of the following hold:

1. Sequential and batch Bayesian updates are equivalent and covered by exact-value tests.
2. Long possible observation sequences do not collapse through numerical underflow.
3. Identifiability rejects invalid, empty, duplicate, and malformed-signature inputs.
4. State IDs are unambiguous for arbitrary accepted task strings.
5. Built-in random adaptive policy succeeds for seeds `0..99` on the frozen identifiable test world.
6. Adaptive result serialization contains the complete decision tree.
7. Exact DSL signatures are deterministic and composition conflicts fail early.
8. DSL execution consumes concrete input and returns concrete deterministic answers.
9. All targeted and repository-wide tests pass with real recorded counts.
10. Phase 2-6 experiments are rerun only from repaired code with frozen configs and seeds.
11. Reports clearly invalidate superseded evidence and bind replacement artifacts.
12. An independent strict read-only agent review finds no unresolved high-severity correctness or protocol defect.

## 17. Independent Review Gate

Before commit, freeze the intended diff and stop all writers. Spawn an independent strict read-only review agent with no conversation history (`fork_context=false` or equivalent). Provide only:

- this handoff path;
- baseline commit and frozen diff;
- changed paths;
- acceptance criteria;
- protocol constraints;
- tests and experiments already run;
- artifact manifest and hashes.

The reviewer must prioritize:

- Bayesian update equivalence;
- numerical stability;
- vacuous or silently filtered scientific inputs;
- unique state identity;
- adaptive tree validity;
- deterministic versus stochastic solver contracts;
- DSL input/output semantics;
- experiment comparability and result overclaiming.

Resolve every confirmed high-severity finding before requesting commit approval. Do not let the implementation agent self-accept its own work.

## 18. Return Format

Return to `main`:

```text
Changed files:
Tests run and exact results:
Experiments run and exact results:
Artifact paths and SHA-256 hashes:
Historical results invalidated:
Acceptance criteria status:
Independent review findings and resolutions:
Remaining risks or blockers:
```

Do not commit until `main` has reviewed this return package and explicitly opened the commit gate.
