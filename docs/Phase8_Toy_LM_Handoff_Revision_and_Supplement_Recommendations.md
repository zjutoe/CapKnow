# Phase 8 Toy Language Model Bridge
## Handoff 修正与补充建议（供 Codex 参考）

## 0. 文档定位

本文件是对以下 master handoff 的修订备忘录：

```text
Capability_Certificate_Task_Handoff_010_Phase8_Toy_Language_Model_Bridge.md
```

用途：

- 为 `main`、Codex 执行器和独立科学 reviewer 提供明确的修订清单；
- 区分必须先解决的阻断项与非阻断补充项；
- 将 master scientific protocol 拆解为 `gpt-5.5` 执行模型可以逐步实现、测试和交付的工程阶段；
- 防止再次出现“正式协议先于计算可行性和最小实现验证”的问题。

本文件不是新的科学 authority，也不自动修改 master handoff。Codex 不得自行把建议静默合入正式协议。正确流程是：

1. 按本文件提出 source-controlled 修改方案；
2. 返回精确 diff、测试和未决问题；
3. 由 `main` 审核；
4. 由 fresh-context `gpt-5.6-sol` 进行科学复审；
5. 接受后再冻结新的 Phase 8 handoff commit。

---

# 1. 总体评价

当前 Phase 8 handoff 的科学设计已经明显成熟，尤其包括：

- one-checkpoint-per-state；
- executable DSL ground truth；
- 明确的 state-specific training supervision；
- held-out composition；
- A/B/C corpus controls；
- evaluator isolation；
- full-information identifiability before certificate；
- exact fixed/adaptive certificate validation；
- protocol gates 与 empirical outcomes 分离；
- conservative statistical semantics；
- Git、checksum、manifest 和 immutable output root。

因此，master handoff 可以保留为 Phase 8 的总科学协议。

但当前版本仍不适合直接作为一个单体任务交给单一 Codex 执行器完整实现和运行。正式实施前应解决下列 P0 问题。

---

# 2. P0：实施前必须修正的事项

## P0-1：明确 master protocol 不得作为单体工程任务直接执行

### 问题

当前 handoff 同时要求实现：

- DSL binding；
- corpus generator；
- leakage oracle；
- randomized control；
- tokenizer；
- Transformer；
- training；
- evaluator；
- certificate integration；
- metric aggregation；
- formal runner；
- checksums/manifests；
- 288 次正式训练。

即使条款清楚，单次交给一个执行模型仍容易：

- 遗漏少数 invariant；
- 只实现 happy path；
- 提前实现后续模块；
- 将 empirical failure 当作 engineering failure；
- 修改已冻结的研究定义；
- 在 runner/provenance 层出现难以发现的错误。

### 必须修改

在 Section 12 前增加：

```text
This handoff is the master scientific protocol.
It must not be implemented as one monolithic Codex task.
Implementation must proceed only through the accepted sub-handoffs listed below.
Each sub-handoff has an independent stop point, test suite, review gate, and return.
```

建议拆成：

| Sub-handoff | 工程范围 |
|---|---|
| Task 010A | DSL oracle、八任务 program、ground-truth matrix、probe-pack schema |
| Task 010B | A/B/C corpus generator、split oracle、leakage checks、randomized control |
| Task 010C | byte tokenizer、Transformer、独立 sequence-transduction feasibility suite |
| Task 010D | training loop、checkpoint、greedy generation、evaluator isolation |
| Task 010E | behavioral matrix、identifiability、fixed/adaptive certificate integration |
| Task 010F | metrics、threshold sensitivity、aggregation、report reconstruction |
| Task 010G | formal sharded runner、status、manifest、checksum、provenance |
| Task 010H | resource benchmark、capacity acceptance、formal-run authorization |

### 验收要求

- 每个 sub-handoff 只能修改预先列出的 paths；
- 每个 sub-handoff 完成后必须停止；
- 不允许提前实现下一个 sub-handoff；
- 每个阶段返回 changed files、tests、deviations、blockers；
- 科学定义冲突时必须 stop-and-report，不得自行决策。

---

## P0-2：保留现有四样本 overfit test，但增加独立的 held-out sequence-transduction feasibility gate

### 问题

当前 `Capacity smoke control` 只要求：

> 小模型在 500 步内记住四条 byte-copy records。

这只能证明：

- forward/backward 可运行；
- response-only mask 大致正确；
- 模型能记忆极小训练集。

但正式任务要求模型在未见 payload 上：

- 定位随机 16-hex operand；
- 复制或变换字符串；
- 从结构化 prompt 中提取字段；
- 输出 exact compact JSON；
- 在 64-token generation 限制内完全正确。

如果正式模型失败，仅靠现有 smoke 无法区分：

1. capability structure 未学习；
2. primitive 未学习；
3. held-out string transduction 未学习；
4. exact JSON generation 失败；
5. architecture/training budget 本身不足。

### 必须新增

增加一个完全独立于 Phase 8 DSL task 的非科学 feasibility suite。

建议包含四个 task families：

1. **Held-out byte copy**
   - prompt 中出现一个随机 16-hex string；
   - response 必须精确复制该字符串。

2. **Field extraction**
   - prompt 中包含若干 key-value fields；
   - response 返回指定 field 的 compact JSON value。

3. **Boolean mapping**
   - 输入布尔条件；
   - 输出 canonical `true` 或 `false`。

4. **Compact JSON emission**
   - 输入若干随机 operands；
   - 输出固定 schema 的 compact JSON array/object。

约束：

- 不得包含 Phase 8 task IDs、DSL programs、templates 或 payloads；
- train/eval operands 和 templates 必须分离；
- 使用 Phase 8 相同 tokenizer、model implementation、optimizer family、generation path；
- 配置必须 source-controlled；
- 结果属于 feasibility evidence，不属于 Phase 8 scientific evidence；
- 不允许根据 Phase 8 正式结果修改该 suite。

### 未决决策

Codex 不得自行决定 feasibility pass threshold。

必须由 `main` 与 SOL reviewer 在独立 mini-contract 中冻结：

- train/eval record 数；
- training steps；
- exact-match threshold；
- small/medium 哪些配置必须通过；
- 失败后是否允许修改 architecture 或 training budget。

### 建议修改位置

- 将原 Section 9.3 改名为：
  `Software overfit smoke control`
- 新增 Section 9.4：
  `Held-out sequence-transduction feasibility control`

---

## P0-3：在正式 288-run grid 前增加非科学 resource benchmark

### 问题

正式 grid 包含：

\[
288 \times 1500 = 432000
\]

个 optimizer steps。

每个 run 在 5 个 checkpoints 上评估 512 prompts：

\[
288 \times 5 \times 512 = 737280
\]

次 prompt generations。

最大 generation-token allowance 约为：

\[
737280 \times 64 = 47185920
\]

此外还有 checkpoint、raw generation、corpus、logs 和 manifest。

当前 handoff 没有测量：

- 一个完整 small/base shard 的 wall time；
- medium/large 的 wall time；
- 512 prompts × 5 checkpoints 的 generation cost；
- peak RSS/VRAM；
- checkpoint 和 raw generation 大小；
- 全 grid 的保守存储与运行上界。

这会重演“协议冻结后才发现计算不可行”的风险。

### 必须新增

在 formal execution permission 前运行 source-controlled、non-evidence resource benchmark。

建议至少有两个 benchmark fixtures：

```text
benchmark-small-base
benchmark-medium-large
```

要求：

- 使用与正式实现相同的 model、training loop、checkpoint、generation 和 artifact path；
- 使用独立 dummy corpus，不使用 Phase 8 scientific outcomes；
- workload 在 batch、sequence length、training steps、checkpoint count、generation count 上匹配代表性正式 shard；
- 不输出或解释 Phase 8 mastery/certificate 指标。

必须记录：

- training wall time；
- evaluation wall time；
- CPU time；
- peak RSS；
- peak VRAM；
- checkpoint bytes；
- raw generation bytes；
- final artifact bytes；
- per-stage breakdown；
- estimated shard cost；
- estimated complete-grid cost。

建议保守外推：

```text
estimated_total = measured_or_modelled_total * 1.25
```

### 正式授权前必须冻结

- concurrency；
- wall-time ceiling；
- GPU-hour ceiling；
- peak VRAM ceiling；
- peak RSS ceiling；
- temporary disk ceiling；
- final disk ceiling。

资源 ceiling breach 属于 operational failure，不属于 scientific outcome。

---

## P0-4：将单一 all-or-nothing formal root 改为预声明 immutable shards

### 问题

当前 handoff 使用一个：

```text
artifacts/phase8_toy_lm_bridge/formal_001
```

运行全部 288 个 trainings。

如果第 280 个 run 因临时 GPU、磁盘或进程错误失败：

- 已完成结果不能静默 resume；
- root 必须保留；
- 可能被迫重跑整个 grid。

这对长周期实验不合理。

### 必须修改

预先冻结 18 个原子 scientific shards，每个 shard 包含：

- 一个 condition/scale/seed；
- 全部 16 个 state checkpoints；
- 全部 5 个 evaluation steps；
- 自己的 `status.json`、`progress.json`、`DONE.json` 或 `FAILED.json`；
- 自己的 shard manifest。

#### Core shards：9 个

```text
core_A_small_base_seed0
core_A_small_base_seed1
core_A_small_base_seed2

core_B_small_base_seed0
core_B_small_base_seed1
core_B_small_base_seed2

core_C_small_base_seed0
core_C_small_base_seed1
core_C_small_base_seed2
```

#### Scale shards：9 个

```text
scale_A_small_large_seed0
scale_A_small_large_seed1
scale_A_small_large_seed2

scale_A_medium_base_seed0
scale_A_medium_base_seed1
scale_A_medium_base_seed2

scale_A_medium_large_seed0
scale_A_medium_large_seed1
scale_A_medium_large_seed2
```

`A/small/base` 与 core 共享，不重复训练。

### 推荐路径

```text
artifacts/phase8_toy_lm_bridge/formal_001/
    shards/
        core_A_small_base_seed0/
        ...
    aggregate/
    shards_manifest.json
    manifest.json
```

### 运行与失败语义

- 每个 shard 是不可覆盖的原子运行单元；
- top-level aggregator 只读取 `DONE` shards；
- 任一 required shard 失败，最终 scientific aggregation 必须 block；
- 已完成 shard 保留，不得重跑；
- 失败 shard 的 rerun 必须使用新 attempt root，例如：
  `core_A_small_base_seed0_attempt002`；
- rerun 需要新的显式授权和 manifest binding；
- 不允许删除或替换失败证据。

这不是 silent resume，而是预注册的分片式正式运行。

---

## P0-5：加入 predeclared scientific interpretation hierarchy

### 问题

若模型连 primitives 都没有学会，则：

- `MEMORY_SEARCH` 失败不能被解释为 composition generalization 失败；
- matrix collision 不能直接解释为 knowledge structure 不存在；
- certificate 不存在不能直接解释为 certificate hypothesis 失败。

当前 handoff虽要求报告 primitive、seen composition 和 held-out composition，但还需要明确的解释顺序。

### 必须新增

在 Section 10 或 Section 17 增加：

```text
Interpretation hierarchy
```

#### Level 1：Primitive acquisition

报告：

- primitive positive-cell success；
- primitive false positives；
- primitive refusals；
- relevant parent primitives `MEMORY` 与 `SEARCH` 的表现。

若 primitives 本身未形成稳定行为，held-out composition 结果必须描述为：

```text
held-out composition not cleanly interpretable because prerequisite primitive behavior was not established
```

不得描述为：

```text
the model failed compositional generalization
```

#### Level 2：Seen composition acquisition

若 primitives 已有行为，但三种 seen composition 均较差：

结论只能是：

```text
the model did not establish stable seen-composition behavior
```

#### Level 3：Held-out composition

只有在同时报告 primitive 和 seen-composition 行为的情况下，才能解释 `MEMORY_SEARCH`。

不得只展示 held-out task 的单一数字。

#### Level 4：Behavioral identifiability

matrix collision 表示：

```text
this frozen binary probe family does not uniquely identify the 16 checkpoints at this threshold
```

不得直接表述为：

```text
the models have no structured capability states
```

#### Level 5：Certificate transfer

只有 identifiable matrix 才允许 fixed/adaptive certificate。

certificate mismatch 只说明：

```text
the neural behavioral certificate does not match the ground-truth certificate under this frozen probe family and threshold
```

不得外推到真实 LLM 或一般知识结构。

### 重要说明

以上是 report interpretation conditions，不是新的 engineering acceptance gates，也不是为了让 scientific outcome “通过”。

---

## P0-6：明确 Phase 8 的 ground-truth certificate 是被设计成 primitive-only 的

### 问题

当前 16 states 是四个 primitive 的完整 powerset，并且存在四个 primitive probes。

空状态和每个 singleton state 形成单坐标 witness，因此：

- 四个 primitive probes 都是必要的；
- 唯一最小 fixed certificate 必然是：
  `MEMORY, SEARCH, FILTER, CONDITION`；
- certificate size 必然是 4。

因此 Phase 8 主要测试：

> 训练后的 checkpoint population 是否保留一个预先设计的 primitive-state certificate。

它不测试：

> composition 是否使 certificate 变得更小。

`MEMORY_SEARCH` 检验的是 held-out composition behavior，而不是 ground-truth certificate 压缩来源。

### 必须新增到 report boundaries

```text
The ground-truth fixed certificate is intentionally forced by the four-bit powerset state design and the four primitive probes. Phase 8 tests whether this designed primitive-state certificate transfers to neural behavior. It does not test whether composition creates a smaller certificate than primitive probing.
```

对应中文报告必须明确：

> Phase 8 的 certificate transfer 是结构保持实验，不是 composition-induced certificate compression 实验。

---

# 3. P1：建议补充但不阻断实施的事项

## P1-1：记录 A/C 的 per-state target-length 与 answer-token 统计

### 原因

Condition C 保持：

- 每个 state 的 composite positive 数；
- 每个 prompt 的 positive state 数；
- aggregate byte histogram；
- aggregate token histogram。

但不保证：

- 每个 state 内 answer-token identity；
- 每个 state 内 answer length distribution。

因此 A–C 差异可能部分混入 per-state target difficulty。

### 建议新增诊断

按 condition/state/task family 记录：

- response bytes 总数；
- response tokens 总数；
- mean/median/min/max target length；
- positive target length；
- `unable` 比例；
- A 与 C 的 paired target-length difference。

这些指标：

- 只作诊断；
- 不作为 acceptance gate；
- 不用于 post hoc 选择结果。

---

## P1-2：把 large corpus 明确命名为 fixed-compute data-diversity comparison

Base 和 large 都使用：

```text
training_steps=1500
batch_size=64
```

因此：

- large corpus 提供更多 unique instances；
- 但每条 instance 的平均 exposure 更少；
- 不是纯粹的“更多训练 token”实验；
- 不是 scaling-law experiment。

建议报告用语：

```text
fixed-compute corpus-diversity comparison
```

避免称为：

```text
data scaling result
```

---

## P1-3：增加 primary/sensitivity response matrix stability 指标

现有 thresholds：

```text
32/64
48/64
52/64 primary
64/64
```

建议在每个 population/step 额外报告：

- sensitivity matrix 与 primary matrix 的 Hamming distance；
- 每个 threshold 的 row collision 数；
- 每个 threshold 是否 identifiable；
- 每个 threshold 的 certificate existence；
- certificate size 与 task-set overlap；
- primary certificate 对 threshold 的稳定性。

这些只作预声明 sensitivity analysis，不得选择最有利 threshold 作为主结果。

---

## P1-4：区分 task-family query 和 raw generation cost

现有 handoff 已规定：

- 一个 certificate query 是 64-prompt task-family battery；
- 不是一次 generation。

实现和报告中应同时输出：

```text
family_query_cost
raw_generation_cost = family_query_cost * 64
```

并避免：

- 将 family-query 数与单 prompt API 调用成本直接比较；
- 将 adaptive depth 误称为模型调用次数。

---

# 4. 对 master handoff 各 section 的建议修改位置

## Section 9

保留：

```text
9.3 Software overfit smoke control
```

新增：

```text
9.4 Held-out sequence-transduction feasibility control
9.5 Non-scientific resource benchmark
```

---

## Section 10

新增：

```text
10.4 Interpretation hierarchy
```

包含：

- primitive；
- seen composition；
- held-out composition；
- identifiability；
- certificate。

---

## Section 12

在 implementation paths 前新增：

```text
Engineering decomposition and mandatory sub-handoffs
```

列出 Task 010A–010H。

---

## Section 14

将单 root formal run 改成：

- immutable shard inventory；
- shard-level terminal state；
- top-level aggregate manifest；
- failed-shard versioning；
- no overwrite/no deletion。

---

## Section 15

增加 implementation/protocol gates：

- sequence-transduction feasibility accepted；
- resource benchmark accepted；
- resource ceilings frozen；
- all required shards present and valid；
- aggregate manifest only after shard closure。

这些仍然不能把 scientific outcome 变成 acceptance gate。

---

## Section 16

增加 expected evidence：

```text
artifacts/phase8_toy_lm_bridge/feasibility_001/
artifacts/phase8_toy_lm_bridge/benchmark_001/
artifacts/phase8_toy_lm_bridge/formal_001/shards_manifest.json
artifacts/phase8_toy_lm_bridge/formal_001/aggregate/summary.json
artifacts/phase8_toy_lm_bridge/formal_001/manifest.json
```

---

## Section 17

增加：

1. ground-truth certificate 是由 powerset + primitive probes 强制产生的；
2. Phase 8 不测试 composition-induced certificate compression；
3. large corpus 是 fixed-compute diversity comparison；
4. held-out composition claim 必须连同 primitive 与 seen-composition evidence 一起解释；
5. A/C 未匹配 per-state answer-token identity，相关统计只作诊断。

---

# 5. 推荐工程执行顺序

```text
Task 010A
DSL oracle and immutable evaluation pack
        ↓
fresh scientific review
        ↓
Task 010B
Corpus A/B/C and leakage/control validation
        ↓
fresh scientific review
        ↓
Task 010C
Tokenizer/model + independent transduction feasibility
        ↓
feasibility decision
        ↓
Task 010D
Training/checkpoint/evaluator
        ↓
Task 010E
Behavioral matrix and certificate integration
        ↓
fresh scientific review
        ↓
Task 010F
Metrics/aggregation/report reconstruction
        ↓
Task 010G
Sharded formal runner and provenance
        ↓
Task 010H
Resource benchmark and formal authorization
        ↓
Formal scientific shards
        ↓
Aggregate and final scientific review
```

---

# 6. Codex 不得自行决定的事项

遇到下列情况必须 stop-and-report：

1. feasibility threshold 尚未冻结；
2. resource ceilings 尚未冻结；
3. shard inventory 与 master grid 不一致；
4. independent transduction suite 失败；
5. medium/large benchmark 超出资源预算；
6. A/C control 无法达到 degree-preserving constraints；
7. train/eval semantic payload split 出现 collision；
8. evaluation prompt 超出 256-token contract；
9. deterministic PyTorch operation 无法满足；
10. accepted Phase 2–7 API 与 Phase 8 要求冲突；
11. scientific result无法解释，但协议和实现本身未失败。

Codex 不得：

- 自动增加 training steps；
- 自动改变 model size；
- 调整 mastery threshold；
- 放宽 exact-match evaluator；
- 删除失败 shard；
- 合并 seeds；
- 用 logits 修复 collision；
- 改动 held-out composition；
- 为获得 positive result 修改 corpus。

---

# 7. Codex 返回格式

每个修订或 sub-handoff 完成后返回：

```text
Scope completed:
Changed paths:
Scientific contract changes:
Engineering-only changes:
Tests run and exact results:
New artifacts:
Unresolved decisions:
Protocol deviations:
Resource observations:
Independent review required:
Recommended next authorized step:
```

若失败：

```text
Problem:
Minimal reproduction:
Evidence:
Likely cause:
Scientific or engineering classification:
Decision required from main:
Work explicitly not attempted:
```

---

# 8. 修订后的总体 verdict

## 作为 master scientific protocol

```text
PASS WITH CONDITIONS
```

## 作为立即交给单一执行模型的 implementation task

```text
REJECT
```

直至完成：

1. sub-handoff 拆分；
2. independent sequence-transduction feasibility control；
3. resource benchmark；
4. immutable sharded formal execution；
5. interpretation hierarchy；
6. certificate-scope boundary。

完成以上修订后，Phase 8 将更适合逐步实现，也能避免正式实验失败后无法判断究竟是：

- scientific hypothesis failure；
- primitive acquisition failure；
- sequence-transduction failure；
- resource infeasibility；
- orchestration/provenance failure。
