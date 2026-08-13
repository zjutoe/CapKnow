# Phase 2-6 GPT-5.3 与 GPT-5.5 修订质量对比

## 目的

本文比较 GPT-5.3 与 GPT-5.5 分别执行
`Capability_Certificate_Task_Handoff_007_Correctness_Repair_and_Revalidation.md`
后的工程质量，重点关注复杂跨阶段修订中的正确性、实验协议、证据可信度和返工风险。

这是一组同项目、同基线、同 handoff 下的案例对比，不足以代表两个模型在所有任务上的一般能力。

## 审查记录归属

当前仓库包含三个需要区分的审查记录：

| 记录 | 被审查对象 | 结论 | 证据位置 |
| --- | --- | --- | --- |
| A | GPT-5.3 执行的第一版修订 | `REJECT` | handoff 顶部的 `Independent Review Result (2026-08-08)` |
| B | GPT-5.5 重做并修复多轮审查问题后的版本 | `ACCEPT` | `phase2_6_revalidation_report.md` 的 `Independent Review` |
| C | 对 GPT-5.5 版本 `source_snapshot_sha256=5ba7640055d3f2b2790b29fe10bf4a0a901b30358c12d637a9ec766c85d096ab` 进行的 fresh-context 严格审查 | `REJECT` | 本文记录的 2026-08-09 审查结果 |

记录 B 不能作为 GPT-5.3 实现的审查结论：它引用了修订后的
`source_snapshot_sha256=0a4f9fb3b2e6a2b02a63dea938a7c248253574b838194cd03ce42489b4af65cd`，
并明确声称 GPT-5.3 审查中发现的 provenance、adaptive validation、DSL 和报告问题已经修复。

因此，模型实现质量的主要对比是 A 与 C；B 与 C 的差异则用于衡量独立 reviewer
对同一 GPT-5.5 修订包的漏检风险。记录 C 是历史审查对象，不指向后续
`source_snapshot_sha256=fa374cc1bf3e7b31ae442cf91e35467d9cab41f6fdedbca84488cae341bf361a`
修复包。

## 总体结果

| 维度 | GPT-5.3 修订（记录 A） | GPT-5.5 修订（记录 C） |
| --- | --- | --- |
| 最终 gate | `REJECT` | `REJECT` |
| 确认发现 | 1 High、6 Medium、2 Low | 1 High、3 Medium、2 Low |
| 核心算法与数据模型 | 仍有 adaptive tree validator 等语义缺陷 | handoff 的主要算法修复通过审查 |
| DSL | 缺失 `LOOP` enforcement，且会把 malformed program 伪装成 capability failure | 上述问题已修复；仍有 capability value 和 loop count 的类型边界缺口 |
| Phase 5 | 12 个 sequential/batch 条件失败，最低一致率 0.86，报告却声称全部通过 | fixed 路径一致性通过，但 adaptive 路径没有被独立 batch recomputation gate 覆盖 |
| 实验 provenance | 五个 manifest 绑定了错误 source/command，所有实验不能从声明来源复现 | source snapshot、命令、dirty state、输出和 manifest hash 均核对通过 |
| Phase 6 evidence | held-out composition 实际已存在于默认规则 | held-out composition 已修正，但脚本仍缺少冻结 expected-output oracle |
| 报告可信度 | 存在结果与报告直接矛盾、metric 语义误标 | 数值与 JSON 对齐；最新 `REJECT` 使原 `ACCEPT` 状态变为过期 |

两版都没有达到无需返工即可提交的标准，但失败性质不同。GPT-5.3 的问题包括已经发生的
语义错误、错误 provenance 和与产物矛盾的科学结论；GPT-5.5 的剩余问题主要集中在验收
gate 不够强，以及少数输入边界没有 fail early。

## GPT-5.3 修订的审查结果

### High

- 五个 revalidation manifest 声称实验来自 baseline commit，但实际导入的是未提交工作树；
  manifest 还漏记了真实命令中的 `PYTHONPATH=.`。输出 hash 只能证明输出字节未变，不能证明
  这些输出由声明的 source 产生。该问题使整套重验证证据失去可复现来源。

### Medium

1. `validate_adaptive_certificate` 可以接受不完整、无进展的 decision tree。
2. `LoopNode` 声明需要 `LOOP`，executor 却不检查该 capability。
3. DSL signature generation 把所有 `InvalidProgramError` 转为响应 `0`，混淆 malformed program
   与 capability absence。
4. Phase 5 有 12 个 sequential/batch MAP consistency 条件低于 1.0，最低 0.86，报告仍声称全部满足。
5. Phase 4 把 worst-case `query_count` 聚合后标成 average query depth。
6. Phase 6 的 held-out `MEMORY -> SEARCH` 已存在于默认 `RETRIEVAL` 规则，不是真正 held-out。

### Low

- 缺少 too-long response signature 回归测试。
- adaptive serialization 测试使用自比较，无法发现对称结构被同时遗漏。

### 质量判断

GPT-5.3 修订解决了部分关键问题，包括 Bayesian history double-counting、log-space posterior、
canonical state ID 和 composition collision，但没有稳定维持跨模块契约。最严重的问题不是普通
边界 bug，而是实验来源声明错误以及报告结论与 artifact 直接矛盾。这会让维护者在测试通过的
情况下接受错误科学证据，属于高返工、高误导风险。

## GPT-5.5 修订的最新严格审查结果

本次审查使用与主线程相同的模型，`fork_context=false`，只提供 handoff、冻结 working-tree
change、artifact、验收标准和已完成验证。审查期间主线程保持只读，未重跑测试或实验。

### High

1. `scripts/revalidation_phase5_robustness.py:202` 只对 fixed traces 检查 sequential/batch
   consistency；从 `scripts/revalidation_phase5_robustness.py:224` 开始的 adaptive runs 没有等价
   检查。零噪声 gate 只比较 MAP。原始 double-counting defect 即使复发，也可能保持 MAP 不变，
   同时错误抬高 confidence 并提前停止。

   最小修正是展平每个 adaptive `query_history`，从原始 prior 重新计算 batch posterior，比较完整
   posterior、MAP 和 confidence，将失败纳入 gate，并重跑 Phase 5。

### Medium

1. `capability_certificate_lab/dsl/executor.py:125` 对 capability value 使用 `bool()`；
   `{"ADD": "false"}` 会被当成具备 `ADD`。应只接受真实 `bool` 并补回归测试。
2. `scripts/revalidation_phase6_dsl.py:21` 冻结了 held-out input，却没有冻结 expected outputs；
   `scripts/revalidation_phase6_dsl.py:79` 只记录 executor 返回值。primitive、composite、held-out、
   transfer 和 reproducibility 结果都应有明确 oracle，失配时脚本必须失败，并重跑 Phase 6。
3. `phase2_6_revalidation_report.md` 仍记录上一轮 `ACCEPT`。该记录在当时是一次真实历史审查，
   不是凭空预写；但本轮发现确认缺陷后，它已经不能代表当前 gate 状态。

### Low

1. `capability_certificate_lab/dsl/executor.py:108` 只检查 `iterations <= 0`，没有要求非布尔正整数。
   `1.5` 会在 `range()` 中偶发式失败，`True` 会执行一次。
2. `capability_certificate_lab/probabilistic/response_model.py:48` 使用 `random() <= p_yes`。
   当 `p_yes == 0` 且 RNG 返回 `0.0` 时，会生成数学上不可能的正响应；应使用 `< p_yes`。

### 已确认通过的部分

- incremental Bayesian update、log-space normalization 和 impossible-observation semantics 正确。
- identifiability validation、canonical state identity、adaptive splitting、recursive tree validation
  和完整 serialization 与 handoff 一致。
- malformed DSL propagation、composition collision detection、deterministic signatures 和
  `LOOP` capability enforcement 已修复。
- source snapshot 独立重算一致；五个 manifest 的命令、配置、Python 版本、dirty state 和 hash
  完整；十个 JSON 文件及报告引用 hash 一致。
- Phase 2-6 报告中的数值与 JSON evidence 对齐，历史无效结果已明确标记。

### 质量判断

GPT-5.5 相比 GPT-5.3 显著降低了核心实现错误和 evidence provenance 风险。当前审查没有发现
handoff 中主要数学修复失效，也没有发现报告数值与 artifact 再次矛盾。

但 GPT-5.5 仍把部分“运行成功并记录输出”当成了“由独立 oracle 验收”。这在普通业务代码中
可能只是测试覆盖不足，在本项目中会直接削弱 scientific revalidation：adaptive posterior 的
confidence 与 stopping behavior 没有被独立重算，Phase 6 输出也没有由冻结期望值 gate。
因此当前 `REJECT` 是合理的提交阻断，而不是形式性意见。

## 两个模型在本任务复杂度下的表现差异

### 1. 跨模块一致性

GPT-5.3 可以完成局部修复，但在 validator、executor、simulator、experiment script 和 report
之间留下多处契约断裂。GPT-5.5 对 handoff 的跨模块追踪明显更完整，主要核心修复已形成一致闭环。

### 2. 科学证据意识

GPT-5.3 的 provenance 错误和 Phase 5 报告矛盾会导致错误结果被当作有效证据，这是本次对比中
最严重的质量差异。GPT-5.5 正确建立了 source snapshot、command、config 和 output hash 绑定，
也正确废弃旧结果，但没有把所有关键语义转成可失败的 acceptance oracle。

### 3. 测试与验收设计

GPT-5.3 的测试缺口包括 handoff 明确要求的具体断言。GPT-5.5 增加了更完整的 regression tests，
但仍漏掉 adaptive-path posterior equivalence 和 Phase 6 expected-output gates。这说明 GPT-5.5
更擅长修复已明确指出的 defect，面对“实验如何证明实现正确”时仍需要独立 reviewer 施加压力。

### 4. 剩余错误的严重程度

GPT-5.3 留下的是已发生的错误行为和误导性结论；GPT-5.5 留下的高风险项主要是回归可能逃逸
当前 gate。后者仍然必须修复，但其实现成熟度和证据可信度高于前者。

## 对同一 GPT-5.5 包出现 `ACCEPT` 与 `REJECT` 的解释

`phase2_6_revalidation_report.md` 中的 reviewer 主要 spot-check 已知修复和 provenance，未发现确认
缺陷，因此给出 `ACCEPT`。本轮 reviewer 进一步检查“每个关键 claim 是否由独立 oracle 直接
gate”，发现 adaptive Phase 5 consistency 和 Phase 6 expected outputs 没有覆盖。

这不是 source regression，而是 reviewer coverage 差异。它说明复杂科学软件不能依赖单次、
宽泛的“严格审查”标签；review prompt 必须显式要求逐条验证 claim-to-oracle mapping，尤其检查：

1. 修复是否在 fixed 和 adaptive 两条路径都被独立验证；
2. experiment script 是否会在语义错误时失败，而不只是成功写出 JSON；
3. 报告中的每个结论是否能指向一个冻结 expected value 或数学 invariant；
4. 最新 review verdict 是否替换或明确 supersede 旧 gate 状态。

## 结论

在这一次同基线、同 handoff 的复杂修订任务中，GPT-5.5 的工作质量明显优于 GPT-5.3：主要
算法和结构性契约更完整，provenance 可复现，报告与 artifact 一致，严重错误数量和性质均改善。

不过，GPT-5.5 仍未达到可以跳过独立审查的水平。它的剩余缺陷集中在 scientific acceptance
gate 和边界类型契约，足以阻断提交。当前建议是继续以 GPT-5.5 作为此复杂度任务的主要执行
模型，同时保留 fresh-context strict review，并把 path-specific invariant 与 expected-output
oracle 明确写入 handoff 和实验脚本。

本文比较的记录 C 提交门禁状态：`REJECT`。后续修复包的当前门禁以
`phase2_6_revalidation_report.md` 为准。
