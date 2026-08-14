# Capability Certificate 项目研究目标与阶段成果回顾

更新时间：2026-08-14

## 1. 项目研究目标

本项目研究的核心问题是：能否用尽可能少、但具有可验证区分力的任务响应，识别一个系统所处的潜在能力状态，并给出可复现的“能力证书”（capability certificate）。

项目采用受控的 knowledge space 作为实验环境。设任务全集为 \(Q\)，一个能力状态为 \(K\subseteq Q\)，所有允许状态构成 \(\mathcal K\subseteq 2^Q\)。对状态 \(K\) 执行任务 \(q\) 后得到响应 \(Y(K,q)\)。研究按以下顺序展开：

1. 能否明确生成、验证和模拟结构化能力空间；
2. 完整任务响应能否唯一识别状态；
3. 唯一识别状态最少需要哪些固定任务；
4. 根据先前回答动态选题能否降低评估成本；
5. 存在失误、猜测和随机性时，能否进行可靠的概率推断；
6. 抽象能力标签能否落到可执行语义，并支持能力组合；
7. 什么样的声明状态族能产生固定 certificate 压缩，以及这种压缩和自适应压缩有何区别。

这里的 certificate 有两种主要形式：

- 固定 certificate：所有被评估对象回答同一组任务，该任务子集足以区分全部候选状态；
- 自适应 certificate：下一任务由此前回答决定，最终决策树的每个叶节点唯一对应一个状态。

在噪声环境中，certificate 不再意味着一次观测后的逻辑唯一性，而是指在给定响应模型和先验下达到目标后验置信度的观测过程。

本项目目前是一个小规模、完全可枚举、人工定义结构的研究实验室。它研究的是能力测量方法的形式性质，还没有证明真实 LLM 的能力可以被这些状态空间准确表示，也没有开展神经模型训练或自动能力结构发现。

## 2. 核心对象之间的关系

- `Task` 是可执行或可观测的探针。
- `KnowledgeState` 是一个候选能力配置，表示哪些任务或 primitive capability 已具备。
- `KnowledgeSpace` 是研究中允许出现的全部状态及其结构约束。
- response matrix 的行是状态、列是任务、单元格是该状态对该任务的响应。
- response signature 是 response matrix 中某个状态对应的一整行或选定列上的子向量。
- certificate 是仍能让所有状态 signature 保持不同的任务集合或自适应提问策略。

因此，knowledge space 定义“可能是什么”，task 定义“可以观察什么”，response model 定义“观察如何产生”，certificate 定义“最少观察什么才能作出区分”。

## 3. 阶段成果总览

证据边界：Phase 2-4 已在 clean commit `1bfc6ced88cf3397f240c3e79d1996955e9d589f` 上正式重跑；Phase 5/6 已在 clean source repair commit `66c0efade42d85c2ca9c5eca1d3cdb4fc19e3d40` 上正式重跑，并记录在 evidence commit `1bfc6ced88cf3397f240c3e79d1996955e9d589f`。合并后的 Phase 2-6 evidence package 冻结于 `8039dcfa88a1a6a1856b19c5301bd74e1616e76c`，并于 2026-08-13 通过 fresh-context `gpt-5.6-sol` 最终严格只读科学验收。Phase 7 formal evidence/report commit `e1c3a542cede8f873832149f0f205302e088a925` 于 2026-08-14 通过 fresh-context strict read-only final scientific review。

| 阶段 | 研究问题 | 阶段结果（含历史结果） | 结论边界 |
| --- | --- | --- | --- |
| Phase 1 | 能否构造和验证结构化能力空间 | 建立 chain、tree、unstructured 世界及确定性响应模拟 | 只证明实验框架可用 |
| Phase 2 | 完整响应能否唯一识别状态 | 三个基准世界可识别；人工 collision 世界不可识别 | 只适用于声明的状态总体和任务语义 |
| Phase 3 | 最小固定任务集有多大 | chain/tree/unstructured 的最小规模分别为 4/4/3 | 当前基准没有固定任务压缩；求解器仍是穷举 |
| Phase 4 | 自适应提问能否降低成本 | structured worlds 的平均深度低于固定规模；unstructured 无优势 | 优势是小型确定性世界中的平均成本优势 |
| Phase 5 | 噪声下能否可靠推断 | 建立稳定 Bayesian 推断和 fixed/adaptive noisy assessment；通过一致性 oracle | 尚不能据此声称噪声下 adaptive 普遍优于 fixed |
| Phase 6 | 能力能否具有可执行和组合语义 | primitive、组合、held-out 组合均可确定执行；证书可转移到 primitive DSL world | 语义和组合规则仍由人工定义 |
| Phase 7 | 何时存在固定 certificate 压缩 | 两个完整 block family 的固定 certificate 只需每个 block 一个代表任务；prefix family 的自适应平均深度低于固定规模 | 只适用于人工声明的小型确定性 block worlds；已通过 final scientific review |

## 4. Phase 1：Knowledge Space Core

Phase 1 建立了后续研究的最小基础设施：任务全集、知识状态、知识空间、状态校验、结构生成器和确定性响应模拟器。第一版响应语义为：

\[
Y(K,q)=\mathbf 1[q\in K]
\]

已实现三类世界：

- chain world：能力按先修链逐步累积；
- tree world：子能力要求父能力；
- unstructured world：使用完整幂集 \(2^Q\) 作为无结构对照组。

阶段结果表明，这三类有限 knowledge space 可以被显式生成、验证并转换为 response matrix。历史阶段测试为 `6 passed`。

这一阶段没有研究 certificate，也没有证明结构一定带来评估压缩。已知输入限制是：tree generator 会忽略映射中的 `parent -> []`，当前明确假设不会输入“没有 child 的父节点”。

## 5. Phase 2：Full-information Identifiability

Phase 2 首先检查最基本的可识别性：如果询问全部任务，两个不同状态是否仍可能得到同一响应 signature。若存在这种 collision，则任何固定或自适应 certificate 都无法在当前观测语义下唯一识别状态。

当前重验证结果如下（绑定 `1bfc6ced88cf3397f240c3e79d1996955e9d589f`，合并 evidence package 已通过最终独立审查）：

| 世界 | 任务数 | 状态数 | 唯一 signature 数 | 可识别 |
| --- | ---: | ---: | ---: | --- |
| chain | 4 | 5 | 5 | 是 |
| tree | 4 | 7 | 7 | 是 |
| unstructured | 3 | 8 | 8 | 是 |
| artificial full-vector collision（自定义非单射响应函数） | 2 | 2 | 1 | 否 |

该 artifact 支持的受限结论是：在其 membership 响应语义和声明的状态总体下，三个标准世界具有 full-information identifiability。人工 collision 世界不是 membership 响应的反例，而是通过自定义 `response_signature_fn` 刻意让不同状态产生相同完整向量，用来验证审计器能够发现不可识别输入。这不是“所有结构化能力空间都可识别”的一般性证明；该结论已随合并 evidence package 通过最终独立审查。

修订后的契约把 `KnowledgeSpace.valid_states` 视为科学实验声明的总体。空总体、非法状态、重复状态以及长度或取值非法的响应 signature 都会直接失败，不再通过静默过滤制造虚假的可识别结论。状态 ID 使用规范 JSON 数组，已覆盖含逗号、引号、反斜杠和花括号的 task ID。

## 6. Phase 3：Exact Fixed Certificate

Phase 3 将固定 certificate 写成 minimum hitting set：对每一对状态 \(K_i,K_j\)，至少选择一个能让二者响应不同的任务。实现包括 exact、greedy 和 random 三种求解器，并由独立 validator 检查所选任务是否分离全部状态对。

当前 exact solver 采用按 certificate 大小枚举任务子集的穷举方案。这与 handoff 最初建议的整数规划不同，但已经被明确接受为中小规模实验的临时实现。

以下是绑定 `1bfc6ced88cf3397f240c3e79d1996955e9d589f` 的当前重验证结果，合并 evidence package 已通过最终独立审查：

- chain：最小固定 certificate 为 4，等于全部 4 个任务；
- tree：最小固定 certificate 为 4，等于全部 4 个任务；
- unstructured：最小固定 certificate 为 3，等于全部 3 个任务。

因此，这些基准没有显示固定 certificate 的任务数压缩。结构约束减少了合法状态数量，但并不自动意味着可以删掉某些固定任务；例如 chain 中相邻状态只在新增的那个任务上不同。greedy 和 random baseline 在这些实例上均能返回有效 certificate，但不能优于 exact optimum。

## 7. Phase 4：Adaptive Certificate

Phase 4 把固定任务集合扩展为决策树。每一步维护当前候选状态集合，只允许选择尚未询问且能把候选集合分成两个非空分支的任务，直到叶节点只剩一个状态。

当前重验证结果如下（绑定 `1bfc6ced88cf3397f240c3e79d1996955e9d589f`，合并 evidence package 已通过最终独立审查）：

| 世界 | 固定最小规模 | 策略 | 平均深度 | 最坏深度 | 有效运行率 |
| --- | ---: | --- | ---: | ---: | ---: |
| chain | 4 | balanced / entropy | 2.4000 | 3 | 1.0 |
| tree | 4 | balanced / entropy | 2.8571 | 3 | 1.0 |
| unstructured | 3 | balanced / entropy | 3.0000 | 3 | 1.0 |

在该 artifact 中，random policy 在 100 个种子上也全部有效，但 chain 和 tree 的平均深度更高，最大最坏深度为 4。

该 artifact 支持的受限结构性推论是：在 chain 和 tree 这两个结构化世界中，自适应提问降低了按状态等权计算的平均问题数；在完整幂集构成的 unstructured world 中，自适应方法没有获得同样优势。它为“结构先验可以被自适应评估利用”提供了实验支持，但只适用于当前的小规模、确定性、状态等权设置；该结论已随合并 evidence package 通过最终独立审查。

validator 已修订为递归检查两个分支、候选集合是否严格缩小、任务是否重复以及叶节点是否唯一。报告也已把平均深度和最坏深度分开，废弃了曾将 worst-case query count 描述为平均成本的旧表述。

## 8. Phase 5：Probabilistic Certificate

Phase 5 引入 slip 和 guess：已具备能力时仍可能答错，未具备能力时也可能猜对。系统基于观测历史计算状态后验，并支持固定任务和基于期望熵下降的自适应查询。

本阶段首先完成的是推断语义和验证闭环：

- likelihood 在 log space 中计算并用 log-sum-exp 归一化；
- 数学上不可能的观测保持显式零后验，不进行平滑掩盖；
- adaptive update 只把新观测作为新证据，不重复计算历史；
- sequential posterior 与从完整历史重新计算的 batch posterior 一致；
- MAP 使用统一且确定的 tie rule；
- 零噪声 fixed/adaptive MAP 与确定性结果一致；
- 零噪声 adaptive query count 对每个状态都与确定性 entropy tree 的叶深度一致。

实验覆盖 chain、tree、unstructured，7 组 slip/guess，100 个随机种子，每个任务 1、3、5 次尝试。全部 fixed/adaptive consistency rate 均为 `1.0`，后验差异最大值为 `1.5543122344752192e-15`，低于 `1e-12` gate。

在对称高噪声 `slip=guess=0.30`、每个任务尝试 5 次时，fixed 方法的 MAP accuracy 分别为 chain `0.74`、tree `0.67`、unstructured `0.74`。这些是有限采样下的描述性结果，不应被解释为一般噪声定律。

Phase 5 当前支持的结论是：概率响应、Bayesian posterior、重复 probe 和 adaptive querying 已形成自洽且可执行的实验框架；零噪声极限与确定性框架一致。当前结果还不足以回答“噪声下 adaptive 是否普遍优于 fixed”或“重复 probe 是否总能以更低成本达到目标置信度”。现有曲线仅保存为数值 JSON，且抽样波动中的非单调性没有被包装成正面结论。

## 9. Phase 6：DSL Bridge

Phase 6 将抽象 capability label 落到确定性的可执行程序。当前 primitive 包括 `ADD`、`COMPARE`、`MEMORY`、`SEARCH`、`FILTER`、`LOOP` 和 `CONDITION`，并支持 sequence、condition、loop 和显式 composition rule。

主要结果包括：

- 七个 primitive 均有明确输入类型、输出类型和冻结的期望输出；
- `MEMORY` 与 `SEARCH` 可以组合执行 `SEQ[MEMORY,SEARCH]`，无需状态中额外存在 `RETRIEVAL` 标签；
- held-out 的 `FILTER + CONDITION -> FILTER_CHECK` 不在默认组合规则中，但能从 primitive capabilities 正确执行；
- pure primitive DSL world 与直接 task world 都包含 128 个状态，exact certificate size 都是 7，且 validator 均返回有效；
- 重复生成 DSL signature 得到相同的冻结向量，证明当前执行路径可复现。

这说明 certificate 框架可以从简单的 capability membership 迁移到具有具体执行输出的 deterministic DSL world；组合任务的可执行性可以由当前实际具备的 primitive 规则决定，而不需要预先声明同名复合能力标签。

这一阶段尚未证明 DSL 能自动发现能力结构，也未建立到真实神经语言模型的桥梁。primitive 语义、输入上下文和组合规则仍是人工设计的，certificate transfer 结果也仅覆盖 pure primitive world。

## 10. Phase 7：Structural Compressibility

Phase 7 构造了两个最小参数化 block world family，用来区分“先修结构减少合法状态数”和“观测列冗余允许固定任务压缩”。

- independent block world：每个 block 内任务同步出现，不同 block 可独立开关，状态总体为所有完整 block union；
- prefix block world：状态总体为空状态和连续 ordered block prefixes。

两个生成器均使用普通 prerequisite rules 表达同步/前缀约束，并用 exhaustive closure oracle 检查：在 `n <= 12` 的冻结网格内，`valid_states` 必须和 `KnowledgeSpace.is_valid_state` 接受的所有子集完全一致。

正式 Phase 7 artifact 由 clean source commit `b44cad06dbe1859008c9ce47f6434ef1ac7774a4` 生成：

- command：`PYTHONPATH=. python scripts/phase7_structural_compressibility.py`；
- result：`artifacts/phase7_structural_compressibility/structural_compressibility.json`，sha256 `9ffd95abfe4ad85ee55aa833342de175d8bebdb39bf9ac808259943282842270`；
- manifest：`artifacts/phase7_structural_compressibility/manifest.json`，sha256 `a496f502f0230089d220a90c86873b0ae1ed258c8a267fa887f960b7c9e6caa0`；
- structured cells：16；
- matched random controls：320；
- structured oracle status：`passed`。

Phase 7 结果显示：

- 两个完整 block family 的 exact fixed certificate size 都等于 block 数 `B`，也就是每个 block 一个代表任务；
- 对 uniform block size `s`，fixed task ratio 为 `1/s`；
- independent block world 的 adaptive 平均和最坏深度都等于 `B`，与 fixed certificate size 相同；
- prefix block world 的 adaptive 平均深度在冻结网格中对 `B >= 2` 均低于 fixed size；adaptive 最坏深度对 `B=2` 等于 fixed size，对 `B>=3` 严格低于 fixed size；
- chain、accepted tree 和 full unstructured controls 均通过 single-coordinate witness oracle，每个任务都 individually indispensable，因此 fixed certificate size 等于 task count。

matched controls 只是描述性参照，不是工程 acceptance gate。独立 block cells 的 fixed size 在本次结果中均低于 matched-control mean；prefix cells 虽然相对 task count 有固定压缩，但在 `B>=3` 时 fixed size 高于 matched-control mean。因此 Phase 7 不支持“结构本身相对随机总体总是更省固定任务”的一般声明。

Phase 7 目前的证据状态是：formal artifacts 已生成，implementation review 已通过，fresh-context final scientific review 于 2026-08-14 给出 `ACCEPT`，未发现 correctness、protocol、provenance、统计口径或科学性过度声明 finding。

## 11. 经过修订后的证据状态

Phase 2-6 曾发现并修复多项会影响科学结论的问题，包括：非法状态被静默过滤、adaptive tree validator 不完整、平均和最坏成本混淆、Bayesian history 重复计数、DSL 只返回 membership 而不执行具体语义，以及实验脚本缺少可失败的 expected-output oracle。

最近一次 source repair 和重验证具有以下状态：

- previous rejected source commit：`dc6a837ac20aad967da76a69d50c0b5b2bfe7379`；
- previous evidence commit：`4e235ab5baff6c9882100b390f39eeb4bf0ac22f`；
- latest source repair commit：`66c0efade42d85c2ca9c5eca1d3cdb4fc19e3d40`；
- accepted Phase 5/6 evidence commit：`1bfc6ced88cf3397f240c3e79d1996955e9d589f`；
- Phase 2-4 evidence 与冻结 package commit：`8039dcfa88a1a6a1856b19c5301bd74e1616e76c`；
- source repair review：fresh-context strict read-only `ACCEPT`；
- latest formal Phase 2 artifact：result `fe0d5c959816fe9b6578eb1c45756f7d57079d914be6a5529e36c5ee5cea3a5c`，manifest `fc31ab28ed66e8f54d808fcae28d94a5c44a8e495785211d219e296c66174bcd`，source `1bfc6ced88cf3397f240c3e79d1996955e9d589f`；
- latest formal Phase 3 artifact：result `6b7186d7da60e7655036f88c39314264e9ee0004b272525ab11e9eae4744edef`，manifest `acbdfaefcc0f64c8eac8db1f2e65744c3575a3cc8335966ca3ec6832d764a3c9`，source `1bfc6ced88cf3397f240c3e79d1996955e9d589f`；
- latest formal Phase 4 artifact：result `1d00e83570164b7253c724cb68e4c5760c283d459a4097621bd88079d96dd39d`，manifest `738fd13570b3ec4b3662ff4b84bf3e3076cebf49b7806af91c7076484237a9d3`，source `1bfc6ced88cf3397f240c3e79d1996955e9d589f`；
- latest formal Phase 5 artifact：result `b554bafa863ace1dd4729ab3b0c435b3dfd0d0e28a3a8b5c9efe43c4b2457d2a`，manifest `dee175d2eee039efece909d6e78d5c84c3bb86b63137a4197a6872e4404c20e2`；
- latest formal Phase 6 artifact：result `252d8043ea73a087b77be919bce91bc91e0b0cfaa40ecaca5887b69006abc5e0`，manifest `814b3cc1320090677fe6d6feb73b5fe3804cb001ff96c445e238a1b4b14b487d`；
- final milestone scientific review：fresh-context `gpt-5.6-sol` strict read-only `ACCEPT`；12 项 acceptance criteria 全部通过，未发现任何未解决的 correctness、protocol、provenance、统计口径或科学性过度声明 finding；
- current evidence gate：`ACCEPTED`。

独立 reviewer 没有自行重跑测试或实验，其结论基于冻结源码、diff、manifest、artifact、JSON 结构与聚合一致性以及已记录验证的只读检查。上一轮 clean evidence package 的 artifact hash 全部匹配，manifest 均绑定当时的 source commit；拒绝原因是报告证据状态不一致、DSL 条件任务声明与执行不一致、DSL 输入依赖 Python 隐式转换，以及公开概率策略接受 boolean attempts。上述问题已在 `66c0efade42d85c2ca9c5eca1d3cdb4fc19e3d40` 中修复并通过 source repair review。Phase 5/6 evidence package 的数值与 provenance 检查通过并已提交；随后 Phase 2-4 也在 clean commit `1bfc6ced88cf3397f240c3e79d1996955e9d589f` 上重跑。最终 reviewer 独立复核了十个当前 result/manifest SHA-256，并确认实现/协议一致、统计口径正确、历史无效证据已隔离且报告没有超出实验边界的科学声明。更早的 source-snapshot 证据包仅保留为历史记录。hash 证明字节身份和来源绑定，不代替数学、实现和实验协议审查。

## 12. 迄今为止的综合研究结论

1. capability certificate 可以在有限、显式 knowledge space 中被严格定义、求解和验证。
2. full-information identifiability 是 certificate 存在的前提，必须在求解前独立检查。
3. Phase 3 结果显示，状态空间有结构不等于固定任务集一定可以压缩；当前三个基准的 exact fixed certificate 都需要全部任务。
4. Phase 4 结果显示，结构可以为自适应评估提供平均成本优势；该优势出现在 chain/tree，而没有出现在 unstructured 对照组。
5. 噪声把“逻辑上唯一识别”转化为“基于模型和先验的后验推断”；正确的增量更新、停止规则和一致性 oracle 是科学结论成立的必要条件。
6. capability metadata 只有在对应当前世界实际可执行规则时才有行为含义。可执行 DSL 比单纯 membership label 提供了更强的验证边界。
7. Phase 7 进一步说明，固定任务压缩需要响应列冗余等更具体的状态总体结构；先修结构本身不足以保证 fixed certificate 变小。
8. 目前最强的结果是受控小世界中的方法可行性和语义一致性，不是对真实 LLM 能力测量有效性的经验结论。

## 13. 仍待回答的研究问题

- 如何把 exact fixed solver 从穷举扩展到更大任务空间，同时保留可验证的最优性或近似界；
- 如何在统一成本预算下比较 noisy fixed、adaptive 和 repeated-probe 方法，并报告置信区间与统计功效；
- 如何处理非均匀状态先验、任务成本不同和噪声参数未知的情况；
- 如何从数据中学习或检验 capability graph，而不是完全人工给定；
- 如何把 DSL task 映射到真实模型输入输出，并验证 latent capability state 是否具有跨任务预测效度。

## 14. 主要证据文档

- [Phase 1 report](phase1_report.md)
- [Phase 2 report](phase2_report.md)
- [Phase 3 report](phase3_report.md)
- [Phase 4 report](phase4_report.md)
- [Phase 5 report](phase5_report.md)
- [Phase 6 report](phase6_report.md)
- [Phase 7 report](phase7_report.md)
- [Phase 2-6 correctness repair and revalidation](phase2_6_revalidation_report.md)
