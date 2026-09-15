# Capability Knowledge Space Certificate Laboratory

本项目研究一个受控问题：如果系统的能力状态具有结构，能否只观察少量任务响应，便识别其完整能力状态，并给出可复现、可验证的 **capability certificate（能力证书）**？

当前项目是小规模、可枚举、人工定义结构的研究实验室。Phase 2–7 已形成经过独立审查的确定性、概率性与结构压缩证据；Phase 8 的 Toy LM 可行性门槛失败并按有效负结果停止；Phase 9 目前仅有已提交的协议提案，尚未实现或实验验证。项目没有真实或开放 LLM 的验证结果。

## 研究对象

设任务全集为 \(Q\)，一个能力状态为 \(K\subseteq Q\)，允许的状态总体为 \(\mathcal K\subseteq 2^Q\)。任务响应 \(Y(K,q)\) 将潜在状态映射为可观察行为。

- `Task`：可执行或可观察的探针。
- `KnowledgeState`：候选能力配置。
- `KnowledgeSpace`：允许状态及其结构约束。
- response signature：一个状态在全部或部分任务上的响应向量。
- fixed certificate：对所有对象询问同一任务子集，仍能区分全部候选状态。
- adaptive certificate：根据先前回答继续选题，以决策树叶节点唯一识别状态。

研究遵循“先验证可识别性，再讨论低成本证书”的顺序。若不同状态在完整任务集上仍产生相同 signature，固定或自适应 certificate 都无法唯一恢复状态。噪声条件下，目标则转为在给定响应模型与先验下进行后验推断。

## 当前阶段与证据状态

| 阶段 | 研究内容 | 当前结论 |
| --- | --- | --- |
| Phase 1 | knowledge space、状态校验与确定性响应 | 已建立 chain、tree、unstructured 世界及基础模拟器 |
| Phase 2 | full-information identifiability | 三个标准世界可识别；人工构造的完整向量 collision 可被审计器发现 |
| Phase 3 | exact fixed certificate | chain/tree/unstructured 的最小规模分别为 `4/4/3`，均需要全部任务 |
| Phase 4 | adaptive certificate | 状态等权时，chain/tree 平均深度低于固定规模；unstructured 无同样优势 |
| Phase 5 | noisy probabilistic inference | Bayesian fixed/adaptive 推断闭环与零噪声一致性已验证；尚无普遍优劣结论 |
| Phase 6 | executable DSL bridge | primitive、组合和 held-out 组合具有确定执行语义；规则仍由人工定义 |
| Phase 7 | structural compressibility | block world 中固定压缩成立；prefix family 在测试网格上还有自适应平均节省 |
| Phase 8 | Toy LM feasibility | `11/24` 单元通过，正式结果为 `FAILED`/stopped；无模型 selection、无 010D |
| Phase 9 | decomposed learned-behavior bridge | 已提交协议提案；尚未实现、运行或形成实验 artifact |

Phase 2–6 的修复与重验证 evidence package 已通过最终独立科学验收；Phase 7 也已通过最终独立科学审查。Phase 8 被接受的是一个有效的失败/停止里程碑，而非 learned-behavior bridge 的成功证据。

## 已支持的主要结果

### 固定证书不由“有结构”自动保证

Phase 3 的三个基准世界中，exact fixed certificate 都需要询问全部任务：chain 为 4、tree 为 4、unstructured 为 3。结构约束可以减少合法状态数，但只要每个任务都有一对仅靠该任务才能区分的状态，它仍不可从固定证书中删除。

### 自适应评估可利用部分结构

Phase 4 在确定性响应和状态等权条件下得到：

| 世界 | 固定最小规模 | entropy 平均深度 | 最坏深度 |
| --- | ---: | ---: | ---: |
| chain | 4 | 2.4000 | 3 |
| tree | 4 | 2.8571 | 3 |
| unstructured | 3 | 3.0000 | 3 |

这支持 chain/tree 小世界中的平均成本优势，不支持自适应方法在任意结构或先验下都更优。当前公开 API 可直接使用确定性的 `entropy` 策略；也保留 `random` 策略。历史 artifact 中的 `balanced` 标签属于冻结记录，不应作为当前用法。

### 固定压缩需要更具体的观测冗余

Phase 7 构造 independent block 与 prefix block 两类完整状态族。若有 \(B\) 个 block、每个 block 含 \(s\) 个同步任务，两个 family 的 exact fixed certificate 都只需每个 block 选一个代表任务：规模为 \(B\)，占全部任务的比例为 \(1/s\)。

在已测试网格中：

- independent block 的 adaptive 平均与最坏深度均为 \(B\)，没有额外节省；
- prefix block 的 adaptive 平均深度低于 fixed size；
- matched random controls 仅作描述性参照，不构成因果或普遍“结构压缩”结论。

这些结果只覆盖确定性 membership observation、状态等权和小型人工总体。

### Toy LM 桥接尚未通过前提

Phase 8 的 `feasibility_005` 在 small/medium 模型、三个固定种子和四类序列任务上要求每个单元达到 `52/64` exact matches，结果为：

| Family | 通过单元 |
| --- | ---: |
| Hex Copy | 5/6 |
| Named-value JSON | 0/6 |
| Boolean JSON | 6/6 |
| Array JSON | 0/6 |
| 合计 | 11/24 |

因此该里程碑正确发布 `FAILED`：没有 selection，也没有进入 010D 或后续正式 learned-behavior 实验。D3 postmortem 虽 operationally `DONE`，其类别是 non-evidence，不能改变失败结论或授权后续实验。该结果只说明冻结的 learner/configuration 未通过冻结门槛，并不证明 learned-behavior certificate 不可能。

## Phase 9：分解式 learned-behavior 提案

[Phase 9 handoff](docs/Capability_Certificate_Task_Handoff_011_Phase9_Decomposed_Learned_Behavior_Bridge.md) 将下一步拆成三个依次设门的研究问题：

1. learner 能否泛化 elementary transformations；
2. 仅改变训练语料，能否实现声明的 capability states；
3. learned checkpoint population 能否保留已知 block-world fixed/adaptive certificate 结构。

提案明确不在 Phase 9 测试 composition；只有前三个问题成功后，composition 才可能成为单独的后续里程碑。当前文档是协议提案，不代表实现、实验许可、审查接受或科学结果。

## 项目结构

```text
capability_certificate_lab/
  knowledge_space/   # Task、KnowledgeState、KnowledgeSpace
  generators/        # chain、tree、unstructured、block worlds
  simulator/         # 确定性响应
  validation/        # 状态与 identifiability 检查
  certificate/       # fixed 与 adaptive solver、validator
  probabilistic/     # 噪声响应、posterior、adaptive assessment
  dsl/               # primitive、程序、组合与执行器
  lm_bridge/         # Phase 8 Toy LM 组件
examples/            # 最小可运行示例
scripts/             # 重验证与正式分析入口
tests/               # 单元与语义测试
docs/                # 研究计划、handoff、报告与文档索引
  phase8/            # Phase 8 handoff、状态和证据索引
```

研究计划、阶段报告和 handoff 集中存放在 `docs/`，入口见[文档索引](docs/README.md)。`artifacts/` 中的正式结果必须结合对应 manifest、源码 commit 和报告解释。

## 快速开始

仓库没有 packaging metadata。请从仓库根目录运行，并显式设置 `PYTHONPATH=.`。已验证的开发环境使用 Python 3.13.9；这是已验证版本，不是声明的最低版本。核心模块只使用标准库，测试需要 `pytest`，Toy LM 相关代码需要 `torch`。

运行现有示例：

```bash
PYTHONPATH=. python examples/create_chain_world.py
PYTHONPATH=. python examples/run_identifiability_audit.py
```

直接比较一个小型 chain world 的 fixed 与 adaptive certificate：

```bash
PYTHONPATH=. python - <<'PY'
from capability_certificate_lab.certificate import (
    solve_adaptive_certificate,
    solve_exact_certificate,
)
from capability_certificate_lab.generators import generate_chain_world

world = generate_chain_world(["A", "B", "C", "D"])
fixed = solve_exact_certificate(world)
adaptive = solve_adaptive_certificate(world, policy="entropy")

print(f"states={len(world.valid_states)}")
print(f"fixed={fixed.certificate_size}, valid={fixed.valid}")
print(
    f"adaptive_avg={adaptive.average_depth:.1f}, "
    f"adaptive_worst={adaptive.worst_case_depth}, valid={adaptive.valid}"
)
PY
```

预期输出中的核心数值为 `states=5`、`fixed=4`、`adaptive_avg=2.4`、`adaptive_worst=3`，且两个 validator 均为 `True`。

运行普通测试：

```bash
PYTHONPATH=. python -m pytest -q
```

`scripts/` 下的正式实验入口会绑定 clean committed source 和固定 evidence root，不应当作日常 quickstart 命令运行。

## 证据与进一步阅读

- [文档总索引](docs/README.md)
- [原始研究计划](docs/Capability_Knowledge_Space_Certificate_Laboratory_Research_Plan.md)
- [研究进展与阶段成果总览](docs/Capability_Certificate_Research_Progress_Summary.md)
- [Phase 2–6 correctness repair 与重验证报告](docs/phase2_6_revalidation_report.md)
- [Phase 7 structural compressibility 报告](docs/phase7_report.md)
- [Phase 8 当前状态与证据索引](docs/phase8/README.md)
- [Phase 9 decomposed learned-behavior 协议提案](docs/Capability_Certificate_Task_Handoff_011_Phase9_Decomposed_Learned_Behavior_Bridge.md)
- [Phase 1](docs/phase1_report.md)、[Phase 2](docs/phase2_report.md)、[Phase 3](docs/phase3_report.md)、[Phase 4](docs/phase4_report.md)、[Phase 5](docs/phase5_report.md)、[Phase 6](docs/phase6_report.md) 历史阶段报告

解释结果时应以重验证报告和当前进展总结为准；较早阶段报告保留历史价值，但其中已被修正或失效的结果不能替代当前 evidence package。

## 仍待回答的问题

- exact solver 如何扩展到更大任务空间，同时保留可验证的最优性或近似界；
- noisy fixed、adaptive 与 repeated-probe 方法如何在统一成本预算下比较；
- 非均匀状态先验、异质任务成本和未知噪声参数会怎样改变 certificate；
- 能否从行为数据学习或检验 capability graph，而非完全人工指定；
- 分解后的 learned-behavior bridge 能否依次通过泛化、状态实现和证书保留；
- 受控结果能否最终迁移到真实模型，并具有跨任务预测效度。
