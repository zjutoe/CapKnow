# Task Handoff 006
# Capability Knowledge Space Certificate Laboratory

## Phase 6: DSL Bridge and Executable Capability World

---

# 0. Task Objective

本任务建立在：

- Task 001: Knowledge Space Core Engine
- Task 002: Identifiability Audit Engine
- Task 003: Exact Fixed Certificate Solver
- Task 004: Adaptive Certificate Solver
- Task 005: Probabilistic Capability Certificate

之上。

目标：

> 将抽象 knowledge space 转换为具有执行语义的 DSL world，研究 capability structure 是否不仅能表示“会不会”，还能表示“为什么会”和“如何组合”。

此前阶段：

```
Task capability
        ↓
Knowledge state
        ↓
Certificate
```

本阶段加入：

```
Capability
        ↓
Executable operation
        ↓
Task generation
        ↓
Observed behavior
```

---

# 1. Research Question

## RQ1

Knowledge Space 中的结构是否可以通过可执行规则表达？

---

## RQ2

如果 capability 是由 primitive operation 组合产生，certificate 是否仍然有效？

---

## RQ3

DSL world 是否可以作为连接 symbolic structure 与 neural language model 的桥梁？

---

# 2. Scope

## Implement

实现：

1. typed primitive operations；
2. DSL syntax；
3. DSL executor；
4. capability composition rules；
5. task generator；
6. execution-based response simulator。

---

## Excluded

禁止：

- neural network；
- Transformer；
- language model training；
- graph neural network；
- automatic structure discovery。

本阶段结构由人工定义。

---

# 3. Conceptual Model

## 3.1 Primitive Capability

定义基础操作：

例如：

```
ADD
COMPARE
MEMORY
SEARCH
FILTER
LOOP
CONDITION
```

每个 primitive 具有：

- execution semantics；
- input/output type；
- difficulty；
- capability requirement。

---

## 3.2 Composite Capability

能力由组合规则产生：

例如：

```
MEMORY + SEARCH
        ↓
RETRIEVAL
```

```
RETRIEVAL + CONDITION
        ↓
PLANNING
```

组合规则必须显式保存：

```
CapabilityGraph
```

---

# 4. DSL Design

## 4.1 DSL Program

定义：

```
Program =
    PrimitiveOperation
    Composition
    Sequence
    Condition
    Loop
```

示例：

```
SEARCH(memory)

IF condition:
    FILTER(data)
```

---

## 4.2 Program Execution

实现：

```python
execute(program, input)
```

输出：

```
correct answer
```

---

# 5. Capability State

模型不再直接拥有：

\[
K\subseteq Q
\]

而拥有：

primitive capability state：

\[
C_M
\]

例如：

```
MEMORY: 0.9
SEARCH: 0.7
LOOP: 0.3
```

---

Composite capability 根据规则生成：

```
RETRIEVAL = f(MEMORY, SEARCH)
```

---

# 6. Model Response Simulation

响应由执行能力决定。

例如：

如果任务需要：

```
RETRIEVAL
```

则成功概率：

取决于：

```
MEMORY
SEARCH
composition rule
execution noise
```

而不是直接读取 task label。

---

接口：

```python
simulate_task_response(
    model_state,
    program
)
```

---

# 7. Task Generator

生成：

## Primitive Tasks

测试：

单个 primitive。

例如：

```
ADD(3,5)
```

---

## Composite Tasks

测试：

组合能力。

例如：

```
SEARCH + MEMORY
```

---

## Generalization Tasks

重点：

训练：

```
MEMORY task
SEARCH task
```

测试：

```
RETRIEVAL task
```

验证 compositional generalization。

---

# 8. Capability Certificate Evaluation

复用 Task 003-005：

比较：

## Flat Knowledge Space

任务直接对应能力标签。

---

## DSL Capability World

任务由程序执行产生。

---

研究：

certificate 是否仍然存在。

---

# 9. Required Experiments

## Experiment A: Primitive Execution

验证：

primitive capability 可以测量。

---

## Experiment B: Composition

验证：

composite capability 是否需要多个 primitive。

---

## Experiment C: Held-out Composition

训练：

```
A task
B task
```

测试：

```
A+B task
```

观察：

结构是否支持组合泛化。

---

## Experiment D: Certificate Transfer

比较：

直接任务 certificate

vs

DSL generated task certificate。

---

# 10. Tests

## Test 1

Primitive execution correctness。

---

## Test 2

Composition execution correctness。

---

## Test 3

Invalid program rejection。

---

## Test 4

Same capability state reproducibility。

---

## Test 5

Composition task does not leak primitive label。

---

# 11. Deliverables

代码：

```
dsl/
    primitives.py
    program.py
    executor.py
    composition.py
    task_generator.py
    simulator.py
```

---

测试：

```bash
pytest tests/
```

---

报告：

```
phase6_report.md
```

包含：

1. DSL design；
2. primitive definitions；
3. composition rules；
4. task generation；
5. certificate results；
6. limitations。

---

# 12. Failure Reporting

遇到困难不要修改 capability 定义。

反馈：

## Problem

描述：

- DSL；
- program；
- execution；
- expected result；
- actual result。

---

## Evidence

提供：

- failing program；
- execution trace；
- response matrix。

---

## Hypothesis

可能原因：

- primitive semantics 不充分；
- composition rule 不合理；
- task generation 泄漏。

---

## Decision Needed

需要人工决定：

- 是否修改 DSL；
- 是否增加 primitive；
- 是否改变 composition rule。

---

# 13. Completion Criteria

完成：

1. DSL 可以表达 primitive 和 composite capability；
2. executor 正确运行；
3. task 可以由 program 生成；
4. response 不再直接由 capability label 生成；
5. certificate 方法可以迁移到 DSL world；
6. phase6_report.md 完成。

完成后停止。

不要实现：

- toy language model；
- neural training；
- LLM evaluation。

这些属于后续阶段。
