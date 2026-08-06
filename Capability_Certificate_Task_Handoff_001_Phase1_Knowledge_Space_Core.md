# Task Handoff 001
# Capability Knowledge Space Certificate Laboratory

## Phase 1: Knowledge Space Core Engine

## 0. Objective

本阶段目标：

构建最小 Knowledge Space 实验框架，用于研究：

> 具有结构约束的 capability space 是否存在可识别的能力状态。

本阶段不实现：

- neural network
- LLM
- DSL
- certificate search
- graph recovery
- optimization

只建立后续实验基础。

---

# 1. Research Question

验证：

\[
\mathcal K \subseteq 2^Q
\]

是否可以被明确生成和验证。

---

# 2. Scope

## Implement

- Task universe
- Knowledge state
- Knowledge space generators
- State validator
- Deterministic response simulator
- Unit tests

## Do not implement

- adaptive assessment
- information gain
- noisy response
- ML models

---

# 3. Project Structure

建议：

```
capability_certificate_lab/

├── knowledge_space/
├── generators/
├── simulator/
├── validation/
├── tests/
└── examples/
```

---

# 4. Data Model

## Task Universe

定义：

\[
Q={q_1,...,q_n}
\]

要求：

- task id 唯一；
- 支持序列化；
- 支持索引。


---

## Knowledge State

表示：

\[
K\subseteq Q
\]

要求：

支持：

- membership
- set operations
- serialization


---

## Knowledge Space

表示：

\[
\mathcal K
\]

保存：

- tasks
- valid states
- generation rules
- metadata


---

# 5. Generators

## Chain World

例如：

```
A -> B -> C -> D
```

合法状态：

```
{}
{A}
{A,B}
{A,B,C}
{A,B,C,D}
```

要求：

子能力必须包含父能力。

---

## Tree World

例如：

```
        A
      /   \
     B     C
```

要求：

child state requires parent state。


---

## Unstructured World

Negative control:

\[
\mathcal K=2^Q
\]

用于验证：

没有结构时不会产生虚假压缩。

---

# 6. Validation

实现：

```python
validate_state(state)
```

检查：

- task 是否存在；
- prerequisite 是否满足；
- composition constraint 是否满足。

---

# 7. Response Simulator

第一版：

\[
Y(K,q)=1[q\in K]
\]


接口：

```python
simulate_response(state, task)
```


生成：

\[
Y\in\{0,1\}^{M\times N}
\]


---

# 8. Required Tests

## Chain

4 nodes:

合法状态数量：

\[
5
\]


非法：

```
{B}
{A,C}
```

必须拒绝。


---

## Tree

验证：

所有 child state 都满足 parent constraint。


---

## Negative Control

n=5:

状态数：

\[
32
\]


---

## Response

验证：

response 与 membership 完全一致。

---

# 9. Example

运行：

```bash
python examples/create_chain_world.py
```

输出：

```
Tasks:
A B C D

States:
{}
{A}
{A,B}
{A,B,C}
{A,B,C,D}
```

---

# 10. Deliverables

提交：

## Code

所有新增文件。


## Tests

运行：

```bash
pytest tests/
```


## Report

生成：

```
phase1_report.md
```

包含：

- 实现内容；
- 文件列表；
- 测试结果；
- 示例输出；
- 遇到的问题；
- 需要决策的问题。

---

# 11. Problem Reporting

遇到困难不要修改研究定义。

反馈格式：

## Problem

发生什么。

## Evidence

包括：

- traceback
- failing test
- reproduction

## Hypothesis

可能原因。

## Decision Needed

需要人工确认的问题。

---

# 12. Completion Criteria

完成条件：

1. 三种 knowledge space generator 可运行；
2. 状态验证正确；
3. response matrix 可生成；
4. tests 全通过；
5. phase1_report.md 完成。

完成后停止。

不要实现后续：

- certificate solver
- adaptive testing
- noisy model
- graph discovery

这些属于后续阶段。
