# Task Handoff 002
# Capability Knowledge Space Certificate Laboratory

## Phase 2: Identifiability Audit Engine

## 0. Task Objective

本任务建立在 Task 001 Knowledge Space Core Engine 基础上。

目标：

> 在 full-information 条件下，验证 capability knowledge space 是否可以由完整任务响应唯一识别。

本阶段不研究：

- certificate size；
- query selection；
- adaptive testing；
- noisy response；
- machine learning。

核心问题：

\[
Given\ all\ task\ responses,\ is\ capability\ state\ identifiable?
\]

---

# 1. Motivation

如果存在：

\[
K_1 \neq K_2
\]

但：

\[
Y(K_1,Q)=Y(K_2,Q)
\]

则：

即使询问全部问题，也无法区分两个能力状态。

因此必须先验证：

Full-information identifiability。

---

# 2. Scope

## Implement

实现：

1. response signature generation；
2. collision detection；
3. equivalent state grouping；
4. identifiability report；
5. tests。

## Excluded

禁止实现：

- certificate solver；
- adaptive testing；
- information gain；
- noisy response；
- graph recovery；
- neural model。

---

# 3. Dependencies

依赖 Task 001：

- knowledge_space
- generators
- simulator

必须能够：

- 枚举合法 states；
- 生成 deterministic responses。

---

# 4. Core Concepts

## 4.1 Response Signature

对于状态：

\[
K_i
\]

定义：

\[
R_i=(Y(q_1),...,Y(q_n))
\]

实现：

```python
response_signature(state)
```

---

## 4.2 Collision

两个状态：

\[
K_i \neq K_j
\]

但：

\[
R_i=R_j
\]

说明：

任务集合不足以区分状态。

---

# 5. Identifiability Checker

实现：

```python
check_identifiability(knowledge_space)
```

返回：

```python
IdentifiabilityReport
```

---

# 6. Report Schema

必须包含：

## Basic

```json
{
 "num_tasks": 8,
 "num_states": 32,
 "num_unique_signatures": 32
}
```

## Result

```json
{
 "identifiable": true
}
```

## Collision

若存在：

```json
{
 "collision_count": 1,
 "collision_groups": [
   ["state1","state2"]
 ]
}
```

## Compression Ratio

计算：

\[
\frac{unique\ signatures}{total\ states}
\]

---

# 7. Algorithms

## Brute Force Enumeration

流程：

```
for each state:
    generate response
    hash signature
    group identical signatures
```

---

## Collision Detection

输出：

所有：

\[
|group|>1
\]

的状态组。

---

## Stable State ID

要求：

相同 KnowledgeSpace 多次运行产生一致 ID。

---

# 8. Required Tests

## Test 1: Chain

结构：

```
A -> B -> C -> D
```

状态：

```
{}
{A}
{A,B}
{A,B,C}
{A,B,C,D}
```

要求：

无 collision。

---

## Test 2: Artificial Collision World

构造：

两个不同状态产生相同 response。

验证：

collision 被发现。

---

## Test 3: Tree

验证：

合法 tree states 可识别。

---

## Test 4: Negative Control

Unstructured:

\[
\mathcal K=2^Q
\]

验证：

状态 signature 唯一。

---

# 9. Example

运行：

```bash
python examples/run_identifiability_audit.py
```

输出：

```
Knowledge Space:
type: chain
tasks: 4
states: 5

Unique signatures:
5

Identifiable:
True

Collisions:
0
```

---

# 10. Deliverables

提交：

## Code

新增：

```
validation/
identifiability/
```

---

## Tests

运行：

```bash
pytest tests/
```

---

## Report

生成：

```
phase2_report.md
```

包含：

1. 实现内容；
2. identifiability 定义；
3. 测试结果；
4. collision 示例；
5. 当前限制。

---

# 11. Problem Reporting

遇到困难不要修改研究定义。

反馈：

## Problem

描述：

- world；
- state；
- signature；
- error。

## Evidence

提供：

- failing test；
- response matrix；
- collision group。

## Hypothesis

可能原因。

## Decision Needed

需要人工决定的问题。

---

# 12. Completion Criteria

完成条件：

1. 任意 KnowledgeSpace 可以运行 audit；
2. 可以发现 collision；
3. 输出 collision groups；
4. chain/tree world 测试通过；
5. artificial collision world 测试通过；
6. phase2_report.md 完成。

完成后停止。

不要实现：

- certificate search；
- adaptive assessment；
- noisy inference。

这些属于后续任务。
