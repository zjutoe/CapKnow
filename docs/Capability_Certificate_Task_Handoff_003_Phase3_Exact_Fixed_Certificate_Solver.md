# Task Handoff 003
# Capability Knowledge Space Certificate Laboratory

## Phase 3: Exact Fixed Certificate Solver

## 0. Objective

基于：

- Task 001 Knowledge Space Core Engine
- Task 002 Identifiability Audit Engine

实现固定 capability certificate 求解。

目标：

寻找最小问题集合：

S ⊂ Q

使任意两个不同能力状态：

K_i != K_j

均满足：

K_i ∩ S != K_j ∩ S

即：

少量固定问题可以唯一识别能力状态。

---

# 1. Research Question

本阶段研究：

> 在能力结构已知且可识别的情况下，理论最少需要多少固定问题完成能力评估？

对应固定 benchmark 场景。

---

# 2. Scope

## Implement

- fixed certificate definition
- exact solver
- certificate validation
- size analysis
- tests

## Excluded

禁止：

- adaptive certificate
- information gain
- noisy response
- graph recovery
- neural model

---

# 3. Mathematical Formulation

对于状态对：

K_i, K_j

定义：

D_ij = {q : Y(K_i,q) != Y(K_j,q)}

certificate S 必须满足：

S ∩ D_ij != empty

对于所有状态对成立。

这是 minimum hitting set 问题。

---

# 4. Exact Solver

使用 binary integer programming。

变量：

x_q ∈ {0,1}

表示问题 q 是否进入 certificate。

目标：

minimize:

sum(x_q)

约束：

对于每个状态 pair:

sum(q in D_ij)x_q >= 1

---

接口：

```python
solve_exact_certificate(knowledge_space)
```

返回：

CertificateResult

---

# 5. Result Schema

必须包含：

- task 数量
- state 数量
- certificate size
- selected tasks
- validation result
- separated state pairs

示例：

```json
{
 "certificate_size":4,
 "valid":true,
 "separated_pairs":4950
}
```

---

# 6. Validation

实现：

```python
validate_certificate(
    knowledge_space,
    certificate
)
```

验证：

所有状态 pair 是否被区分。

---

# 7. Required Tests

## Test 1: Chain

A -> B -> C -> D

验证：

得到最优 certificate。


## Test 2: Independent Tasks

所有任务独立。

验证：

certificate 接近 |Q|。


## Test 3: Structured Space

大量状态但少量关键节点。

验证：

certificate 明显小于任务数量。


## Test 4: Non-identifiable Input

collision world。

solver 必须拒绝：

Certificate undefined.

---

# 8. Baselines

实现：

## Random selection

随机选择问题。

## Greedy separating set

每轮选择：

覆盖最多未区分 state pairs 的问题。

比较：

- exact optimum
- greedy
- random

---

# 9. Metrics

报告：

## Certificate size

|S|

## Compression ratio

|S| / |Q|

## Validation accuracy

必须 100%

## Runtime

记录：

- tasks
- states
- state pairs
- solver time

---

# 10. Experiments

## Experiment A

比较：

- chain
- tree
- DAG

观察：

结构复杂度与 certificate size。


## Experiment B

增加：

- task 数量
- state 数量

观察 scaling。


## Experiment C

structured vs unstructured

验证：

cheap assessment 是否来自结构。

---

# 11. Deliverables

代码：

```
certificate/
    exact_solver.py
    validator.py
    result.py
```

测试：

```bash
pytest tests/
```

报告：

```
phase3_report.md
```

包含：

- 方法
- 示例 certificate
- 测试结果
- runtime
- 限制

---

# 12. Failure Reporting

遇到问题不要修改研究定义。

反馈：

## Problem

描述失败。

## Evidence

提供：

- world
- states
- pair count
- solver output

## Hypothesis

可能原因。

## Decision Needed

需要人工决定的问题。

---

# 13. Completion Criteria

完成：

1. exact fixed certificate 可求解；
2. certificate 可验证；
3. collision input 可拒绝；
4. random/greedy baseline 可比较；
5. phase3_report.md 完成。

完成后停止。

不要实现：

- adaptive certificate
- noisy certificate
- neural model
