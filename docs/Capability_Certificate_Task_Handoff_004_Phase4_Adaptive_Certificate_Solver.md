# Task Handoff 004
# Capability Knowledge Space Certificate Laboratory

## Phase 4: Adaptive Certificate Solver

## 0. Objective

基于：

- Task 001 Knowledge Space Core Engine
- Task 002 Identifiability Audit Engine
- Task 003 Exact Fixed Certificate Solver

研究 adaptive certificate。

目标：

> 研究动态提问是否可以降低能力评估成本。

固定 certificate：

所有模型回答相同问题集合。

Adaptive certificate：

下一问题依赖此前回答：

q(t+1)=π(previous answers)

模拟真实面试过程。

---

# 1. Research Question

比较：

固定评估：

C_fixed

与：

自适应评估：

C_adaptive


核心问题：

结构化能力空间是否支持低成本动态评估？

---

# 2. Scope

## Implement

实现：

- candidate state tracking
- decision tree
- query selection policy
- adaptive validation
- evaluation metrics


## Excluded

禁止：

- noisy response
- neural model
- LLM
- graph recovery
- learned policy

---

# 3. Dependencies

依赖：

```
knowledge_space/
validation/
certificate/
```

要求：

KnowledgeSpace 必须：

- full-information identifiable
- 可生成 response signature

---

# 4. Adaptive Assessment

维护候选状态：

V_t ⊆ K


初始：

V_0 = K


选择问题：

q_t


得到回答：

y_t


更新：

V_(t+1)={K∈V_t:Y(K,q_t)=y_t}


终止：

|V_t|=1

---

# 5. Query Policies

实现：

## Random

随机选择问题。

Baseline。

---

## Entropy Reduction

选择最大减少状态不确定性的问题：

q*=argmax information gain


---

## Balanced Split

选择使回答分支最均衡的问题：

min max(|V+|,|V-|)

---

# 6. Decision Tree

实现：

```python
DecisionNode
```

字段：

- question
- yes_child
- no_child
- candidate_states


输出：

```python
AdaptiveCertificate
```

包含：

- worst-case depth
- average depth
- node count

---

# 7. Validation

实现：

```python
validate_adaptive_certificate(tree)
```

要求：

对于每个：

K∈𝒦

沿 tree：

最终定位唯一状态。

---

# 8. Required Tests

## Chain World

A -> B -> C -> D

比较：

fixed certificate

vs

adaptive tree


---

## Tree World

验证：

branching structure。

---

## Unstructured World

验证：

无结构情况下 adaptive 不应产生异常优势。


---

## Reproducibility

相同 seed：

生成相同 tree。

---

# 9. Metrics

## Query Cost

Worst case:

max depth


Average:

E(depth)


## Improvement

比较：

C_fixed / C_adaptive


## Other

- tree size
- runtime
- memory

---

# 10. Experiments

## Experiment A

Chain:

fixed vs adaptive


## Experiment B

不同结构：

- chain
- tree
- DAG
- AND/OR


## Experiment C

Scaling:

增加：

- task 数
- state 数
- branching factor


观察：

adaptive complexity。

---

# 11. Deliverables

代码：

```
certificate/
    adaptive_solver.py
    decision_tree.py
    policies.py
```


测试：

```bash
pytest tests/
```


报告：

```
phase4_report.md
```

包含：

- 方法
- tree 示例
- fixed/adaptive 对比
- scaling 结果
- 限制

---

# 12. Failure Reporting

不要修改研究定义。

反馈：

## Problem

描述问题。


## Evidence

提供：

- world
- state 数
- tree
- query sequence
- runtime


## Hypothesis

可能原因。


## Decision Needed

需要人工决定的问题。

---

# 13. Completion Criteria

完成：

1. adaptive tree 可生成；
2. tree 可验证；
3. random/entropy/balanced 可比较；
4. fixed vs adaptive 可报告；
5. phase4_report.md 完成。

停止。

不要实现：

- noisy adaptive testing
- learned query policy
- neural bridge

这些属于后续阶段。
