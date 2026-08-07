# Task Handoff 005
# Capability Knowledge Space Certificate Laboratory

## Phase 5: Probabilistic Capability Certificate

## 0. Objective

基于：

- Task 001 Knowledge Space Core Engine
- Task 002 Identifiability Audit Engine
- Task 003 Exact Fixed Certificate Solver
- Task 004 Adaptive Certificate Solver

将确定性 knowledge space 扩展到 probabilistic assessment。

目标：

研究：

> 在存在回答错误、猜测和随机性的情况下，能力 certificate 是否仍然存在。

---

# 1. Research Question

## RQ1

结构化能力空间在 noisy response 下是否仍可识别？

## RQ2

adaptive assessment 是否仍优于 fixed benchmark？

## RQ3

重复 probe 是否可以降低评估成本？

---

# 2. Scope

## Implement

- probabilistic response model
- posterior state inference
- noisy fixed certificate
- noisy adaptive assessment
- uncertainty metrics

## Excluded

禁止：

- neural model
- LLM
- learned query policy
- graph recovery

---

# 3. Dependencies

依赖：

```
knowledge_space/
validation/
certificate/
```

需要：

- KnowledgeSpace
- fixed certificate
- adaptive solver

---

# 4. Probabilistic Response Model

状态：

K

任务：

q

响应：

Y∈{0,1}

---

## Slip

如果：

q∈K

则：

P(Y=1|K,q)=1-s_q

---

## Guess

如果：

q∉K

则：

P(Y=1|K,q)=g_q

---

# 5. Response Simulator

实现：

```python
simulate_probabilistic_response(
    state,
    task,
    parameters
)
```

支持：

- single attempt
- repeated attempts

---

# 6. Posterior Inference

维护：

P(K|Y)

使用：

Bayes update

\[
P(K|Y)\propto P(Y|K)P(K)
\]


实现：

```python
infer_state_posterior(observations)
```

输出：

- state probabilities
- MAP state
- entropy
- confidence

---

# 7. Probabilistic Certificate

## Fixed

给定：

S⊂Q


观察：

Y_S


要求：

\[
P(K^*|Y_S)>1-\delta
\]


---

## Adaptive

每轮选择：

\[
q_t
\]

最大化：

posterior uncertainty reduction。


---

# 8. Query Policies

实现：

## Random

随机选择。

## Entropy Reduction

最大降低：

H(K)

## Expected Error Reduction

最大降低：

state prediction error

---

# 9. Metrics

## State Identification

- MAP accuracy
- posterior confidence
- entropy


## Certificate Cost

记录：

- average queries
- worst-case queries
- queries to confidence threshold


例如：

达到：

P(K*)>0.95

需要多少问题。

---

## Robustness

测试：

- slip
- guess
- repeated attempts

---

# 10. Experiments

## Experiment A

Noise scaling:

改变：

- slip
- guess

观察 certificate degradation。

---

## Experiment B

Fixed vs Adaptive

比较不同 noise level 下：

adaptive advantage。

---

## Experiment C

Repeated probing

比较：

一次回答

vs

多次回答。

---

## Experiment D

Structured vs Unstructured

验证：

cheap assessment 是否依赖结构。

---

# 11. Tests

## Test 1

无噪声：

结果应接近 Task 003/004。

## Test 2

高 slip：

识别准确率下降。

## Test 3

高 guess：

避免错误高估。

## Test 4

Posterior normalization：

\[
\sum_K P(K|Y)=1
\]

---

# 12. Deliverables

代码：

```
probabilistic/
    response_model.py
    posterior.py
    noisy_certificate.py
    adaptive_policy.py
```


测试：

```bash
pytest tests/
```


报告：

```
phase5_report.md
```

包含：

- noise model
- posterior inference
- fixed/adaptive comparison
- robustness curves
- limitations

---

# 13. Failure Reporting

不要修改概率定义。

反馈：

## Problem

描述：

- world
- noise
- posterior
- query sequence

## Evidence

提供：

- logs
- failing tests
- metrics

## Hypothesis

可能原因：

- noise 太大
- certificate 不鲁棒
- state space 太复杂

## Decision Needed

需要人工决定的问题。

---

# 14. Completion Criteria

完成：

1. probabilistic response model 可运行；
2. posterior inference 正确；
3. noisy fixed certificate 可评估；
4. noisy adaptive certificate 可评估；
5. robustness report 完成。

完成后停止。

不要实现：

- DSL bridge
- toy LM
- neural capability discovery

这些属于后续阶段。
