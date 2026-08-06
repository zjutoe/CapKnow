# Capability Knowledge Space Certificate Laboratory

## Objective

研究问题：

> 如果能力状态具有结构，是否存在少量问题组成的 capability
> certificate，使模型完整能力状态可以被低成本识别？

形式化：

任务集合：

Q={q1,...,qn}

能力状态：

K⊆Q

合法状态集合：

𝒦⊆2\^Q

目标研究：

是否存在：

certificate S⊂Q

使得少量问题回答足以识别 K。

------------------------------------------------------------------------

# 1. Experimental Philosophy

第一阶段不训练神经网络。

目标不是模拟 LLM，而是验证：

1.  结构化能力空间是否产生 cheap assessability；
2.  certificate 是否客观存在；
3.  哪些结构性质决定评估成本。

------------------------------------------------------------------------

# 2. Knowledge Space Generation

构造多种 synthetic knowledge spaces。

## Chain

    A
    |
    B
    |
    C
    |
    D

测试线性 prerequisite。

## Tree

           A
         /       B     C

测试分支结构。

## DAG

    Memory ----             -> Retrieval
    Search -----/

测试共享依赖。

## AND/OR Composition

例如：

A+B -\> C

A or B -\> D

测试组合能力。

## Negative Control

无结构：

𝒦=2\^Q

验证 cheap assessment 是否依赖结构。

------------------------------------------------------------------------

# 3. Model States

每个 synthetic model 对应一个状态：

K_i ∈ 𝒦

初始响应：

Y(i,q)=1\[q∈K_i\]

随后加入：

-   slip error
-   guessing error

形成 probabilistic knowledge state。

------------------------------------------------------------------------

# 4. Identifiability First

在研究 cheap certificate 前，先检查：

不同状态是否可以被完整任务集合区分：

K1≠K2

必须满足：

Y(K1,Q)≠Y(K2,Q)

若存在 collision：

记录等价状态，不强行恢复。

------------------------------------------------------------------------

# 5. Certificate Types

## Fixed Certificate

寻找：

S⊂Q

满足：

K1∩S ≠ K2∩S

对应固定 benchmark。

------------------------------------------------------------------------

## Adaptive Certificate

下一问题依赖历史回答：

q(t+1)=π(previous answers)

对应面试式评估。

------------------------------------------------------------------------

## Approximate Certificate

允许剩余不确定状态行为接近：

d(K1,K2)\<ε

对应 routing 场景。

------------------------------------------------------------------------

# 6. Algorithms

比较：

## Oracle

小规模：

-   exhaustive search
-   integer programming
-   dynamic programming

获得理论最优。

## Heuristics

-   random sampling
-   graph centrality
-   prerequisite bottleneck
-   fringe-based selection
-   information gain
-   greedy separating set

------------------------------------------------------------------------

# 7. Evaluation Metrics

## Certificate size

固定：

\|S\|

自适应：

decision tree depth

## Identification

-   exact state accuracy
-   posterior entropy
-   remaining candidate states

## Behavioral utility

预测完整任务域行为：

d_Q(K,K_hat)

------------------------------------------------------------------------

# 8. Experiments

## Experiment 1

Structured vs unstructured:

验证结构是否降低 certificate complexity。

------------------------------------------------------------------------

## Experiment 2

Fixed vs adaptive:

验证面试式评估是否优于固定 benchmark。

------------------------------------------------------------------------

## Experiment 3

Graph property analysis:

研究：

-   depth
-   width
-   branching
-   alternative paths

对 certificate 的影响。

------------------------------------------------------------------------

## Experiment 4

Boundary question analysis:

比较：

-   random
-   centrality
-   fringe
-   information gain

------------------------------------------------------------------------

# 9. Experimental Phases

## Phase 1

Small exact world:

\|Q\|=8

枚举全部状态和最优 certificate。

------------------------------------------------------------------------

## Phase 2

Medium world:

\|Q\|=12-16

使用近似算法。

------------------------------------------------------------------------

## Phase 3

Noise world:

加入：

-   slip
-   guess
-   stochastic mastery

------------------------------------------------------------------------

# 10. Success Criteria

项目成功需要证明：

1.  结构化 knowledge space 比无结构空间具有更小 certificate；

2.  adaptive certificate 显著优于 fixed benchmark；

3.  certificate complexity 与结构性质存在规律；

4.  结论在噪声环境下仍成立。

------------------------------------------------------------------------

# 11. Future Bridge

后续：

Knowledge Space

↓

Typed Graph / DSL

↓

Synthetic Tasks

↓

Toy Language Models

↓

Open LLM

目标：

研究真实模型是否也具有可检测的 capability structure。

------------------------------------------------------------------------

# 12. Codex Implementation Order

不要实现神经网络。

首先实现：

1.  knowledge space generator；
2.  state enumerator；
3.  response simulator；
4.  identifiability checker；
5.  exact certificate solver；
6.  heuristic solvers；
7.  evaluation pipeline。

第一阶段目标：

> 证明结构化能力空间是否客观支持低成本能力评估。
