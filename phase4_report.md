# Phase 4 Report: Adaptive Certificate Solver

## 实施内容

- 新增 `certificate/decision_tree.py`：
  - `DecisionNode` 结构与 `to_dict()`
  - `tree_signature()`，用于决策树等价性校验
- 新增 `certificate/policies.py`：
  - `select_random_question`
  - `select_entropy_reduction_question`
  - `select_balanced_split_question`
  - `resolve_policy`
- 新增 `certificate/adaptive_solver.py`：
  - `solve_adaptive_certificate(...)`
  - `validate_adaptive_certificate(...)`
- 新增 `AdaptiveCertificate` 结果结构（`certificate/result.py`）：
  - `root`
  - `worst_case_depth`
  - `average_depth`
  - `node_count`
  - `valid`
- 导出 API：
  - `certificate/__init__.py` 增加 `AdaptiveCertificate`、`solve_adaptive_certificate`、`validate_adaptive_certificate`、`DecisionNode`、`tree_signature`
- 新增 `tests/test_adaptive_certificate.py`

## 方法

- 候选状态集合 `V_t` 初始化为知识空间全部合法状态。
- 每一步通过策略函数在未问集合中选择问题：
  - `entropy`：优先最大化熵（信息增益）；
  - `balanced`：优先最小化 `max(|V+|, |V-|)`；
  - `random`：按 RNG 随机采样。
- 按返回答案切分候选状态并递归构建 `DecisionNode`。
- 生成树后计算：
  - `node_count`
  - `worst_case_depth`
  - `average_depth`（按等权状态平均）
- `validate_adaptive_certificate` 逐状态重放路径，要求：
  - 每条路径问答一致
  - 叶节点唯一命中该状态

## 示例树（Chain 世界）

`generate_chain_world(["A", "B", "C", "D"])`（balanced/entropy 两种策略一致）：

```
{'question': 'B',
 'yes_child': {'question': 'C',
               'yes_child': {'question': 'D',
                             'yes_child': {'question': None, 'candidate_state_ids': ['{A,B,C,D}']},
                             'no_child': {'question': None, 'candidate_state_ids': ['{A,B,C}']}},
               'no_child': {'question': None, 'candidate_state_ids': ['{A,B}']}},
 'no_child': {'question': 'A',
              'yes_child': {'question': None, 'candidate_state_ids': ['{A}']},
              'no_child': {'question': None, 'candidate_state_ids': ['{}']}}}
```

## 固定 vs 自适应对比（采样结果）

- Chain 世界：
  - `fixed`: 4（exact）
  - `adaptive`: `worst_case_depth=3`, `average_depth=2.4`（balanced/entropy）
- 无结构世界（3任务）：
  - `fixed=3`
  - `adaptive worst_case_depth=3`（entropy）
- Tree 世界示例：
  - 也能成功建树并通过 `validate_adaptive_certificate`
  - `worst_case_depth=3`, `average_depth≈2.857`, `node_count=13`

## 验证命令与结果

- 命令：`python -m pytest tests/`
- 结果：`20 passed in 0.03s`

## 局限与下一步

- 当前实现默认状态空间使用等概率状态分布计算平均深度；未考虑状态先验偏置。
- 未实施 noisy response 或 learned policy（按计划排除）。
- 未进行规模扩展实验（task 数/状态数扫参），当前为功能实现与可复现性验证。
