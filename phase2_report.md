# Phase 2 Report: Identifiability Audit

## 实现内容

- 新增 `validation/identifiability` 审计模块。
- 新增 `response_signature(state, task_ids)`。
- 新增 `check_identifiability(knowledge_space)`。
- 新增稳定状态 ID 及压缩比指标。
- 新增 `IdentifiabilityReport` 数据结构与 `to_dict`。
- 新增示例：`examples/run_identifiability_audit.py`。
- 新增单测：`tests/test_identifiability.py`。

## identifiability 定义

- 对每个合法状态 K 生成完整响应签名 R(K) = (Y(q1),...,Y(qn))。
- 若存在 K_i ≠ K_j 且 R(K_i)=R(K_j)，则判定出现 collision。
- 当 collision 组数为 0 时，定义为可识别（identifiable）。

## 测试结果

- 运行 `python -m pytest tests/test_identifiability.py`
- 收集并执行 4 项：全部通过
  - `test_chain_space_is_identifiable`
  - `test_artificial_collision_is_detected`
  - `test_tree_space_is_identifiable`
  - `test_unstructured_state_signatures_are_unique`
- 示例脚本命令为 `python examples/run_identifiability_audit.py`（从仓库根目录直接运行）。

## Collision 示例

- 人工世界测试通过 `response_signature_fn` 注入验证 collision 的发现能力。

## 当前限制

- 审计默认按 `KnowledgeSpace.valid_states` 进行，并默认只对 `is_valid_state` 为真的状态进行统计。
- 人工 collision 用例依赖于可选的签名构造函数注入。
