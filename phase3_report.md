# Phase 3 Report: Exact Fixed Certificate Solver

## 实现内容

- 新增 `certificate/` 模块
  - `certificate/exact_solver.py`
  - `certificate/validator.py`
  - `certificate/result.py`
- 新增 `solve_exact_certificate`, `solve_greedy_certificate`, `solve_random_certificate`。
- 新增 `validate_certificate`。
- 新增固定任务集合结果结构 `CertificateResult`。
- 新增 `tests/test_certificate.py`。

## 接口定义

- `solve_exact_certificate(knowledge_space, response_signature_fn=response_signature) -> CertificateResult`
- `solve_greedy_certificate(knowledge_space, response_signature_fn=response_signature) -> CertificateResult`
- `solve_random_certificate(knowledge_space, response_signature_fn=response_signature, seed=None) -> CertificateResult`
- `validate_certificate(knowledge_space, certificate, response_signature_fn=response_signature) -> bool`

## 示例结果与统计

- `CertificateResult` 包含：
  - `task_count`
  - `state_count`
  - `certificate_size`
  - `selected_tasks`
  - `valid`
  - `separated_pairs`
  - `total_pairs`
  - `runtime_ms`
  - `method`
- `exact` 解法通过枚举证书规模 + 逐层子集搜索实现 minimum hitting set 的精确解。
- `greedy` 与 `random` 为基线方案，均输出可验证证书（前提是返回 `valid=True`）。

## 测试结果

- 运行 `python -m pytest tests/`
- 共 15 项，通过 `15 passed`。

## 基线对比

- `solve_exact_certificate`: 返回最小证书。
- `solve_greedy_certificate`、`solve_random_certificate`: 提供 baseline，通常在同一实例中不优于 exact。

## 已知限制

- 当前 exact 解法是穷举规模搜索，复杂度随任务数指数增长；适合中小规模的确定性世界。
- non-identifiable 输入在当前任务语义下可通过 `response_signature_fn` 人工构造场景覆盖（默认 membership 响应下，去重且合法的 `valid_states` 通常可识别）。
