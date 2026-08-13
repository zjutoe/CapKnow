# Phase 3 Report: Exact Fixed Certificate Solver

Status: the clean revalidation below is bound to source commit `1bfc6ced88cf3397f240c3e79d1996955e9d589f`. The combined Phase 2-6 evidence package remains pending independent review.

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

## Revalidation Summary

Current clean artifact: `artifacts/phase2_6_revalidation/phase3_fixed_regression.json` (source commit `1bfc6ced88cf3397f240c3e79d1996955e9d589f`; result `6b7186d7da60e7655036f88c39314264e9ee0004b272525ab11e9eae4744edef`; manifest `acbdfaefcc0f64c8eac8db1f2e65744c3575a3cc8335966ca3ec6832d764a3c9`).

| world | tasks | states | exact certificate size | exact valid | greedy valid | random valid | separated pairs |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| chain | 4 | 5 | 4 | true | true | true | 10 |
| tree | 4 | 7 | 4 | true | true | true | 21 |
| unstructured | 3 | 8 | 3 | true | true | true | 28 |

## 已知限制

- 当前 exact 解法是穷举规模搜索，复杂度随任务数指数增长；适合中小规模的确定性世界。
- non-identifiable 输入在当前任务语义下可通过 `response_signature_fn` 人工构造场景覆盖（默认 membership 响应下，去重且合法的 `valid_states` 通常可识别）。
