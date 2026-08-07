# Phase 5 Report: Probabilistic Certificate

## 实施内容

- 新增 `capability_certificate_lab/probabilistic/response_model.py`
  - `ResponseNoiseModel(slip, guess)`
  - `response_probability(state_has_task, noise)`
  - `simulate_probabilistic_response(state_has_task, noise, attempts=1, rng=None)`
- 新增 `capability_certificate_lab/probabilistic/posterior.py`
  - `infer_state_posterior(knowledge_space, observations, noise, prior=None)`
  - `PosteriorResult(state_count, state_posteriors, map_state, entropy, confidence, observation_count)`
- 新增 `capability_certificate_lab/probabilistic/adaptive_policy.py`
  - `select_random_question`
  - `select_entropy_reduction_question`
  - `select_expected_error_reduction_question`
  - `resolve_policy`
- 新增 `capability_certificate_lab/probabilistic/noisy_certificate.py`
  - `NoisyFixedCertificate`
  - `NoisyAdaptiveCertificate`
  - `solve_noisy_fixed_certificate(...)`
  - `solve_noisy_adaptive_certificate(...)`
- 新增导出 `capability_certificate_lab/probabilistic/__init__.py`
- 新增 `tests/test_probabilistic.py`

## 方法

- 以 task 成功/失败为二值输出，定义 slip/guess 噪声：
  - 若任务在状态内：`P(Y=1)=1-slip`
  - 若任务不在状态内：`P(Y=1)=guess`
- 在 `infer_state_posterior` 中直接使用贝叶斯更新：
  - 先验来自均匀分布或 `prior` 输入
  - 对每个观测独立乘积似然
  - 归一化后输出后验分布、MAP、熵、最大后验置信度
- `solve_noisy_fixed_certificate`
  - 给定已选问题和观测，多次观测可直接展开为多个独立样本
  - 使用 `confidence >= 1 - delta` 判定是否通过
- `solve_noisy_adaptive_certificate`
  - 使用 posterior 初始化；
  - 逐步调用 policy 选择问题；
  - 模拟真实状态响应（含重复尝试）并更新 posterior；
  - 直到达到置信阈值或耗尽问题/上限
- `expected_error` 与 `entropy` policy 都使用后验分布和重复探测（`attempts_per_query`）条件下的一步响应计数分支做期望减益评估。

## 测试结果

- 文件：`tests/test_probabilistic.py`
- 覆盖：
  - 无噪声时与确定性 baseline 一致性
  - slip 高噪声导致识别不稳定
  - 高猜测率下避免过度高置信
  - 后验归一化
  - 重复观测的后验收敛性质

## 风险与限制

- 本实现未引入任何 learned policy（按阶段排除项处理）；
- 重复 probing 当前默认按独立重复观测纳入同一 query 的贝叶斯更新；
- 未对大规模状态空间运行完整鲁棒性曲线实验（保留为后续实验计划，当前报告先给出实现与单元级鲁棒性验证）。

## 未完成/后续

- 未覆盖 handoff 要求中的完整 `robustness curves` 与固定/自适应噪声扫描曲线（实验脚本与指标汇总计划中预留）。
- 当前版本实现了重复观测后验更新与策略评估的一致性，但未接入持久实验产物导出流程。
