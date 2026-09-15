# 研究文档索引

本目录集中保存研究计划、阶段 handoff、审查记录和报告。代码、运行示例与实验 artifact 的路径仍以仓库根目录为基准，分别位于 `scripts/`、`examples/` 和 `artifacts/`。

历史文档正文与 artifact manifest 中的路径记录的是其原始 commit 和当时的目录布局，因此不做批量改写。需要核对历史文件时，可用 Git 的 `commit:path` 定位；当前文档从本目录读取。代码与测试必须区分冻结的历史 `commit:path` 和当前文件系统路径，不应使用当前 `HEAD` 查询已经迁移的历史路径。此次整理只改变文档位置，不改变科学结论、授权边界或冻结 artifact。

## 总览

- [研究计划](Capability_Knowledge_Space_Certificate_Laboratory_Research_Plan.md)
- [研究进展与阶段成果总览](Capability_Certificate_Research_Progress_Summary.md)
- [Phase 8 handoff 修订与补充建议](Phase8_Toy_LM_Handoff_Revision_and_Supplement_Recommendations.md)
- [Phase 2–6 审查比较](phase2_6_gpt53_gpt55_review_comparison.md)

## 阶段 handoff

- [Phase 1：Knowledge Space Core](Capability_Certificate_Task_Handoff_001_Phase1_Knowledge_Space_Core.md)
- [Phase 2：Identifiability Audit](Capability_Certificate_Task_Handoff_002_Phase2_Identifiability_Audit.md)
- [Phase 3：Exact Fixed Certificate Solver](Capability_Certificate_Task_Handoff_003_Phase3_Exact_Fixed_Certificate_Solver.md)
- [Phase 4：Adaptive Certificate Solver](Capability_Certificate_Task_Handoff_004_Phase4_Adaptive_Certificate_Solver.md)
- [Phase 5：Probabilistic Certificate](Capability_Certificate_Task_Handoff_005_Phase5_Probabilistic_Certificate.md)
- [Phase 6：DSL Bridge](Capability_Certificate_Task_Handoff_006_Phase6_DSL_Bridge.md)
- [Correctness Repair and Revalidation](Capability_Certificate_Task_Handoff_007_Correctness_Repair_and_Revalidation.md)
- [Second Review Repair](Capability_Certificate_Task_Handoff_008_Second_Review_Repair.md)
- [Phase 7：Structural Compressibility](Capability_Certificate_Task_Handoff_009_Phase7_Structural_Compressibility.md)
- [Phase 8：Toy Language Model Bridge](Capability_Certificate_Task_Handoff_010_Phase8_Toy_Language_Model_Bridge.md)
- [Phase 9：Decomposed Learned Behavior Bridge](Capability_Certificate_Task_Handoff_011_Phase9_Decomposed_Learned_Behavior_Bridge.md)

## 报告

- [Phase 1](phase1_report.md)
- [Phase 2](phase2_report.md)
- [Phase 3](phase3_report.md)
- [Phase 4](phase4_report.md)
- [Phase 5](phase5_report.md)
- [Phase 6](phase6_report.md)
- [Phase 7](phase7_report.md)
- [Phase 2–6 correctness repair 与重验证](phase2_6_revalidation_report.md)

## Phase 8

[Phase 8 文档与证据索引](phase8/README.md)汇总了细分任务、停止条件、审查状态和冻结证据边界。Phase 8 的全部专项文档与只读 verifier source 均位于 `docs/phase8/`。

新的 handoff 和报告应直接放在 `docs/`；未来与 Phase 9 类似的阶段专项文档应放在 `docs/phase9/`。
