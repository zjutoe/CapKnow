# Phase 6 Report: DSL Bridge and Executable Capability World

## DSL Design
- 定义 `Program` AST：
  - `PrimitiveNode`：封装基础指令。
  - `SequenceNode`：顺序执行多个子程序。
  - `ConditionNode`：基于条件分支的程序节点。
  - `LoopNode`：固定次数循环节点。
- `Program` 接口统一提供：
  - `required_primitive_ids()`
  - `to_task_id()`
  - `to_dict()`
- 依赖关系与执行语义集中在单一递归 `execute(program, state_has, input_context=None)` 中实现。

## Primitive Definitions
- 在 `primitives.py` 中新增基础能力定义：
  - `ADD`
  - `COMPARE`
  - `MEMORY`
  - `SEARCH`
  - `FILTER`
  - `LOOP`
  - `CONDITION`
- 每个 primitive 使用 `PrimitiveOperation` 元数据结构表达 `op_id / input_type / output_type / difficulty / requires`。
- 提供 `PRIMITIVES` 与 `get_primitive`、`list_primitives` 等访问接口。

## Composition Rules
- 在 `composition.py` 中定义可组合规则模型：
  - `CompositionRule(left, right, result)`。
  - `DEFAULT_COMPOSITION_RULES`：
    - `MEMORY + SEARCH -> RETRIEVAL`
    - `RETRIEVAL + CONDITION -> PLANNING`
- `CapabilityGraph` 记录规则集合；`resolve_composition` 构建查找表。
- `build_composite_program` 目前返回 `SequenceNode` 组合实现（保持最小语义，便于最小可运行基线）。

## Task Generation
- 新增 `task_generator.py`：
  - `generate_dsl_primitive_task_map()` 生成 primitive 任务程序映射。
  - `_build_composite_rules(...)` 由组合规则闭包构造复合程序。
  - `generate_dsl_world(include_composite=True, composition_rules=...)`：
    - 生成 `KnowledgeSpace`；
    - 根据 `prerequisites` 过滤有效状态；
    - 返回 `(world, task_programs)`。
- 规则与世界元信息写入 `generator_rules` 与 `metadata`（含 composition 与 prereq 约束）。

## Certificate Migration Results
- 对接 `solve_exact_certificate` 的签名函数通过 `make_dsl_response_signature(task_programs)` 提供，返回：
  - 对每个 state 与有序任务集合执行 DSL 程序；
  - 支持 `noise` 与 `rng` 参数（与现有 response 接口兼容）。
- `tests/test_dsl.py` 覆盖：
  - primitive 执行；
  - composition 执行；
  - 无效程序拒绝；
  - 世界可复现性；
  - 复合任务与 primitive 的响应关系；
  - 证书求解链路可运行（在组合 world 上验证可执行；在 pure primitive world 上验证 exact certificate 可通过）。

## Limitations
- 当前复合程序采用最小实现（`SequenceNode` 连接），未引入复杂控制语义分支扩展，以遵循最小实现约束。
- 复合世界不额外声明“可辨识性必须成立”作为约束；是否可辨识由证书求解与报告链路直接验证。
- 目前仅提交了阶段内最小可运行骨架：未扩展到 LLM/学习型语义，也未引入额外搜索策略。
