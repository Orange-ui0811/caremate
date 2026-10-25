# 架构与实现

本文档介绍 CareMate 的模块职责、源码入口、运行约束和验证方式。代码定位表见 [源码索引](source-map.md)。

## 模块导航

| 模块 | 说明 | 关键代码 |
| --- | --- | --- |
| Workflow 与 Agent | [路由、状态、工具循环与暂停恢复](workflow-and-agent.md) | [图定义](../../app/graph/build.py)、[节点](../../app/graph/nodes.py) |
| RAG | [知识处理、检索、证据评分与评估](rag-and-confidence.md) | [检索](../../app/core/retrieval.py)、[置信度](../../app/core/confidence.py) |
| 上下文 | [历史恢复、滑窗、摘要与指代补全](context-engineering.md) | [记忆](../../app/core/memory.py)、[摘要](../../app/core/summarizer.py) |
| 知识反馈 | [问题池、人工审核与主题分类](feedback-and-classifier.md) | [归并](../../app/core/flywheel.py)、[审核](../../app/api/review.py) |
| 模型请求治理 | [调用池](../model-pool.md) | [模型封装](../../app/core/llm.py) |

## 组件分工

LangChain 组织提示词、模型调用、结构化输出和工具消息；LangGraph 管理节点、共享状态、条件分支和循环。物流与售后工具通过 MCP 接入，Langfuse 回调记录受支持的运行事件。

一次业务请求先补全问题、识别意图，再按固定映射分流。需要由程序保证的步骤放入工作流；主 Agent 根据已有材料选择工具并组织回答。多个节点调用模型不等于多个自治 Agent。

## 数据与运行边界

- MySQL 保存业务会话、知识、工单与反馈；SQLite checkpointer 保存图执行状态。两者用途不同。
- 订单、商品、物流及部分售后查询使用模拟数据。工单写入 MySQL，退款表单不执行资金退款。
- 九类意图用于在线流程分流；17 类主题分类用于旁路知识缺口分析。
- 证据门控控制知识分支是否进入生成阶段，不能保证所有路线的最终回答都正确。
- Token 报表来自观测数据；金额分析取决于模型价格和计费口径。

## 验证方式

先按 [部署说明](../../DEPLOY.md) 准备环境，再根据模块选择测试和评估脚本。运行前确认真实服务、模型调用及写入操作的前置条件。

- 状态与分流：[状态测试](../../tests/graph/test_state.py)、[路由测试](../../tests/graph/test_routing.py)。
- 工具行为：[注册测试](../../tests/tools/test_registry.py)、[执行引擎测试](../../tests/tools/test_engine.py)。
- 检索质量：[固定问集](../../tests/data/eval_ch04.jsonl)、[评估脚本](../../scripts/eval_ch04.py)。
- 上下文策略：[合成案例](../../tests/data/context_engineering_v1.json)、[评估](../../scripts/eval_context_engineering.py)、[审计](../../scripts/audit_context_eval.py)。
- 观测接入：[回调测试](../../tests/core/test_observability.py)。

本机运行报告与会话记录未包含在公开源码中。比较结果时应同时记录代码版本、数据版本、模型配置、失败数量和统计分母；单次成功请求不能代表稳定成功率。

历史设计位于 `docs/superpowers/`，描述不同阶段的实现计划。当前行为以源码为准，旧节点名、意图数量和处理路线不能直接当作当前配置。
