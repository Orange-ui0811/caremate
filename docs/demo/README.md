# 功能演示与验证导航

阅读顺序：[首页](../../README.md) → [Workflow](../architecture/workflow-and-agent.md) → [工具执行](../engineering/tool-execution.md) → [RAG](../architecture/rag-and-confidence.md) → [上下文](../architecture/context-engineering.md) → [实验](../evaluation/README.md)。环境准备见[运行手册](runbook.md)。

## 物流工具组合

- **输入：**`订单 1001 的物流到哪了？`
- **预期路由：**物流 → 主 Agent。
- **必要数据：**query_order 获取 tracking_no，query_logistics 使用该物流单号查询；不能把订单号当成物流单号。
- **观察：**查看实际工具名、参数、回执与最终回答；工具未成功时不能宣称已查询成功。
- **验证：**[工具定义](../../app/tools/builtin/orders.py)、[物流 MCP](../../mcp_servers/logistics_server.py)、[执行测试](../../tests/tools/test_engine.py)。

## 退货选单

- **输入：**`我要退货，但我不记得订单号了。`
- **预期路由：**退款退货 → fetch_order 暂停选单 → 恢复 → retrieve_policy → 主 Agent。
- **操作：**选择本人列表中的订单，观察恢复后的政策咨询；展示退款表单后停止提交。
- **约束：**不能编造订单号，不能跳过订单归属校验，展示表单不表示实际退款。
- **验证：**[图编排](../../app/graph/build.py)、[选单节点测试](../../tests/graph/test_ch06_nodes.py)。

## 证据不足场景

- **输入：**`你们卖活体宠物吗？`
- **预期行为：**如果进入知识分支且没有充分依据，则兜底并提供人工入口；如果被分到“其他”，则观察固定引导。
- **约束：**一次输入进入固定引导不能证明检索门控生效；需要查看真实路由与检索记录。
- **验证：**[最高重排分](../../app/core/confidence.py)、[证据门控测试](../../tests/graph/test_ch09_confidence_gate.py)。稳定验证门控时使用隔离测试，真实模型路由以实际输出为准。

## 工单确认与取消

- **输入：**`帮我建售后工单，猫砂盆漏电，需要人工处理。`
- **操作：**等待工单预览，点击取消。
- **预期行为：**确认前不调用创建函数；取消后不执行该写操作，并回传取消上下文。
- **验证：**[确认与取消测试](../../tests/graph/test_ch08_confirm_ticket.py)、[本次回归记录](assets/core-tests-20261103.md)。真实数据库无新增的结论需另行比较同一会话的数据库记录，不能只看自然语言回答。

## 知识问答与后台

输入 `MH-LP100 的废砂盒多久倒一次？`，打开引用核对使用条件与频率。随后打开 /observability 查看在用阈值、可答通过率与应拒放行率，打开 /review 查看已有待审问题及召回快照。仅展示已有数据时不运行校准、不核准新答案。

## 证据与版本

真实模型 Demo 的执行路径会变化；上述内容是测试输入和验收条件，不是预录成功结果。本次提供[隔离回归记录](assets/core-tests-20261103.md)，没有将测试替身输出标为真实业务 Trace。历史评测的日期与口径见[实验目录](../evaluation/README.md)，上下文新回查的集成状态见[版本表](../architecture/context-engineering.md#版本与材料)。
