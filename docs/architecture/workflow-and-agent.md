# Workflow 与 Agent

CareMate 使用外层工作流与单主 Agent 配合的架构。工作流规定必要步骤和分支；Agent 在允许的执行条件下选择工具，并根据结果继续处理。

## 九类意图与五条路线

公共入口为 `resolve_reference → classify_intent`。意图分类输出一个主标签，随后读取 [INTENT_TO_ROUTE](../../app/graph/routing.py) 分流。

| 意图 | 路线 | 后续处理 |
| --- | --- | --- |
| 商品咨询 | knowledge | retrieve_knowledge → confidence_check → main_agent 或 fallback_reply |
| 退款退货、售后 | refund_flow | fetch_order → retrieve_policy → main_agent |
| 订单、物流、人工 | business | main_agent |
| 投诉 | escalate | complaint_reply |
| 闲聊、其他 | fallback_script | script_reply |

[build.py](../../app/graph/build.py) 定义节点和边。确定性指执行顺序和分支条件由代码规定，不表示节点内部模型输出固定。当前未实现通用的多意图任务拆分。

## 状态与历史

[ConversationState](../../app/graph/state.py) 包含消息、意图、订单、证据、步数和前端动作。节点返回局部字段更新；`messages` 使用 `add_messages` 合并，`trace` 使用自定义合并函数。

[runtime.py](../../app/graph/runtime.py) 为新用户轮重置临时输出。继续已有会话时，`_new_turn_input` 从 MySQL 恢复已展示的历史消息；中断恢复不走这条历史重建逻辑，以保留当前工具请求和暂停状态。

## ReAct 式工具循环

[main_agent](../../app/graph/nodes.py#L316) 获取工具清单，使用 LangChain 的 `bind_tools()` 绑定定义，并通过 `ainvoke()` 请求模型。模型可能直接回答，也可能产生 `AIMessage.tool_calls`。

[should_continue](../../app/graph/routing.py#L30) 检查是否存在工具调用以及是否达到步数上限。允许继续时进入 [agent_tools](../../app/graph/nodes.py#L338)，后端执行工具，将结果写成 `ToolMessage`，再回到主节点。

这一结构对应“决策 → 行动 → 观察结果 → 再决策”。当前不要求模型显式输出 Thought 文本；执行依据是结构化工具请求。每次主模型调用增加一个 step，工具执行本身不增加该计数。

达到上限后直接进入日志节点，没有额外的总结生成节点，因此强制停止不代表任务已完成。

## 业务工具与 MCP

静态定义包含五个内置工具：`query_order`、`query_product`、`query_faq`、`submit_refund`、`create_ticket`。两台 MCP 服务分别提供物流查询，以及保修、退货进度查询。

[注册表](../../app/tools/registry.py) 合并内置与动态发现的工具；[MCP 客户端](../../app/tools/mcp_client.py) 在服务不可达时跳过对应工具。[执行引擎](../../app/tools/engine.py) 负责参数校验、可信参数注入、超时、有限重试、结果格式化与审计。

主 Agent 绑定当时可用的全量工具，没有按意图裁剪工具白名单。只有 `create_ticket` 被登记为写工具，未知名称默认按读处理；接入新的外部服务时需要重新核对权限语义。

## 人工确认与恢复

`fetch_order` 先提取订单号并检查归属，缺少有效订单时通过 `interrupt()` 返回选择器。`create_ticket` 参数有效时先返回工单预览，用户确认后才执行写入。恢复使用 `Command(resume=...)` 和对应会话的 `thread_id`。

SQLite checkpointer 保存图运行进度。恢复时中断节点会从头执行，所以中断前应避免不可重复的副作用。检查点不自动提供外部写操作的幂等保证。

`submit_refund` 在工具节点中被转换为退款工单表单，不经过普通工具执行路径进行支付退款。

## 观测与排障

[观测模块](../../app/core/observability.py) 在图上挂载 Langfuse 回调。`langfuse_session_id` 关联会话，意图标签支持按业务类别聚合。[成本脚本](../../scripts/cost_by_intent.py) 汇总 Token 与去重 trace 数。

定位拒答时，依次检查意图、路由、检索结果、证据门控以及是否进入主 Agent；只有进入工具循环后，才进一步检查工具请求与执行结果。模型能查询某类业务，不代表所有路线都会到达该模型节点。

知识路线的四信号评分与语义自检没有被退款政策路线完整复用。提示词中的“根据证据回答”和“不要重复查询”属于行为引导，不能代替代码约束或最终事实校验。
