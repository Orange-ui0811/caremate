# 面试源码速查：LangChain、LangGraph、MCP 与 Langfuse

核对日期：2026-10-25。先用自己的话回答，再点击对应源码证明。此页只引用源码，不依赖本机数据库、会话或评估产物，适合作为 GitHub 展示入口。

链接的 `#L` 为本次核对的 GitHub 行号；代码变化后按函数名重新定位。面试前可在 GitHub 源码页按 `y` 转为提交固定链接，再选择要展示的行，避免分支变化使证据漂移。

## 1. LangChain 如何组织模型、提示词、结构化结果与工具调用？

口述：我用 LangChain 统一提示词、模型调用、结构化结果和工具消息。业务规则由项目定义，框架负责调用与组合机制。

| 追问 | 点击源码 | 找到的关键内容 | 需要说明的边界 |
| --- | --- | --- | --- |
| 模型在哪里创建？ | [get_chat_model](../../app/core/llm.py#L74) | 模型配置与 ChatOpenAI 封装 | 配置和兼容逻辑是项目代码 |
| 提示词怎么填参数？ | [INTENT_CLASSIFY_PROMPT](../../app/core/prompts.py#L207) | ChatPromptTemplate、query/history | 业务分类规则由提示词内容定义 |
| 如何得到固定字段？ | [structured](../../app/core/llm.py#L161)、[_Intent](../../app/core/intent.py#L13) | with_structured_output、Pydantic 字段 | 结构合法不代表分类正确 |
| 提示词与模型怎么串起来？ | [classify](../../app/core/intent.py#L19) | prompt 与 model 通过管道组合、ainvoke | 解析失败有项目兜底 |
| 工具如何发给模型？ | [main_agent](../../app/graph/nodes.py#L316) | get_all_specs、bind_tools、ainvoke | 绑定当时可用的全部业务工具 |
| Python 函数怎么成为工具？ | [query_order](../../app/tools/builtin/orders.py#L16) | 函数上方的 @tool、参数、说明 | 实际业务逻辑仍需自行实现 |

## 2. LangGraph 如何组织流程、状态、分支与 Agent 循环？

口述：外层图规定业务路线与必要步骤；内层通过主模型和工具节点构成 ReAct 式循环。检查点负责状态保存，interrupt/resume 处理用户选择与确认。

| 追问 | 点击源码 | 找到的关键内容 | 需要说明的边界 |
| --- | --- | --- | --- |
| 整体流程在哪里？ | [_builder](../../app/graph/build.py#L8) | StateGraph、add_node、固定边、条件边 | 图节点不等于独立 Agent |
| 九类到五路线在哪里？ | [INTENT_TO_ROUTE](../../app/graph/routing.py#L7) | 意图到路线的固定映射 | 当前一次输出一个主意图 |
| 节点之间怎么传数据？ | [ConversationState](../../app/graph/state.py#L18) | messages、order_data、evidence、steps | messages 使用 add_messages 合并 |
| 为什么叫 ReAct 式循环？ | [main_agent](../../app/graph/nodes.py#L316)、[agent_tools](../../app/graph/nodes.py#L338) | 模型选择行动，后端执行，ToolMessage 返回结果 | 没有强制输出显式 Thought 文本 |
| 什么时候停止？ | [should_continue](../../app/graph/routing.py#L30) | 无 tool_calls 或达到步数上限 | 达上限停止不代表成功，也无额外总结节点 |
| 缺订单怎么办？ | [fetch_order](../../app/graph/nodes.py#L66) | 归属检查、interrupt 订单选择 | 恢复时节点会重跑，暂停前避免副作用 |
| 状态怎么保存和恢复？ | [init_graph](../../app/graph/runtime.py#L33)、[resume_turn](../../app/graph/runtime.py#L158) | SQLite checkpointer、thread_id、Command(resume) | 检查点不自动保证外部写操作幂等 |
| 新用户轮与工具循环的历史有何不同？ | [_new_turn_input](../../app/graph/runtime.py#L123) | 新轮恢复 MySQL 中已展示的对话；resume 保留中断状态 | 历史恢复与当前轮工具结果回传是两条逻辑 |

## 3. MCP 怎样接入独立物流和售后服务？

口述：服务端通过 FastMCP 暴露工具，客服后端通过 MultiServerMCPClient 发现工具，转换为 LangChain 工具对象并合并注册。模型发出调用请求后，由后端经 MCP 执行，模型不直接连接服务。

| 追问 | 点击源码 | 找到的关键内容 | 需要说明的边界 |
| --- | --- | --- | --- |
| 独立服务定义在哪里？ | [物流工具](../../mcp_servers/logistics_server.py#L22)、[售后工具](../../mcp_servers/aftersales_server.py#L20) | @mcp.tool、FastMCP、服务启动入口 | 物流、保修、退货进度目前为模拟数据 |
| 客服怎样发现工具？ | [get_client](../../app/tools/mcp_client.py#L54)、[fetch_mcp_specs](../../app/tools/mcp_client.py#L67) | get_tools、发现超时、不可达跳过 | 动态发现有延迟与可用性代价 |
| 本地和远程工具怎么合并？ | [get_all_specs](../../app/tools/registry.py#L96) | 内置优先、按名称合并、MCP 来源 | 静态定义共 5 个内置、3 个 MCP；在线数量随发现结果变化 |
| 谁校验和执行？ | [execute_tool_call](../../app/tools/engine.py#L101) | 参数校验、可信参数注入、超时、工具 ainvoke、审计 | MCP 本身不代替业务授权 |
| 写操作如何限制？ | [WRITE_TOOLS](../../app/tools/registry.py#L15)、[agent_tools](../../app/graph/nodes.py#L338) | create_ticket 确认流、submit_refund 表单分支 | 未登记名称默认读；当前不是完整生产权限体系 |

## 4. Langfuse 如何记录运行并统计 Token？

口述：图编译后挂 Langfuse 的 LangChain 回调，用 session ID 关联会话，用意图标签分组，沿实际运行轨迹排查分类、检索、工具和生成阶段，再从 Metrics API 汇总用量。

| 追问 | 点击源码 | 找到的关键内容 | 需要说明的边界 |
| --- | --- | --- | --- |
| 回调在哪里挂？ | [attach_observability](../../app/core/observability.py#L55)、[init_graph](../../app/graph/runtime.py#L33) | graph.with_config(callbacks) | 未配置时原样返回图 |
| 同一会话怎么关联？ | [run_turn](../../app/graph/runtime.py#L144) | langfuse_session_id 与 thread_id | 前者观测归组，后者检查点状态标识 |
| 意图如何进 trace？ | [tag_intent](../../app/core/observability.py#L64) | propagate_attributes、intent 标签及 event | 意图标签是统计维度，不是准确率保证 |
| Token 报表如何计算？ | [_query](../../scripts/cost_by_intent.py#L43)、[_aggregate](../../scripts/cost_by_intent.py#L55) | Metrics API、按意图汇总、trace 去重 | count 是 trace 数；报表不直接核对货币账单 |
| 主 Agent 的 tokens_used 是全部费用吗？ | [main_agent](../../app/graph/nodes.py#L316) | usage_metadata、逐步累加 | 该字段主要累计主 Agent，不等于全链路总消耗 |

## 一条工具链的投屏顺序

以“订单 1001 的物流到哪里了”为讲解示例，不声称这是本次刚执行的线上记录：

1. 打开 `build.py`，说明订单、物流分支如何进入主 Agent。
2. 打开 `nodes.py` 的 `main_agent`，指出工具绑定和模型调用。
3. 打开 `routing.py` 的 `should_continue`，指出工具请求如何触发循环。
4. 打开 `engine.py` 的 `execute_tool_call`，指出模型请求与后端执行的区别。
5. 打开 `mcp_client.py` 和物流服务，说明物流单号如何传给独立服务。
6. 回到 `agent_tools`，指出 ToolMessage 与返回主节点的固定边。
7. 最后打开 `observability.py`，说明这条链怎样留下观测记录。

函数与接口是实现证据；实际成功率、耗时和成本需要带日期、配置与样本范围的实验记录，不能由源码存在推导。
