# Workflow 与 Agent

CareMate 使用外层工作流与单主 Agent 配合的架构。工作流规定必要步骤和分支；主 Agent 根据当前材料选择工具，并结合执行结果继续处理。本文从流程编排、模型交互、工具接入和运行观测四个方面说明实现，再用一条订单物流请求串起调用关系。

源码核对基线：[e9fa3a2](https://github.com/Orange-ui0811/caremate/commit/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6)。下文源码链接固定到该提交，便于核对函数和行号；其他模块入口见 [源码索引](source-map.md)。

## 组件分工与设计依据

| 组件 | 负责的机制 | 项目定义的内容 | 主要源码入口 |
| --- | --- | --- | --- |
| LangGraph | 共享状态、节点、边、条件分支、循环、检查点和中断恢复 | 意图映射、必经步骤、继续与停止条件 | [_builder](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/build.py#L8)、[ConversationState](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/state.py#L18)、[routing.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/routing.py) |
| LangChain | 消息与提示词模板、模型调用、结构化结果解析、工具绑定 | 提示词中的业务规则、输出字段、上下文材料及模型兼容逻辑 | [prompts.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/prompts.py#L207)、[llm.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/llm.py#L74)、[main_agent](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/nodes.py#L316) |
| MCP、SDK 与适配器 | 独立服务的工具暴露、发现和远程调用，转换为 LangChain 工具对象 | 物流与售后业务实现、服务连接、执行校验和结果格式化 | [物流服务](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/mcp_servers/logistics_server.py#L22)、[mcp_client.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/tools/mcp_client.py#L47)、[engine.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/tools/engine.py#L101) |
| Langfuse | 通过回调记录运行事件、模型交互和可获得的用量 | 会话关联、意图标签、Token 聚合与排障口径 | [observability.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/observability.py#L36)、[runtime.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/runtime.py#L33)、[cost_by_intent.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/scripts/cost_by_intent.py#L43) |

这种划分让必要的业务顺序由程序保证，同时保留模型组合工具和组织回答的能力。固定步骤、模型交互、工具执行和运行记录各有独立入口，便于分别修改和定位故障。多个节点调用模型，不等于存在多个自治 Agent。

## LangGraph：流程、状态与 Agent 循环

### 九类意图与五个处理分支

[_builder](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/build.py#L8) 创建 `StateGraph(ConversationState)`，通过 `add_node` 注册节点，通过 `add_edge` 连接固定步骤，通过 `add_conditional_edges` 根据状态选择下一节点。

公共入口为 `resolve_reference → classify_intent`。意图分类输出一个主标签，随后读取 [INTENT_TO_ROUTE](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/routing.py#L7) 分流：

| 意图 | 分支 | 后续处理 |
| --- | --- | --- |
| 商品咨询 | `knowledge` | `retrieve_knowledge → confidence_check → main_agent` 或 `fallback_reply` |
| 退款退货、售后 | `refund_flow` | `fetch_order → retrieve_policy → main_agent` |
| 订单、物流、人工 | `business` | `main_agent` 按需要选择工具 |
| 投诉 | `escalate` | `complaint_reply` 固定回复与操作入口 |
| 闲聊、其他 | `fallback_script` | `script_reply` 固定引导 |

五个分支可以汇入同一主 Agent，最终回答通常经过 `log → END`。当前一次识别一个主意图，未实现通用的多意图任务拆分。

**确定性指流程顺序和分支条件由代码规定。** 节点内部仍可调用模型。例如退款分支固定先取单、再检索政策；`retrieve_policy` 内部可以先用模型扩写检索问题。一般订单物流分支则直接进入 Agent，查询工具的组合由模型根据工具说明与返回结果决定。

### 状态、历史与暂停恢复

[ConversationState](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/state.py#L18) 保存消息、意图、订单、证据、步数和前端动作。节点返回局部字段更新；`messages` 通过 LangGraph 的 `add_messages` 合并，业务决策摘要 `trace` 通过项目的 `merge_dict` 合并。

[_new_turn_input](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/runtime.py#L123) 在新用户轮重置临时输出；继续已有会话时，从 MySQL 恢复已展示的对话。上下文窗口和摘要的具体处理见 [上下文工程](context-engineering.md)。

[init_graph](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/runtime.py#L33) 创建 SQLite `AsyncSqliteSaver`，再编译带 checkpointer 的图。运行时的 `thread_id` 标识对应的图状态；MySQL 保存业务会话和工单，两者用途不同。

[fetch_order](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/nodes.py#L66) 提取订单号并检查归属，缺少有效订单时调用 `interrupt()` 返回订单选择器。`create_ticket` 参数有效时，工具节点先暂停并展示工单预览；用户确认后才执行写入。[resume_turn](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/runtime.py#L158) 使用 `Command(resume=...)` 继续对应会话，新轮历史重建不会覆盖这次恢复中的工具请求和暂停状态。

恢复时中断节点会从头执行，因此中断前应避免不可重复的副作用；checkpointer 本身不提供外部写操作的幂等保证。

### ReAct 式工具循环

[main_agent](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/nodes.py#L316) 调用模型后，[should_continue](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/routing.py#L30) 检查最后一条消息：存在 `tool_calls` 且未达到步数上限时进入 [agent_tools](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/nodes.py#L338)。工具节点返回 `ToolMessage`，固定边 `agent_tools → main_agent` 让模型根据结果继续处理。

这一结构对应“决策 → 行动 → 观察结果 → 再决策”。执行依据是结构化工具请求；当前不要求模型显式输出 Thought 文本。每次主模型调用增加一个 step，工具执行本身不增加该计数。

没有工具请求或达到步数上限时进入日志节点。达到上限后没有额外的总结生成节点，因此强制停止不代表任务已完成。

## LangChain：提示词、模型、结构化结果与工具消息

### 提示词到结构化意图结果

[INTENT_CLASSIFY_PROMPT](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/prompts.py#L207) 使用 `ChatPromptTemplate.from_messages` 定义 system 和 human 消息，并以 `{history}`、`{query}` 填入历史与当前问题。业务分类规则写在项目提示词中；LangChain 提供模板填充和调用组合机制。

[classify](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/intent.py#L19) 将模板与结构化模型串成调用链：

```python
model = llm.structured(_Intent, model=settings.intent_model or None)
r: _Intent = await (INTENT_CLASSIFY_PROMPT | model).ainvoke(
    {"query": query, "history": history or "(无)"})
```

[get_chat_model](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/llm.py#L74) 创建继承自 `ChatOpenAI` 的项目封装，统一配置模型、服务地址、流式用量和兼容参数。[structured](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/llm.py#L161) 调用 `with_structured_output(schema, method="function_calling")`，结合 Pydantic schema 解析并校验返回字段。

[_Intent](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/intent.py#L13) 将意图限定为九类，将 `confidence` 限定在 0–1。字段合法不代表分类正确；`confidence` 是模型返回的判断把握，并非经过校准的正确率。当前路由按 `intent` 查表，`confidence` 用于保存和观测，没有在此按阈值切换模型或分支。意图结果解析失败时，项目返回“其他”和 0 分。

这里的 function calling 用于取得结构化数据；业务工具请求还需经过后面的工具执行流程。

### 消息列表到工具调用请求

[_agent_messages](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/nodes.py#L274) 组织 `SystemMessage`、历史窗口以及本轮任务、摘要、检索证据和订单材料。[main_agent](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/nodes.py#L316) 获取工具清单，再绑定工具并请求模型：

```python
specs = await registry.get_all_specs()
model = get_chat_model(streaming=True).bind_tools([s.tool for s in specs])
msgs = _agent_messages(state)
ai: AIMessage = await model.ainvoke(msgs, config)
```

`bind_tools` 将工具名、说明和参数 schema 加入模型请求。模型返回 `AIMessage`，其中可以包含普通回答或 `tool_calls`；模型给出调用请求后，由后端执行工具。

内置工具通过 LangChain 的 `@tool` 转为工具对象，例如 [query_order](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/tools/builtin/orders.py#L16)。后端将执行结果封装为带 `tool_call_id` 的 `ToolMessage`，使模型能将结果对应到原请求。LangChain 提供这些调用和消息接口，业务实现、执行顺序与授权规则由项目负责。

## MCP：独立业务服务的暴露、发现与调用

[物流服务](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/mcp_servers/logistics_server.py#L22) 使用 MCP Python SDK 的 `FastMCP` 与 `@mcp.tool()` 暴露 `query_logistics(tracking_no)`，通过 `mcp.run(transport="streamable-http")` 提供服务。物流和 [售后服务](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/mcp_servers/aftersales_server.py#L20) 运行在独立进程中，当前使用模拟数据。

客户端使用 `langchain-mcp-adapters` 提供的 `MultiServerMCPClient`。从服务端到实际执行的路径为：

1. [_connections / get_client](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/tools/mcp_client.py#L47) 配置物流、售后两个连接并创建客户端。
2. [fetch_mcp_specs](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/tools/mcp_client.py#L67) 调用 `get_tools` 发现工具，转换为项目的 `ToolSpec`，附加来源、执行策略和结果格式化函数。单个服务发现失败时跳过该服务。
3. [get_all_specs](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/tools/registry.py#L96) 合并内置与 MCP 工具，重名时内置优先，再由主 Agent 绑定给模型。
4. [execute_tool_call](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/tools/engine.py#L101) 查找工具、校验参数、检查确认状态并注入可信参数，随后在超时控制下执行 `spec.tool.ainvoke(args)`。MCP 工具对象经适配器向对应服务发起请求。
5. 执行结果经格式化、审计后转为 `ToolMessage`，由图送回主 Agent。失败也以工具结果返回，供模型修正参数或说明情况。

### 工具清单与执行边界

| 来源 | 工具 | 当前行为 |
| --- | --- | --- |
| 内置查询 | `query_order`、`query_product`、`query_faq` | 订单、商品和知识查询；订单身份由后端注入并检查归属 |
| 内置业务入口 | `submit_refund`、`create_ticket` | 前者在工具节点转换为退款工单表单；后者经用户确认后写入 MySQL |
| MCP | `query_logistics`、`query_warranty`、`query_return_status` | 独立服务提供物流、保修和退货进度模拟查询 |

静态业务定义共 5 个内置工具和 3 个 MCP 工具；实际可用数量取决于服务发现结果。当前主 Agent 绑定当时可用的全量工具，没有按意图裁剪工具白名单。

参数校验、身份注入、确认、超时、有限重试、结果格式化与审计由项目执行层实现。只读工具可以按配置重试暂时性故障，写工具不自动重试。`submit_refund` 的表单分支不执行资金退款。

[WRITE_TOOLS](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/tools/registry.py#L15) 当前只登记 `create_ticket`；未列入写清单的已注册工具默认按读处理。未注册的工具请求由引擎拒绝。接入新服务时，需要同时核对其真实副作用与项目权限登记，MCP 协议本身不替代业务授权。

## Langfuse：运行记录、会话关联与用量分析

[_make_handler / attach_observability](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/observability.py#L36) 创建 Langfuse 的 LangChain `CallbackHandler`，通过 `graph.with_config({"callbacks": [...]})` 挂到编译后的图上。[init_graph](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/runtime.py#L33) 在应用启动时完成挂载；配置不齐全时返回原图。回调设置 `run_inline=True`，用于在异步执行中保持观测上下文的传播。

运行时在 [run_turn](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/runtime.py#L144) 和 [_stream_events](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/runtime.py#L170) 传入 `langfuse_session_id`；[tag_intent](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/observability.py#L64) 追加意图 metadata、标签和事件。

| 标识或记录 | 作用 |
| --- | --- |
| `thread_id` | LangGraph checkpointer 使用的状态标识，用于继续图执行 |
| `langfuse_session_id` | 将同一业务会话的多轮运行归组；代码使用同一个会话 ID 字符串，但用途不同 |
| Langfuse Trace | 回调覆盖的一次运行轨迹；同一 Session 可以包含多条 Trace |
| `ConversationState.trace` | 项目保存的业务决策摘要，如路由、证据闸结果；它不是 Langfuse Trace 对象 |

Trace 用于检查实际经过的节点、模型输入输出、工具请求及返回事件。未执行分支需要结合图定义理解；普通 Python 函数、数据库访问和任意 HTTP 请求也不会仅因挂了回调就全部获得独立观测。

[cost_by_intent.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/scripts/cost_by_intent.py#L43) 从 Metrics API 查询 observations 的 Token，用意图标签分组，并按 `traceId` 去重计数。主 Agent 状态中的 `tokens_used` 主要累计主模型调用，与全链路用量口径不同。金额分析还依赖模型价格配置和服务商计费规则，显示 0 不能直接解释为免费。

## 一条订单物流请求如何经过四个组件

以“订单 1001 的物流到哪里了”为例，下表说明可能的执行过程。订单须属于当前用户，工具顺序以实际模型输出为准；这是一条源码阅读路径，不是已完成请求的观测记录。

| 阶段 | 执行过程 | 对应组件与源码 |
| --- | --- | --- |
| 入口与分流 | 补全指代、识别意图；识别为物流后进入 `business → main_agent` | LangGraph：[图定义](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/build.py#L25)、[意图映射](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/routing.py#L7) |
| 请求模型 | 组织历史和当前材料，绑定可用工具；模型可以请求 `query_order` | LangChain：[main_agent](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/nodes.py#L316) |
| 获取订单 | 图进入工具节点，后端注入用户身份并检查订单归属，返回订单和 `tracking_no` | LangGraph 与项目执行层：[agent_tools](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/nodes.py#L338)、[query_order](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/tools/builtin/orders.py#L16) |
| 查询物流 | 订单结果以 `ToolMessage` 回到模型；模型可以用 `tracking_no` 请求 `query_logistics`，后端通过 MCP 调独立服务 | LangChain、MCP 与执行层：[engine](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/tools/engine.py#L101)、[物流服务](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/mcp_servers/logistics_server.py#L22) |
| 生成与结束 | 物流结果再次回到主模型；没有工具请求时进入 `log → END`，保存回复 | LangGraph 与 LangChain：[循环与结束边](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/build.py#L43)、[log_node](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/nodes.py#L426) |
| 核对运行 | 在对应 Session 中找到本次请求的 Trace，核对模型工具请求、实际执行事件、结果与用量 | Langfuse：[观测接入](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/observability.py#L55)、[会话标识](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/runtime.py#L170) |

订单工具到物流工具的先后关系由数据依赖和工具说明引导；图只固定主模型与工具节点的往返。与之相比，退款分支中的 `fetch_order → retrieve_policy` 是固定边，必须先取得订单材料再取政策。

## 约束与排障

定位拒答时，先检查意图、路由、检索结果、证据门控以及是否进入主 Agent；进入工具循环后，再检查工具是否被发现、请求参数、执行状态和返回材料。将同一次运行的业务日志、工具审计与 Langfuse 记录结合起来，区分流程分流、模型判断和执行故障。

商品知识分支的最高重排分粗筛与生成前语义自检，没有被退款政策分支完整复用。当前证据检查也不等于对全部最终回答进行事实校验。提示词中的“根据证据回答”和“不要重复查询”属于行为引导，应与固定边、权限门、参数校验和步数限制分别理解。检索与证据评分的详细实现见 [RAG 与置信度](rag-and-confidence.md)。
