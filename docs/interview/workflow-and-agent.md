# Workflow 与 Agent：讲解、源码与现场演示

按 2026-10-23 的工作区代码核对。本文描述系统已经实现的行为，不代表任何人的个人贡献声明。对外介绍时，应另行如实说明自己的学习、复现和修改范围。

## 60 秒口述

> 客小助 · CareMate 是一个面向电商售后的智能客服演示系统。它把固定业务步骤和模型的工具选择结合起来：LangGraph 负责共享状态、意图分流、必要检索以及暂停恢复；主 Agent 则在这些路径中决定是否调用工具、如何根据结果回答。
>
> 比如商品知识问题先检索，再通过置信度和证据充分性检查；退货请求先拿订单，再查政策，缺订单时暂停让用户选择。物流问题则由 Agent 先查订单拿物流单号，再通过 MCP 调物流服务。创建人工工单需要先展示预览，用户确认后才写入数据库。
>
> LangChain 负责消息、提示词、模型调用、结构化输出和工具绑定；Langfuse 记录链路及模型用量。这里是一个主 Agent 配合确定性 Workflow，订单与物流使用模拟业务数据，工单实际写入 MySQL。它重点验证的是可约束、可观察、可评估的 Agent 应用流程。

不要在这段口述中宣称已经实现 Multi-Agent、真实电商支付退款或工业设备参数自动优化。

## 四个组件各管什么

| 组件 | 当前职责 | 代码证据 | 不应混淆的边界 |
|---|---|---|---|
| LangGraph | State、节点和边、按意图分支、Agent 工具循环、`interrupt` / `Command(resume=...)`、持久检查点 | [build.py](../../app/graph/build.py)、[routing.py](../../app/graph/routing.py)、[runtime.py](../../app/graph/runtime.py)、[state.py](../../app/graph/state.py) | 图检查点使用 SQLite；业务会话、工单等使用 MySQL。图中的节点可以调用 LLM。 |
| LangChain | `ChatPromptTemplate`、消息对象、`ChatOpenAI`、Pydantic 结构化输出、`bind_tools` | [prompts.py](../../app/core/prompts.py)、[llm.py](../../app/core/llm.py)、[intent.py](../../app/core/intent.py)、[nodes.py](../../app/graph/nodes.py) | 模型提出工具调用；实际执行和权限确认由项目代码负责。 |
| MCP | 从独立物流、售后进程发现工具，调用工具，转换返回字段 | [mcp_client.py](../../app/tools/mcp_client.py)、[logistics_server.py](../../mcp_servers/logistics_server.py)、[aftersales_server.py](../../mcp_servers/aftersales_server.py) | MCP 是工具连接协议；不会自动给系统提供业务安全、推理正确性或多智能体协作。当前服务返回模拟数据。 |
| Langfuse | 图回调记录运行链路，按会话关联多轮，附加意图标签，统计模型用量 | [observability.py](../../app/core/observability.py)、[runtime.py](../../app/graph/runtime.py)、[cost_by_intent.py](../../scripts/cost_by_intent.py) | 项目成本报表主要统计 Token。实际金额取决于模型定价配置；不能把 Token 报表说成已核对账单。 |

关键表述：**确定性指程序约束流程顺序，不表示节点内的模型输出固定。** `retrieve_policy` 内先用模型扩写查询，再检索；只要进入退款分支，“拿订单 → 检索政策 → 进入 Agent”的顺序仍由代码固定。

## 九类意图到五类处理出口

以下以 [INTENT_TO_ROUTE](../../app/graph/routing.py) 和 [图构建](../../app/graph/build.py) 为准。“五出口”是意图路由的五个分支名，不是五个最终回答节点。

| 意图 | 路由值 | 路径 | 进入主 Agent |
|---|---|---|---|
| 商品咨询 | `knowledge` | `retrieve_knowledge → confidence_check`，证据强进入 Agent，证据弱进入 `fallback_reply` | 条件进入 |
| 退款退货 | `refund_flow` | `fetch_order → retrieve_policy → main_agent` | 是 |
| 售后 | `refund_flow` | 当前与退款共用取单、检索政策路径 | 是 |
| 订单 | `business` | `main_agent` 按需要调用订单工具 | 是 |
| 物流 | `business` | `main_agent` 按需要组合订单、物流工具 | 是 |
| 人工 | `business` | `main_agent` 可发起创建工单确认流 | 是 |
| 投诉 | `escalate` | `complaint_reply` 固定安抚文字，提供人工、工单入口 | 否 |
| 闲聊 | `fallback_script` | `script_reply` 固定引导 | 否 |
| 其他 | `fallback_script` | `script_reply` 固定引导 | 否 |

公共前置路径为 `resolve_reference → classify_intent`，最终回答通常经过 `log → END`。意图判断是九选一，不是九个 Agent，也不是多标签意图分解。解析意图失败时归为“其他”；路由函数对未知标签的默认值则是 `business`，二者是不同层的防御逻辑。

注意两个实际边界：

- 四信号置信度与语义充分性检查位于商品知识路；`retrieve_policy` 当前没有复用这两道检查。不要把“生成前证据闸”说成所有分支都覆盖。
- `main_agent` 绑定完整注册工具清单。订单查询后查物流，依赖 `query_order` 返回的 `tracking_no` 和工具说明；它不是图上写死的两个连续节点。必要步骤的硬约束与提示词引导要分开解释。

## 投屏时按这个顺序走

先在 GitHub 打开本页，点击对应源码；再切本地聊天页 `http://127.0.0.1:8000/`。每个独立场景新建会话，防止前一段的指代和待恢复状态影响下一段。除非演示上下文，避免复用旧会话。

下面是候选输入与观察点，不是保证模型每次输出完全相同的自动化验收结果。除以下两轮聊天和 MCP 工具发现外，本节的其他聊天输入尚未在本轮检查中实际执行。现场前应按当前模型和配置彩排；请求等待期间可切回源码讲解。不要现场重跑完整评测或训练。

### 本轮实际验证记录（2026-10-23）

使用专门新建的测试会话 `115`，通过 `/api/chat` 验证了以下两轮。耗时为本次请求的观测值，不是稳定延迟承诺。

| 验证项 | 本次结果 | 可以说明什么 |
|---|---|---|
| 首轮询问 MH-LP100 废砂盒清理频率 | 37.98 秒完成，首字约 33.36 秒，返回回答及 10 条引用 | 本次知识问答链路成功返回；引用数量不代表答案每一句都正确，也不等于检索 Recall。 |
| 同会话追问“它支持 App 远程监控吗？” | 113.07 秒后返回固定拒答，0 条引用 | 当前这条追问不适合当稳定成功演示；可以用于解释分阶段排障和拒答门控。 |
| `fetch_mcp_specs()` 工具发现 | 实际发现 `query_logistics`、`query_warranty`、`query_return_status` | 两台服务通过 MCP 客户端完成工具发现；这不等于这三种业务聊天都已端到端验证。 |
| Langfuse 采集 | 两轮 trace 分别有 36、33 个 observations，均有关联记录；两条 `total_cost` 均为 0 | 证明链路采集存在，不能证明实际费用已正确计量；observations 数也不等于模型调用次数。 |

第二轮检查点中的指代补全正确；证据分数约 `0.2698`，超过在用阈值 `0.26`，但 `fallback_source=self_check`、`evidence_strong=false`。因此这次的直接拒答位置是后续语义充分性闸，不能归因于四信号分数没过线，也不能说“指代补全正确就一定答对”。这份终态只能定位拒答分支；要区分证据确实不足、自评误拒或自评调用失败的安全降级，还需要结合该节点的 trace 和错误记录判断。

可在 Langfuse 按会话 `115` 或以下 trace ID 查找：

- 首轮：`df0b831e764787c384df6a410908fbb5`
- 追问：`275a9a3a5690e3add84337b0600dd9cb`

一次成功与一次失败都保留展示价值。建议先展示已保存的成功回答和引用，再用失败请求讲“先查指代、再查数值闸、最后查语义闸”的排障顺序；不要现场反复重试，直到碰巧答对才当成稳定能力。

### 1. 商品知识：程序固定检索与证据门控

**首选展示：**本轮已经返回回答的 MH-LP100 废砂盒清理频率测试记录，并展开引用原文。

**待彩排的独立输入：**“智能猫砂盆 Pro MH-LP100 有没有 App 远程监控？”该完整问题来自 [检索评估集 B2](../../tests/data/eval_ch04.jsonl)，本轮没有单独验证。不要用它的完整实体问法，替代上面已失败的“它支持 App”追问并声称追问已成功。

**看什么：**检索活动、答案的引用及引用原文。若当前证据闸拒答，就解释实际拒答原因，不把兜底包装成答对。

**怎么讲：**“这条路径不能跳过检索直接让主 Agent 凭记忆回答。检索结果先算置信度，再检查语义上是否足够回答；通过后将编号证据交给 Agent。”

**源码：**[retrieve_knowledge 与 confidence_check](../../app/graph/nodes.py)、[confidence.py](../../app/core/confidence.py)、[selfcheck.py](../../app/core/selfcheck.py)。

### 2. 订单物流：Agent 组合本地与 MCP 工具

**输入：**“订单 1001 的物流到哪了？”

**看什么：**`query_order → query_logistics → 回答` 的实际工具轨迹；Langfuse 中同一轮的模型与工具调用。在聊天页只看到简短工具提示时，用准备好的本轮 trace 或 `/api/agent` 返回轨迹展示具体入参结果，不依靠答案文字猜测调用顺序。

**怎么讲：**“这里先查订单取得物流单号，再传给独立物流 MCP 服务。模型负责组合工具，统一执行引擎校验参数、设定超时并记录审计。”

**源码：**[query_order](../../app/tools/builtin/orders.py)、[query_logistics](../../mcp_servers/logistics_server.py)、[统一执行引擎](../../app/tools/engine.py)。

**演示数据：**[business.py](../../app/tools/business.py) 给每个非空用户标识提供 `1001`、`2002` 两笔模拟订单，因此网页生成的随机用户标识也可直接使用。状态、金额和物流单号按固定种子生成，并非真实商家订单。不要声称有线上电商业务量。

### 3. 退货选单：Workflow 的暂停与恢复

**输入：**在新会话中说“我要退货，但我不记得订单号了。”

**看什么：**订单选择卡；选择 `1001` 后继续执行政策检索及回答。本段不提交退款表单。

**怎么讲：**“缺订单时不是让模型猜。`fetch_order` 发出 `interrupt`，检查点保留状态；用户选择后，前端请求续跑接口，`Command(resume=...)` 把订单号交回节点。节点从头重跑，所以中断之前只做只读操作，并再次校验订单归属。”

**源码：**[fetch_order、retrieve_policy](../../app/graph/nodes.py)、[resume_turn、stream_resume](../../app/graph/runtime.py)、[actions.py](../../app/api/actions.py)。

**限制：**选单和政策检索顺序确定，但最终是否建议退款由模型结合模拟订单和证据生成。当前“退款提交”实际创建类型为“退款”的工单；没有调用支付网关，也没有执行资金退回。

### 4. 工单预览：写操作确认门

**输入：**“我的智能猫砂盆一开机就跳闸，请帮我建一个售后工单，交给人工跟进。”

**看什么：**包含工单类型、问题描述的确认预览。停在预览处即可证明“提出操作”和“执行操作”分离；如需演示拒绝分支，选择取消，不点击确认提交。

**怎么讲：**“Agent 发出 `create_ticket` 只代表建议。参数校验通过后暂停，只有用户明确确认，执行引擎才获得 `confirmed=True`。写工具超时不自动重试，因为超时不等于没有执行，盲目重试可能重复建单。”

**源码与测试：**[agent_tools](../../app/graph/nodes.py)、[execute_tool_call](../../app/tools/engine.py)、[确认与取消的图级测试](../../tests/graph/test_ch08_confirm_ticket.py)。

**限制：**这是当前 Agent 工具执行路径的确认门。项目还有前端按钮调用的动作接口，不能据此声称系统已经完成生产级身份认证、完整授权及幂等保障。

### 5. Langfuse：用一次具体请求解释排障

切到 `http://127.0.0.1:3000/`，打开刚才会话对应的 trace。依次展示意图分流、模型输入、工具请求和耗时、模型 usage；按 `langfuse_session_id` 解释同一会话如何关联多轮。投屏前准备可展示的测试 trace，避免暴露真实会话、凭据或环境配置。

本轮会话 `115` 已验证实际 trace 采集，具体 ID 与 observations 数见上表；两条 `total_cost=0`，目前只能展示链路与用量，不能把零金额说成免费，也不能声称实际费用已经正确计算。如果现场新 trace 没及时出现，展示这两条已核验记录，注明是先前运行。`/observability` 的意图成本图来自已生成报表，不保证刚才请求立即出现在图中。

## 面试官追问时必须答清楚

| 追问 | 对应回答 | 证据 |
|---|---|---|
| 为什么不用一个自由 Agent 做完？ | 要求商品知识先检索、退款先取单查政策、写操作先确认，这些要求由图和执行引擎承担；工具组合与自然语言回复交给模型。 | [build.py](../../app/graph/build.py)、[engine.py](../../app/tools/engine.py) |
| 怎么防止 Agent 打转？ | 每轮模型调用累加 `steps`，达到 `max_agent_steps` 就停止工具循环，默认配置为 6；Token 记录用于观测，并不是这里的停止条件。 | [routing.py](../../app/graph/routing.py)、[config.py](../../app/config.py) |
| 达到步数上限会保证一个完整答案吗？ | 当前仅强制退出循环并记录最后回答，没有额外的总结生成节点。若最后一条仍是工具调用，可能没有完整文字；这是可改进点。 | [should_continue](../../app/graph/routing.py)、[resolve_answer](../../app/graph/nodes.py) |
| 模型能伪造确认参数吗？ | `confirmed` 不在模型工具参数中，由图恢复后的代码传给执行引擎；身份和会话参数也由执行侧注入。 | [engine.py](../../app/tools/engine.py) |
| 为什么检查点是 SQLite，业务库却是 MySQL？ | 检查点保存图运行状态以恢复流程，MySQL 保存业务会话、工单、知识反馈等，两者职责不同。当前是单机演示部署。 | [runtime.py](../../app/graph/runtime.py)、[repository.py](../../app/db/repository.py) |
| 工具扩展不改核心代码，怎么实现？ | 启动扫描内置模块，模块导入即注册；MCP 工具通过客户端动态获取，两者合并为 `ToolSpec`。 | [registry.py](../../app/tools/registry.py)、[mcp_client.py](../../app/tools/mcp_client.py) |
| 工具权限有没有缺口？ | 当前只把已登记的 `create_ticket` 视为写工具，未知名称默认只读，只适合现有受控模拟服务。接入外部系统前需要默认拒绝和明确的权限元数据。 | [WRITE_TOOLS、permission_for](../../app/tools/registry.py) |
| 主 Agent 是否只能调用当前意图的工具？ | 当前绑定全量工具，并未按分支做工具白名单。硬约束在图前置步骤、执行校验和确认门；部分行为仍依赖提示词。 | [main_agent](../../app/graph/nodes.py) |

## 与招聘岗位如何衔接

| 岗位方向 | 现有可迁移经验 | 尚未实现、应作为下一步 |
|---|---|---|
| 感知—诊断—决策—执行 | 输入规范化、意图/路径分流、检索证据、受控工具执行、轨迹审计的工程分层 | 没有打印图像、传感器时序或真实打印机接入；不能宣称已实现工业诊断闭环。 |
| Multi-Agent | 有显式状态图与一个主 Agent 的工具循环，可以解释节点接口和调度约束 | 多个 LLM 调用节点不等于多个自治 Agent；没有角色协商、委派或多 Agent 通信机制。 |
| Agent Skills | 有工具名、描述、参数 schema、来源、权限及执行策略的统一注册 | 工具注册中心不等于 Skills 系统。需进一步封装适用条件、诊断步骤、输入输出、资源和测试，并建立版本、注册、发现机制。 |
| Harness Engineering | 有循环上限、超时、有限重试、人工确认、模型池、回放、可观测与评测等具体治理措施 | 不宜把这些集合称为已经完成完整工业治理体系。还需要身份权限、设备操作安全边界、幂等、恢复和回滚验证。 |
| Prompt 优化和 TDAD | 提示词在代码中集中管理，存在自动评测脚本与语义任务样本 | 没有发现 TDAD 自优化流水线或完整提示词版本发布系统；应先确认招聘团队对该方法的定义、实验基线及验收口径。 |
| 知识库与数据 | 检索、拒答、反馈归集与人工审核模式可迁移到故障案例知识 | 打印参数、耗材、机型、环境与故障因果关系需要真实领域数据；不能只替换客服提示词就认为迁移完成。 |

项目另有 [上下文与 Harness 策略对照](../context-comparison-method.md)：它是隔离的 Python 文本策略迁移实验，未运行原 TypeScript Harness 的完整运行时；现有报告没有证明显著优于当前策略。这个实验可以作为技术调研与实验边界意识的例子，不宜包装成线上 Harness 已替换现有客服。

可用的一句岗位衔接：

> 我能把这个项目中“证据驱动的判断、受控工具执行和可追溯评测”的方法迁移到排障任务。下一步需要先明确故障数据、参数修改权限和验收标准，再把单个故障的诊断流程封装成可复用能力；当前项目还没有真实打印机闭环，我会把这一差距讲清楚。

## 文档使用提醒

`docs/superpowers/` 保存的是分阶段设计和实现计划，不全部等于当前行为。例如 [ch05 早期设计](../superpowers/specs/2026-07-15-ch05-workflow-agent-design.md) 仍描述七类意图、四出口和旧节点名；[ch08 工具设计](../superpowers/specs/2026-07-17-ch08-tool-system-design.md) 更接近当前工具架构，但仍应回到源码确认。现场优先展示本文和对应源码，历史文档用于解释演变，不用于证明当前指标。
