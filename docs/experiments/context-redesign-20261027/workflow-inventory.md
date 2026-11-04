# 客服 Workflow 的上下文需求盘点

审查日期：2026-10-27。第 1—7 节记录本轮重新设计前的源码及已保存实验复核；该盘点未调用模型、读取客户会话或密钥。“当前”在这些章节中指审查时的源码及配置默认值，不代表检查过进程中的环境覆盖值。随后新增的实验代码与共享流程修复列在第 8 节，避免将设计前状态和新增实现混为一谈。

## 1. 设计起点：客服每轮需要完成什么

本项目的记忆首先要服务于**选对订单或商品、理解本轮动作、遵守仍有效的操作限制、用本轮业务证据回答**。流畅地复述长对话是辅助能力，不能代替这些任务条件。

具体优先级如下：

1. **对象与关系**：不只保存订单号，还要保留“哪个订单对应哪件商品、家里还是办公室、先发还是后发、后来改成哪个”的原句和更新顺序。
2. **动作与否定**：区分“查物流”“申请退款”“不要退款”“只撤回另一单的工单限制”；不能把曾经谈过的动作当成本轮待办。
3. **事实的来源与时间**：用户陈述、历史客服表述、本轮订单/物流工具结果、知识库政策，不具有相同权威性。
4. **当前任务的完成条件**：用户要求“再查物流”，需要本轮完成查询；记住上一次查询内容不等于完成当前任务。
5. **合理成本**：意图分类不需要承载整段证据，证据充分性检查不需要看全部对话；压缩应逐节点进行，并统计维护、重试及回查的成本。

这些要求来自现有业务路径，而不是为了套用某一种通用 Agent 记忆架构。

## 2. 真实执行路径与实验 reader 的边界

真实图定义在 [build.py](E:/Mewhelp/app/graph/build.py:8)。固定入口是：

```text
用户消息 → resolve_reference → classify_intent
  ├─ 商品咨询 → retrieve_knowledge → confidence_check
  │                               ├─ 证据充分 → main_agent
  │                               └─ 证据不足 → fallback_reply
  ├─ 退款退货 / 售后 → fetch_order → retrieve_policy → main_agent
  ├─ 订单 / 物流 / 人工 → main_agent
  ├─ 投诉 → complaint_reply
  └─ 闲聊 / 其他 → script_reply

main_agent → 有工具调用且未达步数上限 → agent_tools → main_agent
           → 无工具调用或达到步数上限 → log → END
固定回复出口 → log → END
```

九类意图到五个出口的映射见 [routing.py](E:/Mewhelp/app/graph/routing.py:7)。`main_agent` 直接绑定工具调用模型，可以直接回复，也可以进入 ReAct 工具循环；不必先生成另一份 `ServicePlan`。见 [nodes.py](E:/Mewhelp/app/graph/nodes.py:387) 和 [routing.py](E:/Mewhelp/app/graph/routing.py:30)。

此前公共 reader 的 `coref → ServicePlan → 只读夹具 → reply` 是控制实验执行链。`ServicePlan` 定义在 [eval_context_comparison.py](E:/Mewhelp/scripts/eval_context_comparison.py:56)，**不是线上图中的必需节点**。其 plan/answer 各 4,200 的上下文预算不能解释为客服新增两个固定调用，也不能直接换算为真实客服每轮调用数。

## 3. 已实现的记忆基础

### 3.1 默认 legacy：摘要边界切窗，而非每个环节都保留八轮

源码默认 `context_policy="legacy"`，摘要触发为新增 30 条消息、摘要后保留最近 8 个用户轮、原文裁剪上限 3,000 估算 Token。见 [config.py](E:/Mewhelp/app/config.py:71)。

- 完整可展示对话保存在数据库；新轮从数据库重新加载用户和客服消息，丢弃上一轮工具中间消息作为本轮输入。本轮进行中的工具序列仍由图保留。见 [runtime.py](E:/Mewhelp/app/graph/runtime.py:165)。
- `memory.build_window` 先按摘要的消息 ID 边界切窗，再按估算 Token 截断；被裁成空时会回退原窗口，因此 3,000 不是该函数在所有情况下都严格保证的硬上限。见 [memory.py](E:/Mewhelp/app/core/memory.py:45)。
- coref/intent 在这个窗口上再取 `prior[-6:]`，实际是最多 **6 条消息**，不是 6 个完整问答轮；前面追加已有滚动摘要。见 [nodes.py](E:/Mewhelp/app/graph/nodes.py:46)。
- 主 Agent 读取裁剪后的原文，并插入摘要、本轮原始诉求、检索证据和退款订单数据。摘要明确标为背景，不是待办或已核验事实。见 [nodes.py](E:/Mewhelp/app/graph/nodes.py:293)、[memory.py](E:/Mewhelp/app/core/memory.py:68)。
- 滚动摘要消耗模型调用：旧摘要与新增滑出区间生成新摘要，成功后更新边界。轮后后台调度，返回时不等待完成；下一轮不保证马上得到最新摘要。见 [summarizer.py](E:/Mewhelp/app/core/summarizer.py:49)、[summarizer.py](E:/Mewhelp/app/core/summarizer.py:92)、[runtime.py](E:/Mewhelp/app/graph/runtime.py:202)。

### 3.2 source：有来源的派生状态与按节点装配，尚不是完善的语义记忆

启用 source 才会准备会话快照，保存可见消息前缀，按消息水位排除未来消息。快照由原文重放构建；来源有消息 ID、角色、片段及引用。用户陈述与客服表述不同型。见 [context_runtime.py](E:/Mewhelp/app/core/context_runtime.py:77)、[context_state.py](E:/Mewhelp/app/core/context_state.py:97)。

当前 source 节点上下文预算为：coref 4,000、intent 1,800、商品检索改写 4,000、Agent 历史包最多 9,800 估算 Token；Agent 先扣掉系统提示、本轮证据、工具 schema 及本轮工具后缀的开销。完整请求另受 16,000 估算输入预算约束，输出配置 8,192。见 [context_runtime.py](E:/Mewhelp/app/core/context_runtime.py:56)、[nodes.py](E:/Mewhelp/app/graph/nodes.py:227)、[nodes.py](E:/Mewhelp/app/graph/nodes.py:342)、[config.py](E:/Mewhelp/app/config.py:78)。估算值不等于供应商实际 usage。

当前装配器会保留候选对象、有效限制的原始引用；词面触发相关历史查找，再加入可容纳的近期完整问答单元。摘要只作为背景，原文超过预算才尝试加入，不能改变状态。必要材料本身超预算会阻断而不静默删除。见 [context_state.py](E:/Mewhelp/app/core/context_state.py:307)。

source 的摘要维护按未覆盖原文达到 12,000 估算 Token 后处理固定区间，目标 1,000，保留最近 8 个用户轮；摘要生成仍有模型费用，且后台完成前不能把它算成“已可见”。见 [config.py](E:/Mewhelp/app/config.py:84)、[summarizer.py](E:/Mewhelp/app/core/summarizer.py:128)。

来源回查工具包括查历史、读原文、读工具回执，但只有 source 且开启 `context_recovery_tools_enabled` 时才向 Agent 暴露，默认关闭。工具完整回执保存与当前短预览分开；保存失败不能重新执行成功写操作。见 [nodes.py](E:/Mewhelp/app/graph/nodes.py:391)、[tools/context.py](E:/Mewhelp/app/tools/context.py:17)、[engine.py](E:/Mewhelp/app/tools/engine.py:146)。

## 4. 逐节点需求表

“必须”指业务正确性所需信息，不意味着每项都要重复作为自然语言发送给模型；身份、确认、归属校验应在后端执行。

| 节点 / 当前实现 | 必须的上下文 | 可选的上下文 | 权威来源与有效期 | 误带历史风险 | 建议的最小装配 |
|---|---|---|---|---|---|
| 会话入口 / `_new_turn_input` | 用户与会话绑定、本轮原话、消息水位、已展示历史 | 有效摘要段及引用 | 本轮数据库可见前缀；排除未来消息 | checkpoint 中旧答案、旧工具结果、旧订单字段残留 | 继续重置本轮输出；历史存储与当前输入分开。原话、改写和派生对象分字段保存。依据：[runtime.py:100](E:/Mewhelp/app/graph/runtime.py:100)、[runtime.py:165](E:/Mewhelp/app/graph/runtime.py:165) |
| 指代消解 / `resolve_reference` | 本轮原话、被指代对象的相关原句、关系、更正与切换顺序 | 最近完整问答；定位远期证据的摘要提示 | 用户原始表述决定其指代/诉求；更晚明确更正覆盖旧候选，业务真实性仍需工具 | 旧 candidate 强压新原句；只存两个订单号却丢掉家里/办公室对应关系；将本轮“再查”改写掉 | 优先相关原句+短近期窗口；补全只补对象与必要限定，保留动作/否定/刷新要求。找不到关系先回查或澄清。现有 coref 每轮调用一次，没有显式请求的代码短路。依据：[nodes.py:173](E:/Mewhelp/app/graph/nodes.py:173)、[coref.py:8](E:/Mewhelp/app/core/coref.py:8) |
| 意图分类 / `classify_intent` | 原始当前动作、已消解问题、最少任务背景 | 最近一次相关转折，如“物流改退款” | 本轮用户诉求；意图仅对本轮有效 | 被旧退款问题拉回退款路；把历史投诉当本轮投诉 | 小视图：本轮原话+改写+必要的上一轮背景；通常无需整段摘要和工具大回执。依据：[nodes.py:196](E:/Mewhelp/app/graph/nodes.py:196)、[intent.py:22](E:/Mewhelp/app/core/intent.py:22) |
| 商品检索改写 / `query_understanding.understand` | 已消解问题、商品/型号和当前限定条件 | 支持这些字段的少量关系原句 | 用户原话说明问什么；型号/规格需由知识库或业务工具核实 | 旧型号或旧需求混进检索词；错误 coref 结果被继续标准化 | 只给本轮检索目标与可追溯原句。source 已可附加历史；legacy 只用改写后的问题。依据：[nodes.py:220](E:/Mewhelp/app/graph/nodes.py:220)、[query_understanding.py:25](E:/Mewhelp/app/core/query_understanding.py:25) |
| RAG 召回 / 四信号评分 | 本轮标准问题、扩展词、候选文档分数 | 商品/政策范围约束 | 当次知识库召回；不是用户聊天中的“政策” | 历史回答被当 KB；相似但不同型号条款被混用 | 检索目标与引用分开；若业务要比较多对象，明确每个证据对应对象，不靠含混的“这款”。依据：[nodes.py:243](E:/Mewhelp/app/graph/nodes.py:243) |
| 证据充分性 / `selfcheck` | 标准问题、当次召回证据 | 若问题遗漏必要限定，先修正问题而非塞全部历史 | 知识库证据支持结论；查询与证据组合当轮有效 | 把常识、历史客服答案或摘要当支撑材料 | 维持小而专一输入；拒绝/故障归因分开。该节点当前只接收 question+evidence。依据：[selfcheck.py:24](E:/Mewhelp/app/core/selfcheck.py:24)、[nodes.py:255](E:/Mewhelp/app/graph/nodes.py:255) |
| 退款取单 / `fetch_order` | 确定订单、后端用户归属、实时订单快照 | 若对象有歧义，订单选择器 | 归属与订单状态来自业务接口；本轮重新读取 | 原话含多个四位数却取到首个；历史已授权误当本次授权 | 对象未确定时保留现有选择器；不要把自由文本中的任意数字直接视为可靠选择。依据：[nodes.py:69](E:/Mewhelp/app/graph/nodes.py:69)、[nodes.py:78](E:/Mewhelp/app/graph/nodes.py:78) |
| 退款政策检索 / `retrieve_policy` | 本轮退换诉求、当前订单状态、当次政策证据 | 商品类别/购买条件等实际政策依赖字段 | 业务订单状态+知识库政策；不能由历史摘要覆盖 | 旧“可退”答复被当持续承诺，旧状态下政策被复用 | 保留确定性取单→政策检索。当前扩写 seed 仅为问题+status，新增字段须先证实实际政策需要。依据：[nodes.py:95](E:/Mewhelp/app/graph/nodes.py:95) |
| 业务 Agent / `main_agent` 首步 | 本轮原话、对象与动作、有效限制、必要历史来源、可用工具 schema | 背景摘要、可回读来源目录 | 本轮诉求确定要做什么；历史是理解材料，不是执行清单 | 延续旧任务、把客服旧话当工具事实、无新查询直接输出物流 | 按 route 装配；本轮任务放在稳定位置；旧状态附时间且标记“非本轮核验”。见下一节共享 freshness 修复。依据：[nodes.py:293](E:/Mewhelp/app/graph/nodes.py:293)、[nodes.py:331](E:/Mewhelp/app/graph/nodes.py:331) |
| 工具执行 / `agent_tools` + `engine` | 工具名/参数、后端用户身份、归属、有效禁止项、实际确认记录 | 用于解释失败的原始约束引用 | 后端鉴权/确认与当轮工具执行；摘要永不产生授权 | 把“之前同意”当新的确认；模型伪造确认；过期卡片继续执行 | 执行前和恢复后统一校验；记忆只可拒绝写入，不能授予权限。依据：[nodes.py:420](E:/Mewhelp/app/graph/nodes.py:420)、[context_runtime.py:99](E:/Mewhelp/app/core/context_runtime.py:99)、[runtime.py:129](E:/Mewhelp/app/graph/runtime.py:129) |
| Agent 工具后续与最终回答 | 本轮真实 tool call/结果/失败状态、证据引用、原始本轮诉求、有效限制 | 对理解回复有价值的少量历史 | 工具结果只证明观察时状态；写成功须有真实回执 | 将失败说成成功；本轮查询失败后引用旧记录却不说明；裁剪掉 ToolMessage 配对 | 优先保留当前工具调用与结果配对；只读大回执可外置后按需回读；最新状态任务只接受本轮有效证据。当前 source 已先为整个轮内后缀预留预算。依据：[nodes.py:342](E:/Mewhelp/app/graph/nodes.py:342)、[engine.py:146](E:/Mewhelp/app/tools/engine.py:146) |
| 固定回复 / 证据不足 / 投诉 | 当前路由、失败来源、必要当前对象 | 解释当前无法完成的最少背景 | 本轮图状态 | 把“已转人工/已完成”写成并未执行的结果 | 主要是代码话术，通常不应再增加历史模型调用。依据：[nodes.py:125](E:/Mewhelp/app/graph/nodes.py:125)、[nodes.py:147](E:/Mewhelp/app/graph/nodes.py:147) |
| 后台摘要 | 固定历史区间、已存在摘要或固定段、边界 ID、源 hash | 无需当前实时工具状态的全量副本 | 原文为准；摘要为派生背景，不能修改授权或否定状态 | 滚动摘要累积错误、不同对象关系合并、把未执行说成已执行 | 保留原文和可回读范围；摘要独立计费，记录完成时间及何轮可见；实时更正先进入当前证据，不等待后台。依据：[summarizer.py:128](E:/Mewhelp/app/core/summarizer.py:128) |

## 5. 目前已观察到的三个设计缺口

### 5.1 旧 candidate 与更新原句同时可见，模型仍选旧对象

source 用有限字面规则从用户句子生成 candidate；未知自然表达不会自动使旧候选失效。当前视图会把候选渲染成“当前对象候选”并固定其原句来源。相关代码在 [context_state.py](E:/Mewhelp/app/core/context_state.py:158)、[context_state.py](E:/Mewhelp/app/core/context_state.py:322)。

已有复核案例：较新原句“之前那张凭证暂告一段落，接下来咨询的配送对象换成920061”在三个阶段都可见，但候选仍是920060，CS 最终查了旧订单。见 [CS_FAILURE_REVIEW.md](E:/Mewhelp/data/verification/context-compact-20261027/CS_FAILURE_REVIEW.md)。

**建议：**不把正则候选当作当前对象真值。每轮根据当前问题及有时间顺序的原文证据确定目标；旧候选只能做检索提示。存在较新不同对象或自然切换却无法判定时，标记冲突/待解析，不能持续强化旧值。没有必要为了做这件事额外新增通用规划 LLM，可先复用指代消解和已有意图环节，并校验其输出是否有来源支持。

### 5.2 多动作限制与部分撤回没有可靠表达

当前按逗号/分号拆句，`_action` 对一段只返回一个动作；对象缺省时回退候选；撤回需匹配有限直接肯定规则。见 [context_state.py](E:/Mewhelp/app/core/context_state.py:53)、[context_state.py](E:/Mewhelp/app/core/context_state.py:184)。

真实 Workflow 复核已经看到：`订单710003不要提交退款，也不要创建工单` 的第二个动作成了对象未确定限制；`只撤回订单710004的工单限制；订单710003的限制继续保留` 未准确表达撤回。见 [WORKFLOW_REVIEW.md](E:/Mewhelp/data/verification/context-compact-20261027/WORKFLOW_REVIEW.md)。这些是状态正确性缺口，不是单纯 Token 不够；旧物流任务分数没有覆盖它们。

**建议：**用独立的 `对象 × 动作 × 否定/撤回 × 来源` 事件表示，保留限制原句和更新原句。无法准确拆分时视为未解析限制并要求核实；撤销禁止项不等于获得执行授权。评测必须单列状态更新正确性和写操作禁止正确性，不能用一个只读查询通过率代替。

### 5.3 记住旧物流结果，不能替代“本轮再查”

订单/物流都直达 `main_agent`，而 `should_continue` 在没有 tool calls 时直接结束。当前没有代码保证“再查物流”必须得到本轮查询回执。见 [routing.py](E:/Mewhelp/app/graph/routing.py:14)、[routing.py](E:/Mewhelp/app/graph/routing.py:38)。

之前 source/compact 两组各 27 轮都在相同 2 轮遗漏本轮物流查询；对象正确、没有写操作，但复述旧物流。静态夹具每轮结果相同，不能靠最终文字区分是否真正刷新。见 [WORKFLOW_REVIEW.md](E:/Mewhelp/data/verification/context-compact-20261027/WORKFLOW_REVIEW.md)。

**建议作为所有 arm 共享的流程修复：**在业务路增加确定性的本轮查询要求，或等价的执行守卫，不把它计为新记忆策略的收益。

1. 从本轮原话、既有 coref/intent 输出判断“查询当前物流/再查/刷新”的任务；“刚才查到了什么”是历史回顾，不能误触发新查询。该规则读取原话，避免改写丢掉“再查”后失效。
2. 明确目标订单；对象不清晰先澄清，归属由后端校验。不要相信旧 candidate，也不要为通过评测向某组注入金标准订单。
3. 复用 `query_order` 取得 `tracking_no`，再调用 `query_logistics`。真实物流工具接收物流单号，不能把订单号直接传给它。见 [orders.py](E:/Mewhelp/app/tools/builtin/orders.py:16)、[logistics_server.py](E:/Mewhelp/mcp_servers/logistics_server.py:22)。
4. 经现有统一执行引擎记录本轮 `message_id、order_id、tracking_no、tool_call_id、observed_at、status`。若本轮已有同对象的有效订单/物流结果，可直接满足要求，避免重复调用。
5. 成功则供 Agent 组织回复；失败则保留失败事实，可以说明旧结果的观察时间，但不能声称本轮查得最新。
6. 在所有对照组使用完全相同的判定、执行、回执装配和计费规则；另报“旧流程→共享修复”的影响。

最小接入点是在业务分支与 `main_agent` 之间加一个 `prepare_business_evidence` 只读节点，或在既有业务 Agent 前增加等价守卫。这里是设计建议，不是当前已经有的节点。该方案复用已有模型调用，不需要再多一次专用规划调用。

真实项目当前订单与物流服务也是固定种子演示数据，见 [business.py](E:/Mewhelp/app/tools/business.py:1)、[logistics_server.py](E:/Mewhelp/mcp_servers/logistics_server.py:29)。新实验需要让夹具随轮改变状态和时间，才能测试陈旧证据问题；仍需如实标注是模拟业务系统，而非线上生产业务验证。

## 6. 下一版最适合保留的职责边界

建议方案可以解释为：**按节点装配的任务证据上下文**。

- **完整原文层**：负责可追溯与恢复，尽量不重复送进每个调用。
- **当前任务层**：保留原始诉求、已消解诉求、对象来源、动作、有效禁止项；派生字段不能压过较新的原始更正。
- **近期原文层**：提供连续对话语境；按完整问答单元裁剪，避免只剩客服答复失去用户条件。
- **按需证据层**：依据当前问题选中相关原句，必须带上关系与更正上下文，而非只召回孤立实体。
- **本轮业务证据层**：由 Workflow 与工具执行建立，明确对象、时间、成功/失败；实时状态任务不能从背景摘要生成事实。
- **可恢复摘要层**：只概括远期背景，标记来源区间；是否值得维护由包含维护费用的实验决定，不能默认每轮总结。

这套职责分工的逻辑是：**原文负责可核对，任务信息负责当前范围，工具负责业务事实，Workflow 负责必要执行，摘要负责成本。** 模型仍负责语义理解与回答，但来源校验、授权和业务流程不能全靠模型读完长历史后自行判断。

## 7. 对统一评测的直接约束

1. **主实验跑真实图。** 保留公共 reader 作为隔离记忆效应的辅助实验，两个结果单独报告。
2. **共享业务守卫。** 物流 freshness、确认、归属、工具 schema 等对各 arm 一致，不把执行修复包装成记忆提升。
3. **每阶段记录实际可见证据。** 不只记录最终历史包；旧结果已说明 coref 缺失关系后，后续看到证据也未必修复错误。
4. **核对更新而不只识别实体。** 覆盖自然切换、订单更正、多动作否定、部分撤回、历史回顾与重新查询、同类多对象关系、工具失败后回复。
5. **统计完整调用链。** coref、intent、改写/扩写、selfcheck、所有 Agent 步骤、摘要维护、额外回查、协议重试分别计数，并汇总供应商 usage。上下文装配估算预算单列，不能作为实际付费 Token。
6. **统计真实完成与等待。** 报对象/动作/工具参数/回执新鲜度/答案依据/澄清质量；端到端延迟及后台维护时间分开，API 故障不混成记忆能力差异。
7. **保护简单场景。** 短对话必须有独立子组；此前紧凑方案省 Token 主要来自重复背景压力轨迹，短回归反而略增，不应宣称普遍省钱。

当前最需要验证的是：不用容易过时的候选值强化模型，同时用更小的节点视图保住真正相关的原句、关系和有效限制，能否在真实 Workflow 中保持或改善任务完成，并降低全部调用和维护 Token。它是一项有明确失败案例支撑的待检验假设，不是已经证明的新结论。

## 8. 本轮新增实现与冻结前检查

新增 [context_task.py](E:/Mewhelp/app/core/context_task.py:100) 是纯函数实验装配器，支持完整历史、同预算滑窗、滑窗与摘要、任务原文检索四个投影。task 不新增 LLM 调用，不采用旧 candidate 为当前对象，不进行每轮语义状态维护；按照当前问题词面检索用户原文，再扩大相邻完整轮与共享实体的一跳关系。每条引用记录消息 ID、角色、偏移和内容 hash。该方法没有直接解决所有复杂自然语言约束；它保留原文供现有模型理解，后端授权与禁止项仍需公共守卫。

冻结前检查修正了两类装配问题：当 retrieval 传入消解后的问题时，仍按当前消息 ID/可见前缀位置排除当前原话，避免把它当作历史重复发送；预算同时检查消息形式与带角色前缀的 history+query 形式，取两者较大值。系统提示、工具 schema、本轮工具回执及 JSON 封装另由调用方和完整请求检查覆盖。13 项构造单测通过；它们验证实现约束，不代表任务准确率。

新增 [business_evidence.py](E:/Mewhelp/app/graph/business_evidence.py:19) 为所有实验组共享的可选只读流程修复：物流意图下校验唯一订单和归属，先查订单取得 tracking_no，再查物流，保留本轮查询元数据。静态检查确认 AI tool_calls 与 ToolMessage 成对，用户身份由执行引擎注入，状态在每轮重置并按消息水位使用，没有新增写操作授权。

冻结前也修正了显式“不查询”、仅回顾过去结果，以及大工具回执外置后 tracking_no 位于 preview.fields 的情况。该触发器仍是有限字面规则，不等于完整语义规划；多对象与复杂否定需要保守处理，不能用单次实验宣称全部边界已解决。此共享修复的收益不得算作 task 原文检索的独有收益。

[analyze_context_unified.py](E:/Mewhelp/scripts/analyze_context_unified.py) 读取冻结结果而不重新评分，分组统计所有已记录的 API Token、维护、重试和缺失 usage。task 的检索关闭、来源标签关闭消融只在同一冻结前缀上离线重装配，输出原文 ID 和估算 Token 变化；**不调用模型，所以不生成消融后的任务通过率**。真实任务差异仍需以五组同一 Workflow 的推理记录为准。
