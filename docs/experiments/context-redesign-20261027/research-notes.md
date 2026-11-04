# 上下文工程依据核对（2026-10-27）

本笔记区分三件事：外部资料实际提出的原则、参考项目本地代码实际实现的机制，以及拟移植到 MewHelp 的设计。这里只核对资料和源码，没有运行模型，也没有把外部基准成绩当作本项目成绩。

## 1. 可以直接支持本项目设计的外部证据

| 原始来源 | 核实到的内容 | 对客服的具体用途与边界 |
| --- | --- | --- |
| [LangChain：Context Engineering，2025-07-02](https://www.langchain.com/blog/context-engineering-for-agents) | 将策略归为写入、选择、压缩、隔离；对象包括指令、知识和工具反馈。 | 原文存档是写入，按节点选历史是选择，长工具回执预览/历史摘要是压缩。现有 Workflow 已提供职责分离，不必为了“隔离”增加多 Agent。分类不是效果保证。 |
| [Anthropic：Effective context engineering，2025-09-29](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents) | 强调每次推理的有效信息集合、按需检索、工具结果清理；过度压缩会损失之后才显得重要的细节，应先验证关键内容召回再减少冗余。 | 先保证更正、约束作用对象、最新回执在相关节点可见，再压低预算；允许旧的完整回执退出常驻上下文，但保留读取入口。不能据此断言摘要总优于原文。 |
| [Manus：Lessons from Building Manus，2025-07-18](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) | 稳定提示前缀有利缓存；外置内容保留 URL/路径，使内容仍可恢复。 | 客服使用已有数据库消息 ID、字符区间、哈希和回执 ID；无需让客户接触文件系统。缓存收益必须看 Gitee 返回的缓存 usage，不能只凭 Token 减少推断账单或延迟下降。 |
| [LongMemEval 原论文](https://arxiv.org/abs/2410.10813) | 检查信息提取、多会话推理、时间推理、知识更新和信息不足时的拒答；把记忆链路拆为索引、检索、阅读。 | 分开检查“选中的原文是否有依据”“模型是否使用了正确关系”“最终工具和答复是否正确”；加入用户更正、旧状态失效、多个对象需要澄清的场景。本文的新客服数据只是内部合成集，不应称为 LongMemEval 复现。 |
| [MemoryAgentBench 原论文](https://arxiv.org/abs/2507.05257) | 通过增量多轮交互考察检索、测试时学习、长程理解、选择性遗忘。 | 将本轮生成的答复与状态真正带到下一轮；更新/撤回不能靠每轮重新提供金标准历史伪装成功。每个方法维护自己的轨迹，外部故障单独报告。 |

这些材料支持的是“最少但足够的信息、可恢复的存储、增量更新和完整评估”，没有提出适用于本客服的统一最佳窗口大小。本轮不直接采用用户此前引用的“85% 模型窗口触发摘要”：该数值不是以上资料共同结论，也不是本项目成本最优点。

## 2. QueryMind 本地代码：值得借鉴业务状态，不是现成的聊天压缩器

核查位置：`D:\QueryMind`，2026-10-27。以下描述针对本地检出的代码，未启动该应用。

- [agent.py:1531](D:/QueryMind/src/QueryMind/core/agent/agent.py:1531) 的请求构造先执行可配置 conversation filters，再把消息转为 LlmMessage，最后增强上下文；未发现该入口自带一个已启用的固定滑窗/滚动摘要。 [my_agent.py:197](D:/QueryMind/my_agent.py:197) 装配默认记忆增强、Schema/SQL 治理和文件会话存储，没有传入 conversation_filters。因此不能把它概括成“QueryMind 已实现完整自动压缩方案”。
- [default.py:46](D:/QueryMind/src/QueryMind/core/enhancer/default.py:46) 按最近用户问题检索最多 5 条文字记忆，作为 user-side Memory Advisory 添加；不再追加到 system prompt，异常时保留原消息。[default.py:69](D:/QueryMind/src/QueryMind/core/enhancer/default.py:69) 是实际检索入口。适合借鉴“历史经验只是辅助”，但本实现并没有证明每条建议都带可验证原始来源或严格总 Token 上限。
- [governance.py:292](D:/QueryMind/src/QueryMind/core/agent/governance.py:292) 从执行状态产生紧凑 request metadata，保留调用/成功/失败计数、重复检索、锁定原因和最新 schema 快照；[governance.py:327](D:/QueryMind/src/QueryMind/core/agent/governance.py:327) 再生成对应提示块。这是“由工具结果更新业务状态”的实例，适合映射为客服当前订单、实时查询成功/失败、是否尚需核验，而不是依靠反复总结自然语言。
- [run_sql.py:73](D:/QueryMind/src/QueryMind/tools/run_sql.py:73) 把完整结果保存成 CSV，给模型的结果正文使用最多约 1000 字符预览及文件名。适合迁移为“完整物流/订单回执留库，本轮只发关键字段+回执引用”；字符截断不等于完整请求 Token 预算，也不能无条件截掉正确答案需要的字段。

本项目已有 MySQL/来源表和 Milvus，不因该参考项目使用 Mem0/Neo4j 就增加一套跨会话图记忆。短客服会话首先需要同一客户、同一订单、有效时点的一致性；跨用户经验建议与订单业务事实必须隔离。

## 3. DeepSeek Harness 本地代码：借鉴预算检查和可恢复投影

核查位置：`D:\harness\deepseek-harness`，2026-10-27。它是通用 Agent 运行时，本项目只能迁移部分机制；本文没有声称在客服里运行了完整 Harness。

- [config.ts:19](D:/harness/deepseek-harness/packages/compaction/compaction-basic/src/config.ts:19) 当前默认压力比例为 0.8、原文保留比例为 0.16；[config.ts:128](D:/harness/deepseek-harness/packages/compaction/compaction-basic/src/config.ts:128) 按路由模型容量计算预算，并支持绝对 retainTokens。这些是该运行时策略参数，不是客服优化实验的答案。
- [index.ts:275](D:/harness/deepseek-harness/packages/compaction/compaction-basic/src/index.ts:275) 在达到压力后先执行可选的无模型工具结果裁剪，再重新计量；仍超阈值才选历史范围摘要，并在每次压缩后再次测量。可直接借鉴“先删重复/过期回执，再考虑额外模型摘要”，减少每轮维护调用。
- [region.ts:110](D:/harness/deepseek-harness/packages/compaction/compaction-basic/src/region.ts:110) 保留最近尾部，选择边界时不拆开 assistant tool-call/result 配对。客服压缩也必须保留本轮尚未消费完的工具调用与结果，不能只按字符串截末尾。
- [region.ts:411](D:/harness/deepseek-harness/packages/compaction/compaction-basic/src/region.ts:411) 将摘要包装成最终 checkpoint 后重新计算；若包装后的摘要没有比被替代内容更小则拒绝。可直接迁移为“预算计算包括来源标记、标题、系统指令、工具 schema 与回复预留”，避免正文看似放得下、最终请求放不下。
- [region.ts:431](D:/harness/deepseek-harness/packages/compaction/compaction-basic/src/region.ts:431) 检查异步摘要期间原始 surface 是否变化，避免旧摘要覆盖新状态。客服对应版本号/消息水位和单会话维护互斥。
- [session-reference/index.ts:59](D:/harness/deepseek-harness/packages/context/session-reference/src/index.ts:59) 把回读资料标为只读非指令快照；[同文件:84](D:/harness/deepseek-harness/packages/context/session-reference/src/index.ts:84) 提供有预算的精确引用读取。MewHelp 适合使用已核验归属的本会话消息引用，不默认启用跨会话读取。

不迁入整个插件运行时、通用文件系统工具、多 Agent 调度、任意跨会话引用。它们对当前九类意图客服的收益尚未验证，且增加维护、权限和评测变量。

## 4. 模型与供应商规格边界

本次访问 [DeepSeek 官方 API 起始页](https://api-docs.deepseek.com/quick_start/pricing) 发现它实际显示的当前模型入口为 `deepseek-flash`，并明确旧 `deepseek-v4-flash` API 名在其官方服务中已被映射到 V4.1-Flash-1017。官方模型能力不能直接等同于 Gitee 同名模型的部署限制。

Gitee 的 [官方 API 文档入口](https://ai.gitee.com/docs/openapi/v1) 本次网页提取仅返回前端导航；尝试具体模型页未取得可靠模型规格。故**本次没有独立确认 Gitee 端点的最大输入、最大输出、缓存和精确模型修订版**，也没有调用模型来探测极限。应在实验 manifest 保存供应商、请求 model 名、返回 model 名（若有）、思考开关、max_tokens、日期和实际 usage，不擅自改模型。

建议区分三种量：供应商硬上限用于避免非法请求；服务预算用于控制正常延迟和成本；按节点的历史/证据预算用于分配服务预算。即使模型有百万级窗口，也不代表客服应把原文堆到数十万 Token 才整理。

## 5. 对 MewHelp 的可解释设计建议

1. 完整消息和回执保持存档。派生状态记录对象、原始来源、有效/已替代状态；用户陈述、模型猜测、工具确认不同权威等级。派生状态不获得写操作授权。
2. 本轮问题、明确更正/撤回和未完成的工具调用优先保留。旧助手话语尤其是“已发货”“已退款”只代表历史陈述；“现在/再查”应经 Workflow 获取本轮回执。
3. 短问题优先装配紧凑原文窗口。遇到指代不明、旧对象、来源状态冲突才检索同会话历史；返回原句及相邻关系链，不能仅匹配一个订单号便宣布找到答案。
4. 各节点使用同一原始事实和消息水位，按职责装配：消解需要用户更正和对象关系；意图需要消解后的诉求；检索需要商品与约束；业务工具需要当前目标及核验状态；回答需要本轮可信证据。保留用户原问题，防止消解错误不可逆地污染后续节点。
5. 超预算先去重、缩短无关助手寒暄、外置旧回执；随后优先级截取原句，摘要仅作可回读的背景补充。若关键证据无法安全放入或对象仍冲突，澄清而不是用过时状态强行选定目标。
6. 使用现有代码的确定性装配与业务执行校验，避免为了“先规划上下文”额外固定增加一次模型调用。只有实验发现确定性回查不足，才单独评估模型检索规划的边际收益。

以上是设计假设，尚不是本轮结果。需要同时对照默认 legacy、完整原文、紧凑窗口、紧凑窗口+摘要，以及新的选择/来源策略；与旧实验相比，必须在同一真实 Workflow 上记全消解、意图、查询改写、证据检查、Agent、维护、重试的实际输入/输出 Token。先固定数据和评分，再运行，不能看新结果修改金标准。

## 6. 对实验的具体要求

- 效果拆分：目标对象、动作、最新证据、应澄清、工具是否实际执行、是否出现未授权写提议/执行。记住订单号不等于完成物流查询。
- 原文选择单测：更正/撤回、两个对象关系、超预算单条消息、回执失效、来源查找是否能恢复。实际模型结果再检查工具和答复，避免依赖自报“记住了”。
- 同预算组件消融与真实默认整套方案比较分开报告。不同窗口、维护频次、提示词、思考模式同时变化时不能把差异归因于某一个组件。
- 同一案例连续跟进，使用该方法自己产生的历史。初始历史是人工合成事实，既往所有轮次生成费用未测；应只称为“初始化维护+本次连续追问”的工作负载。
- 成本汇总实际 usage，包括摘要、额外回读、结构化重试；输出、思考和缓存字段分别保存。缺少 usage 就标未知，不用估算伪装实测。
- 保留接口故障和语义答错原记录；接口失败率、全样本任务通过率和可比较子集分别报告，不能只筛成功记录提高分数。
- 小合成集用于定位机制和内部回归，不足以证明线上泛化或真实客户满意度；若结果接近，优先较简单且低成本方案，附失败案例解释。
