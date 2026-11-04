# CareMate：逐字稿与展示操作

更新：2026-11-03。公开源码核对基线：[5c1840b](https://github.com/Orange-ui0811/caremate/commit/5c1840b)。主讲约12～15分钟，真实模型等待时间另留余量。引用块可直接口述；其余为操作与备答。配套[操作台本](interview-runbook.md)，旧版详细材料保存在[参考稿](interview-reference-20261028.md)。本稿仅保存在本地。

## 1. 客服功能与整体架构｜0:00～0:45

**操作：**打开[GitHub首页](https://github.com/Orange-ui0811/caremate)，指功能介绍，再滚动到系统架构图。首页指标此时不展开。

> 我介绍一下客小助 CareMate。它主要支持商品和政策问答、订单物流查询、退换货咨询，以及需要人工跟进时的工单确认。用户没解决的问题还可以进入后台，经过归并和人工审核，作为补充知识的依据。
>
> 接下来，我先介绍客服的整个 Workflow 架构。这里由 LangGraph 管流程和状态，LangChain 组织模型消息与调用，MCP 接入独立的物流、售后服务，Langfuse 查看调用记录。它们围绕同一条请求配合工作。
>
> 比如退货必须先确定订单、再查政策，这些步骤由程序规定；查订单之后还需不需要查物流，则可以由主 Agent 根据工具结果继续判断。

## 2. 从流程到模型，再到受控工具执行｜0:45～3:30

**操作：**首页执行图 → [build.py](https://github.com/Orange-ui0811/caremate/blob/main/app/graph/build.py)，指 StateGraph、意图条件边、fetch_order→retrieve_policy、agent_tools→main_agent。

> 用户消息进来，先补全指代，再识别九类主意图，映射到五条路线。商品咨询先查知识和检查证据；退款退货、售后先取订单再查政策；订单、物流和人工请求进入主 Agent；投诉和闲聊等由固定话术处理。
>
> 对应代码，StateGraph 定义状态图，add_edge 连接固定顺序，add_conditional_edges 根据状态选择分支。主 Agent 提出工具请求后，进入工具节点，工具结果回来再交给模型。没有工具请求，或者达到步数上限，就结束这轮。
>
> 所以这是一个主 Agent 配合工作流。确定性描述的是流程受到代码约束，节点内部也可以调用模型，比如政策检索前的查询扩写。

**操作：**打开[prompts.py](https://github.com/Orange-ui0811/caremate/blob/main/app/core/prompts.py)，搜索 INTENT_CLASSIFY_PROMPT；随后打开[intent.py](https://github.com/Orange-ui0811/caremate/blob/main/app/core/intent.py)，指模板组合与 ainvoke 字典。

> 流程规定了下一步去哪，节点具体怎么问模型，就看 LangChain。以意图识别为例，这里用 ChatPromptTemplate.from_messages 定义 system 和 human 两类消息。system 放九类意图的规则，human 模板留出 history 和 query。
>
> 在 intent.py，ainvoke 的字典把历史和当前问题填进同名变量。中间的竖线把模板和模型组合起来，先生成消息，再调用模型。
>
> 结果使用 Pydantic 的 _Intent 定义，包含九选一的 intent 和零到一的 confidence。llm.structured 指定 function calling，让 LangChain 把这个结构转成工具参数定义，接收并校验模型填回的分类结果。这里的 confidence 是模型自报的判断把握；路由使用的是 intent。

**操作：**[llm.py](https://github.com/Orange-ui0811/caremate/blob/main/app/core/llm.py)搜索 with_structured_output；[nodes.py](https://github.com/Orange-ui0811/caremate/blob/main/app/graph/nodes.py)搜索 main_agent，指 get_all_specs、bind_tools、_agent_messages、ainvoke。

> 主 Agent 也是调用模型，但提供的是业务工具。get_all_specs 收集合并后的工具，bind_tools 把名称、用途和参数交给模型，_agent_messages 组织系统规则、历史和本轮材料。模型返回 AIMessage，如果包含 tool_calls，后端执行后用 ToolMessage 回传结果，再让模型继续判断。
>
> 意图分类借用工具参数格式交回结果，业务工具调用则要执行函数，这两种 function calling 的后续处理不同。

**操作：**打开[工具执行说明](https://github.com/Orange-ui0811/caremate/blob/main/docs/engineering/tool-execution.md)，指执行图；再点 registry.py、engine.py。MCP 只展开[客户端](https://github.com/Orange-ui0811/caremate/blob/main/app/tools/mcp_client.py)的 get_tools 和[物流服务](https://github.com/Orange-ui0811/caremate/blob/main/mcp_servers/logistics_server.py)的 tracking_no。

> 当前有五个内置业务工具，物流与售后两个 MCP 服务再提供三个工具。MCP 客户端负责发现并适配成 LangChain 工具，因此本地函数和远程服务可以进入同一个工具列表。
>
> 模型提出调用后，执行引擎查注册表、校验参数、检查写操作确认，再注入后端的身份信息，处理超时、重试和审计。创建工单会先暂停展示预览，确认后才写库；写操作超时不自动重试，避免重复创建。退款工具另行转换成前端申请表单。
>
> 这部分有确认、取消、非法参数和超时等回归测试。后面演示时，我会展示选单和取消工单这两个交互。

## 3. RAG：知识存储、选型、检索与证据｜3:30～6:15

**操作：**打开[RAG说明](https://github.com/Orange-ui0811/caremate/blob/main/docs/architecture/rag-and-confidence.md)，按“入库→Embedding→检索”展开；知识目录只停留概览。

> 刚才是订单、物流这些动态信息。商品说明、退货政策、运费规则等相对稳定的资料，则通过知识库回答。MySQL 管理原文和状态，Milvus 提供向量与 BM25 检索。这里选择 Milvus，是因为它支持同一条知识上的 Dense 和 Sparse 两路检索，跟当前方案匹配。
>
> 资料先按 Markdown 标题分节，正文递归切块，再保留完整句子的重叠；表格重复保留表头，按数据行拆分。这样是为了让型号、政策条件和表格字段尽量保留在可理解的检索单元里。

**操作：**打开[五模型Embedding报告](https://github.com/Orange-ui0811/caremate/blob/main/docs/evaluation/embedding/README.md)，指五行总表和输入协议。

> 文本转成向量之前，还要确定模型。这里保存了五模型比较，使用相同的五十二个片段和三百道问题，做纯向量的精确余弦排序。其中二百四十道可答题计算主要质量指标。
>
> BGE-M3 的分组 Recall@5 是98.96%，跨文档多证据题是95.83%，在这份题集上召回较高，所以当前保留 M3。Qwen0.6B 的分组倒数排名更高，因此结论是 M3 符合当前召回目标，各指标仍需分别看。表里的编码耗时包含网络和服务排队。

**操作：**回[retrieval.py](https://github.com/Orange-ui0811/caremate/blob/main/app/core/retrieval.py)，指 hybrid_search、rerank；再开[四策略报告](https://github.com/Orange-ui0811/caremate/blob/main/docs/evaluation/rag/README.md)。

> 只有语义向量还不够，用户会用口语，但型号、数量和条款又需要精确词项。所以在线使用 M3 向量加 Milvus BM25，两路召回后通过 RRF 按排名融合，再用 BGE-Reranker-v2-M3 对问题和候选文本重排，默认输出前十条证据。
>
> 这套组合是否有效，要看四策略对照。三百题分政策、型号、口语、库外和跨文档五桶，每桶六十题；主要检索指标统计二百四十道可答题。标准答案标出必需证据组，多证据问题要分别覆盖。
>
> 报告里，向量、混合、混合加重排的分组 Recall@5 分别是97.3%、94.8%和98.4%；答案要点覆盖分别是73.5%、89.7%和93.7%。所以当前方案的依据是重排后的整体结果，而不是认为加上 BM25 就一定提升所有指标。答案覆盖由模型裁判评价，这些是固定小语料的内部实验结果。

**操作：**[confidence.py](https://github.com/Orange-ui0811/caremate/blob/main/app/core/confidence.py)指 max；[selfcheck.py](https://github.com/Orange-ui0811/caremate/blob/main/app/core/selfcheck.py)指 useful/reason。

> 检索相关，还不一定足够回答。当前第一道检查取候选中的最高有效重排分，与阈值比较。显式取最大值，是因为复合问题可能分开检索、轮转合并，第一条未必最高。
>
> 达到阈值后，再让模型判断证据是否足以支持当前问题，结构化返回 useful 和 reason。不充分就进入兜底。这两道检查是先后执行的，没有再加权成一个分数。阈值配置最后在管理端展示。

## 4. 上下文：为什么摘要之外还要回查｜6:15～7:45

**操作：**打开[上下文说明](https://github.com/Orange-ui0811/caremate/blob/main/docs/architecture/context-engineering.md)及[回查摘录](https://github.com/Orange-ui0811/caremate/blob/main/docs/architecture/code-excerpts/context-recall.md)，再开[四方法报告](https://github.com/Orange-ui0811/caremate/blob/main/docs/evaluation/context/README.md)。

> 前面解决了查什么资料。多轮对话还要知道用户说的是哪个对象，例如先说订单710001，后来更正成710003，中间又聊了其他商品，最后问“更正后的那单”。
>
> 基础方案用近期窗口保留原话，用滚动摘要保留早期背景。但摘要可能丢掉更正关系和限制条件，所以开发验证版增加了原文回查：对当前会话历史用户问句做 BM25 检索，取正分 Top3，再补相邻问答，按消息 ID 去重，加入节点上下文。它没有增加一轮规划模型调用，但补进来的原文仍然会消耗 Token。
>
> 实验用了二十四段合成会话，每段连续追问三次，共七十二次判定。场景包括历史更正、多对象切换和限制反转。除了答案，还检查实际查询对象、是否遵守限制，以及是否有本轮工具回执。
>
> 摘要加滑窗通过59/72，新回查通过70/72；完整历史是71/72。新回查总用量479,290 Token，完整历史记录下界是641,440，至少少162,150，按完整历史下界算约25.3%。这里计入摘要维护成本；完整历史有一次超时缺失用量，所以用下界表达。
>
> 这份结果来自开发验证版，公开仓库提供算法和节点摘录，完整接线还没有全部同步到公开应用。这里展示实验和实现思路，现场客服则以实际运行版本为准。

## 5. 未解决问题如何回到知识维护｜7:45～8:30

**操作：**打开[反馈说明](https://github.com/Orange-ui0811/caremate/blob/main/docs/architecture/feedback-and-classifier.md)，点 flywheel.py 的标准化和归并流程。

> 证据不足或者用户反馈没解决的问题，会进入问题池，尽量保留当时的召回快照。后台对问题标准化、查重归并，形成待审项，人工核准后才补充知识库。
>
> 旁路还设计了十七类多标签主题分类，用于分析知识缺口的分布，跟在线九类意图分流是两个任务。它帮助定位补什么知识，实际解决率提升还需要冻结问题集做补库前后对照。
>
> 接下来切到本地客服，把刚才讲的工具组合和交互实际展示一下。

## 6. 本地演示｜8:30～12:30，按等待时间调整

**操作一：**打开 http://127.0.0.1:8000/ ，新对话输入 `订单 1001 的物流到哪了？`。有实际工具活动时展示工具名与参数。

> 这条请求预期先查订单取得物流单号，再通过 MCP 查物流，最后依据工具结果回答。订单和物流使用模拟数据，但工具发现、调用和回传链路实际运行。

**看到结果后：**对照 tracking_no 与物流回执再解释。工具没执行或失败时，说实际观察到的情况，打开工具定义或已验证测试作为补充。

**操作二：**新对话输入 `我要退货，但我不记得订单号了。`，选一笔本人订单，展示恢复；出现退款表单后不提交。

> 这次缺少订单号，流程先停在选单。选完以后从暂停处继续，取得订单再检索政策。这里展示的是申请入口，实际退款还要走外部业务流程。

**操作三：**时间允许时新对话输入 `帮我建售后工单，猫砂盆漏电，需要人工处理。`，出现预览后点取消。同时打开[确认测试](https://github.com/Orange-ui0811/caremate/blob/main/tests/graph/test_ch08_confirm_ticket.py)。

> 模型可以提出建单，但写入前需要用户确认。这次我点取消。测试里还会检查创建函数没有被调用、执行层留下权限拒绝记录；这比只看模型说“已取消”更能验证控制是否生效。

**操作四：**打开 http://127.0.0.1:8000/observability ，指在用阈值、两条曲线及对应样本分母；再到 /review 打开已有待审项，不核准写回。

> 这里可以查看当前在用阈值，以及固定校准样本在这个阈值下的可答通过率和应拒放行率。它们描述这批样本的分离情况。阈值保存后，主知识分支后续检索会读取新值。待审页保留问题和可获得的召回快照，方便判断到底缺知识还是检索没命中。

**操作五：**打开本地 Langfuse，选择有明确日期的演示会话，停留20～30秒。

> Langfuse 这部分是 AI 协助完成接入的。我主要用它查看调用记录、耗时和已上报的 Token，定位失败发生在哪一步。这里展示的是这条记录实际保存的内容。

> 主要流程就介绍到这里。具体的工具控制、检索对照和上下文任务，都可以继续结合代码展开。

## 备答卡：不计入主讲时间

| 追问 | 回答依据 |
| --- | --- |
| 为什么当前采用单主 Agent？ | 必要步骤已经由图约束；单主 Agent 足以组织现有工具，新增多个 Agent 还需协调协议和独立评估。当前没有 Multi-Agent 成果。 |
| 意图结构化输出也是工具吗？ | 协议层提供 _Intent 工具定义并指定 tool_choice，LangChain 解析 arguments 成 Pydantic 对象；未注册到主 Agent 的业务工具列表，也不执行外部查询。 |
| 意图 confidence 与检索 confidence？ | 前者模型自报；后者是有效候选最高重排分。当前路由按 intent，不能把自报值当统计正确率。 |
| 所有知识入口阈值一致吗？ | 主知识分支使用动态阈值与 max_rerank_v1；query_faq 仍使用首项分数与 rerank_min_score，不能声称已统一。 |
| 新增 MCP 写工具如何授权？ | 当前 WRITE_TOOLS 仅列 create_ticket，其他已发现工具默认读；需明确补副作用声明及默认拒绝策略。执行时未知工具名称会被拒绝，但这不等于未知副作用工具已得到治理。 |
| Tool Registry 等于 Skills 吗？ | 工具定义可调用动作；Skill 还需要任务知识、步骤、约束和依赖。当前没有实现完整 Skills 注册与加载。 |
| 取消后“数据库无新增”怎么证明？ | 当前隔离测试替换 create_ticket 并断言调用数0；真实数据库需要同会话前后记录对比，不能只据模型回复判断。 |
| 写操作是否完整幂等？ | 确认后才执行、写超时不重试；尚无跨网络重放、进程崩溃窗口的完整幂等证明。 |
| 新回查比摘要滑窗更省吗？ | 本次总量479,290对403,048，增加约18.9%；换来59/72到70/72。与完整历史比较才得到至少约25.3%的节省。 |
| 上下文题目是否独立盲测？ | 24段内部合成会话，16段已有回归、8段新增内部场景，单轮对照；问题目录公开，完整预置历史与逐次输出在本地归档。 |
| 来源与贡献怎么回答？ | 基于已有工程学习与扩展，按实际做过的分析、复现、修改和实验说明；仓库变更依据记录具体材料，不把整个基础工程归为个人原创。 |

## 8分钟路线

开场30秒 → 图与LangChain一分半 → 工具执行一分钟 → RAG两分钟 → 上下文一分钟 → 反馈20秒 → 物流与取消演示一分半 → Langfuse10秒。模型等待超时时，使用公开源码与隔离测试说明当前观察，避免重复介绍。
