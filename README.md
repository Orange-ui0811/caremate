# 客小助 · CareMate

基于 **LangGraph + LangChain** 的电商客服应用，将确定性业务流程与 ReAct 式工具 Agent 结合，覆盖知识检索、订单与售后咨询、人工确认、上下文管理和运行观测。

订单、商品、物流及部分售后查询使用模拟数据；工单写入实际数据库，退款入口创建表单和工单，不执行支付退款。
## 从这里阅读

| 你想了解什么 | 入口 |
| --- | --- |
| 理解四个框架的职责与实现 | **[框架源码索引](docs/architecture/source-map.md)** |
| 模块设计、实现边界与验证方式 | [架构文档](docs/architecture/README.md) |
| Workflow、ReAct 与工具执行 | [架构详解](docs/architecture/workflow-and-agent.md) |
| 检索和拒答门控 | [RAG 与置信度](docs/architecture/rag-and-confidence.md) |
| 长对话、滑窗与摘要 | [上下文工程](docs/architecture/context-engineering.md) |
| 待审队列、知识反馈与主题分类 | [反馈与分类器](docs/architecture/feedback-and-classifier.md) |
| RAG、Embedding 与上下文的结果及题目设计 | **[实验与评估](docs/evaluation/README.md)** |
| 安装与本地运行 | [部署说明](DEPLOY.md) |

## 系统做什么

- **Workflow 与 Agent：**九类意图映射到五条路线，知识咨询先检索，退款/售后先确定订单并获取政策，再进入主 Agent；模型通过工具结果逐步决定下一步。
- **RAG：**组合向量与 BM25 检索、RRF 融合、重排和查询改写；知识分支使用候选最高重排分粗筛与生成前语义自检，返回带编号的证据引用。
- **工具执行：**统一注册本地与 MCP 工具，执行参数校验、可信参数注入、超时和有限重试；创建工单需要用户确认。
- **上下文与交互：**当前本地运行方案为滚动摘要＋近期原文滑窗＋相关原文回查，结合指代补全、SSE 和暂停恢复；[上下文工程](docs/architecture/context-engineering.md)说明全历史 BM25、邻轮扩展、消息 ID 去重及公开版本边界。
- **知识反馈：**将未解决问题与可获得的检索快照归集到待审流程，人工审核后补充知识；17 类主题分类器用于旁路归类，不代替在线九类意图识别。
- **运行观测：**接入 Langfuse，按会话与意图追踪执行并统计 Token。金额分析需要正确的模型价格和用量口径。

## 架构分工

| 层 | 技术与职责 | 核心代码 |
| --- | --- | --- |
| HTTP 与流式交互 | FastAPI、SSE | [app/api](app/api)、[runtime.py](app/graph/runtime.py) |
| 业务编排 | LangGraph 状态、分支、检查点与暂停恢复 | [build.py](app/graph/build.py)、[state.py](app/graph/state.py) |
| 模型交互 | LangChain 提示词、模型、结构化输出、工具消息 | [llm.py](app/core/llm.py)、[prompts.py](app/core/prompts.py) |
| Agent 循环 | 主模型决策 → 工具执行 → 结果回传 → 再决策 | [nodes.py](app/graph/nodes.py)、[routing.py](app/graph/routing.py) |
| 工具接入 | 内置工具、MCP 动态发现、统一执行引擎 | [app/tools](app/tools)、[mcp_servers](mcp_servers) |
| 检索与存储 | Milvus、MySQL、SQLite 检查点 | [retrieval.py](app/core/retrieval.py)、[app/db](app/db) |
| 观测 | Langfuse 回调、意图标签与 Token 报表 | [observability.py](app/core/observability.py)、[cost_by_intent.py](scripts/cost_by_intent.py) |

### 五条处理路线

公共前置步骤是“用户问题 → 指代补全 → 意图识别”。

| 意图 | 后续处理 |
| --- | --- |
| 商品咨询 | 知识检索 → 证据判断 → 主 Agent 或固定兜底 |
| 退款退货、售后 | 确定订单 → 检索政策 → 主 Agent |
| 订单、物流、人工 | 主 Agent 按需选择工具 |
| 投诉 | 固定安抚与后续操作入口 |
| 闲聊、其他 | 固定引导话术 |

主 Agent 通过 Function Calling 提出行动请求，后端执行后将结果写成 ToolMessage，再请求模型继续处理。没有工具请求或达到步数上限时结束。当前没有要求显式输出 Thought 文本，也没有独立的多 Agent 协作系统。

### 八个业务工具

| 来源 | 工具 |
| --- | --- |
| 内置 | `query_order`、`query_product`、`query_faq`、`submit_refund`、`create_ticket` |
| 物流 MCP 服务 | `query_logistics` |
| 售后 MCP 服务 | `query_warranty`、`query_return_status` |

实际绑定数量取决于当时 MCP 服务的发现结果。`submit_refund` 在图中转换为前端表单；`create_ticket` 确认后写入 MySQL。

## 本地运行

需要 Python 3.12+（可由 uv 管理）、uv、Docker Compose。Makefile 工作流还需要 Make 与 Bash；Windows 可使用 Git for Windows、GNU Make 和项目启动脚本。模型聊天、嵌入、重排服务分别在配置中设置。

1. 复制 `.env.example` 为 `.env`，填写自己的服务配置。
2. 依照 [DEPLOY.md](DEPLOY.md) 安装依赖、启动容器并初始化业务与知识数据。
3. Windows 可运行 `Start-CareMate.cmd`；使用 Make 的环境可运行下面的基础启动命令。

```bash
uv sync
cp .env.example .env
# 编辑 .env 后再启动
docker compose up -d
make seed
make dev
```

以上是基础服务启动，不能代替知识入库、向量化和可选分类器准备。首次部署请继续按部署说明执行相关初始化；模型权重和本机运行数据不应随源码直接发布。

| 页面/服务 | 默认本机地址 |
| --- | --- |
| 客服聊天 | http://localhost:8000/ |
| 管理入口 | http://localhost:8000/admin |
| 知识库 | http://localhost:8000/kb |
| 待审队列 | http://localhost:8000/review |
| 观测与成本 | http://localhost:8000/observability |
| 主题分类验收 | http://localhost:8000/acceptance |
| Langfuse（单独启动） | http://localhost:3000/ |

这些 localhost 地址指向访问者自己的电脑，不是在线演示站。Compose 配置用于本机演示，公开部署前需要调整默认凭据、鉴权与网络暴露设置。

## 实验结果与问题设计

| 实验 | 固定数据与比较范围 | 报告 |
| --- | --- | --- |
| RAG 四策略 | 300 题、五桶；240 可答题的检索、证据与答案覆盖 | [四策略结果与五桶设计](docs/evaluation/rag/README.md) |
| 五模型 Embedding | 同 52 知识片段和 300 题；纯向量精确余弦排序 | [模型、维度、输入协议与分桶结果](docs/evaluation/embedding/README.md) |
| 上下文四方法 | 同 24 段合成会话，每方法三次连续追问；含维护成本 | [四方法对照与测试场景](docs/evaluation/context/README.md) |

上下文页只展示完整历史、仅滑窗、滚动摘要＋滑窗、滚动摘要＋滑窗＋新回查。三组实验日期和执行链路不同，不能拼用最优指标；原始调用缺失的用量保留为下界。公开的是历史结果展示，不代表当前线上准确率或完整一键复现环境。

## 验证与结果口径

```bash
make test
make help
```

`make test` 执行项目测试；真实模型、检索与分类器评估另有前置条件，参见部署说明和架构文档。测试运行与真实服务验收分开记录，具体环境和检查范围以运行结果为准。

评估数字应与报告日期、样本范围、模型配置一起呈现。历史结果不等于当前请求的成功率；源码包含某功能也不等于已验证该功能的生产可靠性。

## 目录

```text
app/api/          HTTP 与 SSE 接口
app/graph/        状态、节点、路由和图运行
app/core/         模型、检索、上下文、反馈与观测
app/tools/        工具注册、实现、MCP 客户端与执行引擎
app/db/           数据模型与仓储
app/static/       客服与管理页面
mcp_servers/      独立物流和售后服务
scripts/          初始化、评估、训练与报表
sql/              数据库初始化与迁移
tests/            测试与测试数据
docs/architecture/   架构说明及源码索引
```


## 本次公开范围

仓库包含应用基础源码、前端、MCP 服务、部署脚本、测试数据、示例知识、架构文档，以及选定实验汇总和合成题目设计。真实配置、会话数据库、原始客户语料、完整调用日志、缓存、训练权重及私有实验快照未包含。新回查文档描述当前本地运行版本，公开应用源码尚未同步本地全部上下文改动；版本关系见[上下文工程](docs/architecture/context-engineering.md)。

定位实现可使用 [框架源码索引](docs/architecture/source-map.md)，该页面直接链接仓库中的实际函数。
