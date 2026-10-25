# 客小助 · CareMate

基于 **LangGraph + LangChain** 的电商客服演示项目，将确定性业务流程与 ReAct 式工具 Agent 结合，覆盖知识检索、订单与售后咨询、人工确认、上下文管理和运行观测。

本项目用于架构学习、功能复现与扩展展示。订单、商品、物流及部分售后查询使用模拟数据；工单写入实际数据库，退款入口创建表单和工单，不执行支付退款。
## 从这里阅读

| 你想了解什么 | 入口 |
| --- | --- |
| 面试官追问四个框架，快速定位实现 | **[问题 → 源码函数 → 行号速查](docs/interview/framework-source-map.md)** |
| 完整讲解、指标口径与演示准备 | [面试与演示导航](docs/interview/README.md) |
| Workflow、ReAct 与工具执行 | [架构详解](docs/interview/workflow-and-agent.md) |
| 检索和拒答门控 | [RAG 与置信度](docs/interview/rag-and-confidence.md) |
| 长对话、滑窗与摘要 | [上下文工程](docs/interview/context-engineering.md) |
| 待审队列、知识反馈与主题分类 | [反馈与分类器](docs/interview/feedback-and-classifier.md) |
| 安装与本地运行 | [部署说明](DEPLOY.md) |

## 系统做什么

- **Workflow 与 Agent：**九类意图映射到五条路线，知识咨询先检索，退款/售后先确定订单并获取政策，再进入主 Agent；模型通过工具结果逐步决定下一步。
- **RAG：**组合向量与 BM25 检索、RRF 融合、重排和查询改写；知识分支使用证据评分与生成前语义自检，返回带编号的证据引用。
- **工具执行：**统一注册本地与 MCP 工具，执行参数校验、可信参数注入、超时和有限重试；创建工单需要用户确认。
- **上下文与交互：**结合滑动窗口、滚动摘要和指代补全；支持 SSE 事件、会话管理，以及订单选择/工单确认后的流程恢复。
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

## 验证与结果口径

```bash
make test
make help
```

`make test` 执行项目测试；真实模型、检索与分类器评估另有前置条件，参见部署说明和面试文档。本文更新只验证文档链接与源码定位，不代表重新执行了全部测试或在线验收。

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
docs/interview/   面试讲解及源码导航
```


## 本次公开范围

仓库包含应用源码、前端、MCP 服务、部署脚本、测试数据、示例知识和面试文档。真实配置、会话数据库、17 条脱敏问题及其衍生产物、日志、缓存、训练权重、本机评估归档和学习图谱快照未包含。相关报告需按文档在本机重新生成。

面试定位实现请优先使用 [框架源码速查](docs/interview/framework-source-map.md)，该页面直接链接仓库中的实际函数。
