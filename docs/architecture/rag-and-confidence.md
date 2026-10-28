# RAG 与证据评估

知识问答依次经过资料分块、向量化、混合检索、重排、证据检查和回答生成。本页按存储、模型选型、检索策略与评估说明实现。

实验入口：[五模型 Embedding 对比](../evaluation/embedding/README.md) · [RAG 四策略评估](../evaluation/rag/README.md)。

源码核对基线：[fcfdd41](https://github.com/Orange-ui0811/caremate/commit/fcfdd419b4736a189e65b7cee2fdd52691fb194e)。源码链接固定到该提交，实验结果的适用范围见下文。

## 知识与存储选择

示例资料位于 [data/kb](../../data/kb)，包括商品规格、FAQ、退换货政策、售后手册、会员权益和运费发票规则。订单、物流等动态状态由业务工具查询。

| 存储 | 保存内容 | 在本项目中的作用 | 实现 |
| --- | --- | --- | --- |
| MySQL | 知识原文、分类、章节路径与处理状态 | 后台管理、人工审核、更新追踪 | [dualwrite.py](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/app/kb/dualwrite.py#L7) |
| Milvus | dense 向量、检索 text、BM25 sparse 与元数据 | 同集合内完成语义与字面召回，支持分类过滤和混合排序 | [ensure_collection](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/app/kb/milvus_client.py#L41) |

Milvus 的 dense 字段采用余弦相似度，text 使用中文分析器并通过 BM25 Function 生成 sparse 字段。BGE-M3 提供 dense 向量。两库使用知识 ID 对应，向量化由 pending 状态推进，跨存储更新通过状态与重试协调。

## 知识入库

| 处理 | 做法 | 目的 | 源码 |
| --- | --- | --- | --- |
| 标题分节 | 保留一至四级标题路径 | 保留商品、章节与政策主题，形成引用出处 | [build_chunks](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/app/kb/documents.py#L28) |
| 正文切分 | 默认基础块 400 字符，按段落和句子分隔递归切分 | 控制检索单元大小 | [recursive_split](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/app/kb/chunking.py#L16) |
| 句子重叠 | 从上一块末尾补完整句子，目标约 60 字符 | 保留跨块条件与结论；完整句可能超过目标长度 | [apply_sentence_overlap](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/app/kb/chunking.py#L44) |
| 表格拆分 | 默认每块最多 10 行数据，重复表头 | 保留数值与列名的对应关系 | [split_table_rows](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/app/kb/chunking.py#L72) |
| 向量化 | 拼接 category、questions、answer | 为正文附带分类与标题上下文 | [vectorize_pending](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/app/kb/dualwrite.py#L29) |

## Embedding 选型

当前使用 BGE-M3（1024 维）。[五模型对照](../evaluation/embedding/README.md)固定 52 个知识片段和 300 道问题，比较 M3、BGE-small、BGE-large、Qwen3-Embedding-0.6B 与 4B，采用模型对应的前缀和维度进行纯向量精确余弦排序。

- M3 的分组 Recall@5 为 **98.96%**，E 桶跨文档召回为 **95.83%**，为该题集中的较高结果。
- Qwen0.6B 的分组倒数排名@10 为 **0.8624**，高于 M3 的 0.8444。
- 当前保留 M3 的依据是该语料上的证据召回表现；模型对照与完整检索策略对照分别评估。

## 查询与检索

普通单问的主路径：**标准问法 → dense 与 BM25 双路召回 → RRF 融合 → BGE-Reranker-v2-M3 重排 → 证据检查**。

| 阶段 | 当前实现 | 作用 |
| --- | --- | --- |
| 查询理解 | [retrieve_knowledge](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/app/graph/nodes.py#L180)生成标准问法与扩展词；未拆分单问的扩展词追加到 BM25 文本 | 兼顾口语规范化与字面扩展 |
| 双路召回 | dense 与 BM25 默认各取 Top50 | 覆盖语义近似与型号、术语匹配 |
| 融合 | [hybrid_search](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/app/kb/milvus_client.py#L111)使用 RRFRanker，默认保留 50 个候选 | 根据两路排名融合，避免直接相加不同尺度分数 |
| 重排 | [search_knowledge](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/app/core/retrieval.py#L51)对问题与候选文本精排，默认取 Top10 | 从候选中进一步选择相关证据 |
| 多子句 | 按标点拆分子句，分别检索重排，再轮转合并、按片段去重 | 分配多个诉求的证据名额 |

源码版本说明：上述公开 `retrieve_knowledge` 从用户原句发起查询理解；开发运行版本已增加优先使用补全问题的处理。多轮查询需结合相应版本查看。

## 检索策略的实验依据

[RAG 四策略报告](../evaluation/rag/README.md)使用 300 题五桶设计，其中 240 道可答题参与主要质量统计。

| 策略 | 分组 Recall@5 | 答案要点覆盖（模型裁判） |
| --- | ---: | ---: |
| 向量 | 97.3% | 73.5% |
| BM25 | 91.3% | 84.2% |
| 混合 | 94.8% | 89.7% |
| 混合＋重排 | **98.4%** | **93.7%** |

混合加重排在本次对照中取得较高的证据召回与答案要点覆盖，E 桶召回为 93.6%。题目包含政策、型号、口语、库外及跨文档场景；多证据题按必需证据组计分。运行日期、完整分桶与指标定义见报告。

## 生成前两道检查

第一道由 [compute_evidence_confidence](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/app/core/confidence.py) 对最终返回的最多 10 条候选计算最高重排分：`max(clip(rerank_score, 0, 1))`，没有有效候选时为 0；非法值和非有限数值不参与评分。评分版本为 `max_rerank_v1`，不依赖合并顺序，不使用数量、分差或通用关键词加权。

该分数衡量候选相关性，复合问题的完整覆盖由后续语义自检继续判断。代码默认阈值为 `0.2260993940477108`（约 0.2261），来自 2026-10-27 内部实验按原始 ABCD 标签、校准集误放不超过 5% 的选择结果。该值对应上述内部知识快照与标签。管理端保存且评分版本匹配的阈值优先生效；例如保存 `0.226` 后，实际比较就使用 `0.226`，显示时的四舍五入不改变计算精度。

运行阈值与校准报告记录评分版本；旧四信号阈值不会自动用于新评分，旧校准报告会标记过期，不再提供可直接应用的推荐值。

分数达到运行阈值后，第二道调用 [check_sufficient](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/app/core/selfcheck.py)，让模型输出证据是否足够及原因。任何一关未通过都会进入兜底；通过后整理编号证据，交给主 Agent。

`confidence_check` 节点记录判断，实际分流由其后条件边完成。失败问题及可获得的召回快照进入待审流程。

## 阈值与评估

[runtime_settings.py](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/app/core/runtime_settings.py) 持久化运行阈值；后续请求读取更新值。降低阈值不会补充缺失知识，也不会跳过语义自检。

[eval_ch04.jsonl](../../tests/data/eval_ch04.jsonl) 提供固定评估问题；[eval_ch04.py](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/scripts/eval_ch04.py) 比较检索策略，[calibrate_confidence.py](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/scripts/calibrate_confidence.py) 扫描证据阈值。

### 校准来源与选择规则

| 入口 | 分数来源 | 阈值选择规则 | 是否修改在用阈值 |
| --- | --- | --- | --- |
| `make calibrate-confidence` | 当前检索接口实跑；ABC 可答题、D 应拒题参与 | 在 0.05～0.95、步进 0.01 的扫描点中选择 Youden J（可答通过率减应拒放行率）最大值，标记 `youden_j` | 不修改；管理端手动应用 |
| [rebuild_confidence_curve.py](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/scripts/rebuild_confidence_curve.py) | 保存的真实接口候选记录回放，重新计算当前最高分 | 读取实验中校准集误放不超过 5%、优先减少误拒的选定值，再核对计数，标记 `false_accept_cap` | 不修改；管理端手动应用 |

两者的检索来源和选择目标不同，推荐值可能不同。管理端展示来源、选择规则、在用值和该报告分数下的取舍。回放曲线除常规扫描点外包含精确选定值；旧评分报告标为过期，不再提供可应用的推荐值。

回放命令为 `python -m scripts.rebuild_confidence_curve --study <记录目录>`，默认输出到 `data/ch09/reports/`，也可通过 `--output` 指定目录。需要 `samples.json`、`protocol.json`、逐题 `cases/<id>.json`、`knowledge-before.json` 和 `analysis.json`；输入格式示例见 [回放测试](https://github.com/Orange-ui0811/caremate/blob/fcfdd419b4736a189e65b7cee2fdd52691fb194e/tests/test_rebuild_confidence_curve.py)。回放命令读取指定目录中的实验记录，输入文件由运行方准备。

### 2026-10-27 内部验证的口径

该实验采集 300 题的真实接口检索记录，使用 54 个知识片段；各评分方案复用每题同一份候选。ABC 可答 180 题、D 应拒 60 题参与阈值选择与分组五折验证，跨文档 E 桶 60 题不参加阈值选择。

| 最高重排分方案的评估方式 | 可答误拒 | 应拒误放 |
| --- | --- | --- |
| 全 ABCD 拟合约 0.2261 后，在同批校准数据回放 | 5/180（2.78%） | 3/60（5.00%） |
| 分组五折：每折只用其他四折选择阈值，合并留出预测 | 8/180（4.44%） | 3/60（5.00%） |

五折每折独立选择阈值并在留出折统计，E 桶在全部 ABCD 拟合阈值下误拒 2/60。统计对象为第一道相关性粗筛；完整路由、工具和答案生成另行评估。

该题集用于内部调优。负例可答性、部分可答标签与知识冲突仍在复核；隔离争议负例后的分析中，最高分在 ABC 上的误拒优势消失。当前采用最高分的主要依据是简化评分、消除合并顺序及通用关键词奖励的干扰，后续验证包括独立问集与完整在线链路。


| 指标 | 解释 |
| --- | --- |
| 分组 Recall@K | 所需证据组在前 K 项中的覆盖程度 |
| 分组倒数排名 | 每个必需证据组最早命中名次的倒数，先组内、再题间平均 |
| 证据要点覆盖 | 检索材料包含标准要点的程度 |
| 答案要点覆盖 | 生成答案覆盖要点的程度，需说明裁判及评分方法 |
| 可答通过 / 应拒放行 | 给定阈值下的取舍，需同时报告分母与样本构成 |

阈值选择与留出验证分别记录；语义自检和最终生成质量使用各自评估结果。

## 复现与排障

按 [部署说明](../../DEPLOY.md) 完成知识入库、向量化和模型配置后，可使用 `make eval-rag` 与 `make calibrate-confidence` 生成本机报告。运行前检查服务可达性，并记录代码、知识库、模型及数据版本。

拒答排查按 `fallback_source`、评分与召回快照、自检输入、异常日志定位。`self_check` 分支覆盖证据不足及自检调用失败两类情况。

原始调用记录保存在实验归档中。校准文件通过评分版本区分最高重排分与历史四信号方案。
