# RAG 与证据评估

知识问答由入库处理、混合检索、证据评分、生成前自检和回答生成组成。各阶段需要分别验证，召回到资料不等于最终答案正确。

历史结果与测试问题设计：[RAG 四策略评估](../evaluation/rag/README.md)、[五模型 Embedding 对比](../evaluation/embedding/README.md)。两组的日期和链路不同，不能直接拼用指标。

源码核对基线：[e9fa3a2](https://github.com/Orange-ui0811/caremate/commit/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6)。源码链接固定到该提交，实验结果的适用范围见下文。

## 知识入库

示例知识位于 [data/kb](../../data/kb)。[documents.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/kb/documents.py) 与 [chunking.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/kb/chunking.py) 处理标题路径、长正文重叠和表格拆分。[dualwrite.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/kb/dualwrite.py) 协调 MySQL 与 Milvus 的写入及状态更新。

MySQL 保存原文与元数据，Milvus 提供检索索引。两者不构成跨存储强事务；需要关注部分成功、失败状态与重试。

## 查询与检索

[retrieve_knowledge](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/graph/nodes.py#L180) 从当前用户原句发起查询理解，生成标准问句和扩展词，再调用 [search_knowledge](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/retrieval.py)。当前此入口未直接优先使用 `resolved_query`，依赖上下文的短问句需要单独验证。

检索使用向量与 BM25 补充语义和字面匹配，通过 RRF 融合，再执行重排。服务和模型由配置决定。检索策略是否有效应按不同类型问题比较，不能从某个单例推导全面优势。

## 生成前两道检查

第一道由 [compute_evidence_confidence](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/confidence.py) 对最终返回的最多 10 条候选计算最高重排分：`max(clip(rerank_score, 0, 1))`，没有有效候选时为 0；非法值和非有限数值不参与评分。评分版本为 `max_rerank_v1`，不依赖合并顺序，不使用数量、分差或通用关键词加权。

这是相关性粗筛：最高分可能来自某个子问题，不能证明复合问题完整覆盖，也不是答案正确概率。代码默认阈值为 `0.2260993940477108`（约 0.2261），来自 2026-10-27 内部实验按原始 ABCD 标签、校准集误放不超过 5% 的选择结果。它是内部实验候选值，不构成新业务的适用性保证。管理端保存且评分版本匹配的阈值优先生效；例如保存 `0.226` 后，实际比较就使用 `0.226`，显示时的四舍五入不改变计算精度。

运行阈值与校准报告记录评分版本；旧四信号阈值不会自动用于新评分，旧校准报告会标记过期，不再提供可直接应用的推荐值。

分数达到运行阈值后，第二道调用 [check_sufficient](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/selfcheck.py)，让模型输出证据是否足够及原因。任何一关未通过都会进入兜底；通过后整理编号证据，交给主 Agent。

`confidence_check` 节点记录判断，实际分流由其后条件边完成。失败问题及可获得的召回快照进入待审流程。

## 阈值与评估

[runtime_settings.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/app/core/runtime_settings.py) 持久化运行阈值；后续请求读取更新值。降低阈值不会补充缺失知识，也不会跳过语义自检。

[eval_ch04.jsonl](../../tests/data/eval_ch04.jsonl) 提供固定评估问题；[eval_ch04.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/scripts/eval_ch04.py) 比较检索策略，[calibrate_confidence.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/scripts/calibrate_confidence.py) 扫描证据阈值。

### 校准来源与选择规则

| 入口 | 分数来源 | 阈值选择规则 | 是否修改在用阈值 |
| --- | --- | --- | --- |
| `make calibrate-confidence` | 当前检索接口实跑；ABC 可答题、D 应拒题参与 | 在 0.05～0.95、步进 0.01 的扫描点中选择 Youden J（可答通过率减应拒放行率）最大值，标记 `youden_j` | 不修改；管理端手动应用 |
| [rebuild_confidence_curve.py](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/scripts/rebuild_confidence_curve.py) | 保存的真实接口候选记录回放，重新计算当前最高分 | 读取实验中校准集误放不超过 5%、优先减少误拒的选定值，再核对计数，标记 `false_accept_cap` | 不修改；管理端手动应用 |

两者的检索来源和选择目标不同，推荐值可能不同。管理端展示来源、选择规则、在用值和该报告分数下的取舍。回放曲线除常规扫描点外包含精确选定值；旧评分报告标为过期，不再提供可应用的推荐值。

回放命令为 `python -m scripts.rebuild_confidence_curve --study <记录目录>`，默认输出到 `data/ch09/reports/`，也可通过 `--output` 指定目录。需要 `samples.json`、`protocol.json`、逐题 `cases/<id>.json`、`knowledge-before.json` 和 `analysis.json`；输入格式示例见 [回放测试](https://github.com/Orange-ui0811/caremate/blob/e9fa3a25efbeb69dcfbe703826c7c400a10fdfd6/tests/test_rebuild_confidence_curve.py)。本机实验记录未包含在仓库中，命令需要使用者提供自己的记录。

### 2026-10-27 内部验证的口径

该实验采集 300 题的真实接口检索记录，使用 54 个知识片段；各评分方案复用每题同一份候选。ABC 可答 180 题、D 应拒 60 题参与阈值选择与分组五折验证，跨文档 E 桶 60 题不参加阈值选择。

| 最高重排分方案的评估方式 | 可答误拒 | 应拒误放 |
| --- | --- | --- |
| 全 ABCD 拟合约 0.2261 后，在同批校准数据回放 | 5/180（2.78%） | 3/60（5.00%） |
| 分组五折：每折只用其他四折选择阈值，合并留出预测 | 8/180（4.44%） | 3/60（5.00%） |

五折使用各折自己的阈值，不是固定 0.2261 的测试结果；E 桶在全部 ABCD 拟合阈值下误拒 2/60。以上仅衡量第一道相关性粗筛，不包含完整意图路由、会话历史、工具调用或最终答案生成，也不等于双闸的端到端误拒率、回答准确率或幻觉率。

题集曾用于项目调优，这属于内部验证；另有负例可答性、部分可答标签和知识冲突等待复核。隔离争议负例的事后分析中，最高分在 ABC 上少误拒的优势不再成立。因此采用最高分主要简化评分并消除合并顺序、分差与通用关键词奖励的干扰，不能仅凭这些结果宣称稳定优于旧方案。新增独立题、标签审核与完整在线链路评估仍需分别开展。


| 指标 | 解释 |
| --- | --- |
| 分组 Recall@K | 所需证据组在前 K 项中的覆盖程度，不等于回答准确率 |
| 分组倒数排名 | 各证据组最早命中名次的倒数，不应与只看首个相关项的 MRR 混用 |
| 证据要点覆盖 | 检索材料包含标准要点的程度 |
| 答案要点覆盖 | 生成答案覆盖要点的程度，需说明裁判及评分方法 |
| 可答通过 / 应拒放行 | 给定阈值下的取舍，需同时报告分母与样本构成 |

阈值校准不运行完整的在线生成链路，不能将其结果直接视为包含语义自检与回答生成的端到端误拒率。阈值选择集与独立测试集应分开。

## 复现与排障

按 [部署说明](../../DEPLOY.md) 完成知识入库、向量化和模型配置后，可使用 `make eval-rag` 与 `make calibrate-confidence` 生成本机报告。运行前检查服务可达性，并记录代码、知识库、模型及数据版本。

拒答排查先看 `fallback_source`、分数、各信号和召回快照，再检查自检输入与异常日志。`self_check` 分支可能反映证据不足或自检调用失败，不能仅凭分支名判定唯一根因。

公开仓库不包含本机原始报告与会话记录。上述内部结果用于说明阈值选择与验证口径；旧四信号报告应保留历史版本标识，不能将其中的阈值或性能数字直接套用于当前评分。
