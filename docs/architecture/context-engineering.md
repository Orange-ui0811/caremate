# 上下文工程：滚动摘要、滑窗与原文回查

当前开发运行方案已经替换为 **滚动摘要＋近期原文滑窗＋相关原文回查**，指代补全和主 Agent 继续使用现有 Workflow。新回查补充窗口外、摘要可能遗漏的原话关系，不额外增加规划模型调用。

四种方法的实验结果、问题设计和成本口径见 [上下文四方法报告](../evaluation/context/README.md)。本次公开的是报告与当前方案说明；公开应用源码的基础快照尚未同步本地全部新回查实现，下述新配置不能视为当前公开旧代码已经支持的完整运行承诺。

## 源码与实现对应

下面保留逐段源码入口。**基础源码**链接固定到已公开提交，避免后续行号漂移；**新回查代码摘录**来自当前本地实现，尚不代表整套应用源码已同步。

| 要理解的步骤 | 对应代码 | 阅读重点 |
| --- | --- | --- |
| 摘要边界与 Token 滑窗 | [memory.build_window](https://github.com/Orange-ui0811/caremate/blob/984339471d71c44c9513bfd5fe87b69a3a289d43/app/core/memory.py#L45) | 先按摘要覆盖边界取原文，再限制窗口大小 |
| 滚动摘要维护 | [compute_boundary](https://github.com/Orange-ui0811/caremate/blob/984339471d71c44c9513bfd5fe87b69a3a289d43/app/core/summarizer.py#L33)、[run_summary](https://github.com/Orange-ui0811/caremate/blob/984339471d71c44c9513bfd5fe87b69a3a289d43/app/core/summarizer.py#L54)、[maybe_schedule_summary](https://github.com/Orange-ui0811/caremate/blob/984339471d71c44c9513bfd5fe87b69a3a289d43/app/core/summarizer.py#L82) | 触发条件、保留近期轮次，以及摘要成功后推进边界 |
| 指代消解 | [coref.resolve](https://github.com/Orange-ui0811/caremate/blob/984339471d71c44c9513bfd5fe87b69a3a289d43/app/core/coref.py#L5)、[resolve_reference 节点](https://github.com/Orange-ui0811/caremate/blob/984339471d71c44c9513bfd5fe87b69a3a289d43/app/graph/nodes.py#L153) | 历史与当前问题怎样进入提示词，模型失败时怎样回退 |
| 新一轮历史恢复 | [runtime._new_turn_input](https://github.com/Orange-ui0811/caremate/blob/984339471d71c44c9513bfd5fe87b69a3a289d43/app/graph/runtime.py#L123) | 持久化原话怎样恢复到图状态 |
| 基础版节点输入拼装 | [_history_text](https://github.com/Orange-ui0811/caremate/blob/984339471d71c44c9513bfd5fe87b69a3a289d43/app/graph/nodes.py#L39)、[_agent_messages](https://github.com/Orange-ui0811/caremate/blob/984339471d71c44c9513bfd5fe87b69a3a289d43/app/graph/nodes.py#L274) | 指代/意图的历史文本与主 Agent 的消息列表分别怎样形成 |
| 新回查：BM25、邻轮扩展、逐节点去重 | [当前实现代码摘录](code-excerpts/context-recall.md#recall) | Top3 命中、前后邻轮、可见消息 ID、来源引用 |
| 新回查怎样接进各节点 | [节点接入代码摘录](code-excerpts/context-recall.md#integration) | 先确定节点窗口，再补充该节点缺少的原话 |

## 三类历史材料各有什么作用

| 材料 | 保留什么 | 主要作用 |
| --- | --- | --- |
| 滚动摘要 | 早期重要事实、对象关系和未完成诉求的压缩文本 | 为远距离纯指代提供背景，减少重复历史输入 |
| 近期原文滑窗 | 最近可容纳的用户与客服原话 | 保留当前话题、最新更正和连续工具交互 |
| 相关原文回查 | 与本轮问题相关的历史原问答及相邻问答 | 补回摘要遗漏、旧实体关系或离开节点近期输入的关键原句 |

数据库中的完整聊天记录仍然保存。模型每次收到的输入是按节点选择的视图，不等于把全部数据库记录直接发送给模型，也不等于删除窗口外消息。

## 新回查怎样工作

1. **限定来源。** 只使用已校验归属的当前会话、当前用户消息水位之前的持久化原问答。当前问题仅作为查询，不进入候选；未来消息、工具消息和无原文锚点内容不作为候选。
2. **查询全可见历史。** 以本轮用户原话对历史用户问句做本地 BM25 排名，不再只搜摘要边界以前的归档前缀。中文双字片段、英文与编号参与词项匹配。
3. **扩展相邻问答。** 取正分 Top3 命中及各自前一、后一问答，单跳扩展，最多 9 个问答单元；不递归，不添加额外实体扩展。
4. **逐节点去重。** 指代、意图和回答节点先完成自己的窗口截取，再按该节点已经可见的原文消息 ID 去重。若只已有某单元的一部分消息，仅补缺少部分；不按文本相同去重，也不因为摘要提到某事实就删除对应原话。
5. **附带来源。** 按历史顺序追加需要的原文，保留用户/客服角色、消息位置与来源引用。没有正分命中或已全部可见时，不追加。

核心选择与去重代码如下，摘自当前 `dialogue_supplement`；完整来源过滤和引用生成见[代码摘录](code-excerpts/context-recall.md#recall)。

```python
    units = _units(rows[:-1])
    ranked = _rank(units, rows[-1]["content"])
    selected, reasons = set(), {}
    for index, _ in ranked[:3]:
        for i, reason in ((index,"bm25_match"),(index-1,"previous_unit"),(index+1,"following_unit")):
            if 0 <= i < len(units):
                selected.add(i)
                reasons.setdefault(i,[]).append(reason)
    candidates = [row for i in sorted(selected) for row in units[i]]
    chosen = [row for row in candidates if row["id"] not in visible]
```

摘要提供概括，回查提供可定位的原话；两者不会自动变成当前业务事实。历史客服说“已签收”不能替代本轮物流回执，历史提及或撤回限制也不代表已经授权退款或建工单。

## 摘要和窗口如何维护

维护过程对应 [摘要实现](https://github.com/Orange-ui0811/caremate/blob/984339471d71c44c9513bfd5fe87b69a3a289d43/app/core/summarizer.py#L33) 与 [窗口实现](https://github.com/Orange-ui0811/caremate/blob/984339471d71c44c9513bfd5fe87b69a3a289d43/app/core/memory.py#L45)。当前开发运行配置如下，配置变化需要重启服务：

```dotenv
CONTEXT_POLICY=legacy
SUMMARY_TRIGGER_MESSAGES=30
CONTEXT_WINDOW_TURNS=3
CONTEXT_WINDOW_MAX_TOKENS=3000
CONTEXT_LEGACY_RECALL_ENABLED=true
CONTEXT_LEGACY_RECALL_MODE=all_visible_neighbors
```

`legacy` 是保留原节点输入方式的配置名，不代表仍然关闭新回查；`all_visible_neighbors` 才指定全可见历史、前后邻轮和节点去重。另一套结构化来源状态策略是独立实验路径，不混入本页四方法结果。

- 摘要边界之后累计 30 条消息时，在回复后后台尝试用“旧摘要＋新滑出原文”更新摘要；成功后保留最近 3 轮原文，失败保留旧摘要与边界。
- 留 3 轮是维护完成后的数量；之后原文继续累积，不是每轮所有节点都固定只看 3 轮。
- 指代消解与意图识别保留原有最近 6 条历史消息的截取；主 Agent 的原文窗口保持约 3,000 估算 Token。摘要、回查、系统提示、工具定义和当轮证据另计，3,000 不是整个请求的硬上限。
- 旧会话下次请求可以从已存原话回查；已有摘要不会因为配置切换被批量重写，下次符合维护条件时再更新。

## 一轮请求里的位置

代码阅读顺序：先看 [历史恢复](https://github.com/Orange-ui0811/caremate/blob/984339471d71c44c9513bfd5fe87b69a3a289d43/app/graph/runtime.py#L123)，再看[新回查的节点接入](code-excerpts/context-recall.md#integration)。

新用户轮先从数据库恢复已展示的原话，再分别形成各节点需要的摘要、近期窗口及回查补充。指代节点尝试补全“那单、摄影棚那件”等指代；意图节点读取相应历史选择路线；回答节点将系统规则、历史和本轮知识/订单/工具材料交给主 Agent。

新用户轮的历史恢复与当前轮工具循环不同：工具循环继续保留本轮 AI 工具请求与 ToolMessage；用户确认/选单通过中断恢复继续原运行状态，不能把它重建成一条全新的普通用户问题。

新回查不改变知识检索、订单归属检查、工具执行或提交确认的业务语义，也不能保证模型一定理解了已经检索到的原句。来源校验失败、没有有效命中和模型理解错误应分别排查。

## 四方法的结果怎样阅读

公开报告只展示以下四种方法：完整历史、仅滑动窗口、滚动摘要＋滑窗、滚动摘要＋滑窗＋新回查。最新混合在冻结的 24 段合成会话、72 次任务上通过 70 次，22 段三问全部通过，总模型 Token 为 479,290，包含摘要维护与记录到的重试。

这些是同数据、同模型、分阶段单次运行的内部结果。完整历史为 71/72，含一次模型接口故障并缺失 usage；不能声称新混合已经等效或超过完整历史。相对摘要＋滑窗，任务完成增加同时 Token 也增加。详细表格、题目设计、分母、成本下界及未通过案例见[四方法实验报告](../evaluation/context/README.md)。

该实验使用真实模型与 LangGraph/SSE，但知识、数据库和动态业务工具使用隔离夹具，并统一开启新鲜业务证据检查。当前部署配置与冻结实验执行条件需要分别记录，表中的规则通过率不代表当前真实客户准确率。

## 排障顺序

先核对恢复的历史、当前水位、摘要覆盖边界，再看某个节点实际已有的原文和补充回查消息 ID；随后检查补全问题、路由、工具对象、回执时间及最终答案。不能仅凭“数据库有摘要”或“回查命中一个编号”认定模型已经使用了正确关系。

完整原始会话、模型请求、私有环境配置和全部历史运行快照未放入公开展示包。公开问题目录是合成数据的设计投影，不是一键重跑环境。
