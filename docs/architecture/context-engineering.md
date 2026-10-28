# 上下文工程：滚动摘要、滑窗与原文回查
原文保存在数据库；滑窗保留近期表达，减少压缩损失；摘要保存较早的背景；原文回查补充摘要可能遗漏的对象关系、更正和限制。摘要边界之后累计30条消息时，在当轮回复结束后后台更新摘要，成功后保留最近3轮原文。每次收到问题，再用本轮原话对当前会话的历史用户问句做BM25检索，取正分Top3，并带上前后相邻问答。最后按消息ID与各节点已有原文去重。

## 源码与实现对应

基础源码链接固定到对应提交；新回查的选择器与节点接入见代码摘录。

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

数据库保留完整聊天记录，各节点从中选择本轮需要的模型输入。窗口裁剪只影响模型输入，原始历史继续保留。

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

摘要与回查用于恢复对象、关系和更正；当前订单状态以本轮业务回执为依据，退款或建工单仍经过对应的授权与确认流程。

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

`legacy` 保留原节点输入方式，`all_visible_neighbors` 选择全可见历史、前后邻轮和节点去重。结构化来源状态策略属于另一条实验路径。

- 摘要边界之后累计 30 条消息时，在回复后后台尝试用“旧摘要＋新滑出原文”更新摘要；成功后保留最近 3 轮原文，失败保留旧摘要与边界。
- 旧会话下次请求可以从已存原话回查；已有摘要不会因为配置切换被批量重写，下次符合维护条件时再更新。

## 一轮请求里的位置

代码阅读顺序：先看 [历史恢复](https://github.com/Orange-ui0811/caremate/blob/984339471d71c44c9513bfd5fe87b69a3a289d43/app/graph/runtime.py#L123)，再看[新回查的节点接入](code-excerpts/context-recall.md#integration)。

新用户轮先从数据库恢复已展示的原话，再分别形成各节点需要的摘要、近期窗口及回查补充。指代节点尝试补全“那单、摄影棚那件”等指代；意图节点读取相应历史选择路线；回答节点将系统规则、历史和本轮知识/订单/工具材料交给主 Agent。

新用户轮从数据库恢复历史；当前轮工具循环保留 AI 工具请求与 ToolMessage；用户确认和选单通过中断恢复继续原运行状态。

回查完成上下文补充，知识检索、订单归属检查、工具执行与提交确认继续使用原业务流程。诊断时分别记录来源校验、检索命中和模型对材料的使用情况。

## 四方法评估

固定 24 段合成会话、每段三次连续追问，对比完整历史、仅滑窗、摘要＋滑窗和摘要＋滑窗＋新回查。新回查组通过 70/72 次任务，22/24 段三问全过，总模型用量为 479,290 Token，包含摘要维护和记录到的重试。

实验使用真实模型与 LangGraph/SSE，数据库、知识与业务工具采用隔离夹具。完整历史组通过 71/72 次任务，用量因一次接口超时缺失 usage 记为下界。完整结果、问题设计和误差分析见[四方法报告](../evaluation/context/README.md)。

## 排障顺序

1. 核对恢复历史的归属、当前消息水位和摘要覆盖边界。
2. 检查节点窗口已有消息 ID、回查命中与补充原文。
3. 对照指代补全、路由、工具对象、本轮回执和最终答案。

## 版本与材料

| 材料 | 对应版本与内容 |
| --- | --- |
| 仓库 `app/` | 已公开的基础上下文源码，包含历史恢复、窗口、摘要与指代消解 |
| 本页与[回查摘录](code-excerpts/context-recall.md) | 2026-10-28 开发运行版本的算法与节点接入；完整新回查依赖及应用接线尚未同步到 `app/` |
| [评估报告](../evaluation/context/README.md) | 冻结条件下的四方法结果、用量与场景统计 |
| [问题目录](../evaluation/context/questions.json) | 合成样本的目的、长度、三次问题与预设金标；完整预置历史和逐次输出保存在实验归档 |
