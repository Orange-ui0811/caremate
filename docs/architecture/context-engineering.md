# 上下文管理

上下文模块在历史保留、模型输入长度和实体一致性之间取舍，主要包括会话历史恢复、消息窗口、滚动摘要和指代补全。

## 数据流

1. [runtime._new_turn_input](../../app/graph/runtime.py#L123) 在已有会话的新用户轮，从 MySQL 恢复已展示的用户和客服消息。
2. [memory.py](../../app/core/memory.py) 按摘要边界选择近期消息，再使用近似 Token 预算裁剪。
3. [coref.resolve](../../app/core/coref.py) 结合摘要与近期历史，把“它、最开始那笔订单”等指代补成完整问题。
4. [nodes._agent_messages](../../app/graph/nodes.py#L274) 组装固定系统提示、历史窗口及本轮材料。
5. [summarizer.py](../../app/core/summarizer.py) 在轮后检查是否需要后台更新摘要。

持久化聊天记录与模型实际收到的上下文不是同一概念。限制模型窗口并不等于从数据库删除旧消息；当前轮工具结果也需要与新用户轮历史恢复分开处理。

## 摘要与窗口

摘要使用消息 ID 记录覆盖边界，避免旧摘要与近期原文无边界地重复堆叠。触发阈值、保留轮数和窗口预算由配置决定。

摘要更新是有损且异步的：失败时保留旧摘要及边界，下一轮也可能在后台更新完成前继续使用旧摘要。进程内防重复标记不等于跨进程持久任务队列。

`_history_text` 在窗口基础上取最近若干条消息，并添加摘要；其 `max_turns` 参数实际切分消息列表，不能简单解释成完整的用户/客服轮数。当前主 Agent 的摘要和证据作为用户侧材料组织，具体位置以 `_agent_messages` 为准。

指代补全失败时回落原句，保证流程可以继续，但不保证语义已完整。指代补全成功也不等于后续检索和回答一定成功。

## 对照实验

[context_engineering_v1.json](../../tests/data/context_engineering_v1.json) 提供固定合成历史。[eval_context_engineering.py](../../scripts/eval_context_engineering.py) 比较完整历史、仅滑窗、摘要加滑窗；[audit_context_eval.py](../../scripts/audit_context_eval.py) 复算已保存结果。

进一步的多策略比较见 [方法说明](../context-comparison-method.md)、[评估脚本](../../scripts/eval_context_comparison.py) 和 [语义审计](../../scripts/audit_context_semantics.py)。策略迁移实验不等于替换在线运行时。

结果应分开报告：

- 规则或语义判定的通过率，以及实体更正、否定、时间更新等分组表现。
- 最后一次请求的输入 Token 与摘要维护消耗。
- 模型调用失败、超时和成功请求的输出质量。
- 完整历史的效果上限与摘要方案的信息损失。

最后一次请求的 Token 减少不能直接表述为整个会话的同等降本；规则通过率也不是生产客服整体准确率。

## 运行与诊断

可按 [部署说明](../../DEPLOY.md) 使用 `make eval-context` 生成实验产物。测试案例、模型配置、代码版本和原始输出应共同留存，已有结果的审计与重新调用模型需要区分。

排查实体遗漏时，检查新轮恢复的历史、摘要覆盖边界、窗口内容、补全后的问题以及下游节点实际使用的查询。`history_ctx` 与 `model_ctx` 反映不同调用点的材料，不能仅以数据库存在摘要就推断它已进入模型输入。
