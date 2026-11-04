# 当前客服：思考开启/关闭配对实验

日期：2026-10-25。仅隔离实验，不修改线上模型或上下文策略。

## 要回答的问题

当前摘要窗口方案的失败是否能通过开启思考减少？来源状态方案是否也受益？效果变化需要多少思考Token和响应时间？

当前源码的摘要提示词、记忆与节点拼装已不同于10月23日，因此本轮关闭组全部重新生成。旧分数不作为本轮因果对照。

## 固定设计

- 数据：仓库`make_suite()`生成的合成数据，每个场景族固定选择variant 0（24轮）和variant 2（64轮），8类共16条。按位置选样，不按旧分数挑选。
- 每条两个因果追问，共32次追问/组，四组128条结果。
- C_off/C_on：当前摘要触发、提示词、窗口与指代可见范围；摘要、改写、规划、回答全部使用该组模式。
- S_off/S_on：8000估算Token原文窗口＋1800估算Token来源状态。状态由本地规则生成，不调用模型，开关不会改变这部分输入。
- 当前Gitee DeepSeek-V4-Flash，同一工具夹具、提示词和评分；不开供应商模型故障转移。
- 使用`chat_template_kwargs`中的`thinking`与`enable_thinking`，关闭组均为false，开启组均为true；清除继承的其他extra_body开关。不指定reasoning_effort，由该端点默认处理。以usage确认思考实际发生，不把请求参数当作生效证据。初次顶层开关预检未生效，保留在verification原目录；正式运行必须提供通过验证的预检证据。
- 每阶段相同8192总输出上限（包括思考和最终输出），非摘要/摘要输入估算均须≤16000。原有512/1024输出预算不适合直接用于开关对照。
- 请求温度：摘要/改写0.3，规划/回答0，两组请求一致；不保证上游实际采样实现一致。
- 最多4路后台调用，最小启动间隔1秒。沿用当前模型池90秒调用期限，外层100秒。超时/截断说明预算或服务限制，不能解释为纯记忆失败。
- 固定种子打乱会话顺序，同会话四组交错执行，维护状态分别保存。
- 只对传输故障立即重试一次；空输出、格式错误、截断、语义答错保留。缓存契约不匹配即报错。

## 报告与审计

任务规则通过、动作/对象正确、指代、分场景；输入、总输出、思考输出与非思考输出；摘要维护的单独开销；三阶段等待P50/P95；接口/格式/截断失败。输出Token已含思考，不能再把思考Token加一次。

全部逐请求保存usage、参数、工具调用、输出、finish_reason、错误类型/HTTP状态与尝试。离线重演摘要边界、来源状态及每次请求，检查输入哈希、完整性和评分；S组开关的原始上下文必须一致。按16条会话做配对bootstrap，不将同会话两次追问视作独立样本。

本轮仍是已知合成回归集和共享只读测试执行器，不是生产端到端或独立新场景泛化测试。旧规则与现有宽松规则同时保留；有已知否定引用措辞误判，不事后修改评分提高结果。

历史轮次是给定文本，成本仅覆盖维护和两次追问。延迟不包含实验节流/排队与后台维护，不是线上首字延迟。本轮验证的是整条调用链的开关，不能直接证明只在某个阶段开思考也有同等收益。

## 复现

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_context_thinking.py -q
.\.venv\Scripts\python.exe -u -m scripts.eval_context_thinking --preflight --output data/verification/context-thinking-preflight-20261025-v2
.\.venv\Scripts\python.exe -m scripts.eval_context_thinking --prepare --verified-preflight data/verification/context-thinking-preflight-20261025-v2 --output data/ch07/context-thinking/20261025-gitee-v1
.\.venv\Scripts\python.exe -u -m scripts.eval_context_thinking --output data/ch07/context-thinking/20261025-gitee-v1
.\.venv\Scripts\python.exe -m scripts.eval_context_thinking --audit --output data/ch07/context-thinking/20261025-gitee-v1
```

DeepSeek官方思考模式文档说明模式与推理预算的关系，但Gitee兼容层行为仍以本次实测为准：https://api-docs.deepseek.com/guides/thinking_mode/ 。
