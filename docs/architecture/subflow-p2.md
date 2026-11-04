# P2：业务子流程接入、中断与恢复

2026-11-04。承接P1订单/物流复用实验；保持原单意图与旧多意图节点可用。

## 保存与启用

P1恢复点：preserved/p1-20261104分支及p1-verified-20261104标签，提交5015cc64ae1886942840703ca0afcdf65051109e。当前开发仍在feature/multi-intent-subflows，本地原客服E:/Mewhelp继续运行原版本。

新分支通过进程配置MULTI_INTENT_SUBFLOWS_ENABLED=true，或build_graph(saver, subflows=True)选择新调度；会话已有的multi_intent_enabled继续决定单意图/多意图。新配置默认false，用于隔离试运行，没有改写.env或接管localhost。暂停期间重启应保持相同子流程配置和同一SQLite检查点文件。

## 实际架构

旧安全前置与完整任务规划 → sf_schedule → sf_prepare → 原Agent只读子图/退款准备/投诉准备 → sf_schedule → 独立选单或确认节点 → 回到调度。

1. 安全规则与安全事件复用现有前置节点，在规划调用之前持久化与输出。
2. 规划、原文校验、订单对象绑定复用task_planner和order_context。任务用会话+原始消息+任务键生成稳定ID。
3. sf_prepare将缺参任务保存为WAITING_USER/select_order，继续调度其他PENDING事项，不在准备节点中interrupt。
4. 原P1只读子图提取为execute_read，订单/物流仍复用main_agent、agent_tools及统一执行引擎；业务答案按稳定任务事件保存。
5. 退款复用原fetch_order核验与现有退款准备函数；政策仍经原retrieve_policy真实检索。投诉复用原投诉预览准备。两者均只准备审核工单，分别进入confirm_submission，不自动执行资金退款，也不声称具体订单退款资格已获批准。
6. 待所有可执行事项准备结束，sf_select才interrupt；用户选单后重新校验归属。只有原文解析出的同一reference_group共享选单，独立订单或“另一笔”不借号。选择结果与可归属的用户陈述进入business_context，供新一轮恢复指代，不保存操作授权。
7. sf_wait只返回当前未处理操作的预览。恢复必须带对应operation_id和严格布尔confirmed；请求不允许附加修改工单参数。
8. sf_commit复用已有事务提交，SQL中以稳定operation_id保存参数摘要、工单和回执；同会话行锁串行化竞争，工作流记录主键防止同一操作重复建单。
9. 写入后响应丢失等暂时异常会保留同一操作与预览，重新生成等待确认卡。重试返回已保存的原工单；如果原提交实际成功，之后点击取消也不会被描述为撤销成功工单。后续事项可继续处理。

任务状态为PENDING/RUNNING/WAITING_USER/DONE/FAILED/CANCELLED。查询DONE以本轮工具回执为据；退款/投诉DONE以本地审核工单回执为据；待付款订单无需退款的解释是独立outcome。任务状态不直接生成用户清单。

## 会话与安全边界

复用现有run_turn/stream_turn、resume、SSE和actions路由。用户消息每轮落库一次，内部子流程消息不追加为客户对话；可见结果与确认回执由workflow.event稳定保存。

原validate_resume继续检查当前用户、会话、最新消息水位、interrupt_id和操作类型；旧卡片或被新消息取代的确认被拒绝。每次提交复验退款订单归属和写入约束。一个事项取消/政策失败后仍调度兄弟事项；已DONE不重新执行。

SQLite保存任务阶段与待确认决策，MySQL保存对话、计划历史和事务回执。它们不是一个跨库原子事务；幂等回执用于处理SQL提交领先于图检查点的窗口。P2不宣称外部工具Exactly-Once。

## 验证方式

新增tests/graph/test_subflow_tasks.py：真实测试MySQL、真实SQLite关闭重开、模型与只读工具桩，以及ASGI HTTP/SSE。覆盖共享选单、跨轮追问、两次重建、独立确认、取消、政策故障、非法恢复、旧卡失效、新消息取代、并发重试、SQL提交成功后的响应丢失。

scripts/probe_subflow_tasks.py：真实Gitee DeepSeek-V4-Flash、BGE-M3、BGE-Reranker、现有Milvus知识检索和本地HTTP MCP；自动建立新合成MySQL库和SQLite文件，通过实际API路由验证。MCP仍为项目演示业务数据；未启动浏览器或网络监听服务。

从实验source目录运行：

```
E:/Mewhelp/.venv/Scripts/python.exe -m pytest tests/graph/test_subflow_tasks.py --basetemp=data/verification/p2-reproduce
E:/Mewhelp/.venv/Scripts/python.exe scripts/probe_subflow_tasks.py
```

真实探针读取已有knowledge集合，先确认存在才检索；只在独立合成SQL库写入工单，运行后保留证据与测试库。它不会建立/更新知识内容或将真实用户会话发送到接口。

## 留待后续

P3：统一自然答复、避免多子任务重复邀约、完善工具与检索活动展示。P4：固定120场景新旧版本对比。当前专项结果不代表120场景完成率。

P1只读Agent内部中途崩溃仍可能重试读取；本阶段主要验证任务边界、用户中断和本地工单幂等。持续数据库故障、操作系统断电/SQLite损坏、真实外部工单系统及跨部署切换迁移不在本次证明范围。
