# P0/P1：单意图业务子流程复用实验

本地分支：feature/multi-intent-subflows。基线为 preserved/current-20261104，提交 2b8ec065171e9020982a646dd02221cac0568c23。

原目录 E:/Mewhelp 没有 Git 元数据。本实验在 data/branches/multi-intent-subflows/source 建立独立 Git 仓库，保存当前应用、测试、脚本、SQL、文档和依赖文件共519份。该提交是当前工作区快照，不继承 GitHub 提交历史。运行凭据仅作本地忽略文件保存，不进入提交。

## 入口与执行

`app.graph.read_subflows.build_pilot(checkpointer)` 是独立实验入口，没有接管现有 build_graph、HTTP API 或网页开关。localhost:8000 继续运行原版本。

调用方提供 user_id、conversation_id、current_message_id、query，以及归属于该用户和会话的 business_context。上下文必须来自后端核验记录，不能直接信任客户端传入对象；这里不是公开HTTP接口。规划复用 task_planner，订单指代复用 order_context.bind_tasks，执行复用原 nodes.main_agent 和 nodes.agent_tools。两者仍使用既有 registry 和 engine。

计划只支持 order/logistics；其他事项明确标记尚未接入，缺失或歧义订单标记 WAITING_USER，不自动猜选。P1不实现用户选单恢复，因此这种等待只返回状态，不弹出可恢复卡片。

调度先订单后物流；每项使用独立 ConversationState，当前消息仅包含任务及已核验订单。原复合消息不会反复入库，兄弟任务的工具消息也不混入。共享订单身份，不共享授权，也不把历史状态当作本轮查询结果。物流子任务自行取得当前订单回执后，再查询回执中的物流号；P1因此可能有一次额外订单读取，尚未优化同轮查询缓存。

工具权限在模型绑定和工具执行前双重限制。订单任务只允许query_order；物流任务允许query_order/query_logistics。不同订单、未取得本任务订单回执的物流号、退款和工单工具均拒绝。原单意图调用不提供task_scope，仍沿用原行为。

完成以实际成功工具回执及目标订单核对，不采信模型口头声称完成。待付款订单返回UNPAID结果及暂无物流进度说明，不因未调用物流工具判为失败。当前演示夹具可能为待付款订单也生成运单号，因此禁止据此冒充已有物流。

## 持久化和恢复边界

父图在每项执行返回后保存任务状态与结果。验证通过执行后暂停、关闭SQLite连接、重建图、从同thread恢复，确认只执行剩余任务。子流程不调用run_turn/log_node，不写对话消息；业务审计仍经过原执行引擎。

P1子图使用checkpointer=False。父图保存的是任务边界，不保证Agent内部每步恢复；中途进程崩溃可以重试只读查询。本阶段不能据此声称已完成P2的interrupt/resume、写操作Exactly-Once或用户消息生命周期集成。

## 验证与复现

在实验source目录，使用原工作区虚拟环境Python：

```
E:/Mewhelp/.venv/Scripts/python.exe -m pytest tests/graph/test_read_subflows.py tests/graph/test_agent_nodes.py tests/graph/test_ch08_confirm_ticket.py tests/graph/test_routing.py tests/graph/test_build.py tests/graph/test_agent_view.py tests/graph/test_context_scope.py tests/core/test_order_context.py tests/core/test_task_planner.py --basetemp=data/verification/reproduce
E:/Mewhelp/.venv/Scripts/python.exe scripts/probe_read_subflows.py data/verification/new-live-run
```

真实探针使用合成问题、Gitee真实模型与本地HTTP MCP。订单和物流服务是项目演示数据，不是真实商业系统。工具审计改存实验文件，SQLite和调用池存实验目录；不创建生产会话、工单或修改知识库。每次复现使用新输出目录。

首轮发现订单身份确认漏项和待付款误判，原始结果保留在data/verification/live-p1。后续测试记录分别保存，不覆盖首轮失败。

## 后续阶段

P2接入真实会话生命周期、准备/等待分离、选单、退款投诉确认和幂等提交；P3统一自然答复与SSE，避免子任务答复重复邀约；P4冻结后比较单意图基线与新版本。原120场景尚未重跑，历史指标不代表本分支。
