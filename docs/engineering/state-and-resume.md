# 状态、中断与恢复

[ConversationState](../../app/graph/state.py)携带消息、意图、订单、证据和执行计数。[runtime.py](../../app/graph/runtime.py)使用 SQLite AsyncSqliteSaver 保存图检查点，thread_id 标识会话运行；恢复使用 Command(resume=...)。

## 两类交互

| 场景 | 暂停前 | 恢复值 | 恢复后 |
| --- | --- | --- | --- |
| 退货缺少订单 | fetch_order 获取当前用户可选订单，interrupt 展示列表 | 用户选择的订单号 | 读取所选订单，再检索政策 |
| 创建人工工单 | agent_tools 预检参数，interrupt 展示工单草稿 | confirmed 布尔值组成的字典 | 确认后才写入；取消则返回拒绝结果 |

节点恢复时可能重新执行中断前代码，所以创建工单的副作用安排在确认之后。后端校验订单归属；身份不由模型参数决定。退款申请入口另行由前端表单提交。

## 回归证据

[test_ch08_confirm_ticket.py](../../tests/graph/test_ch08_confirm_ticket.py)覆盖确认前未执行、确认后一次创建、取消不调用创建函数、缺参数不弹确认卡、同轮多个工单仅执行首个、错误恢复形状按取消处理。测试使用 InMemorySaver 和被替换的模型/仓储，因此证明图控制行为，不代表真实 MySQL 事务验收。

选单节点见[test_ch06_nodes.py](../../tests/graph/test_ch06_nodes.py)。真实 SQLite 检查点、浏览器恢复与数据库写入按[本地指南](../demo/runbook.md)单独验证。当前未提供重复 HTTP 提交、崩溃重放的全链路幂等性证明。
