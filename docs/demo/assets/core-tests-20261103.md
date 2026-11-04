# 核心回归记录 · 2026-11-03

结果：**92 passed，0 failed，0 errors**，pytest 用时 **8.82 秒**。机器可读附件：[JUnit XML](core-tests-20261103.xml)。

## 版本与环境

- 应用与测试源码：[5c1840b](https://github.com/Orange-ui0811/caremate/commit/5c1840b)。本次修改仅涉及文档与 CI，未修改应用和测试实现。
- 本地 Windows，Python 3.13.3；pytest 9.1.1、LangGraph 1.2.9、langchain-openai 1.3.5。
- CHAT_MODEL 使用 offline-test；模型地址设为本机不可用端口，密钥使用占位值。模型、MCP 工具发现及仓储写入由测试替身隔离，不使用真实模型密钥。
- pytest 临时目录使用本工作区新建目录，运行配置通过 fixture 隔离；未启动数据库集成夹具。

## 入选测试

```bash
python -m pytest \
  tests/graph/test_routing.py \
  tests/graph/test_ch08_confirm_ticket.py \
  tests/tools/test_engine.py \
  tests/tools/test_registry.py \
  tests/tools/test_mcp_client.py \
  tests/tools/test_order_ownership.py \
  tests/core/test_confidence.py \
  tests/core/test_runtime_settings.py \
  -q --junitxml=core-tests.xml
```

占位环境变量与依赖安装见[CI配置](../../../.github/workflows/ci.yml)。Windows 本机遇到系统临时目录权限冲突时，可通过 `--basetemp=<新的可写目录>` 指定独立临时目录。

## 工单取消验证

`test_confirm_false_denied_and_audited` 使用测试模型发起 create_ticket，先暂停生成预览，再以 confirmed=false 恢复。测试断言创建函数调用列表为空，审计出现权限拒绝，取消信息进入返回消息。

该证据证明隔离环境中的执行控制行为。真实模型工具选择、浏览器交互、SQLite 持久恢复与 MySQL 无新增记录需要按[本地指南](../runbook.md)另外验收。GitHub Actions 的 Linux 执行结果以该提交对应的工作流运行记录为准。
