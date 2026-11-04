# 知识反馈与主题分类

反馈流程把未解决问题转为可审核的知识缺口；主题分类辅助分析积累的问题类型。在线九类意图分流与旁路十七类主题分析分别承担实时处理和知识维护。

## 问题归集与审核

低置信度、生成前自检不足和用户未解决反馈进入问题池，并尽量保留召回快照。相关入口为 [fallback_reply](../../app/graph/nodes.py#L130) 与 [feedback.py](../../app/api/feedback.py)。

[flywheel.py](../../app/core/flywheel.py) 对问题标准化、查重归并，形成或更新待审项。出现次数帮助定位反复出现的知识缺口；快照辅助区分知识缺失与检索不准确。

模型生成的内容是草稿。[review.py](../../app/api/review.py) 接收人工核准后的答案，再通过 [dualwrite.py](../../app/kb/dualwrite.py) 写入知识存储。知识发布经过人工核准。

MySQL 与 Milvus 分别写入，状态标记、重试和补偿处理部分失败；并发审核及补偿失败属于一致性检查项。

## 九类意图与十七类主题

| 任务 | 用途 | 执行位置 |
| --- | --- | --- |
| 九类意图识别 | 决定当前请求走哪条业务路线 | 在线工作流 |
| 十七类多标签主题分类 | 分析已归并问题的主题分布与知识缺口 | 旁路批处理及独立推理服务 |

统一主题表在 [taxonomy.py](../../app/core/taxonomy.py)。一句“尺码不合适想退货”可以涉及多个主题，因此分类器使用多标签输出；主题计数相加可能大于问题数量。

## 数据与训练

[build_corpus.py](../../scripts/ch10/build_corpus.py) 准备语料，[corpus_lib.py](../../scripts/ch10/corpus_lib.py) 提供清洗与划分，[build_dataset.py](../../scripts/ch10/build_dataset.py) 组织训练集增强。

训练采用多标签配置、multi-hot 标签和独立 sigmoid 输出，见 [train.py](../../scripts/ch10/train.py)。阈值在验证集上扫描，再用于测试评估。分类阈值与 RAG 证据阈值属于不同任务。

当前推理的空标签兜底位于 [inference_lib.py](../../scripts/ch10/inference_lib.py)：所有标签未过线时选择最高分标签。该行为是空标签回退策略，独立于标签是否超过阈值。

仓库提供语料处理、训练与评估脚本；问题池数据、模型权重和训练产物由运行方管理。数据划分检查覆盖近重复样本与来源交叉。

## 推理与评估

[export_onnx.py](../../scripts/ch10/export_onnx.py) 导出模型并检查转换一致性，[serve.py](../../scripts/ch10/serve.py) 提供独立推理服务，[classify_pool.py](../../scripts/ch10/classify_pool.py) 对符合条件的归并问题执行旁路分类。

[evaluate.py](../../scripts/ch10/evaluate.py) 统计 precision、recall、micro/macro-F1 及各类表现。micro-F1 汇总标签层面的 TP、FP、FN，报告同时提供各主题指标和错例。

ONNX 导出报告记录测试输入、输出偏差和容差；推理耗时与吞吐通过单独性能测试衡量。

## 反馈效果验证

反馈效果采用独立的补库前后对照：冻结问题集、代码与模型配置，保存两版知识，比较同题的解决、误答和回归结果。主题分类 F1 用于评价标签预测。

本地页面包括 `/review`、`/topics` 和 `/acceptance`。处理新问题、审核写回、重新训练等操作会调用服务或修改数据，调试时宜使用独立样本与可恢复的数据环境。
