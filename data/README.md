# Data

`data/cache/` 被 Git 忽略，用于存放按照 [训练数据契约](../docs/training.md) 生成的 memmap
cache。项目不把公开数据集副本、teacher embedding 或 checkpoint 提交到 Git。

建议把数据准备实现为独立命令/作业，并让输出 `manifest.json` 至少记录：数据源及 revision、
license、过滤和去重规则、split seed、tokenizer SHA-256、teacher id/revision、各语言切片数量、
数组 SHA-256。训练 runner 只消费该稳定契约。
