# Data

`data/cache/` 被 Git 忽略，用于存放按照 [训练数据契约](../docs/training.md) 生成的 memmap
cache。项目不把公开数据集副本、teacher embedding 或 checkpoint 提交到 Git。

正式数据先规范化为 source-neutral JSONL，每行必须提供 `group_id`、`query`、`document`、
`variant`、`source`、`source_revision` 和 `source_license`；可选 `id` 与 `split`。数据源下载是
可替换 adapter，这一边界避免训练器依赖具体 Hugging Face dataset script。

```bash
uv run --extra data browser-embedding prepare cache \
  --recipe recipes/cache/multilingual-200k.yaml
uv run --extra data browser-embedding prepare teacher \
  --recipe recipes/teachers/multilingual-minilm-l12-v2.yaml
uv run --extra data browser-embedding prepare validate \
  --cache data/cache/multilingual-200k
```

输出 `manifest.json` 记录数据 revision/license、确定性 split 与 selection hash、tokenizer SHA-256、
各语言切片数量和所有数组 SHA-256；teacher 独立记录固定模型 revision、行选择 hash、归一化审计
与文件 SHA-256。训练 runner 只消费该稳定契约。
