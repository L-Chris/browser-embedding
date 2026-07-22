# 训练与数据契约

## 最小链路

```bash
uv sync --extra dev
uv run browser-embedding inspect --config configs/smoke.yaml
uv run browser-embedding train --config configs/smoke.yaml --max-batches 1
uv run browser-embedding inspect-model --model artifacts/runs/smoke/model.bem
```

smoke data 是确定性、可学习的 query/document pair。它不会证明模型质量，只证明下面这条
链路没有断裂：

```text
config → data → forward → loss → backward → optimizer/scheduler
       → validation → metrics → best/last checkpoint → BEM2 export
```

正式训练使用：

```bash
uv sync --extra data --extra dev
uv run browser-embedding prepare cache --recipe recipes/cache/multilingual-200k.yaml
uv run browser-embedding prepare teacher \
  --recipe recipes/teachers/multilingual-minilm-l12-v2.yaml
uv run browser-embedding prepare validate --cache data/cache/multilingual-200k
uv run browser-embedding train --config configs/multilingual.yaml
```

准备层和训练层是两个进程边界。准备层可以下载数据和 teacher；训练 runner 不联网，只读取已经
通过校验的不可变 cache。

仓库中的小型例子可以独立验证 tokenizer → memmap → manifest：

```bash
uv run --extra data browser-embedding prepare cache --recipe recipes/cache/example.yaml
uv run --extra data browser-embedding prepare validate --cache data/cache/example
```

## Canonical pair v1

数据源 adapter 输出 UTF-8 JSONL，每行结构如下：

```json
{
  "id": "optional-stable-id",
  "group_id": "same-concept-or-document-group",
  "query": "如何修复 TypeScript 类型错误？",
  "document": "Fixing TypeScript type errors",
  "variant": "mixed_en",
  "source": "example-source",
  "source_revision": "immutable-revision",
  "source_license": "SPDX-or-upstream-license",
  "split": null
}
```

省略 `split` 时按 `seed + group_id` 确定性生成 90/5/5 split；同一 group 永远不会跨集合。
准备器进行 NFKC、空白规范化、exact dedup 和按语言切片的确定性抽样。正式数据源可变化，但
canonical schema 不随数据平台变化。

## Memmap cache v1

正式数据 adapter 接受一个自描述目录：

```text
data/cache/multilingual-200k/
├── manifest.json
├── texts.jsonl             teacher adapter 的行对齐文本，不由训练器读取
├── input_ids.npy           uint32 [N, 2, max_sequence_length]
├── attention_mask.npy      uint8  [N, 2, max_sequence_length]
├── split_codes.npy         uint8  [N], train=0/validation=1/test=2
├── metadata.jsonl          id/group/source/license/split，不含训练张量
└── teachers/
    ├── multilingual-minilm-l12-v2.npy
    ├── multilingual-minilm-l12-v2.manifest.json
    └── multilingual-minilm-l12-v2.progress.json
```

第二维固定为 `[query, document]`。token cache 必须已经包含 `[QRY]` / `[DOC]` 前缀；
teacher 向量固定为 L2-normalized float16。`group_id` 相同的文档会被 InfoNCE 当作多个正例，而不是
batch 内负例。`variant` 必须来自：

- `en_en`
- `zh_zh`
- `zh_en`
- `en_zh`
- `mixed_en`
- `mixed_zh`

数据准备和 teacher 推理是 adapter，不进入训练 runner。这样更换数据集、teacher 或远程
缓存实现时，不需要修改模型与训练循环。

Teacher job 必须在 recipe 中固定 40 位模型 revision；不能运行时追踪浮动 `main`。中断时只在
selection hash、模型 revision、shape 和 device 全部一致时续跑。当前默认 teacher 来自旧 pilot
的对照结论，但新模型只在 384 维执行 pointwise/relational 蒸馏；64/128/256/384 都执行 student
retrieval loss，避免把非 Matryoshka teacher 的低维前缀误当成监督目标。

## 运行产物

每个 run 目录包含：

```text
manifest.json              完整 config、环境、预算和数据 provenance
metrics.jsonl              每 epoch 的 loss、retrieval、QAT health
checkpoint-001.pt          按 save_every 保存
last.pt                    最近可恢复状态
best.pt                    best_metric 最优状态
model.bem                  best checkpoint 的浏览器权重
model.bem.json             section offsets、shape 和 SHA-256
```

checkpoint 是训练内部格式，包含 optimizer、scheduler 和 Python/NumPy/PyTorch/DataLoader
RNG；BEM2 是部署格式，不包含任何训练状态。二者有独立 schema version。

## 正式质量门槛

训练链路跑通不等于模型可发布。正式权重至少应满足：

1. 六个语言切片分别报告 Recall@1/3/10、MRR@10、NDCG@10；
2. 不能只看 macro，任一跨语言切片显著退化都应阻止发布；
3. fp32 warmup、ternary QAT、INT4 导出和 Rust forward 逐级记录质量差；
4. 384 维与各 Matryoshka 维度分别评估；
5. 浏览器真机记录冷启动、p50/p95、峰值内存和包体，而非用 Python 延迟代替。
