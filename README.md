# browser-embedding

一个从零设计的、浏览器优先的中英多语 embedding 模型项目。目标不是做通用大模型，
而是在浏览器内以很小的下载体积和稳定的 CPU/WASM 延迟完成中文、英文及中英混合检索。

当前里程碑实现了可执行的训练架构与模型导出契约；不附带训练后的正式权重。

## 设计亮点

- 2-bit ternary 线性权重，部署时只保存 `{-1, 0, +1}` 和缩放因子。
- 词嵌入采用 INT4 per-row 量化；因子化 embedding 避免词表主导模型体积。
- 一个 Transformer block 重复执行，像 ALBERT 一样共享参数，以计算换下载体积。
- Matryoshka 表征同时训练 64/128/256/384 维输出，调用方按延迟与索引体积取舍。
- `[QRY]` / `[DOC]` 角色建模和 en→en、zh→zh、zh→en、en→zh、mixed 检索评估。
- Python 训练、`.bem` 模型格式、Rust/WASM 运行时以同一个 deployment contract 为边界。

## 快速验证

使用 `uv`（推荐）：

```bash
uv sync --extra dev
uv run browser-embedding inspect --config configs/smoke.yaml
uv run browser-embedding train --config configs/smoke.yaml
uv run pytest
```

也可使用普通虚拟环境：

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/browser-embedding train --config configs/smoke.yaml
```

smoke 配置只使用确定性的合成中英 pair，不下载数据集或 teacher。它会完整经过
数据加载、前向、Matryoshka 蒸馏/检索损失、反向传播、验证、checkpoint 和 `.bem`
导出，因此适合作为 CI 的训练链路检查。

正式训练从 `configs/multilingual.yaml` 开始。数据准备适配器的输入契约见
[架构文档](docs/architecture.md)，训练阶段不与具体数据源或 teacher 实现耦合。

离线制品通过同一个 CLI 构建和审计：

```bash
uv run --extra data browser-embedding prepare cache \
  --recipe recipes/cache/multilingual-200k.yaml
uv run --extra data browser-embedding prepare teacher \
  --recipe recipes/teachers/multilingual-minilm-l12-v2.yaml
uv run --extra data browser-embedding prepare validate \
  --cache data/cache/multilingual-200k
```

仓库包含一个可贯通 Python 与 WASM 的 32k Unigram tokenizer candidate。它可用于架构开发，
但其来源语料包含早期评估 seed，因此正式质量训练前应通过已提交 recipe 从无评估污染的语料重训。

## 文档

- [系统与代码架构](docs/architecture.md)
- [训练与数据契约](docs/training.md)
- [模型格式和浏览器运行时](docs/runtime.md)
