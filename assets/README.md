# Assets

训练与浏览器发布使用同一份 32k NFKC + ByteLevel Unigram tokenizer：

```text
assets/
├── tokenizer.json
└── tokenizer.manifest.json
```

词表必须固定 `[PAD]=0`、`[UNK]=1`、`[CLS]=2`、`[SEP]=3`、`[QRY]=4`、`[DOC]=5`。
训练 cache、Python 多语评估和 Rust/WASM runtime 都读取这一个 artifact。当前文件是用于推进
代码架构的 candidate：词表和运行时兼容性已经验证，但原始语料包含早期评估 seed，因此不能
用于正式质量声明。`tokenizer.manifest.json` 明确记录了这一限制。

正式替换通过下面的无评估污染 recipe 完成，生成器会记录语料 revision/license、文件 hash、
特殊 token、unknown-token 审计和实际字节数：

```bash
uv run --extra data browser-embedding prepare tokenizer \
  --recipe recipes/tokenizer/zh-en-32k.yaml
```

测试用的 13-token tokenizer 位于 `tests/fixtures/tokenizer.json`，只用于验证 WASM 链路，
不能用于正式训练。
