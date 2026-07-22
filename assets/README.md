# Assets

正式训练与浏览器发布需要把同一份 32k WordPiece tokenizer 放在这里：

```text
assets/
└── tokenizer.json
```

词表必须固定 `[PAD]=0`、`[UNK]=1`、`[CLS]=2`、`[SEP]=3`、`[QRY]=4`、`[DOC]=5`。
训练 cache、Python 多语评估和 Rust/WASM runtime 都读取这一个 artifact。tokenizer 进入版本
控制前应记录训练语料版本、normalizer/pre-tokenizer 配置、词表 hash 和中英 mixed 文本的
unknown-token 审计。

测试用的 13-token tokenizer 位于 `tests/fixtures/tokenizer.json`，只用于验证 WASM 链路，
不能用于正式训练。
