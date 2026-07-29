# BEM2 与浏览器运行时

## BEM2 文件

BEM2 是面向固定 encoder 图的紧凑格式。它不是通用神经网络 IR；省去通用 runtime 的
图解析与算子调度，换取更小包体和可预测执行。

```text
0 .. 63                 64-byte little-endian header
64 .. 64+body_length    ordered weight sections
last 32 bytes           SHA-256(header + body)
```

header 包含 vocab、sequence、embedding/hidden/FFN/output 宽度、head 数、共享 block
重复次数、embedding 编码、padding id 与最多 8 个 Matryoshka 维度。body section 顺序：

1. token embedding（INT4/INT8/fp16）
2. position embedding（fp16）
3. embedding LayerNorm（fp16）
4. embedding projection（2-bit ternary）
5. attention LayerNorm
6. fused QKV（2-bit ternary）
7. attention output（2-bit ternary）
8. FFN LayerNorm
9. FFN up（2-bit ternary）
10. FFN down（2-bit ternary）
11. final LayerNorm
12. output projection + bias（fp16）

INT4 每 byte 保存两个 two's-complement nibble，范围 `[-7, 7]`，每个 token row 带一个
fp16 scale。ternary 每 byte 保存四个 2-bit code：`00=0, 01=+1, 10=-1`，每个矩阵带
一个 fp32 AbsMean scale。

## Runtime 结构

`browser-embedding-core` 已提供无文件系统依赖的 BEM2 parser、校验、量化 kernel 和完整 reference
forward；`browser-embedding-wasm` 只负责 Hugging Face tokenizer 与 JS ABI。文本在 Rust 内加
`[QRY]` / `[DOC]` 前缀，避免 JS 和训练端各自维护 tokenization 规则。

WASM 构造模型时会用 BEM2 header 的 `max_sequence_length` 覆盖 tokenizer truncation 参数，
因此超长文本在 post-processor 之前截断，`[CLS]`、角色 token 和尾部 `[SEP]` 都会保留；不能在
tokenization 完成后简单切掉 token 数组尾部。

TypeScript 层通过依赖注入加载 wasm-bindgen 产物，适配 bundler、CDN 或 Service Worker
缓存，而不把具体加载方式写死在模型 API 中：

```ts
const model = await createEmbedder({
  model: new URL("./model.bem", import.meta.url),
  tokenizer: new URL("./tokenizer.json", import.meta.url),
  loadBindings: () => import("./generated/browser_embedding_wasm.js"),
  defaultDimension: 128,
});

const query = model.embed("TypeScript 类型错误怎么修复？", "query");
const document = model.embed("Fixing TypeScript type errors", "document");
console.log(model.similarity(query, document));
```

构建 npm 包时会从根目录单向复制 tokenizer 与 manifest 到 `dist/`，根目录 artifact 始终是
唯一来源。正式发布包会同时携带 WASM、BEM2 和 tokenizer，业务方不需要安装 Python、
ONNX Runtime 或原生依赖。资源仍拆成独立文件，便于浏览器缓存和模型热更新；“自包含”指
npm 包闭包，不是强制把数 MB 权重复制进 WASM code section。

生成 wasm-bindgen 绑定：

```bash
cargo install wasm-bindgen-cli --version 0.2.126 --locked
pnpm build:wasm
```

## 性能演进顺序

Rust graph 同时是 Python/Rust parity 的基准和实际 WASM 热路径。已完成的优化不改变
BEM2 下载格式：

1. `BrowserModel` 构造时只校验和解析一次 BEM2，不在每次 `embed()` 重算 SHA-256；
2. 初始化时将共享 2-bit ternary matrices 解包为只读 fp32 `{-1, 0, +1}` runtime weights，
   并将 LayerNorm、position embedding 与输出 projection 的 fp16 参数预解码；
3. WASM SIMD128 使用四路 fp32 accumulator 执行 ternary matrix dot product；
4. 短文本始终按实际 token length 执行，不补齐到 128。

下一阶段是在不改变公开同步 API 的前提下复用 attention/FFN scratch buffer。只有统一
benchmark 证明有稳定收益后才考虑 WebGPU backend。

任何优化必须通过 Python quantized forward ↔ Rust native ↔ WASM 三方 parity；只有浏览器
真机 benchmark 能决定是否保留优化。
