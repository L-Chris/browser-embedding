# 架构说明

## 1. 目标与边界

browser-embedding 只解决一个问题：在浏览器内完成中文、英文和中英混合文本的语义向量化。
它不追求通用 NLP 能力，也不把训练框架带进浏览器。架构优先级固定为：

1. 下载体积和冷启动可控；
2. CPU/WASM 是可靠基线，后续 SIMD 优化不改变模型格式；
3. 检索质量覆盖 en→en、zh→zh、zh→en、en→zh 和 code-switch；
4. 训练、量化、导出与运行时的数值规则不能各自演化；
5. 实验必须可恢复、可追溯，部署预算必须在训练前验证。

正式配置当前的静态估算为：

| 指标 | 数值 |
|---|---:|
| 实际可训练参数 | 3,473,088 |
| 逻辑深度参数量（计入 3 次 block） | 4,062,912 |
| BEM2 模型文件 | 1,853,940 bytes |
| 模型 + WASM + tokenizer 预算 | 6,053,940 bytes |
| batch=1 推理工作集上界 | 7,807,656 bytes |

这些是由配置和 BEM2 section 公式直接计算出的契约值，不是营销估算。bundle 采用当前
约 3.0 MB WASM 的 3.2 MB 上界和 tokenizer 的 1 MB 上界；工作集还计入 JS/Rust 双份模型
bytes 与 tokenizer heap。实际导出器会断言模型文件字节数与预算器完全一致。

## 2. 系统分层

```text
configs/*.yaml
    │ 强类型校验 + 浏览器预算
    ▼
┌──────────────────────── Python / training ────────────────────────┐
│ Data adapter → BrowserEncoder → MatryoshkaObjective → Runner      │
│                                   │                               │
│                          versioned checkpoint                     │
│                                   │                               │
│                             BEM2 exporter                         │
└───────────────────────────────────┼───────────────────────────────┘
                                    │ model.bem + tokenizer.json
                                    ▼
┌──────────────────────── Rust / runtime ───────────────────────────┐
│ BEM2 parser → quantized kernels → fixed encoder graph → pooling   │
│                       tokenizers crate                → L2 vector │
└───────────────────────────────────┬───────────────────────────────┘
                                    │ wasm-bindgen
                                    ▼
┌────────────────────── TypeScript / product API ───────────────────┐
│ createEmbedder → embed(text, role, dimension) → similarity       │
└───────────────────────────────────────────────────────────────────┘
```

依赖方向只有向下：CLI 可以认识 runner，runner 可以认识模型和 data port；模型不知道
checkpoint、文件路径、teacher、W&B 或浏览器。Rust core 不认识 JavaScript 和文件系统，
同一套 parser/graph 可在原生测试和 WASM 中运行。

## 3. 模型结构

默认模型如下：

```text
token ids [B, T]
  ├─ token embedding [32000, 96]     部署为 INT4 + fp16 row scale
  └─ learned position [128, 96]      部署为 fp16
          ↓ LayerNorm
          ↓ ternary projection 96 → 192
          ↓
  SharedEncoderBlock × 3（同一组参数重复执行）
    ├─ pre-LN
    ├─ fused QKV ternary linear
    ├─ 3 heads × 64-dim self-attention
    ├─ ternary output linear + residual
    ├─ pre-LN
    └─ ternary 192 → 384 → 192, tanh-GELU + residual
          ↓ final LayerNorm
          ↓ masked mean pooling
          ↓ fp16 output projection 192 → 384
          ↓ L2 normalize
embedding [B, 384]
```

### 因子化 embedding

词表宽度是 96，而不是 Transformer 隐层的 192。对 32k 词表，INT4 token table 约
1.6 MB；若直接使用 256 维表则约 4.2 MB。一次很小的 ternary projection 换来了更低的
下载体积和缓存压力。

### 共享 block

模型保存一个 block，前向重复三次。下载大小接近一层，表达深度接近三层。共享并不一定
适合所有质量目标，因此 `num_repeats` 是实验参数；但 BEM2 只存一份 block，这是部署
契约，不会因为 repeats 增加而扩大文件。

### Matryoshka 输出

训练同时约束 `[64, 128, 256, 384]` 四个归一化前缀。浏览器 API 只允许这些已训练宽度：
搜索建议或移动端索引可用 64/128，质量优先可用 256/384。截断后必须再次 L2 normalize。

### 为什么输出层保留 fp16

输出层是内部表征到 teacher 空间的桥，参数只占约 145 KB。将其 ternary 化的节省有限，
质量风险却集中，因此 v1 contract 明确保留 fp16。此决定是格式字段而不是散落在 packer
里的特殊判断。

## 4. 训练架构

`TernaryLinear` 从模型创建时就存在。量化 warmup 使用 strength=0，QAT 使用 strength=1：

```text
q = clamp(round(w / mean(abs(w))), -1, 1)
w_forward = w + strength * stop_gradient(q * scale - w)
```

这样有三个直接收益：

- warmup、QAT 和 resume 的 `state_dict` 结构相同；
- 不依赖第三方模块替换器，也不会漏量化某个新增 linear；
- exporter 与 runtime 只需识别一种 ternary 规则。

目标函数在每个 Matryoshka 维度上计算并取平均：

- pointwise cosine distillation：对齐 teacher 向量；
- relational distillation：对齐 batch 内相似度几何；
- symmetric multi-positive InfoNCE：学习 query/document 检索，并用 `group_id` 避免假负例。

## 5. 代码职责

| 模块 | 唯一职责 |
|---|---|
| `config.py` | 实验 schema、跨字段校验、部署预算 |
| `model.py` | 与浏览器图一致的纯 PyTorch encoder |
| `quantization.py` | ternary STE、INT4 和 bit packing 的唯一数学来源 |
| `data.py` | pair data port、memmap adapter、synthetic smoke adapter |
| `objectives.py` | 可组合的 Matryoshka 蒸馏/检索损失 |
| `evaluation.py` | pair retrieval 与六切片多语评估 |
| `checkpoint.py` | 独立版本的训练状态与 RNG 恢复 |
| `export.py` | checkpoint → BEM2；不包含训练逻辑 |
| `runner.py` | application orchestration，不定义模型或数据格式 |
| `browser-embedding-core` | BEM2 校验、section layout、reference kernels/graph |
| `browser-embedding-wasm` | tokenizer 与 wasm-bindgen 边界 |
| `browser-embedding` | 资源加载和稳定 TypeScript API |

## 6. 允许变化与必须一起变化的部分

一般训练策略、数据源和 teacher 可以独立变化。以下修改属于 deployment contract 变更，
必须提升 BEM2 version，并同时修改 Python exporter、Rust parser/reference graph 和 parity test：

- 层顺序、矩阵 shape、激活函数或 pooling；
- 量化公式、scale 粒度、bit 编码；
- position/role 的建模方式；
- header 字段或 section 顺序。

这一边界是整个重构的核心：实验代码可以快速变化，而浏览器运行时只跟随显式的格式版本。
