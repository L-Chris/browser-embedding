# Frozen retrieval evaluation

`retrieval-frozen-v1` is the release-quality complement to the small
development set in `eval/retrieval`. It evaluates the complete candidate
corpus for five test slices:

- MLQA Retrieval `eng-eng`, `zho-zho`, `eng-zho`, and `zho-eng` test splits;
- BEIR SciFact test queries and qrels against the complete SciFact corpus.

Pinned upstream views:

- [MLQA Retrieval at `cf59ddd8`](https://huggingface.co/datasets/mteb/MLQARetrieval/tree/cf59ddd8f4aaf39ce1869361e09252698c340945)
- [SciFact corpus and queries at `b3b53356`](https://huggingface.co/datasets/BeIR/scifact/tree/b3b5335604bf5ee3c4447671af975ea25143d4f5)
- [SciFact qrels at `2938d17d`](https://huggingface.co/datasets/BeIR/scifact-qrels/tree/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029)

The suite manifest pins immutable Hugging Face dataset commits. The committed
lock additionally pins SHA-256 digests for every downloaded source and every
normalized JSONL artifact. A normal preparation run fails if any byte, row
count, or normalization result differs.

## Test-set policy

Use this suite only for aggregate release comparisons. Do not use its test
queries for checkpoint selection, prompt or prefix tuning, threshold tuning,
or per-query error analysis. The scorer deliberately does not serialize query
texts, rankings, or per-query metrics. Any change to a source revision,
normalization rule, qrel, or metric definition requires a new suite id rather
than an in-place lock update.

## Prepare and verify

```bash
pnpm eval:frozen:prepare
pnpm test:frozen
```

The first command downloads only the files named in
`retrieval-frozen-v1.json`, normalizes them under the ignored
`eval/frozen/cache/` directory, and verifies them against
`retrieval-frozen-v1.lock.json`.

`--update-lock` exists solely to initialize a reviewed new suite revision. It
must not be used to make an unexpected verification failure disappear.

## Encode engines

Build the browser Node WASM target and the fork Ternlight Node package first,
then encode the frozen data:

```bash
pnpm eval:frozen:embed:browser
pnpm eval:frozen:embed:fork
```

Encode the installed official upstream packages with explicit identities:

```bash
node eval/frozen/embed.mjs --engine ternlight --engine-id ternlight-official-base \
  --ternlight-module ../ternlight-upstream/target/npm-official/node_modules/@ternlight/base/pkg-node/tern_engine.js

node eval/frozen/embed.mjs --engine ternlight --engine-id ternlight-official-mini \
  --ternlight-module ../ternlight-upstream/target/npm-official/node_modules/@ternlight/mini/pkg-node/tern_engine.js
```

Each output manifest fingerprints the Git commit, JS/WASM/model/tokenizer
artifacts, role API behavior, and every little-endian Float32 embedding matrix.
The official 0.1.0 packages expose a symmetric `embed` API; the report records
that fallback rather than claiming role-aware behavior.

## Score comparisons

```bash
uv run python eval/frozen/score.py \
  --left eval/frozen/cache/retrieval-frozen-v1/embeddings/browser-embedding/manifest.json \
  --right eval/frozen/cache/retrieval-frozen-v1/embeddings/ternlight-fork/manifest.json \
  --output eval/frozen/results/browser-vs-fork.json
```

Repeat with the official Base and Mini manifests. Scoring uses exact dot
products against every corpus document, exact Top-100 with stable tie breaking, multi-positive
and graded qrels, and paired dataset-stratified bootstrap confidence intervals.
The primary statistic is macro nDCG@10; MAP@100, MRR@10, and Recall@1/3/10/100
are secondary metrics. Raw similarity values are never compared across model
spaces.
