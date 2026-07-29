# browser-embedding vs Ternlight benchmark

This benchmark compares the two shipped WASM runtimes on the same host and the
same 384-dimensional bilingual input corpus. It reports cold initialization,
steady-state latency, throughput, process memory, and complete download payload
size (raw, gzip, and brotli).

The engines run in separate fresh Node.js processes. `browser-embedding` payload
size includes its JS glue, WASM, `.bem` model, and tokenizer. Ternlight payload
size includes its JS glue and WASM because its model and tokenizer are embedded
in that WASM; counting its source assets again would double-count them.

## Build the benchmark artifacts

The wasm-bindgen CLI must match the crate version in each lockfile. At the time
of writing, browser-embedding uses `0.2.126` and the sibling Ternlight checkout
uses `0.2.122`.

```bash
# browser-embedding (from this repository)
cargo build --release --target wasm32-unknown-unknown -p browser-embedding-wasm
wasm-bindgen \
  target/wasm32-unknown-unknown/release/browser_embedding_wasm.wasm \
  --target nodejs \
  --out-dir target/benchmark-node \
  --out-name browser_embedding_wasm

# Ternlight (from ../ternlight)
PKG=packages/base TARGET=nodejs bash scripts/build-engine.sh
```

If the appropriate wasm-bindgen executables are not on `PATH`, invoke each one
by its full path. The default model path is
`artifacts/runs/ternlight-100k/model.bem`, which is gitignored and therefore may
need to be overridden on another machine.

## Run

```bash
pnpm benchmark:compare > eval/benchmarks/results/local.json
```

The JSON report goes to stdout; a compact comparison table goes to stderr. The
results directory is ignored because measurements are machine-specific.

Override any artifact path or increase repetitions when needed:

```bash
node eval/benchmarks/compare.mjs \
  --browser-model artifacts/runs/my-run/model.bem \
  --rounds 100
```

Use `node eval/benchmarks/compare.mjs --help` for all path overrides.

## Interpretation

- Compare ratios only within the same report. Results quoted from different
  machines, Node versions, build profiles, or corpora are not interchangeable.
- Payload means all runtime resources transferred to a browser, not checkpoint
  size. gzip and brotli compress each resource separately, matching HTTP usage.
- Node.js is a reproducible WASM baseline, not a substitute for Chrome, Safari,
  and Firefox measurements. Browser claims require an additional browser run.
- This benchmark does not measure retrieval quality. Performance/size results
  should be read alongside multilingual retrieval and similarity evaluation.
