# Retrieval-quality comparison

This harness compares `browser-embedding` and Ternlight by ranking the same
documents for the same queries. It never compares raw cosine values between
models because each model defines a different embedding space.

The default development dataset is Ternlight's 32-intent bilingual
four-quadrant set. It produces 192 paired queries across `en_en`, `zh_zh`,
`zh_en`, `en_zh`, `mixed_en`, and `mixed_zh`. Both engines use their role-aware
query/document entry points at 384 dimensions.

## Run

Build both Node.js WASM artifacts as described in
[`../benchmarks/README.md`](../benchmarks/README.md), then run:

```bash
pnpm eval:retrieval:compare > eval/retrieval/results/local.json
```

JSON goes to stdout and a compact table goes to stderr. Use `--dataset PATH`
or the other options shown by `node eval/retrieval/compare.mjs --help` to
override artifacts.

The report contains:

- Recall@1/3/10, MRR@10, nDCG@10, mean rank, and positive margin per slice;
- macro averages with browser-embedding minus Ternlight deltas;
- paired, language-stratified bootstrap confidence intervals for macro
  nDCG@10;
- per-query rank wins/ties/losses and the ten largest wins and losses;
- dataset, commit, model, tokenizer, and WASM fingerprints.

## Interpretation

This dataset validates the comparison machinery and is suitable for fast
regression checks. It is not sufficient for a release-quality claim: it is
small, has only 32 documents per slice, and has been visible to historical
evaluation/model-selection workflows.

A formal evaluation should replace it with a frozen dataset containing at
least 1,000 held-out queries, 10,000 or more documents, hard negatives, exact
and near-duplicate checks against training data, and relevance judgments that
were never used for checkpoint selection. The same harness can then consume
the frozen four-quadrant JSONL without changing model-side code.
