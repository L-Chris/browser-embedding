import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { arch, cpus, platform } from "node:os";
import { dirname, relative, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

import {
  METRIC_NAMES,
  aggregateResults,
  bootstrapMacroNdcgDelta,
  compareRanks,
  macroAverage,
  rankResult,
} from "./metrics.mjs";

const EVAL_DIR = dirname(fileURLToPath(import.meta.url));
const BROWSER_ROOT = resolve(EVAL_DIR, "../..");
const WORKSPACE_ROOT = resolve(BROWSER_ROOT, "..");
const DEFAULTS = {
  dataset: resolve(
    WORKSPACE_ROOT,
    "ternlight/eval/multilingual/data/four_quadrant_v1.jsonl",
  ),
  browserModule: resolve(BROWSER_ROOT, "target/benchmark-node/browser_embedding_wasm.js"),
  browserModel: resolve(BROWSER_ROOT, "artifacts/runs/ternlight-100k/model.bem"),
  browserTokenizer: resolve(BROWSER_ROOT, "assets/tokenizer.json"),
  ternlightModule: resolve(WORKSPACE_ROOT, "ternlight/packages/base/pkg-node/tern_engine.js"),
  bootstrapIterations: 10_000,
};

const SLICE_DEFINITIONS = [
  ["en_en", "query_en", "doc_en"],
  ["zh_zh", "query_zh", "doc_zh"],
  ["zh_en", "query_zh", "doc_en"],
  ["en_zh", "query_en", "doc_zh"],
  ["mixed_en", "query_mixed", "doc_en"],
  ["mixed_zh", "query_mixed", "doc_zh"],
];

function parseArgs(argv) {
  const options = { ...DEFAULTS };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index + 1];
    switch (argv[index]) {
      case "--dataset": options.dataset = resolve(value); index += 1; break;
      case "--browser-module": options.browserModule = resolve(value); index += 1; break;
      case "--browser-model": options.browserModel = resolve(value); index += 1; break;
      case "--browser-tokenizer": options.browserTokenizer = resolve(value); index += 1; break;
      case "--ternlight-module": options.ternlightModule = resolve(value); index += 1; break;
      case "--bootstrap-iterations":
        options.bootstrapIterations = Number.parseInt(value, 10);
        index += 1;
        break;
      case "--help": options.help = true; break;
      default: throw new Error(`unknown argument: ${argv[index]}`);
    }
  }
  if (!Number.isInteger(options.bootstrapIterations) || options.bootstrapIterations < 1) {
    throw new Error("--bootstrap-iterations must be a positive integer");
  }
  return options;
}

function usage() {
  return `Usage: node eval/retrieval/compare.mjs [options]

Compares browser-embedding and Ternlight retrieval ranking on the same bilingual
query/document pools. JSON goes to stdout and a compact summary to stderr.

Options:
  --dataset PATH                 four-quadrant JSONL dataset
  --browser-module PATH          browser-embedding wasm-bindgen Node loader
  --browser-model PATH           BEM2 model
  --browser-tokenizer PATH       browser-embedding tokenizer
  --ternlight-module PATH        Ternlight wasm-bindgen Node loader
  --bootstrap-iterations N       paired bootstrap samples (default: 10000)
  --help                         show this message
`;
}

function requireFiles(paths) {
  const missing = paths.filter((path) => !existsSync(path));
  if (missing.length > 0) {
    throw new Error(`missing evaluation artifact(s):\n${missing.map((path) => `  ${path}`).join("\n")}`);
  }
}

function loadDataset(path) {
  const required = new Set([
    "id", "domain", "query_en", "query_zh", "query_mixed", "doc_en", "doc_zh",
  ]);
  const lines = readFileSync(path, "utf8").split(/\r?\n/u).filter((line) => line.trim());
  const records = lines.map((line, index) => {
    const record = JSON.parse(line);
    const missing = [...required].filter(
      (field) => typeof record[field] !== "string" || !record[field].trim(),
    );
    if (missing.length > 0) {
      throw new Error(`${path}:${index + 1} missing non-empty fields: ${missing.join(", ")}`);
    }
    return record;
  });
  if (records.length < 20) throw new Error("quality comparison requires at least 20 intents");
  if (new Set(records.map((record) => record.id)).size !== records.length) {
    throw new Error("dataset contains duplicate intent ids");
  }
  return { records, canonicalBytes: Buffer.from(lines.join("\n") + "\n", "utf8") };
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function gitCommit(path) {
  try {
    return execFileSync("git", ["rev-parse", "--short", "HEAD"], {
      cwd: path,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return "unknown";
  }
}

function dot(left, right) {
  if (left.length !== right.length) throw new Error("embedding dimensions differ");
  let score = 0;
  for (let index = 0; index < left.length; index += 1) score += left[index] * right[index];
  return score;
}

function cachedEmbedder(embed) {
  const cache = new Map();
  return (text, role) => {
    const key = `${role}\u0000${text}`;
    if (!cache.has(key)) {
      const vector = Float32Array.from(embed(text, role));
      if (vector.length !== 384) throw new Error(`expected 384 dimensions, got ${vector.length}`);
      cache.set(key, vector);
    }
    return cache.get(key);
  };
}

function loadEngines(options) {
  const require = createRequire(import.meta.url);
  const browserBindings = require(options.browserModule);
  const browserModel = new browserBindings.BrowserModel(
    readFileSync(options.browserModel),
    readFileSync(options.browserTokenizer),
  );
  const ternlight = require(options.ternlightModule);
  const ternlightQuery = ternlight.embed_query ?? ternlight.embedQuery ?? ternlight.embed;
  const ternlightDocument =
    ternlight.embed_document ?? ternlight.embedDocument ?? ternlight.embed;
  if (typeof ternlightQuery !== "function" || typeof ternlightDocument !== "function") {
    throw new Error("Ternlight module does not expose role-aware embedding functions");
  }
  return {
    browser_embedding: {
      build: browserModel.model_info(),
      embed: cachedEmbedder((text, role) => browserModel.embed(text, role, 384)),
    },
    ternlight: {
      build: ternlight.config_summary?.() ?? ternlight.engineInfo?.() ?? "unknown",
      embed: cachedEmbedder((text, role) =>
        role === "query" ? ternlightQuery(text) : ternlightDocument(text)),
    },
  };
}

function buildSlices(records) {
  return Object.fromEntries(
    SLICE_DEFINITIONS.map(([name, queryField, documentField]) => [
      name,
      {
        name,
        queries: records.map((record) => ({
          query_id: record.id,
          domain: record.domain,
          text: record[queryField],
        })),
        documents: records.map((record) => ({
          document_id: `${record.id}:${documentField === "doc_en" ? "en" : "zh"}`,
          text: record[documentField],
        })),
        relevantIndices: records.map((_record, index) => index),
      },
    ]),
  );
}

function evaluateEngine(engine, slices) {
  const started = performance.now();
  const evaluated = {};
  for (const [name, slice] of Object.entries(slices)) {
    const documents = slice.documents.map((document) => engine.embed(document.text, "document"));
    const queries = slice.queries.map((query, queryIndex) => {
      const queryVector = engine.embed(query.text, "query");
      const scores = documents.map((document) => dot(queryVector, document));
      const rank = rankResult(scores, slice.relevantIndices[queryIndex]);
      return {
        query_id: query.query_id,
        domain: query.domain,
        query: query.text,
        relevant_document_id: slice.documents[slice.relevantIndices[queryIndex]].document_id,
        rank: rank.rank,
        recall_at_1: rank.recall_at_1,
        recall_at_3: rank.recall_at_3,
        recall_at_10: rank.recall_at_10,
        mrr_at_10: rank.mrr_at_10,
        ndcg_at_10: rank.ndcg_at_10,
        positive_score: rank.positive_score,
        best_negative_score: rank.best_negative_score,
        positive_margin: rank.positive_margin,
        top_documents: rank.top_indices.map((index) => ({
          document_id: slice.documents[index].document_id,
          score: scores[index],
        })),
      };
    });
    evaluated[name] = {
      metrics: aggregateResults(queries, documents.length),
      queries,
    };
  }
  return {
    build: engine.build,
    dimension: 384,
    evaluation_ms: performance.now() - started,
    macro: macroAverage(evaluated),
    slices: evaluated,
  };
}

function metricDeltas(browser, ternlight) {
  return Object.fromEntries(
    METRIC_NAMES.map((name) => [
      `macro_${name}`,
      browser.macro[`macro_${name}`] - ternlight.macro[`macro_${name}`],
    ]),
  );
}

function percent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function renderSummary(report) {
  const browser = report.results.browser_embedding;
  const ternlight = report.results.ternlight;
  const bootstrap = report.comparison.bootstrap_macro_ndcg_at_10;
  const ranks = report.comparison.rank_outcomes;
  const rows = METRIC_NAMES.map((name) => {
    const key = `macro_${name}`;
    const delta = `${(report.comparison.metric_deltas[key] * 100).toFixed(2)} pp`;
    return `${key.padEnd(24)}${percent(browser.macro[key]).padEnd(20)}${percent(ternlight.macro[key]).padEnd(20)}${delta}`;
  });
  return [
    "",
    "Retrieval quality: browser-embedding vs Ternlight",
    `dataset: ${report.dataset.intents} intents, ${report.dataset.queries} paired queries, 6 slices`,
    "",
    "metric                  browser-embedding   Ternlight           delta (percentage points)",
    ...rows,
    "",
    `bootstrap nDCG delta: ${(bootstrap.estimate * 100).toFixed(2)} pp  95% CI [${(bootstrap.lower * 100).toFixed(2)}, ${(bootstrap.upper * 100).toFixed(2)}] pp  ${bootstrap.verdict}`,
    `rank outcomes: browser wins=${ranks.browser_wins} ties=${ranks.ties} losses=${ranks.browser_losses}`,
    "",
    "This 32-intent development set validates the harness; it is not a held-out release claim.",
    "",
  ].join("\n");
}

const options = parseArgs(process.argv.slice(2));
if (options.help) {
  process.stdout.write(usage());
  process.exit(0);
}
requireFiles([
  options.dataset,
  options.browserModule,
  options.browserModel,
  options.browserTokenizer,
  options.ternlightModule,
]);
const dataset = loadDataset(options.dataset);
const slices = buildSlices(dataset.records);
const engines = loadEngines(options);
const browser = evaluateEngine(engines.browser_embedding, slices);
const ternlight = evaluateEngine(engines.ternlight, slices);
const report = {
  schema_version: 1,
  timestamp: new Date().toISOString(),
  dataset: {
    path: relative(WORKSPACE_ROOT, options.dataset).replaceAll("\\", "/"),
    sha256: sha256(dataset.canonicalBytes),
    intents: dataset.records.length,
    slices: Object.keys(slices),
    queries: dataset.records.length * Object.keys(slices).length,
    documents_per_slice: dataset.records.length,
    status: "development smoke set; not held out from all historical model selection",
  },
  protocol: {
    dimension: 384,
    similarity: "dot product of each engine's own L2-normalized embeddings",
    roles: "role-aware query and document APIs",
    relevance: "one paired relevant document per query",
    candidate_pool: "all target-language documents in the slice",
    raw_scores_cross_model_comparable: false,
  },
  host: {
    platform: platform(),
    arch: arch(),
    node: process.version,
    cpu: cpus()[0]?.model ?? "unknown",
  },
  commits: {
    browser_embedding: gitCommit(BROWSER_ROOT),
    ternlight: gitCommit(resolve(WORKSPACE_ROOT, "ternlight")),
  },
  artifact_sha256: {
    browser_model: sha256(readFileSync(options.browserModel)),
    browser_tokenizer: sha256(readFileSync(options.browserTokenizer)),
    browser_wasm: sha256(readFileSync(resolve(dirname(options.browserModule), "browser_embedding_wasm_bg.wasm"))),
    ternlight_wasm: sha256(readFileSync(resolve(dirname(options.ternlightModule), "tern_engine_bg.wasm"))),
  },
  results: { browser_embedding: browser, ternlight },
  comparison: {
    metric_deltas: metricDeltas(browser, ternlight),
    bootstrap_macro_ndcg_at_10: bootstrapMacroNdcgDelta(
      browser.slices,
      ternlight.slices,
      { iterations: options.bootstrapIterations, seed: 42 },
    ),
    rank_outcomes: compareRanks(browser.slices, ternlight.slices),
  },
  caveats: [
    "The development dataset is small and has been visible to prior evaluation workflows.",
    "Each slice has only 32 candidate documents, so recall@10 is not discriminative enough for release decisions.",
    "Raw cosine scores and margins are meaningful within one engine only, not across embedding spaces.",
    "A formal claim requires a frozen, deduplicated held-out set with at least 1,000 queries and a much larger document pool.",
  ],
};
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
process.stderr.write(renderSummary(report));
