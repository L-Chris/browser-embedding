import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { cpus, arch, platform, release } from "node:os";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { brotliCompressSync, constants, gzipSync } from "node:zlib";
import { performance } from "node:perf_hooks";

const BENCHMARK_DIR = dirname(fileURLToPath(import.meta.url));
const BROWSER_ROOT = resolve(BENCHMARK_DIR, "../..");
const WORKSPACE_ROOT = resolve(BROWSER_ROOT, "..");
const DEFAULTS = {
  browserModule: resolve(BROWSER_ROOT, "target/benchmark-node/browser_embedding_wasm.js"),
  browserWasm: resolve(BROWSER_ROOT, "target/benchmark-node/browser_embedding_wasm_bg.wasm"),
  browserModel: resolve(BROWSER_ROOT, "artifacts/runs/ternlight-100k/model.bem"),
  browserTokenizer: resolve(BROWSER_ROOT, "assets/tokenizer.json"),
  ternlightModule: resolve(WORKSPACE_ROOT, "ternlight/packages/base/pkg-node/tern_engine.js"),
  ternlightWasm: resolve(WORKSPACE_ROOT, "ternlight/packages/base/pkg-node/tern_engine_bg.wasm"),
  rounds: 20,
};

// Kept in this repository so both engines always see byte-identical inputs.
// The set deliberately mixes English, Chinese, mixed-language, short, and long text.
const QUERIES = [
  "reset my password",
  "how do I cancel my subscription",
  "fix npm install error EACCES",
  "how to write unit tests in Rust",
  "kubernetes pod stuck in CrashLoopBackOff",
  "如何重置我的密码",
  "怎么取消订阅并申请退款",
  "为什么 Docker 容器一直崩溃",
  "用 Rust 编写单元测试的方法",
  "如何部署 Next.js 应用到 Vercel",
  "TypeScript 类型错误怎么修复",
  "configure SSH 密钥 for GitHub",
  "机器学习 machine learning 与 deep learning 的区别",
  "跨语言语义检索需要让中文查询和英文文档落在同一个向量空间中。",
  "A browser-first embedding runtime should minimize download size while keeping CPU latency predictable.",
];

function parseArgs(argv) {
  const options = { ...DEFAULTS };
  let worker;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const value = argv[index + 1];
    switch (arg) {
      case "--worker": worker = value; index += 1; break;
      case "--browser-module": options.browserModule = resolve(value); index += 1; break;
      case "--browser-wasm": options.browserWasm = resolve(value); index += 1; break;
      case "--browser-model": options.browserModel = resolve(value); index += 1; break;
      case "--browser-tokenizer": options.browserTokenizer = resolve(value); index += 1; break;
      case "--ternlight-module": options.ternlightModule = resolve(value); index += 1; break;
      case "--ternlight-wasm": options.ternlightWasm = resolve(value); index += 1; break;
      case "--rounds": options.rounds = Number.parseInt(value, 10); index += 1; break;
      case "--help": options.help = true; break;
      default: throw new Error(`unknown argument: ${arg}`);
    }
  }
  if (!Number.isInteger(options.rounds) || options.rounds < 1) {
    throw new Error("--rounds must be a positive integer");
  }
  return { options, worker };
}

function usage() {
  return `Usage: node eval/benchmarks/compare.mjs [options]

Runs browser-embedding and Ternlight in separate Node processes with the same
384-dimensional workload. JSON is written to stdout and a summary to stderr.

Options:
  --browser-module PATH     wasm-bindgen Node.js loader
  --browser-wasm PATH       browser-embedding WASM binary
  --browser-model PATH      BEM2 model (default: artifacts/runs/ternlight-100k/model.bem)
  --browser-tokenizer PATH  browser-embedding tokenizer
  --ternlight-module PATH   Ternlight Node.js loader
  --ternlight-wasm PATH     Ternlight WASM binary
  --rounds N                repetitions of the 15-query corpus (default: 20)
  --help                    show this message
`;
}

function requireFiles(paths) {
  const missing = paths.filter((path) => !existsSync(path));
  if (missing.length > 0) {
    throw new Error(`missing benchmark artifact(s):\n${missing.map((path) => `  ${path}`).join("\n")}`);
  }
}

function percentile(sorted, fraction) {
  const index = Math.min(sorted.length - 1, Math.floor(sorted.length * fraction));
  return sorted[index];
}

function summarize(samples, wallMs) {
  const sorted = [...samples].sort((left, right) => left - right);
  const round = (value) => Number(value.toFixed(3));
  return {
    samples: sorted.length,
    min_ms: round(sorted[0]),
    p50_ms: round(percentile(sorted, 0.5)),
    p95_ms: round(percentile(sorted, 0.95)),
    p99_ms: round(percentile(sorted, 0.99)),
    max_ms: round(sorted.at(-1)),
    mean_ms: round(sorted.reduce((sum, value) => sum + value, 0) / sorted.length),
    throughput_per_second: round(sorted.length * 1000 / wallMs),
  };
}

function memorySnapshot() {
  const memory = process.memoryUsage();
  return {
    rss: memory.rss,
    heap_used: memory.heapUsed,
    external: memory.external,
    array_buffers: memory.arrayBuffers,
  };
}

function subtractMemory(after, before) {
  return Object.fromEntries(
    Object.keys(after).map((key) => [key, Math.max(0, after[key] - before[key])]),
  );
}

function benchmark(embed, rounds) {
  for (let index = 0; index < 10; index += 1) embed(QUERIES[index % QUERIES.length]);
  const memoryBefore = memorySnapshot();
  const samples = [];
  const wallStart = performance.now();
  for (let round = 0; round < rounds; round += 1) {
    for (const query of QUERIES) {
      const started = performance.now();
      const vector = embed(query);
      samples.push(performance.now() - started);
      if (vector.length !== 384) throw new Error(`expected 384 dimensions, got ${vector.length}`);
    }
  }
  const wallMs = performance.now() - wallStart;
  const memoryAfter = memorySnapshot();
  return {
    latency: summarize(samples, wallMs),
    memory_after_benchmark_bytes: memoryAfter,
    memory_benchmark_delta_bytes: subtractMemory(memoryAfter, memoryBefore),
  };
}

function runBrowserWorker(options) {
  requireFiles([options.browserModule, options.browserWasm, options.browserModel, options.browserTokenizer]);
  const require = createRequire(import.meta.url);
  const baseline = memorySnapshot();

  const moduleStarted = performance.now();
  const bindings = require(options.browserModule);
  const moduleLoadMs = performance.now() - moduleStarted;

  const assetsStarted = performance.now();
  const modelBytes = readFileSync(options.browserModel);
  const tokenizerBytes = readFileSync(options.browserTokenizer);
  const assetReadMs = performance.now() - assetsStarted;

  const initStarted = performance.now();
  const model = new bindings.BrowserModel(modelBytes, tokenizerBytes);
  const modelInitMs = performance.now() - initStarted;

  const firstStarted = performance.now();
  const first = model.embed("warmup", "document", 384);
  const firstInferenceMs = performance.now() - firstStarted;
  if (first.length !== 384) throw new Error(`expected 384 dimensions, got ${first.length}`);

  return {
    engine: "browser-embedding",
    build: model.model_info(),
    startup_ms: {
      module_load: moduleLoadMs,
      asset_read: assetReadMs,
      model_init: modelInitMs,
      first_inference: firstInferenceMs,
      total: moduleLoadMs + assetReadMs + modelInitMs + firstInferenceMs,
    },
    memory_startup_delta_bytes: subtractMemory(memorySnapshot(), baseline),
    ...benchmark((text) => model.embed(text, "document", 384), options.rounds),
  };
}

function runTernlightWorker(options) {
  requireFiles([options.ternlightModule, options.ternlightWasm]);
  const require = createRequire(import.meta.url);
  const baseline = memorySnapshot();

  const moduleStarted = performance.now();
  const bindings = require(options.ternlightModule);
  const moduleLoadMs = performance.now() - moduleStarted;
  const embed = bindings.embed_document ?? bindings.embedDocument ?? bindings.embed;
  if (typeof embed !== "function") throw new Error("Ternlight module has no embedding function");

  const firstStarted = performance.now();
  const first = embed("warmup");
  const firstInferenceMs = performance.now() - firstStarted;
  if (first.length !== 384) throw new Error(`expected 384 dimensions, got ${first.length}`);

  return {
    engine: "ternlight",
    build: bindings.config_summary?.() ?? bindings.engineInfo?.() ?? "unknown",
    startup_ms: {
      module_load: moduleLoadMs,
      asset_read: 0,
      model_init: 0,
      first_inference: firstInferenceMs,
      total: moduleLoadMs + firstInferenceMs,
    },
    memory_startup_delta_bytes: subtractMemory(memorySnapshot(), baseline),
    ...benchmark(embed, options.rounds),
  };
}

function sha256(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

function payload(files) {
  const entries = files.map((path) => {
    const bytes = readFileSync(path);
    return {
      path: relative(WORKSPACE_ROOT, path).replaceAll("\\", "/"),
      raw_bytes: bytes.length,
      gzip_bytes: gzipSync(bytes, { level: 9 }).length,
      brotli_bytes: brotliCompressSync(bytes, {
        params: { [constants.BROTLI_PARAM_QUALITY]: 11 },
      }).length,
      sha256: sha256(bytes),
    };
  });
  const sum = (key) => entries.reduce((total, entry) => total + entry[key], 0);
  return {
    files: entries,
    total: {
      raw_bytes: sum("raw_bytes"),
      gzip_bytes: sum("gzip_bytes"),
      brotli_bytes: sum("brotli_bytes"),
    },
  };
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

function workerArgs(options, engine) {
  return [
    fileURLToPath(import.meta.url), "--worker", engine,
    "--browser-module", options.browserModule,
    "--browser-wasm", options.browserWasm,
    "--browser-model", options.browserModel,
    "--browser-tokenizer", options.browserTokenizer,
    "--ternlight-module", options.ternlightModule,
    "--ternlight-wasm", options.ternlightWasm,
    "--rounds", String(options.rounds),
  ];
}

function runIsolated(options, engine) {
  const result = spawnSync(process.execPath, workerArgs(options, engine), {
    encoding: "utf8",
    windowsHide: true,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${engine} worker failed:\n${result.stderr || result.stdout}`);
  }
  return JSON.parse(result.stdout);
}

function ratio(numerator, denominator) {
  return Number((numerator / denominator).toFixed(3));
}

function mib(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(2)} MiB`;
}

function milliseconds(value) {
  return `${value.toFixed(2)} ms`;
}

function renderSummary(report) {
  const browser = report.results.browser_embedding;
  const ternlight = report.results.ternlight;
  const browserSize = report.payload.browser_embedding.total;
  const ternlightSize = report.payload.ternlight.total;
  return [
    "",
    "browser-embedding vs Ternlight (same host, same 384d workload)",
    "",
    "metric                    browser-embedding    Ternlight           ratio (browser/ternlight)",
    `p50 latency               ${milliseconds(browser.latency.p50_ms).padEnd(21)}${milliseconds(ternlight.latency.p50_ms).padEnd(20)}${report.comparison.browser_to_ternlight.p50_latency}`,
    `p95 latency               ${milliseconds(browser.latency.p95_ms).padEnd(21)}${milliseconds(ternlight.latency.p95_ms).padEnd(20)}${report.comparison.browser_to_ternlight.p95_latency}`,
    `throughput                ${String(browser.latency.throughput_per_second + " /s").padEnd(21)}${String(ternlight.latency.throughput_per_second + " /s").padEnd(20)}${report.comparison.browser_to_ternlight.throughput}`,
    `startup                   ${milliseconds(browser.startup_ms.total).padEnd(21)}${milliseconds(ternlight.startup_ms.total).padEnd(20)}${report.comparison.browser_to_ternlight.startup}`,
    `raw payload               ${mib(browserSize.raw_bytes).padEnd(21)}${mib(ternlightSize.raw_bytes).padEnd(20)}${report.comparison.browser_to_ternlight.raw_payload}`,
    `gzip payload              ${mib(browserSize.gzip_bytes).padEnd(21)}${mib(ternlightSize.gzip_bytes).padEnd(20)}${report.comparison.browser_to_ternlight.gzip_payload}`,
    `brotli payload            ${mib(browserSize.brotli_bytes).padEnd(21)}${mib(ternlightSize.brotli_bytes).padEnd(20)}${report.comparison.browser_to_ternlight.brotli_payload}`,
    "",
    "Startup is local Node module/asset initialization, not network download time.",
    "Payload is the complete runtime input: JS glue + WASM + external model/tokenizer assets.",
    "",
  ].join("\n");
}

const { options, worker } = parseArgs(process.argv.slice(2));
if (options.help) {
  process.stdout.write(usage());
  process.exit(0);
}

if (worker) {
  const result = worker === "browser"
    ? runBrowserWorker(options)
    : worker === "ternlight"
      ? runTernlightWorker(options)
      : (() => { throw new Error(`unknown worker engine: ${worker}`); })();
  process.stdout.write(`${JSON.stringify(result)}\n`);
} else {
  requireFiles([
    options.browserModule,
    options.browserWasm,
    options.browserModel,
    options.browserTokenizer,
    options.ternlightModule,
    options.ternlightWasm,
  ]);
  const browser = runIsolated(options, "browser");
  const ternlight = runIsolated(options, "ternlight");
  const browserPayload = payload([
    options.browserModule,
    options.browserWasm,
    options.browserModel,
    options.browserTokenizer,
  ]);
  const ternlightPayload = payload([options.ternlightModule, options.ternlightWasm]);
  const report = {
    schema_version: 1,
    timestamp: new Date().toISOString(),
    workload: {
      dimension: 384,
      role: "document",
      query_count: QUERIES.length,
      rounds: options.rounds,
      samples_per_engine: QUERIES.length * options.rounds,
      isolation: "one fresh Node.js process per engine",
    },
    host: {
      platform: platform(),
      release: release(),
      arch: arch(),
      node: process.version,
      cpu: cpus()[0]?.model ?? "unknown",
    },
    commits: {
      browser_embedding: gitCommit(BROWSER_ROOT),
      ternlight: gitCommit(resolve(WORKSPACE_ROOT, "ternlight")),
    },
    results: { browser_embedding: browser, ternlight },
    payload: { browser_embedding: browserPayload, ternlight: ternlightPayload },
    comparison: {
      browser_to_ternlight: {
        p50_latency: ratio(browser.latency.p50_ms, ternlight.latency.p50_ms),
        p95_latency: ratio(browser.latency.p95_ms, ternlight.latency.p95_ms),
        throughput: ratio(browser.latency.throughput_per_second, ternlight.latency.throughput_per_second),
        startup: ratio(browser.startup_ms.total, ternlight.startup_ms.total),
        raw_payload: ratio(browserPayload.total.raw_bytes, ternlightPayload.total.raw_bytes),
        gzip_payload: ratio(browserPayload.total.gzip_bytes, ternlightPayload.total.gzip_bytes),
        brotli_payload: ratio(browserPayload.total.brotli_bytes, ternlightPayload.total.brotli_bytes),
      },
    },
    caveats: [
      "Latency measures Node.js WASM on one host; a real-browser benchmark is still required before browser performance claims.",
      "Startup excludes network transfer and measures warm operating-system file cache behavior.",
      "Compressed payload sizes compress each HTTP-style resource separately.",
      "This benchmark compares runtime performance and package size, not retrieval quality.",
    ],
  };
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  process.stderr.write(renderSummary(report));
}
