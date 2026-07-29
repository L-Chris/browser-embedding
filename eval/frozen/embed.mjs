import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { createRequire } from "node:module";
import { dirname, relative, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

const EVAL_DIR = dirname(fileURLToPath(import.meta.url));
const BROWSER_ROOT = resolve(EVAL_DIR, "../..");
const WORKSPACE_ROOT = resolve(BROWSER_ROOT, "..");
const DIMENSION = 384;
const DEFAULTS = {
  lock: resolve(EVAL_DIR, "retrieval-frozen-v1.lock.json"),
  cache: resolve(EVAL_DIR, "cache/retrieval-frozen-v1"),
  browserModule: resolve(BROWSER_ROOT, "target/benchmark-node/browser_embedding_wasm.js"),
  browserModel: resolve(BROWSER_ROOT, "artifacts/runs/ternlight-100k/model.bem"),
  browserTokenizer: resolve(BROWSER_ROOT, "assets/tokenizer.json"),
  ternlightModule: resolve(WORKSPACE_ROOT, "ternlight/packages/base/pkg-node/tern_engine.js"),
};

function parseArgs(argv) {
  const options = { ...DEFAULTS };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index + 1];
    switch (argv[index]) {
      case "--engine": options.engine = value; index += 1; break;
      case "--engine-id": options.engineId = value; index += 1; break;
      case "--lock": options.lock = resolve(value); index += 1; break;
      case "--cache": options.cache = resolve(value); index += 1; break;
      case "--output": options.output = resolve(value); index += 1; break;
      case "--browser-module": options.browserModule = resolve(value); index += 1; break;
      case "--browser-model": options.browserModel = resolve(value); index += 1; break;
      case "--browser-tokenizer": options.browserTokenizer = resolve(value); index += 1; break;
      case "--ternlight-module": options.ternlightModule = resolve(value); index += 1; break;
      case "--adopt-untracked": options.adoptUntracked = true; break;
      case "--help": options.help = true; break;
      default: throw new Error(`unknown argument: ${argv[index]}`);
    }
  }
  if (!options.help && !["browser", "ternlight"].includes(options.engine)) {
    throw new Error("--engine must be browser or ternlight");
  }
  options.engineId ??= options.engine === "browser" ? "browser-embedding" : "ternlight";
  if (!/^[a-z0-9._-]+$/iu.test(options.engineId)) {
    throw new Error("--engine-id may contain only letters, digits, dot, underscore, and hyphen");
  }
  options.output ??= resolve(options.cache, "embeddings", options.engineId);
  return options;
}

function usage() {
  return `Usage: node eval/frozen/embed.mjs --engine browser|ternlight [options]

Encodes every frozen query and full corpus into deterministic little-endian
Float32 matrices. Generated matrices live under the ignored evaluation cache.

Options:
  --engine browser|ternlight     engine adapter (required)
  --engine-id ID                 stable output label
  --lock PATH                    committed frozen-suite lock
  --cache PATH                   prepared normalized dataset cache
  --output PATH                  embedding output directory
  --browser-module PATH          browser-embedding Node WASM loader
  --browser-model PATH           browser-embedding BEM2 model
  --browser-tokenizer PATH       browser-embedding tokenizer
  --ternlight-module PATH        Ternlight Node WASM loader
  --adopt-untracked              recover complete matrices from a pre-checkpoint run
  --help                         show this message
`;
}

function sha256Bytes(value) {
  return createHash("sha256").update(value).digest("hex");
}

function sha256File(path) {
  return sha256Bytes(readFileSync(path));
}

function displayPath(path) {
  const rendered = relative(WORKSPACE_ROOT, path).replaceAll("\\", "/");
  return rendered.startsWith("../") ? path.replaceAll("\\", "/") : rendered;
}

function requireFiles(paths) {
  const missing = paths.filter((path) => !existsSync(path));
  if (missing.length > 0) {
    throw new Error(`missing artifact(s):\n${missing.map((path) => `  ${path}`).join("\n")}`);
  }
}

function loadJsonl(path) {
  return readFileSync(path, "utf8")
    .split(/\r?\n/u)
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
}

function gitMetadata(path) {
  try {
    const root = execFileSync("git", ["rev-parse", "--show-toplevel"], {
      cwd: dirname(path),
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
    const commit = execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: root,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
    return { root: displayPath(root), commit };
  } catch {
    return { root: "unknown", commit: "unknown" };
  }
}

function moduleArtifacts(modulePath) {
  const wasm = readdirSync(dirname(modulePath))
    .filter((name) => name.endsWith(".wasm"))
    .sort()
    .map((name) => resolve(dirname(modulePath), name));
  return [modulePath, ...wasm].map((path) => ({
    path: displayPath(path),
    bytes: readFileSync(path).byteLength,
    sha256: sha256File(path),
  }));
}

function loadEngine(options) {
  const require = createRequire(import.meta.url);
  if (options.engine === "browser") {
    requireFiles([options.browserModule, options.browserModel, options.browserTokenizer]);
    const bindings = require(options.browserModule);
    const model = new bindings.BrowserModel(
      readFileSync(options.browserModel),
      readFileSync(options.browserTokenizer),
    );
    return {
      build: model.model_info(),
      role_api: "role-aware query/document",
      artifacts: [
        ...moduleArtifacts(options.browserModule),
        ...[options.browserModel, options.browserTokenizer].map((path) => ({
          path: displayPath(path),
          bytes: readFileSync(path).byteLength,
          sha256: sha256File(path),
        })),
      ],
      git: gitMetadata(options.browserModule),
      embed: (text, role) => model.embed(text, role, DIMENSION),
    };
  }

  requireFiles([options.ternlightModule]);
  const ternlight = require(options.ternlightModule);
  const query = ternlight.embed_query ?? ternlight.embedQuery ?? ternlight.embed;
  const document = ternlight.embed_document ?? ternlight.embedDocument ?? ternlight.embed;
  if (typeof query !== "function" || typeof document !== "function") {
    throw new Error("Ternlight module does not expose an embedding function");
  }
  const roleAware = Boolean(
    ternlight.embed_query ?? ternlight.embedQuery ?? ternlight.embed_document ?? ternlight.embedDocument,
  );
  return {
    build: ternlight.config_summary?.() ?? ternlight.engineInfo?.() ?? "unknown",
    role_api: roleAware ? "role-aware query/document" : "symmetric embed fallback",
    artifacts: moduleArtifacts(options.ternlightModule),
    git: gitMetadata(options.ternlightModule),
    embed: (text, role) => (role === "query" ? query(text) : document(text)),
  };
}

function cachedValidatedEmbedder(embed) {
  const cache = new Map();
  return {
    cache,
    embed(text, role) {
      const key = `${role}\u0000${text}`;
      if (!cache.has(key)) {
        const vector = Float32Array.from(embed(text, role));
        if (vector.length !== DIMENSION) {
          throw new Error(`expected ${DIMENSION} dimensions, got ${vector.length}`);
        }
        for (const value of vector) {
          if (!Number.isFinite(value)) throw new Error("embedding contains a non-finite value");
        }
        cache.set(key, vector);
      }
      return cache.get(key);
    },
  };
}

function encodeRows(rows, role, adapter, label) {
  const output = Buffer.allocUnsafe(rows.length * DIMENSION * Float32Array.BYTES_PER_ELEMENT);
  const started = performance.now();
  for (let index = 0; index < rows.length; index += 1) {
    const vector = adapter.embed(rows[index].text, role);
    Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength).copy(
      output,
      index * DIMENSION * Float32Array.BYTES_PER_ELEMENT,
    );
    if ((index + 1) % 1000 === 0 || index + 1 === rows.length) {
      const elapsed = (performance.now() - started) / 1000;
      const rate = (index + 1) / Math.max(elapsed, 0.001);
      process.stderr.write(
        `\r${label}: ${index + 1}/${rows.length} (${rate.toFixed(1)} rows/s, cache ${adapter.cache.size})`,
      );
    }
  }
  process.stderr.write("\n");
  return output;
}

function writeAtomic(path, bytes) {
  writeFileSync(path, bytes);
}

const options = parseArgs(process.argv.slice(2));
if (options.help) {
  process.stdout.write(usage());
  process.exit(0);
}
requireFiles([options.lock]);
if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
  throw new Error("the frozen embedding format requires a little-endian host");
}

const lockBytes = readFileSync(options.lock);
const lock = JSON.parse(lockBytes);
const engine = loadEngine(options);
const adapter = cachedValidatedEmbedder(engine.embed);
mkdirSync(options.output, { recursive: true });
const resumePath = resolve(options.output, ".resume.json");
const identity = {
  suite_lock_sha256: sha256Bytes(lockBytes),
  engine_id: options.engineId,
  adapter: options.engine,
  dimension: DIMENSION,
  role_api: engine.role_api,
  artifacts: engine.artifacts,
};
const existingMatrices = readdirSync(options.output).filter((name) => name.endsWith(".f32"));
if (existsSync(resumePath)) {
  const previous = JSON.parse(readFileSync(resumePath, "utf8"));
  if (JSON.stringify(previous) !== JSON.stringify(identity)) {
    throw new Error(
      `resume identity differs in ${resumePath}; choose a new --engine-id or clear that generated output`,
    );
  }
} else {
  if (existingMatrices.length > 0 && !options.adoptUntracked) {
    throw new Error(
      `found untracked matrices in ${options.output}; pass --adopt-untracked only when recovering this exact engine run`,
    );
  }
  writeAtomic(resumePath, `${JSON.stringify(identity, null, 2)}\n`);
}
const started = performance.now();
const datasetArtifacts = [];

for (const dataset of lock.datasets) {
  const outputs = {};
  for (const [kind, role] of [["corpus", "document"], ["queries", "query"]]) {
    const source = resolve(options.cache, dataset.artifacts[kind].path);
    requireFiles([source]);
    if (sha256File(source) !== dataset.artifacts[kind].sha256) {
      throw new Error(`${dataset.id} ${kind} hash differs from the frozen lock; rerun prepare.py`);
    }
    const rows = loadJsonl(source);
    if (rows.length !== dataset.artifacts[kind].records) {
      throw new Error(`${dataset.id} ${kind} row count differs from the frozen lock`);
    }
    const filename = `${dataset.id}.${kind}.f32`;
    const output = resolve(options.output, filename);
    const expectedBytes = rows.length * DIMENSION * Float32Array.BYTES_PER_ELEMENT;
    let byteCount;
    let digest;
    if (existsSync(output) && statSync(output).size === expectedBytes) {
      process.stderr.write(`Reusing complete matrix ${filename}\n`);
      byteCount = expectedBytes;
      digest = sha256File(output);
    } else {
      const matrix = encodeRows(rows, role, adapter, `${dataset.id} ${kind}`);
      writeAtomic(output, matrix);
      byteCount = matrix.byteLength;
      digest = sha256Bytes(matrix);
    }
    outputs[kind] = {
      path: filename,
      rows: rows.length,
      columns: DIMENSION,
      bytes: byteCount,
      sha256: digest,
    };
  }
  datasetArtifacts.push({ id: dataset.id, matrices: outputs });
}

const manifest = {
  schema_version: 1,
  suite_id: lock.suite_id,
  suite_lock_sha256: sha256Bytes(lockBytes),
  engine: {
    id: options.engineId,
    adapter: options.engine,
    build: engine.build,
    role_api: engine.role_api,
    dimension: DIMENSION,
    git: engine.git,
    artifacts: engine.artifacts,
  },
  encoding: {
    dtype: "little-endian float32",
    duration_ms: performance.now() - started,
    unique_role_text_pairs: adapter.cache.size,
  },
  datasets: datasetArtifacts,
};
writeAtomic(resolve(options.output, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
process.stdout.write(`${resolve(options.output, "manifest.json")}\n`);
