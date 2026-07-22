export type EmbeddingRole = "query" | "document";
export type MatryoshkaDimension = 64 | 128 | 256 | 384;

export interface WasmModel {
  embed(text: string, role: EmbeddingRole, dimension: number): Float32Array | number[];
  model_info(): string;
}

export interface WasmBindings {
  BrowserModel: new (modelBytes: Uint8Array, tokenizerJson: Uint8Array) => WasmModel;
}

export interface CreateEmbedderOptions {
  model: URL | ArrayBuffer | Uint8Array;
  tokenizer: URL | ArrayBuffer | Uint8Array;
  loadBindings: () => Promise<WasmBindings>;
  defaultDimension?: MatryoshkaDimension;
}

async function loadBytes(source: URL | ArrayBuffer | Uint8Array): Promise<Uint8Array> {
  if (source instanceof Uint8Array) return source;
  if (source instanceof ArrayBuffer) return new Uint8Array(source);
  const response = await fetch(source);
  if (!response.ok) {
    throw new Error(`Failed to load browser-embedding asset ${source}: HTTP ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

export class Embedder {
  readonly info: string;

  constructor(
    private readonly backend: WasmModel,
    private readonly defaultDimension: MatryoshkaDimension = 384,
  ) {
    this.info = backend.model_info();
  }

  embed(
    text: string,
    role: EmbeddingRole = "document",
    dimension: MatryoshkaDimension = this.defaultDimension,
  ): Float32Array {
    if (!text.trim()) throw new Error("text cannot be empty");
    return Float32Array.from(this.backend.embed(text, role, dimension));
  }

  similarity(left: Float32Array, right: Float32Array): number {
    if (left.length !== right.length || left.length === 0) {
      throw new Error("embeddings must have the same non-zero dimension");
    }
    let score = 0;
    for (let index = 0; index < left.length; index += 1) {
      score += left[index]! * right[index]!;
    }
    return score;
  }
}

export async function createEmbedder(options: CreateEmbedderOptions): Promise<Embedder> {
  const [model, tokenizer, bindings] = await Promise.all([
    loadBytes(options.model),
    loadBytes(options.tokenizer),
    options.loadBindings(),
  ]);
  return new Embedder(
    new bindings.BrowserModel(model, tokenizer),
    options.defaultDimension,
  );
}
