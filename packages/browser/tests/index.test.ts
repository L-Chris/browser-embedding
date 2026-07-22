import { describe, expect, it } from "vitest";

import { Embedder, type WasmModel } from "../src/index";

const backend: WasmModel = {
  embed: (_text, _role, dimension) => {
    const output = new Float32Array(dimension);
    output[0] = 1;
    return output;
  },
  model_info: () => "BEM2 test model",
};

describe("Embedder", () => {
  it("forwards role and Matryoshka dimension to WASM", () => {
    const embedder = new Embedder(backend, 128);
    expect(embedder.embed("中英 mixed query", "query")).toHaveLength(128);
    expect(embedder.info).toBe("BEM2 test model");
  });

  it("computes cosine for normalized vectors as a dot product", () => {
    const embedder = new Embedder(backend);
    expect(embedder.similarity(Float32Array.of(1, 0), Float32Array.of(0.5, 0.5))).toBe(0.5);
  });
});
