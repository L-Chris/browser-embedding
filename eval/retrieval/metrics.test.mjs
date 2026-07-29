import assert from "node:assert/strict";
import test from "node:test";

import {
  aggregateResults,
  bootstrapMacroNdcgDelta,
  compareRanks,
  macroAverage,
  rankResult,
} from "./metrics.mjs";

test("rankResult reports stable rank and single-positive metrics", () => {
  const result = rankResult([0.2, 0.9, 0.4], 2, 2);
  assert.equal(result.rank, 2);
  assert.equal(result.recall_at_1, 0);
  assert.equal(result.recall_at_3, 1);
  assert.equal(result.mrr_at_10, 0.5);
  assert.deepEqual(result.top_indices, [1, 2]);
  assert.ok(Math.abs(result.positive_margin + 0.5) < 1e-12);
});

test("aggregate and macro metrics retain slice weighting", () => {
  const first = rankResult([0.9, 0.1], 0);
  const second = rankResult([0.9, 0.1], 1);
  const metrics = aggregateResults([first, second], 2);
  assert.equal(metrics.recall_at_1, 0.5);
  const macro = macroAverage({
    a: { metrics },
    b: { metrics: { ...metrics, ndcg_at_10: 1 } },
  });
  assert.equal(macro.macro_recall_at_1, 0.5);
  assert.equal(macro.macro_ndcg_at_10, (metrics.ndcg_at_10 + 1) / 2);
});

test("paired bootstrap and rank comparison favor a consistently better engine", () => {
  const query = (id, rank, ndcg) => ({
    query_id: id,
    domain: "test",
    query: id,
    rank,
    ndcg_at_10: ndcg,
    top_documents: [{ document_id: "d" }],
    relevant_document_id: "d",
  });
  const browser = {
    en_en: {
      metrics: { ndcg_at_10: 1 },
      queries: [query("q1", 1, 1), query("q2", 1, 1)],
    },
  };
  const ternlight = {
    en_en: {
      metrics: { ndcg_at_10: 0.5 },
      queries: [query("q1", 2, 0.5), query("q2", 2, 0.5)],
    },
  };
  const bootstrap = bootstrapMacroNdcgDelta(browser, ternlight, {
    iterations: 100,
    seed: 7,
  });
  assert.equal(bootstrap.estimate, 0.5);
  assert.equal(bootstrap.lower, 0.5);
  assert.equal(bootstrap.verdict, "superior");
  assert.equal(compareRanks(browser, ternlight).browser_wins, 2);
});
