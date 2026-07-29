export const METRIC_NAMES = [
  "recall_at_1",
  "recall_at_3",
  "recall_at_10",
  "mrr_at_10",
  "ndcg_at_10",
];

export function rankResult(scores, relevantIndex, topK = 10) {
  if (!Array.isArray(scores) || scores.length < 2) {
    throw new Error("ranking requires at least two document scores");
  }
  if (!Number.isInteger(relevantIndex) || relevantIndex < 0 || relevantIndex >= scores.length) {
    throw new Error("relevant document index is outside the score array");
  }
  const ordering = Array.from(scores.keys()).sort(
    (left, right) => scores[right] - scores[left] || left - right,
  );
  const rank = ordering.indexOf(relevantIndex) + 1;
  const positiveScore = scores[relevantIndex];
  const bestNegativeScore = Math.max(
    ...scores.filter((_score, index) => index !== relevantIndex),
  );
  return {
    rank,
    recall_at_1: Number(rank <= 1),
    recall_at_3: Number(rank <= 3),
    recall_at_10: Number(rank <= 10),
    mrr_at_10: rank <= 10 ? 1 / rank : 0,
    ndcg_at_10: rank <= 10 ? 1 / Math.log2(rank + 1) : 0,
    positive_score: positiveScore,
    best_negative_score: bestNegativeScore,
    positive_margin: positiveScore - bestNegativeScore,
    top_indices: ordering.slice(0, topK),
  };
}

export function aggregateResults(results, documentCount) {
  if (results.length === 0) throw new Error("cannot aggregate an empty query result set");
  const mean = (field) =>
    results.reduce((total, result) => total + result[field], 0) / results.length;
  return {
    queries: results.length,
    documents: documentCount,
    recall_at_1: mean("recall_at_1"),
    recall_at_3: mean("recall_at_3"),
    recall_at_10: mean("recall_at_10"),
    mrr_at_10: mean("mrr_at_10"),
    ndcg_at_10: mean("ndcg_at_10"),
    mean_rank: mean("rank"),
    mean_positive_score: mean("positive_score"),
    mean_positive_margin: mean("positive_margin"),
  };
}

export function macroAverage(slices) {
  const entries = Object.values(slices);
  if (entries.length === 0) throw new Error("cannot macro-average zero slices");
  return Object.fromEntries(
    METRIC_NAMES.map((metric) => [
      `macro_${metric}`,
      entries.reduce((total, slice) => total + slice.metrics[metric], 0) / entries.length,
    ]),
  );
}

function mulberry32(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let output = value;
    output = Math.imul(output ^ (output >>> 15), output | 1);
    output ^= output + Math.imul(output ^ (output >>> 7), output | 61);
    return ((output ^ (output >>> 14)) >>> 0) / 4294967296;
  };
}

function quantile(sorted, fraction) {
  const index = Math.min(sorted.length - 1, Math.floor(sorted.length * fraction));
  return sorted[index];
}

export function bootstrapMacroNdcgDelta(
  browserSlices,
  ternlightSlices,
  { iterations = 10_000, seed = 42 } = {},
) {
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("bootstrap iterations must be a positive integer");
  }
  const sliceNames = Object.keys(browserSlices).sort();
  if (sliceNames.length === 0 || sliceNames.some((name) => !ternlightSlices[name])) {
    throw new Error("engines must provide the same non-empty slice set");
  }
  const random = mulberry32(seed);
  const samples = new Array(iterations);
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    let macroDelta = 0;
    for (const name of sliceNames) {
      const browser = browserSlices[name].queries;
      const ternlight = ternlightSlices[name].queries;
      if (browser.length !== ternlight.length || browser.length === 0) {
        throw new Error(`slice ${name} is not paired`);
      }
      let sliceDelta = 0;
      for (let sample = 0; sample < browser.length; sample += 1) {
        const index = Math.floor(random() * browser.length);
        sliceDelta += browser[index].ndcg_at_10 - ternlight[index].ndcg_at_10;
      }
      macroDelta += sliceDelta / browser.length;
    }
    samples[iteration] = macroDelta / sliceNames.length;
  }
  samples.sort((left, right) => left - right);
  const estimate =
    sliceNames.reduce(
      (total, name) =>
        total
        + browserSlices[name].metrics.ndcg_at_10
        - ternlightSlices[name].metrics.ndcg_at_10,
      0,
    ) / sliceNames.length;
  const lower = quantile(samples, 0.025);
  const upper = quantile(samples, 0.975);
  return {
    metric: "macro_ndcg_at_10",
    method: "paired stratified bootstrap by language slice",
    iterations,
    seed,
    estimate,
    confidence_level: 0.95,
    lower,
    upper,
    verdict: lower > 0 ? "superior" : lower >= -0.02 ? "non_inferior" : "regressed",
    non_inferiority_margin: -0.02,
  };
}

export function compareRanks(browserSlices, ternlightSlices) {
  const comparisons = [];
  for (const name of Object.keys(browserSlices).sort()) {
    const browser = browserSlices[name].queries;
    const ternlight = ternlightSlices[name]?.queries;
    if (!ternlight || browser.length !== ternlight.length) {
      throw new Error(`slice ${name} is not paired`);
    }
    for (let index = 0; index < browser.length; index += 1) {
      if (browser[index].query_id !== ternlight[index].query_id) {
        throw new Error(`slice ${name} query order differs between engines`);
      }
      comparisons.push({
        slice: name,
        query_id: browser[index].query_id,
        domain: browser[index].domain,
        query: browser[index].query,
        browser_rank: browser[index].rank,
        ternlight_rank: ternlight[index].rank,
        rank_delta: browser[index].rank - ternlight[index].rank,
        browser_top_document_id: browser[index].top_documents[0].document_id,
        ternlight_top_document_id: ternlight[index].top_documents[0].document_id,
        relevant_document_id: browser[index].relevant_document_id,
      });
    }
  }
  const wins = comparisons.filter((item) => item.rank_delta < 0).length;
  const losses = comparisons.filter((item) => item.rank_delta > 0).length;
  const ties = comparisons.length - wins - losses;
  const summarize = (items) => {
    const groupWins = items.filter((item) => item.rank_delta < 0).length;
    const groupLosses = items.filter((item) => item.rank_delta > 0).length;
    return {
      queries: items.length,
      browser_wins: groupWins,
      ties: items.length - groupWins - groupLosses,
      browser_losses: groupLosses,
      mean_rank_delta:
        items.reduce((total, item) => total + item.rank_delta, 0) / items.length,
    };
  };
  const groupBy = (field) => Object.fromEntries(
    [...new Set(comparisons.map((item) => item[field]))]
      .sort()
      .map((value) => [value, summarize(comparisons.filter((item) => item[field] === value))]),
  );
  const byWorstDelta = [...comparisons].sort(
    (left, right) => right.rank_delta - left.rank_delta || left.slice.localeCompare(right.slice),
  );
  return {
    queries: comparisons.length,
    browser_wins: wins,
    ties,
    browser_losses: losses,
    mean_rank_delta:
      comparisons.reduce((total, item) => total + item.rank_delta, 0) / comparisons.length,
    by_slice: groupBy("slice"),
    by_domain: groupBy("domain"),
    largest_browser_losses: byWorstDelta.filter((item) => item.rank_delta > 0).slice(0, 10),
    largest_browser_wins: byWorstDelta
      .filter((item) => item.rank_delta < 0)
      .slice(-10)
      .reverse(),
  };
}
