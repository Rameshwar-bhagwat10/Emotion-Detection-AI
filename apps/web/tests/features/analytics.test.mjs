import test from "node:test";
import assert from "node:assert/strict";

// Test suite for Phase 13 Analytics, Aggregations, and Explainability logic

test("Expression distribution mathematically sums to 100% and matches total counts", () => {
  const counts = {
    happy: 50,
    neutral: 30,
    surprise: 15,
    sad: 5,
    fear: 0,
    angry: 0,
    disgust: 0,
  };

  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  assert.equal(total, 100);

  const items = Object.entries(counts).map(([emotion, count]) => ({
    emotion,
    count,
    percentage: Math.round((count / total) * 1000) / 10,
  }));

  const sumCounts = items.reduce((acc, it) => acc + it.count, 0);
  assert.equal(sumCounts, total);

  const sumPercentages = items.reduce((acc, it) => acc + it.percentage, 0);
  assert.ok(Math.abs(sumPercentages - 100.0) < 0.2);
});

test("Dominant emotion resolution with deterministic tie-breaking", () => {
  const resolveDominant = (counts) => {
    let dominant = null;
    let maxCount = -1;

    // Alphabetical order for deterministic tie breaking
    const emotions = Object.keys(counts).sort();
    for (const emo of emotions) {
      const c = counts[emo];
      if (c > maxCount) {
        maxCount = c;
        dominant = emo;
      }
    }
    return maxCount > 0 ? dominant : null;
  };

  // Clear dominant
  assert.equal(resolveDominant({ happy: 10, neutral: 5 }), "happy");

  // Tie-breaking: angry vs happy (both 10) -> angry wins alphabetically
  assert.equal(resolveDominant({ happy: 10, angry: 10 }), "angry");

  // All zero
  assert.equal(resolveDominant({ happy: 0, neutral: 0 }), null);
});

test("Confidence histogram bin assignment logic", () => {
  const assignBin = (confidence) => {
    if (confidence < 0.20) return "0-20%";
    if (confidence < 0.40) return "20-40%";
    if (confidence < 0.60) return "40-60%";
    if (confidence < 0.80) return "60-80%";
    return "80-100%";
  };

  assert.equal(assignBin(0.05), "0-20%");
  assert.equal(assignBin(0.20), "20-40%");
  assert.equal(assignBin(0.55), "40-60%");
  assert.equal(assignBin(0.79), "60-80%");
  assert.equal(assignBin(0.80), "80-100%");
  assert.equal(assignBin(0.99), "80-100%");
});

test("Top-K probability ranking for explainability panel", () => {
  const rawProbabilities = {
    angry: 0.02,
    disgust: 0.01,
    fear: 0.04,
    happy: 0.72,
    sad: 0.03,
    surprise: 0.08,
    neutral: 0.10,
  };

  const ranked = Object.entries(rawProbabilities)
    .map(([emotion, probability]) => ({
      emotion,
      probability,
      percentage: (probability * 100).toFixed(1),
    }))
    .sort((a, b) => b.probability - a.probability);

  assert.equal(ranked[0].emotion, "happy");
  assert.equal(ranked[0].percentage, "72.0");
  assert.equal(ranked[1].emotion, "neutral");
  assert.equal(ranked[1].percentage, "10.0");
  assert.equal(ranked[2].emotion, "surprise");
  assert.equal(ranked[2].percentage, "8.0");
});

test("Timeline relative seconds formatting helper", () => {
  const formatOffset = (relativeSeconds) => {
    const min = Math.floor(relativeSeconds / 60);
    const sec = Math.floor(relativeSeconds % 60);
    return `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  };

  assert.equal(formatOffset(0), "00:00");
  assert.equal(formatOffset(15), "00:15");
  assert.equal(formatOffset(65), "01:05");
  assert.equal(formatOffset(360), "06:00");
});

test("Zero-safe analytics formatting handles empty session without NaN", () => {
  const safeAverage = (sum, count) => {
    if (!count || count === 0) return 0.0;
    return Math.round((sum / count) * 1000) / 1000;
  };

  assert.equal(safeAverage(0, 0), 0.0);
  assert.equal(Number.isNaN(safeAverage(0, 0)), false);
  assert.equal(safeAverage(8.24, 10), 0.824);
});
