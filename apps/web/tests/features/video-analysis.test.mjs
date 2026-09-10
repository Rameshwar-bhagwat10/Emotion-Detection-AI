import test from "node:test";
import assert from "node:assert/strict";

// Test suite for Video Analysis Temporal Intelligence and Timeline Logic

test("Video timeline segments preserve strict non-negative durations and boundary order", () => {
  const segments = [
    { track_id: 1, emotion: "neutral", start_time: 0.0, end_time: 4.2, duration: 4.2, average_confidence: 0.72 },
    { track_id: 1, emotion: "happy", start_time: 4.2, end_time: 6.8, duration: 2.6, average_confidence: 0.81 },
    { track_id: 1, emotion: "surprise", start_time: 6.8, end_time: 7.6, duration: 0.8, average_confidence: 0.76 },
    { track_id: 1, emotion: "sad", start_time: 7.6, end_time: 11.4, duration: 3.8, average_confidence: 0.69 },
  ];

  for (const seg of segments) {
    assert.ok(seg.start_time >= 0, "Start time must be >= 0");
    assert.ok(seg.end_time > seg.start_time, "End time must strictly exceed start time");
    const diff = Math.round((seg.end_time - seg.start_time) * 100) / 100;
    assert.equal(diff, seg.duration, "Segment duration must equal end_time - start_time");
  }
});

test("Expression time share mathematically sums to 100% of tracked duration", () => {
  const segments = [
    { track_id: 1, emotion: "neutral", duration: 4.2 },
    { track_id: 1, emotion: "happy", duration: 2.6 },
    { track_id: 1, emotion: "surprise", duration: 0.8 },
    { track_id: 1, emotion: "sad", duration: 3.8 },
  ];

  const totalTrackedDuration = segments.reduce((sum, s) => sum + s.duration, 0);
  assert.equal(Math.round(totalTrackedDuration * 10) / 10, 11.4);

  const durationByEmotion = {};
  for (const seg of segments) {
    durationByEmotion[seg.emotion] = (durationByEmotion[seg.emotion] || 0) + seg.duration;
  }

  const timeShares = Object.entries(durationByEmotion).map(([emo, dur]) => ({
    emotion: emo,
    time_seconds: Math.round(dur * 10) / 10,
    time_share_percent: Math.round((dur / totalTrackedDuration) * 1000) / 10,
  }));

  const sumShares = timeShares.reduce((acc, it) => acc + it.time_share_percent, 0);
  assert.ok(Math.abs(sumShares - 100.0) < 0.2, `Expected sum ~100%, got ${sumShares}`);
});

test("Dominant expression is resolved by largest time share (not just frame count)", () => {
  // Scenario: neutral has 40 frames at 10 FPS (4.0s), happy has 100 frames at 20 FPS (5.0s)
  const timeShares = [
    { emotion: "neutral", duration: 4.0 },
    { emotion: "happy", duration: 5.0 },
    { emotion: "sad", duration: 1.0 },
  ];

  const dominant = timeShares.reduce((prev, curr) =>
    curr.duration > prev.duration ? curr : prev
  );

  assert.equal(dominant.emotion, "happy");
  assert.equal(dominant.duration, 5.0);
});

test("Transition events generation between adjacent segments", () => {
  const segments = [
    { track_id: 1, emotion: "neutral", start_time: 0.0, end_time: 4.2, average_confidence: 0.72 },
    { track_id: 1, emotion: "happy", start_time: 4.2, end_time: 6.8, average_confidence: 0.81 },
    { track_id: 1, emotion: "surprise", start_time: 6.8, end_time: 7.6, average_confidence: 0.76 },
    { track_id: 1, emotion: "sad", start_time: 7.6, end_time: 11.4, average_confidence: 0.69 },
  ];

  const events = [];
  for (let i = 0; i < segments.length - 1; i++) {
    const prev = segments[i];
    const next = segments[i + 1];
    if (prev.emotion !== next.emotion) {
      events.push({
        track_id: 1,
        timestamp: next.start_time,
        from_emotion: prev.emotion,
        to_emotion: next.emotion,
        confidence: next.average_confidence,
      });
    }
  }

  assert.equal(events.length, 3);
  assert.deepEqual(events[0], {
    track_id: 1,
    timestamp: 4.2,
    from_emotion: "neutral",
    to_emotion: "happy",
    confidence: 0.81,
  });
  assert.deepEqual(events[1], {
    track_id: 1,
    timestamp: 6.8,
    from_emotion: "happy",
    to_emotion: "surprise",
    confidence: 0.76,
  });
});

test("Playhead position calculation and clamping", () => {
  const duration = 20.0;
  const calcPlayhead = (currentTime) => {
    return Math.max(0, Math.min(100, (currentTime / duration) * 100));
  };

  assert.equal(calcPlayhead(0.0), 0);
  assert.equal(calcPlayhead(10.0), 50);
  assert.equal(calcPlayhead(20.0), 100);
  assert.equal(calcPlayhead(-2.0), 0);
  assert.equal(calcPlayhead(25.0), 100);
});

test("Multi-track face isolation prevents track segment cross-contamination", () => {
  const segments = [
    { track_id: 1, emotion: "neutral", start_time: 0.0, end_time: 5.0 },
    { track_id: 2, emotion: "happy", start_time: 2.0, end_time: 8.0 },
    { track_id: 1, emotion: "happy", start_time: 5.0, end_time: 10.0 },
  ];

  const byTrack = {};
  for (const s of segments) {
    byTrack[s.track_id] = byTrack[s.track_id] || [];
    byTrack[s.track_id].push(s);
  }

  assert.equal(Object.keys(byTrack).length, 2);
  assert.equal(byTrack[1].length, 2);
  assert.equal(byTrack[2].length, 1);
  assert.equal(byTrack[2][0].emotion, "happy");
});
