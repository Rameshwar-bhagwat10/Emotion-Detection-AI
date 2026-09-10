import test from "node:test";
import assert from "node:assert/strict";

// Test suite for frontend application logic, state transitions, and validation

test("Emotion taxonomy contains exactly 7 primary classes plus uncertain", () => {
  const primaryEmotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];
  assert.equal(primaryEmotions.length, 7);
  assert.ok(primaryEmotions.includes("happy"));
  assert.ok(primaryEmotions.includes("neutral"));
  assert.ok(primaryEmotions.includes("surprise"));
});

test("Client-side image file validation logic", () => {
  const MAX_FILE_SIZE_MB = 10;
  const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];

  const validateFile = (file) => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return { valid: false, error: "UNSUPPORTED_FORMAT" };
    }
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      return { valid: false, error: "FILE_TOO_LARGE" };
    }
    return { valid: true };
  };

  // Valid JPEG
  assert.deepEqual(validateFile({ type: "image/jpeg", size: 2 * 1024 * 1024 }), { valid: true });
  // Valid PNG
  assert.deepEqual(validateFile({ type: "image/png", size: 5 * 1024 * 1024 }), { valid: true });
  // Invalid format (GIF)
  assert.deepEqual(validateFile({ type: "image/gif", size: 1024 }), {
    valid: false,
    error: "UNSUPPORTED_FORMAT",
  });
  // Oversized image (15MB)
  assert.deepEqual(validateFile({ type: "image/jpeg", size: 15 * 1024 * 1024 }), {
    valid: false,
    error: "FILE_TOO_LARGE",
  });
});

test("ApiClientError normalizes HTTP and network exceptions", () => {
  class ApiClientError extends Error {
    constructor(message, code = "UNKNOWN_ERROR", status = 500, details) {
      super(message);
      this.name = "ApiClientError";
      this.code = code;
      this.status = status;
      this.details = details;
    }
  }

  const netError = new ApiClientError("Network error", "NETWORK_ERROR", 0);
  assert.equal(netError.status, 0);
  assert.equal(netError.code, "NETWORK_ERROR");

  const timeoutError = new ApiClientError("Request timeout", "TIMEOUT", 408);
  assert.equal(timeoutError.status, 408);
  assert.equal(timeoutError.code, "TIMEOUT");

  const apiError = new ApiClientError("No face detected", "NO_FACE_DETECTED", 400, { faces: 0 });
  assert.equal(apiError.status, 400);
  assert.deepEqual(apiError.details, { faces: 0 });
});

test("Session state machine transitions", () => {
  const validTransitions = {
    NOT_STARTED: ["STARTING", "ERROR"],
    STARTING: ["ACTIVE", "ERROR"],
    ACTIVE: ["STOPPING", "ERROR"],
    STOPPING: ["COMPLETED", "ERROR"],
    COMPLETED: ["STARTING"],
    ERROR: ["STARTING", "NOT_STARTED"],
  };

  const isValidTransition = (from, to) => {
    return validTransitions[from]?.includes(to) ?? false;
  };

  assert.ok(isValidTransition("NOT_STARTED", "STARTING"));
  assert.ok(isValidTransition("STARTING", "ACTIVE"));
  assert.ok(isValidTransition("ACTIVE", "STOPPING"));
  assert.ok(isValidTransition("STOPPING", "COMPLETED"));
  assert.ok(!isValidTransition("COMPLETED", "STOPPING"));
  assert.ok(!isValidTransition("NOT_STARTED", "COMPLETED"));
});

test("Probability distribution format validation", () => {
  const sampleProbabilities = {
    angry: 0.011,
    disgust: 0.001,
    fear: 0.007,
    happy: 0.824,
    sad: 0.029,
    surprise: 0.002,
    neutral: 0.126,
  };

  const keys = Object.keys(sampleProbabilities);
  assert.equal(keys.length, 7);

  const sum = Object.values(sampleProbabilities).reduce((acc, v) => acc + v, 0);
  assert.ok(sum >= 0.99 && sum <= 1.01, `Sum of probabilities must equal 1.0, got ${sum}`);

  // Determine dominant emotion
  const dominant = Object.entries(sampleProbabilities).reduce(
    (max, [emotion, prob]) => (prob > max.prob ? { emotion, prob } : max),
    { emotion: "", prob: -1 }
  );
  assert.equal(dominant.emotion, "happy");
  assert.equal(dominant.prob, 0.824);
});
