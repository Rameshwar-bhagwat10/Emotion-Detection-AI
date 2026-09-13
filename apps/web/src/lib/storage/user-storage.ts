/**
 * User storage utility for client-side privacy, user isolation, and session persistence.
 *
 * In production (when no centralized user database is configured), browser localStorage acts
 * as the authoritative store for user-created videos and live detection sessions.
 * In local mode, this layer acts as an isolation barrier preventing cross-user data leakage.
 */

export interface UserVideoRecord {
  video_id: string;
  filename: string;
  status: "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED" | "CANCELLED";
  current_stage?: string;
  progress_percent: number;
  duration_seconds: number;
  frames_analyzed: number;
  created_at: string;
  error_message?: string | null;
}

export interface UserSessionRecord {
  id: string;
  name: string;
  status: "active" | "completed" | "cancelled";
  started_at: string;
  ended_at?: string | null;
}

const USER_ID_KEY = "valence_client_user_id";
const USER_VIDEOS_KEY = "valence_user_videos";
const USER_SESSIONS_KEY = "valence_user_sessions";

function isBrowser(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

/**
 * Generate a random RFC4122 v4 compliant UUID in browser environments.
 */
function generateUUID(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Retrieve or generate a deterministic, private UUID for the current client browser.
 */
export function getOrCreateUserId(): string {
  if (!isBrowser()) {
    return "00000000-0000-0000-0000-000000000000";
  }

  try {
    let userId = localStorage.getItem(USER_ID_KEY);
    if (!userId || userId.trim() === "") {
      userId = generateUUID();
      localStorage.setItem(USER_ID_KEY, userId);
    }
    return userId;
  } catch (err) {
    console.warn("localStorage access denied for user ID:", err);
    return generateUUID();
  }
}

/* =========================================================================
   USER VIDEO ANALYSIS RECORDS
   ========================================================================= */

/**
 * Get all videos recorded by the current user in this browser.
 */
export function getUserVideos(): UserVideoRecord[] {
  if (!isBrowser()) return [];

  try {
    const data = localStorage.getItem(USER_VIDEOS_KEY);
    if (!data) return [];
    const parsed = JSON.parse(data);
    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    console.warn("Failed to read user videos from localStorage:", err);
    return [];
  }
}

/**
 * Get Set of video IDs owned by the current user.
 */
export function getUserVideoIds(): Set<string> {
  const videos = getUserVideos();
  return new Set(videos.map((v) => v.video_id));
}

/**
 * Save or prepend a new video record to the user's private collection.
 */
export function saveUserVideo(record: UserVideoRecord): void {
  if (!isBrowser()) return;

  try {
    const current = getUserVideos();
    const existingIndex = current.findIndex((v) => v.video_id === record.video_id);

    if (existingIndex >= 0) {
      current[existingIndex] = { ...current[existingIndex], ...record };
    } else {
      current.unshift(record);
    }

    // Keep up to 50 most recent videos per browser
    const trimmed = current.slice(0, 50);
    localStorage.setItem(USER_VIDEOS_KEY, JSON.stringify(trimmed));
  } catch (err) {
    console.warn("Failed to save user video to localStorage:", err);
  }
}

/**
 * Update attributes of an existing video record (e.g. progress, status, frames).
 */
export function updateUserVideo(videoId: string, patch: Partial<UserVideoRecord>): void {
  if (!isBrowser()) return;

  try {
    const current = getUserVideos();
    const index = current.findIndex((v) => v.video_id === videoId);
    if (index >= 0) {
      current[index] = { ...current[index], ...patch };
      localStorage.setItem(USER_VIDEOS_KEY, JSON.stringify(current));
    }
  } catch (err) {
    console.warn("Failed to update user video in localStorage:", err);
  }
}

/**
 * Remove a video from the user's private history.
 */
export function deleteUserVideo(videoId: string): void {
  if (!isBrowser()) return;

  try {
    const current = getUserVideos();
    const filtered = current.filter((v) => v.video_id !== videoId);
    localStorage.setItem(USER_VIDEOS_KEY, JSON.stringify(filtered));
  } catch (err) {
    console.warn("Failed to delete user video from localStorage:", err);
  }
}

/* =========================================================================
   USER LIVE DETECTION SESSIONS
   ========================================================================= */

/**
 * Get all live detection sessions recorded by the current user.
 */
export function getUserSessions(): UserSessionRecord[] {
  if (!isBrowser()) return [];

  try {
    const data = localStorage.getItem(USER_SESSIONS_KEY);
    if (!data) return [];
    const parsed = JSON.parse(data);
    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    console.warn("Failed to read user sessions from localStorage:", err);
    return [];
  }
}

/**
 * Get Set of session IDs owned by the current user.
 */
export function getUserSessionIds(): Set<string> {
  const sessions = getUserSessions();
  return new Set(sessions.map((s) => s.id));
}

/**
 * Save or prepend a new session record to the user's private collection.
 */
export function saveUserSession(record: UserSessionRecord): void {
  if (!isBrowser()) return;

  try {
    const current = getUserSessions();
    const existingIndex = current.findIndex((s) => s.id === record.id);

    if (existingIndex >= 0) {
      current[existingIndex] = { ...current[existingIndex], ...record };
    } else {
      current.unshift(record);
    }

    // Keep up to 50 most recent sessions per browser
    const trimmed = current.slice(0, 50);
    localStorage.setItem(USER_SESSIONS_KEY, JSON.stringify(trimmed));
  } catch (err) {
    console.warn("Failed to save user session to localStorage:", err);
  }
}

/**
 * Update attributes of an existing session record (e.g. status, ended_at).
 */
export function updateUserSession(sessionId: string, patch: Partial<UserSessionRecord>): void {
  if (!isBrowser()) return;

  try {
    const current = getUserSessions();
    const index = current.findIndex((s) => s.id === sessionId);
    if (index >= 0) {
      current[index] = { ...current[index], ...patch };
      localStorage.setItem(USER_SESSIONS_KEY, JSON.stringify(current));
    }
  } catch (err) {
    console.warn("Failed to update user session in localStorage:", err);
  }
}

/**
 * Remove a session from the user's private history.
 */
export function deleteUserSession(sessionId: string): void {
  if (!isBrowser()) return;

  try {
    const current = getUserSessions();
    const filtered = current.filter((s) => s.id !== sessionId);
    localStorage.setItem(USER_SESSIONS_KEY, JSON.stringify(filtered));
  } catch (err) {
    console.warn("Failed to delete user session from localStorage:", err);
  }
}

/**
 * Clear all private user data in this browser (reset).
 */
export function clearUserData(): void {
  if (!isBrowser()) return;

  try {
    localStorage.removeItem(USER_VIDEOS_KEY);
    localStorage.removeItem(USER_SESSIONS_KEY);
    localStorage.removeItem(USER_ID_KEY);
  } catch (err) {
    console.warn("Failed to clear user data in localStorage:", err);
  }
}
