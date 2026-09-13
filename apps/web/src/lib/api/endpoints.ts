import { apiClient } from "./client";
import { HealthResponse, ReadinessResponse, SessionListResponse, SessionResponse } from "@/types/api";
import { PredictionResponse, StoredPredictionDetail } from "@/types/prediction";
import {
  GlobalAnalytics,
  SessionAnalytics,
  SessionAnalyticsListResponse,
  TimelineAnalytics,
  ConfidenceAnalytics,
  PaginatedPredictionsResponse,
  PredictionFilters,
} from "@/types/analytics";

/**
 * Check basic health and dependency reachability.
 */
export async function getHealth(): Promise<HealthResponse> {
  const res = await apiClient.get<HealthResponse>("/health");
  return res.data;
}

/**
 * Check deep service readiness (ML model + database).
 */
export async function getReadiness(): Promise<ReadinessResponse> {
  const res = await apiClient.get<ReadinessResponse>("/health/ready");
  return res.data;
}

/**
 * Ingest image file upload, execute Phase 09 ML inference, and return predictions.
 */
export async function predictImage(
  file: File | Blob,
  sessionId?: string,
  fileName: string = "upload.jpg"
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("image", file, fileName);
  if (sessionId) {
    formData.append("session_id", sessionId);
  }

  const res = await apiClient.post<PredictionResponse>("/predictions", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return res.data;
}

/**
 * Retrieve a stored prediction by UUID.
 */
export async function getPrediction(predictionId: string): Promise<StoredPredictionDetail> {
  const res = await apiClient.get<StoredPredictionDetail>(`/predictions/${predictionId}`);
  return res.data;
}

/**
 * Create a new analysis session.
 */
export async function createSession(name?: string, userId?: string): Promise<SessionResponse> {
  const payload: { name: string; user_id?: string } = {
    name: name || `Session ${new Date().toLocaleTimeString()}`,
  };
  if (userId) {
    payload.user_id = userId;
  }
  const res = await apiClient.post<SessionResponse>("/sessions", payload);
  return res.data;
}

/**
 * List recent analysis sessions with pagination and optional user isolation filter.
 */
export async function listSessions(
  limit: number = 20,
  offset: number = 0,
  userId?: string
): Promise<SessionListResponse> {
  const params: Record<string, unknown> = { limit, offset };
  if (userId) {
    params.user_id = userId;
  }
  const res = await apiClient.get<SessionListResponse>("/sessions", {
    params,
  });
  return res.data;
}

/**
 * Retrieve a single analysis session by UUID.
 */
export async function getSession(sessionId: string): Promise<SessionResponse> {
  const res = await apiClient.get<SessionResponse>(`/sessions/${sessionId}`);
  return res.data;
}

/**
 * End/finalize an active analysis session.
 */
export async function endSession(sessionId: string): Promise<SessionResponse> {
  const res = await apiClient.post<SessionResponse>(`/sessions/${sessionId}/end`);
  return res.data;
}

/**
 * Get all prediction records for a specific session.
 */
export async function getSessionPredictions(
  sessionId: string,
  limit: number = 100
): Promise<StoredPredictionDetail[]> {
  const res = await apiClient.get<StoredPredictionDetail[]>(`/sessions/${sessionId}/predictions`, {
    params: { limit },
  });
  return res.data;
}

/**
 * Retrieve system-wide global analytics and trends.
 */
export async function getGlobalAnalytics(filters?: {
  start_date?: string;
  end_date?: string;
  model_version?: string;
}): Promise<GlobalAnalytics> {
  const res = await apiClient.get<GlobalAnalytics>("/analytics/overview", {
    params: filters,
  });
  return res.data;
}

/**
 * Retrieve list of historical sessions enriched with computed summary analytics.
 */
export async function getSessionsAnalytics(
  limit: number = 20,
  offset: number = 0,
  status?: string
): Promise<SessionAnalyticsListResponse> {
  const res = await apiClient.get<SessionAnalyticsListResponse>("/analytics/sessions", {
    params: { limit, offset, status },
  });
  return res.data;
}

/**
 * Retrieve session-isolated metrics and emotion distribution.
 */
export async function getSessionAnalytics(sessionId: string): Promise<SessionAnalytics> {
  const res = await apiClient.get<SessionAnalytics>(`/analytics/sessions/${sessionId}`);
  return res.data;
}

/**
 * Retrieve time-bucketed timeline predictions for session visualization.
 */
export async function getSessionTimeline(
  sessionId: string,
  bucketSeconds?: number
): Promise<TimelineAnalytics> {
  const res = await apiClient.get<TimelineAnalytics>(`/analytics/sessions/${sessionId}/timeline`, {
    params: bucketSeconds ? { bucket_seconds: bucketSeconds } : undefined,
  });
  return res.data;
}

/**
 * Retrieve session confidence histogram and uncertainty statistics.
 */
export async function getSessionConfidence(sessionId: string): Promise<ConfidenceAnalytics> {
  const res = await apiClient.get<ConfidenceAnalytics>(`/analytics/sessions/${sessionId}/confidence`);
  return res.data;
}

/**
 * Retrieve historical predictions with filtering and pagination.
 */
export async function getPredictionHistory(
  filters?: PredictionFilters
): Promise<PaginatedPredictionsResponse> {
  const res = await apiClient.get<PaginatedPredictionsResponse>("/history/predictions", {
    params: filters,
  });
  return res.data;
}

// ---------------------------------------------------------------------------
// Video Analysis Endpoints
// ---------------------------------------------------------------------------

import {
  RecentVideoItem,
  VideoAnalysisDetail,
  VideoAnalysisJob,
  VideoAnalysisStatus,
  VideoPredictionsResponse,
  VideoTimelineResponse,
} from "@/types/video-analysis";

/**
 * Upload video file and initiate asynchronous temporal analysis.
 */
export async function uploadVideoForAnalysis(
  file: File | Blob,
  samplingFps: number = 5.0,
  sessionId?: string,
  fileName: string = "video.mp4"
): Promise<VideoAnalysisJob> {
  const formData = new FormData();
  formData.append("video", file, fileName);
  formData.append("sampling_fps", samplingFps.toString());
  if (sessionId) {
    formData.append("session_id", sessionId);
  }

  const res = await apiClient.post<VideoAnalysisJob>("/video/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    timeout: 300000, // 5 minutes timeout for large video container uploads
  });
  return res.data;
}

/**
 * Poll progress and execution status of a video analysis job.
 */
export async function getVideoAnalysisStatus(videoId: string): Promise<VideoAnalysisStatus> {
  const res = await apiClient.get<VideoAnalysisStatus>(`/video/${videoId}/status`);
  return res.data;
}

/**
 * Retrieve detailed report, metadata, and aggregated analytics for a video.
 */
export async function getVideoAnalysisDetail(videoId: string): Promise<VideoAnalysisDetail> {
  const res = await apiClient.get<VideoAnalysisDetail>(`/video/${videoId}`);
  return res.data;
}

/**
 * Retrieve chronological expression segments and transition events for timeline visualization.
 */
export async function getVideoTimeline(
  videoId: string,
  trackId?: number
): Promise<VideoTimelineResponse> {
  const res = await apiClient.get<VideoTimelineResponse>(`/video/${videoId}/timeline`, {
    params: trackId !== undefined ? { track_id: trackId } : undefined,
  });
  return res.data;
}

/**
 * Query paginated frame-level predictions.
 */
export async function getVideoPredictions(
  videoId: string,
  params?: {
    track_id?: number;
    start_time?: number;
    end_time?: number;
    page?: number;
    page_size?: number;
  }
): Promise<VideoPredictionsResponse> {
  const res = await apiClient.get<VideoPredictionsResponse>(`/video/${videoId}/predictions`, {
    params,
  });
  return res.data;
}

/**
 * Query recently analyzed videos for historical selection and playback.
 */
export async function getRecentVideos(limit: number = 20): Promise<RecentVideoItem[]> {
  const res = await apiClient.get<RecentVideoItem[]>("/video/recent", {
    params: { limit },
  });
  return res.data;
}

/**
 * Cancel an active video analysis job.
 */
export async function cancelVideoAnalysis(videoId: string): Promise<{ video_id: string; status: string }> {
  const res = await apiClient.post<{ video_id: string; status: string }>(`/video/${videoId}/cancel`);
  return res.data;
}

/**
 * Get the direct streaming URL for HTML5 video player.
 * Guarantees /api/v1/video/{videoId}/stream is correctly formatted.
 */
export function getVideoStreamUrl(videoId: string): string {
  const rawUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const baseUrl = rawUrl.replace(/\/+$/, "");
  const apiPrefix = baseUrl.endsWith("/api/v1") ? "" : "/api/v1";
  return `${baseUrl}${apiPrefix}/video/${videoId}/stream`;
}


