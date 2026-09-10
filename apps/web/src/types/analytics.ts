/**
 * Strongly typed interfaces for Phase 13 Analytics, Session Metrics, Timeline, and History.
 */

import { EmotionType } from "./emotion";

export interface ExpressionCount {
  emotion: EmotionType | string;
  count: number;
  percentage: number;
}

export interface ExpressionDistribution {
  items: ExpressionCount[];
  dominant_emotion: string | null;
  total_predictions: number;
}

export interface ConfidenceBucket {
  range: string;
  min_val: number;
  max_val: number;
  count: number;
  percentage: number;
}

export interface ConfidenceAnalytics {
  average_confidence: number;
  min_confidence: number;
  max_confidence: number;
  distribution: ConfidenceBucket[];
  class_confidences: Record<string, number>;
  low_confidence_count: number;
  high_confidence_count: number;
}

export interface TimelineBucket {
  bucket_index: number;
  timestamp: string;
  relative_seconds: number;
  prediction_count: number;
  emotion_counts: Record<string, number>;
  dominant_emotion: string | null;
  average_confidence: number;
}

export interface TimelineAnalytics {
  session_id: string;
  bucket_seconds: number;
  total_buckets: number;
  buckets: TimelineBucket[];
}

export interface DailyTrend {
  date: string;
  predictions_count: number;
  sessions_count: number;
}

export interface GlobalAnalytics {
  total_sessions: number;
  total_predictions: number;
  total_faces: number;
  total_duration_seconds: number;
  dominant_expression: string | null;
  average_confidence: number;
  expression_distribution: ExpressionDistribution;
  confidence_analytics: ConfidenceAnalytics;
  model_version_distribution: Record<string, number>;
  recent_trends: DailyTrend[];
}

export interface SessionAnalytics {
  session_id: string;
  name: string | null;
  status: string;
  started_at: string;
  ended_at: string | null;
  duration_seconds: number;
  total_predictions: number;
  total_faces: number;
  prediction_rate_per_minute: number;
  dominant_expression: string | null;
  expression_distribution: ExpressionDistribution;
  confidence_analytics: ConfidenceAnalytics;
  model_versions: string[];
}

export interface SessionSummary {
  id: string;
  name: string | null;
  status: string;
  started_at: string;
  ended_at: string | null;
  duration_seconds: number;
  prediction_count: number;
  dominant_expression: string | null;
  average_confidence: number;
}

export interface SessionAnalyticsListResponse {
  sessions: SessionSummary[];
  total: number;
}

export interface HistoricalPrediction {
  prediction_id: string;
  session_id: string | null;
  request_id: string;
  timestamp: string;
  model_version: string;
  face_id: number;
  emotion: string;
  confidence: number;
  is_uncertain: boolean;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  probabilities: Record<string, number>;
  processing_time_ms: number;
}

export interface PaginatedPredictionsResponse {
  items: HistoricalPrediction[];
  total: number;
  limit: number;
  offset: number;
}

export interface PredictionFilters {
  session_id?: string;
  emotion?: string;
  is_uncertain?: boolean;
  min_confidence?: number;
  max_confidence?: number;
  start_date?: string;
  end_date?: string;
  model_version?: string;
  limit?: number;
  offset?: number;
  sort_by?: "created_at" | "confidence";
  order?: "asc" | "desc";
}
