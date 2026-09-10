import { PredictionEmotion } from "./emotion";
import { BoundingBox } from "./prediction";

export interface VideoMetadata {
  filename: string;
  duration_seconds: number;
  source_fps: number;
  analysis_fps: number;
  width: number;
  height: number;
  total_frames: number;
  frames_analyzed: number;
}

export interface VideoTrack {
  track_id: number;
  first_seen_sec: number;
  last_seen_sec: number;
  total_detections: number;
}

export interface ExpressionSegment {
  id: string;
  track_id: number;
  emotion: PredictionEmotion;
  start_time: number;
  end_time: number;
  duration: number;
  average_confidence: number;
  min_confidence: number;
  max_confidence: number;
  prediction_count: number;
}

export interface ExpressionEvent {
  track_id: number;
  timestamp: number;
  from_emotion: PredictionEmotion;
  to_emotion: PredictionEmotion;
  confidence: number;
}

export interface VideoPredictionItem {
  id: string;
  track_id: number;
  timestamp: number;
  frame_index: number;
  bbox: BoundingBox;
  detection_confidence: number;
  raw_emotion: string;
  raw_confidence: number;
  smoothed_emotion: PredictionEmotion;
  smoothed_confidence: number;
  is_uncertain: boolean;
  probabilities: Record<string, number>;
  raw_probabilities: Record<string, number>;
}

export interface VideoDistributionItem {
  emotion: PredictionEmotion;
  count: number;
  percentage: number;
  time_seconds: number;
  time_share_percent: number;
}

export interface VideoAnalyticsSummary {
  duration_seconds: number;
  tracked_faces_count: number;
  total_predictions_count: number;
  total_transitions_count: number;
  dominant_expression: string;
  dominant_expression_time_share: number;
  average_confidence: number;
  processing_time_seconds: number;
  processing_ratio: number;
  analysis_fps: number;
  expression_distribution: VideoDistributionItem[];
  transition_counts: Record<string, number>;
}

export interface VideoAnalysisJob {
  video_id: string;
  session_id: string | null;
  filename: string;
  status: "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED" | "CANCELLED";
  current_stage: string;
  message: string;
}

export interface VideoAnalysisStatus {
  video_id: string;
  status: "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED" | "CANCELLED";
  current_stage: string;
  progress_percent: number;
  frames_analyzed: number;
  total_frames: number;
  error_message: string | null;
}

export interface VideoTimelineResponse {
  video_id: string;
  track_id: number | null;
  segments: ExpressionSegment[];
  events: ExpressionEvent[];
}

export interface VideoAnalysisDetail {
  video_id: string;
  session_id: string | null;
  status: "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED" | "CANCELLED";
  current_stage: string;
  progress_percent: number;
  metadata: VideoMetadata;
  analytics: VideoAnalyticsSummary | null;
  tracks: VideoTrack[];
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
}

export interface VideoPredictionsResponse {
  video_id: string;
  track_id: number | null;
  total: number;
  page: number;
  page_size: number;
  predictions: VideoPredictionItem[];
}

export interface RecentVideoItem {
  video_id: string;
  filename: string;
  status: "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED" | "CANCELLED";
  current_stage: string;
  progress_percent: number;
  duration_seconds: number;
  created_at: string | null;
  frames_analyzed: number;
  error_message: string | null;
}

