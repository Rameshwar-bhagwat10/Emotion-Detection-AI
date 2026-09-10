/**
 * TypeScript definitions for static image and stored prediction responses.
 */

import { EmotionType, PredictionEmotion } from "./emotion";

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface FacePrediction {
  face_id: number;
  bbox: BoundingBox;
  detection_confidence: number;
  emotion: PredictionEmotion;
  confidence: number;
  is_uncertain: boolean;
  probabilities: Record<EmotionType, number>;
}

export interface PredictionTiming {
  image_loading_ms: number;
  face_detection_ms: number;
  preprocessing_ms: number;
  inference_ms: number;
  postprocessing_ms: number;
  total_ms: number;
}

export interface ModelInfo {
  model_name: string;
  architecture: string;
  optimization_type: string;
  device: string;
}

export interface PredictionResponse {
  status: string;
  request_id: string;
  prediction_id?: string | null;
  session_id?: string | null;
  faces_detected: number;
  faces: FacePrediction[];
  timing: PredictionTiming;
  model_info: ModelInfo;
  created_at: string;
}

export interface StoredPredictionDetail {
  id: string;
  session_id?: string | null;
  request_id: string;
  model_version: string;
  status: string;
  faces_detected: number;
  processing_time_ms: number;
  image_width?: number | null;
  image_height?: number | null;
  created_at: string;
  faces: FacePrediction[];
}
