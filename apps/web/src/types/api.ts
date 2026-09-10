/**
 * TypeScript definitions for FastAPI v1 contracts, health probes, and session management.
 */

export interface DependencyHealth {
  status: "healthy" | "unhealthy" | "unavailable" | "degraded" | "unconfigured";
  database?: string;
  host?: string;
  port?: number;
  supabase_url?: string;
  latency_ms?: number;
  error?: string;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "unhealthy";
  service: string;
  version: string;
  environment: string;
  dependencies: {
    database: DependencyHealth;
    supabase: DependencyHealth;
    redis: DependencyHealth;
  };
}

export interface ReadinessResponse {
  status: "ready" | "not_ready";
  service: string;
  version: string;
  model: "ready" | "not_ready";
  database: "ready" | "not_ready";
  details: {
    model_version: string;
    device: string;
    database: DependencyHealth;
  };
}

export interface SessionResponse {
  id: string;
  user_id?: string | null;
  name?: string | null;
  status: "active" | "completed" | "cancelled";
  started_at: string;
  ended_at?: string | null;
  created_at: string;
}

export interface SessionListResponse {
  sessions: SessionResponse[];
  total: number;
}

export interface SessionCreateRequest {
  name?: string;
  user_id?: string;
}

export interface ApiErrorResponse {
  error: {
    code: string;
    message: string;
    request_id?: string;
    details?: Record<string, unknown>;
  };
}
