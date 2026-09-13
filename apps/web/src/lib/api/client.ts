import axios, { AxiosError, AxiosInstance } from "axios";
import { env } from "@/config/environment";
import { ApiErrorResponse } from "@/types/api";

/**
 * Normalized application error object for consistent handling.
 */
export class ApiClientError extends Error {
  public code: string;
  public status: number;
  public details?: Record<string, unknown>;

  constructor(message: string, code: string = "UNKNOWN_ERROR", status: number = 500, details?: Record<string, unknown>) {
    super(message);
    this.name = "ApiClientError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

/**
 * Centralized Axios instance for FastAPI v1 backend communication.
 */
export const apiClient: AxiosInstance = axios.create({
  baseURL: `${env.apiUrl.replace(/\/+$/, "")}/api/v1`,
  timeout: 15000,
  headers: {
    Accept: "application/json",
  },
});

// Request interceptor to attach persistent client user ID for privacy isolation
apiClient.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    try {
      const userId = localStorage.getItem("valence_client_user_id");
      if (userId) {
        config.headers["X-User-Id"] = userId;
      }
    } catch {
      // Ignore localStorage access restrictions in restricted environments
    }
  }
  return config;
});

// Response interceptor to normalize error structures
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiErrorResponse>) => {
    if (error.response) {
      const data = error.response.data;
      const status = error.response.status;

      if (data && data.error) {
        throw new ApiClientError(
          data.error.message || "An error occurred with the inference API.",
          data.error.code || `HTTP_${status}`,
          status,
          data.error.details
        );
      }

      throw new ApiClientError(
        error.message || `Server returned error ${status}`,
        `HTTP_${status}`,
        status
      );
    } else if (error.code === "ECONNABORTED") {
      throw new ApiClientError(
        "Request timed out. The backend server took too long to respond.",
        "TIMEOUT",
        408
      );
    } else if (error.request) {
      throw new ApiClientError(
        "Unable to connect to the backend server. Please verify FastAPI is running at " + env.apiUrl,
        "NETWORK_ERROR",
        0
      );
    }

    throw new ApiClientError(error.message || "An unexpected error occurred.", "CLIENT_ERROR", 500);
  }
);
