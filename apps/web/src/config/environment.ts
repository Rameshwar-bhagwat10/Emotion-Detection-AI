/**
 * Client and server environment configuration for web application.
 */

const isVercel =
  typeof window !== "undefined" &&
  (window.location.hostname.includes("vercel.app") || window.location.hostname !== "localhost");

export const env = {
  apiUrl:
    process.env.NEXT_PUBLIC_API_URL ||
    (isVercel || process.env.NODE_ENV === "production"
      ? "https://emotion-detection-api-6kys.onrender.com"
      : "http://localhost:8000"),
  wsUrl:
    process.env.NEXT_PUBLIC_WS_URL ||
    (isVercel || process.env.NODE_ENV === "production"
      ? "wss://emotion-detection-api-6kys.onrender.com/api/v1/realtime/emotion"
      : "ws://localhost:8000/api/v1/realtime/emotion"),
  appEnv: process.env.NEXT_PUBLIC_APP_ENV || (isVercel || process.env.NODE_ENV === "production" ? "production" : "development"),
  isProduction: process.env.NODE_ENV === "production" || isVercel,
  isDevelopment: process.env.NODE_ENV === "development" && !isVercel,
};
