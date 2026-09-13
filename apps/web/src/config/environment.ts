/**
 * Client and server environment configuration for web application.
 */

const isClient = typeof window !== "undefined";
const isLocal =
  isClient &&
  (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");

export const env = {
  apiUrl: isLocal
    ? "http://localhost:8000"
    : process.env.NEXT_PUBLIC_API_URL || "https://emotion-detection-api-6kys.onrender.com",
  wsUrl: isLocal
    ? "ws://localhost:8000/api/v1/realtime/emotion"
    : process.env.NEXT_PUBLIC_WS_URL || "wss://emotion-detection-api-6kys.onrender.com/api/v1/realtime/emotion",
  appEnv: isLocal ? "local-production" : "production",
  isProduction: true,
  isDevelopment: false,
};
