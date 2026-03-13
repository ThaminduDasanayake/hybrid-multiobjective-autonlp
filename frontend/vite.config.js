import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  const backendUrl = env.VITE_API_BASE_URL || "http://localhost:8000";

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        "@": fileURLToPath(new URL("./src", import.meta.url)),
      },
    },
    server: {
      proxy: {
        // Forward all /api/* requests to the FastAPI backend.
        // This avoids CORS preflight during development — the browser sees
        // everything as same-origin (localhost:5173).
        // Target is read from VITE_API_BASE_URL (falls back to localhost:8000).
        "/api": {
          target: backendUrl,
          changeOrigin: true,
        },
      },
    },
  };
});
