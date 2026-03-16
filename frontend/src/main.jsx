import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "./index.css";
import App from "./App.jsx";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Don't refetch just because the user switched browser tabs — the backend
      // is not a real-time data source (results only change when a job finishes).
      refetchOnWindowFocus: false,
      // One automatic retry on network errors is enough; more would spam a
      // long-running FastAPI process unnecessarily.
      retry: 1,
      // Surface errors to components rather than throwing to an error boundary.
      throwOnError: false,
    },
    mutations: {
      // Don't retry POSTs automatically — starting a second job or a second
      // ablation because of a transient error would be surprising.
      retry: 0,
    },
  },
});

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
