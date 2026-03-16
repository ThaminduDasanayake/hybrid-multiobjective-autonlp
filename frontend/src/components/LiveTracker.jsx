import { useEffect, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, Loader2, RotateCcw, Square, Terminal } from "lucide-react";
import { streamUrl } from "@/api.js";
import { useCancelJob } from "@/hooks/useApi.js";
import { Alert, AlertDescription } from "./ui/alert.jsx";
import { Button } from "./ui/button.jsx";
import { Progress } from "./ui/progress.jsx";
import StatCard from "./StatCard.jsx";
import StatusBadge from "./StatusBadge.jsx";

const TERMINAL_STATES = new Set(["completed", "failed", "terminated"]);

const LiveTracker = ({ jobId, onFinished }) => {
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("Connecting…");
  const [jobStatus, setJobStatus] = useState("created");
  const [metrics, setMetrics] = useState({
    current_generation: 0,
    total_generations: 0,
    best_f1: 0,
    best_latency_ms: 0,
    cache_hit_rate: 0,
    total_evaluated: 0,
  });
  const [logs, setLogs] = useState([]);
  const [sseError, setSseError] = useState(null);

  const cancelJobMutation = useCancelJob();
  const terminalRef = useRef(null);

  // Auto-scroll the terminal to the bottom whenever logs change.
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  // Open the SSE connection.
  useEffect(() => {
    const es = new EventSource(streamUrl(jobId));

    es.onmessage = (e) => {
      const payload = JSON.parse(e.data);

      if (payload.error) {
        setSseError(payload.error);
        es.close();
        return;
      }

      const { status, logs: newLogs } = payload;

      setProgress(status.progress ?? 0);
      setMessage(status.message ?? "");
      setJobStatus(status.status ?? "created");
      setMetrics({
        current_generation: status.current_generation ?? 0,
        total_generations: status.total_generations ?? 0,
        best_f1: status.best_f1 ?? 0,
        best_latency_ms: status.best_latency_ms ?? 0,
        cache_hit_rate: status.cache_hit_rate ?? 0,
        total_evaluated: status.total_evaluated ?? 0,
      });

      if (newLogs) setLogs(newLogs);
    };

    es.onerror = () => {
      // EventSource fires onerror when the server closes the stream on a
      // terminal state — expected, not an error.
      es.close();
    };

    return () => es.close();
  }, [jobId]);

  const handleCancel = () => {
    cancelJobMutation.mutate(jobId);
  };

  const isTerminal = TERMINAL_STATES.has(jobStatus);

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <StatusBadge status={jobStatus} />
            <span className="truncate font-mono text-xs text-muted-foreground">{jobId}</span>
          </div>
          <p className="mt-1 truncate text-sm text-muted-foreground">{message}</p>
        </div>

        <div className="flex shrink-0 gap-2">
          {!isTerminal && (
            <Button
              variant="destructive"
              size="sm"
              onClick={handleCancel}
              disabled={cancelJobMutation.isPending}
            >
              {cancelJobMutation.isPending ? <Loader2 className="animate-spin" /> : <Square />}
              Stop
            </Button>
          )}

          {isTerminal && (
            <Button size="sm" onClick={onFinished}>
              <RotateCcw />
              New Run
            </Button>
          )}
        </div>
      </div>

      {/*Progress bar*/}
      <div>
        <div className="mb-1.5 flex items-center justify-between text-xs text-muted-foreground">
          <span>Progress</span>
          <span className="tabular-nums font-medium text-foreground">{progress}%</span>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      {/*Stat cards*/}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <StatCard
          label="Generation"
          value={`${metrics.current_generation} / ${metrics.total_generations}`}
          dimmed={metrics.current_generation === 0}
        />
        <StatCard
          label="Cache Hit Rate"
          value={metrics.cache_hit_rate > 0 ? metrics.cache_hit_rate.toFixed(1) : "—"}
          unit={metrics.cache_hit_rate > 0 ? "%" : undefined}
          dimmed={metrics.cache_hit_rate === 0}
        />
        <StatCard
          label="Evaluated"
          value={metrics.total_evaluated || "—"}
          unit={metrics.total_evaluated > 0 ? "pipelines" : undefined}
          dimmed={metrics.total_evaluated === 0}
        />
        <StatCard
          label="Best F1"
          value={metrics.best_f1 > 0 ? metrics.best_f1.toFixed(4) : "—"}
          dimmed={metrics.best_f1 === 0}
        />
        <StatCard
          label="Best Speed"
          value={metrics.best_latency_ms > 0 ? metrics.best_latency_ms.toFixed(2) : "—"}
          unit={metrics.best_latency_ms > 0 ? "ms" : undefined}
          dimmed={metrics.best_latency_ms === 0}
        />
      </div>
      {/* ── Terminal window ─────────────────────────── */}
      <div className="overflow-hidden rounded-xl border border-border">
        <div className="flex items-center gap-1.5 bg-muted px-4 py-2">
          <span className="h-3 w-3 rounded-full bg-destructive/60" />
          <span className="h-3 w-3 rounded-full bg-yellow-400/60" />
          <span className="h-3 w-3 rounded-full bg-green-400/60" />
          <span className="mx-auto font-mono text-xs text-muted-foreground">run_{jobId}.log</span>
          <Terminal size={12} className="shrink-0 text-muted-foreground" />
        </div>

        <div
          ref={terminalRef}
          role="log"
          aria-label="Job terminal output"
          className="h-80 overflow-y-auto bg-background p-4 font-mono text-xs leading-relaxed"
        >
          {logs.length === 0 ? (
            <span className="text-muted-foreground/40">Waiting for output…</span>
          ) : (
            logs.map((line, i) => (
              <div key={i} className="text-primary">
                <span className="mr-2 select-none text-muted-foreground/30">›</span>
                {line}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Error */}
      {sseError && (
        <Alert variant="destructive">
          <AlertCircle />
          <AlertDescription>Stream error: {sseError}</AlertDescription>
        </Alert>
      )}
      {/* Terminal state banners */}
      {jobStatus === "completed" && (
        <Alert className="border-secondary/30 bg-secondary/10 text-secondary [&>svg]:text-secondary">
          <CheckCircle2 />
          <AlertDescription>
            Optimisation complete — go to <strong>History &amp; Analysis</strong> to explore the
            Pareto front.
          </AlertDescription>
        </Alert>
      )}
      {jobStatus === "failed" && (
        <Alert variant="destructive">
          <AlertCircle />
          <AlertDescription>
            Job failed. Check the terminal output above for details.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default LiveTracker;
