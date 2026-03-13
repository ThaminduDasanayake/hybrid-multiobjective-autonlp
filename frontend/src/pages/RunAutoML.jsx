import { useStore } from "../store";
import ConfigForm from "../components/run/ConfigForm";
import LiveTracker from "../components/run/LiveTracker";

export default function RunAutoML() {
  const { activeJobId, setActiveJobId, resetJob } = useStore();

  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-foreground">Run AutoML</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          {activeJobId
            ? "Live progress for the active optimisation run."
            : "Configure and launch a new multi-objective optimisation run."}
        </p>
      </div>

      {activeJobId ? (
        <LiveTracker jobId={activeJobId} onFinished={resetJob} />
      ) : (
        <ConfigForm onJobStarted={setActiveJobId} />
      )}
    </div>
  );
}
