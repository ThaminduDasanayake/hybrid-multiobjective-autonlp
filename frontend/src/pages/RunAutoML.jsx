import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { useStore } from "../store";
import { Button } from "../components/ui/button";
import ConfigForm from "../components/run-automl/ConfigForm.jsx";
import LiveTracker from "../components/run-automl/LiveTracker.jsx";

const RunAutoML = () => {
  const navigate = useNavigate();
  const { activeJobId, setActiveJobId, resetJob } = useStore();

  return (
    <div className="mx-auto max-w-5xl p-8">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => navigate("/")}
        className="mb-3 -ml-2 text-muted-foreground"
      >
        <ArrowLeft size={14} />
        Back to Home
      </Button>

      <div className="mb-6">
        <h1 className="text-2xl font-bold text-foreground">Run AutoML</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          {activeJobId
            ? "Live progress for the active optimization run."
            : "Configure and launch a new multi-objective optimization run."}
        </p>
      </div>

      {activeJobId ? (
        <LiveTracker jobId={activeJobId} onFinished={resetJob} />
      ) : (
        <ConfigForm onJobStarted={setActiveJobId} />
      )}
    </div>
  );
};

export default RunAutoML;
