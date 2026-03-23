import { useNavigate } from "react-router-dom";
import {
  AlertCircle,
  BarChart3,
  Clock,
  Eye,
  FlaskConical,
  Loader2,
  Rocket,
  Target,
} from "lucide-react";
import { useDeleteJob, useJobs } from "../hooks/useApi";
import { useStore } from "../store";
import JobSelectorTable from "../components/JobSelectorTable";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";

const OBJECTIVES = [
  {
    icon: Target,
    title: "F1 Score",
    description: "Maximises classification accuracy using macro-averaged F1 across all classes",
  },
  {
    icon: Clock,
    title: "Latency",
    description: "Minimises inference time to meet real-time deployment constraints",
  },
  {
    icon: Eye,
    title: "Interpretability",
    description: "Favours transparent, explainable models over opaque black-box classifiers",
  },
];

const Home = () => {
  const navigate = useNavigate();
  const activeJobId = useStore((s) => s.activeJobId);

  const { data: jobMap = {}, isLoading: jobsLoading, error: jobsError } = useJobs();
  const completedIds = Object.keys(jobMap);
  const deleteJobMutation = useDeleteJob();

  return (
    <div className="mx-auto max-w-6xl p-8">
      {/* Active job banner */}
      {activeJobId && (
        <button
          onClick={() => navigate("/run")}
          className="mb-6 flex w-full items-center gap-3 rounded-lg border border-primary/30 bg-primary/10 px-4 py-3 text-left text-sm transition-colors hover:bg-primary/15"
        >
          <span className="relative flex h-2.5 w-2.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-primary" />
          </span>
          <span className="font-medium text-foreground">Job running</span>
          <span className="text-muted-foreground">— View progress</span>
        </button>
      )}

      {/* Hero */}
      <div className="mb-10">
        <div className="flex items-center gap-2.5 mb-2">
          <FlaskConical size={28} className="text-primary" />
          <h1 className="text-3xl font-bold text-foreground">T-AutoNLP</h1>
        </div>
        <p className="text-muted-foreground max-w-2xl">
          A multi-objective AutoML system for text classification. T-AutoNLP automates the
          construction of end-to-end NLP pipelines — exploring combinations of vectorisers, scalers,
          dimensionality reduction, and classifiers — using NSGA-II genetic algorithms and Bayesian
          optimisation to find Pareto-optimal trade-offs across three objectives simultaneously.
        </p>

        <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
          {OBJECTIVES.map(({ icon: Icon, title, description }) => (
            <div
              key={title}
              className="rounded-lg border border-border bg-card px-4 py-3"
            >
              <div className="flex items-center gap-2 mb-1">
                <Icon size={15} className="text-primary" />
                <span className="text-sm font-semibold text-foreground">{title}</span>
              </div>
              <p className="text-xs text-muted-foreground">{description}</p>
            </div>
          ))}
        </div>

        <div className="mt-6 flex justify-center">
          <Button onClick={() => navigate("/run")} size="lg">
            <Rocket size={16} />
            Start Experiment
          </Button>
        </div>
      </div>

      {/* Run History */}
      <div>
        <div className="mb-4 flex items-center gap-3">
          <h2 className="text-lg font-semibold text-foreground">Run History</h2>
          {!jobsLoading && completedIds.length > 0 && (
            <span className="rounded-full bg-muted px-2.5 py-0.5 text-xs text-muted-foreground">
              {completedIds.length} run{completedIds.length !== 1 ? "s" : ""}
            </span>
          )}
        </div>

        {jobsLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 size={15} className="animate-spin" />
            Loading jobs…
          </div>
        ) : jobsError ? (
          <Alert variant="destructive">
            <AlertCircle />
            <AlertDescription>{jobsError.message}</AlertDescription>
          </Alert>
        ) : completedIds.length === 0 ? (
          <div className="rounded-xl border-2 border-dashed border-border bg-card p-12 text-center">
            <BarChart3
              className="mx-auto mb-3 text-muted-foreground/30"
              size={36}
              strokeWidth={1}
            />
            <p className="text-sm font-medium text-muted-foreground">No completed runs yet</p>
            <p className="mt-1 text-xs text-muted-foreground/60">
              Start an experiment to see results here.
            </p>
          </div>
        ) : (
          <JobSelectorTable
            jobs={jobMap}
            onDelete={(id) => deleteJobMutation.mutate(id)}
            isDeleting={deleteJobMutation.isPending}
          />
        )}
      </div>
    </div>
  );
};

export default Home;
