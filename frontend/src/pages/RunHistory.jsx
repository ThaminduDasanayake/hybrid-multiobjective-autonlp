import { AlertCircle, BarChart3, Loader2 } from "lucide-react";
import { useDeleteJob, useJobs } from "../hooks/useApi";
import JobSelectorTable from "../components/JobSelectorTable";
import { Alert, AlertDescription } from "../components/ui/alert";

const RunHistory = () => {
  const { data: jobMap = {}, isLoading: jobsLoading, error: jobsError } = useJobs();
  const completedIds = Object.keys(jobMap);
  const deleteJobMutation = useDeleteJob();

  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-foreground">Run History</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Browse completed runs — click View to explore results and run experiments.
        </p>
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
            Run an AutoML experiment first, then return here to explore results.
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
  );
};

export default RunHistory;
