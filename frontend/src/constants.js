import { History, Zap } from "lucide-react";
export const BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

export const NAV_ITEMS = [
  {
    to: "/",
    end: true,
    icon: Zap,
    label: "Run AutoML",
    description: "Launch a new run",
  },
  {
    to: "/history",
    icon: History,
    label: "Run History",
    description: "Browse past jobs",
  },
];

// Config Form
export const DATASETS = [
  { value: "imdb", label: "IMDB (sentiment, 2 classes)" },
  { value: "20newsgroups", label: "20 Newsgroups (topic, 20 classes)" },
  { value: "ag_news", label: "AG News (topic, 4 classes)" },
  { value: "banking77", label: "Banking77 (intent, 77 classes)" },
];

export const DEFAULTS = {
  dataset_name: "imdb",
  max_samples: 2000,
  population_size: 20,
  n_generations: 10,
  bo_calls: 15,
  optimization_mode: "multi_3d",
  disable_bo: false,
};

// Live Tracker
export const TERMINAL_STATES = new Set(["completed", "failed", "terminated"]);

// Shortened preset for a quick smoke-test demo (~3 min run).
export const DEMO_CONFIG = {
  ...DEFAULTS,
  max_samples: 500,
  population_size: 5,
  n_generations: 3,
  bo_calls: 10,
};

// Job Config Card
export const PARAMS_CONFIG = [
  { id: "dataset_name", label: "Dataset" },
  { id: "max_samples", label: "Max Samples" },
  { id: "population_size", label: "Population Size" },
  { id: "n_generations", label: "Generations" },
  { id: "bo_calls", label: "BO Calls" },
];

// Status Badge
export const STYLE_MAP = {
  created: { variant: "outline" },
  running: { variant: "outline", className: "border-primary/30 bg-primary/15 text-primary" },
  completed: { variant: "secondary" },
  failed: { variant: "destructive" },
  terminated: { variant: "ghost" },
};

export const LABEL_MAP = {
  created: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  terminated: "Terminated",
};

// Pareto Heatmap
export const MODEL_LABEL = { logistic: "LR", naive_bayes: "NB", svm: "SVM" };
