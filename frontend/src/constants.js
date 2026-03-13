import { GraduationCap, History, Zap } from "lucide-react";
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
    label: "History & Analysis",
    description: "Browse past jobs",
  },
  {
    to: "/defense",
    icon: GraduationCap,
    label: "Thesis Defense",
    description: "Presentation mode",
  },
];
export const DATASETS = [
  { value: "imdb", label: "IMDB (sentiment, 2 classes)" },
  { value: "20newsgroups", label: "20 Newsgroups (topic, 20 classes)" },
  { value: "ag_news", label: "AG News (topic, 4 classes)" },
  { value: "banking77", label: "Banking77 (intent, 77 classes)" },
];

export const OPT_MODES = [
  { value: "multi_3d", label: "3D: F1, Latency, Interpretability" },
  { value: "multi_2d", label: "2D: F1, Latency" },
  { value: "single_f1", label: "Single: F1 only" },
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

// Shortened preset for a quick smoke-test demo (~3 min run).
export const DEMO_CONFIG = {
  ...DEFAULTS,
  max_samples: 500,
  population_size: 5,
  n_generations: 3,
  bo_calls: 5,
};

export const PARAM_KEYS = [
  "dataset_name",
  "max_samples",
  "population_size",
  "n_generations",
  "bo_calls",
];

export const PARAM_LABELS = {
  dataset_name: "Dataset",
  max_samples: "Max Samples",
  population_size: "Population Size",
  n_generations: "Generations",
  bo_calls: "BO Calls",
};
