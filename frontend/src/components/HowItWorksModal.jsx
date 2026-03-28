import { Button } from "@/components/ui/button.jsx";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog.jsx";
import { Download, GitBranch, HelpCircle, Settings } from "lucide-react";

export default function HowItWorksModal() {
  const steps = [
    {
      icon: <Settings className="h-5 w-5 text-primary" />,
      title: "1. Configure Search",
      description:
        "Define your constraints (like max latency or dataset size) and select your optimization mode. We support fast 2D and rigorous 3D multi-objective Pareto optimization.",
    },
    {
      icon: <GitBranch className="h-5 w-5 text-primary" />,
      title: "2. Multi-Objective Optimization",
      description:
        "Our pipeline uses Bayesian Optimization to warm-start a Genetic Algorithm, balancing F1 Score, Latency, and Interpretability simultaneously.",
    },
    {
      icon: <Download className="h-5 w-5 text-primary" />,
      title: "3. Zero-Lock-in Export",
      description:
        "Once the pipeline converges, explore the Pareto front of models and instantly export the pure Scikit-Learn Python code for any solution you choose.",
    },
  ];

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="lg" className="font-medium">
          <HelpCircle size={16} />
          How it works
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md md:max-w-lg">
        <DialogHeader>
          <DialogTitle className="text-xl">How T-AutoNLP Works</DialogTitle>
          <DialogDescription>
            An intelligent pipeline designed to find the perfect balance between predictive power
            and hardware efficiency.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-6 py-4">
          {steps.map((step, index) => (
            <div key={index} className="flex gap-4 items-start">
              <div className="mt-1 bg-primary/10 p-2.5 rounded-full shrink-0">{step.icon}</div>
              <div>
                <h4 className="font-semibold text-foreground tracking-tight">{step.title}</h4>
                <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
                  {step.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
