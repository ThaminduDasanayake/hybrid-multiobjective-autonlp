import { Label } from "@/components/ui/label.jsx";
import { Slider } from "@/components/ui/slider.jsx";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip.jsx";
import { Info } from "lucide-react";

const SliderField = ({ label, hint, value, min, max, step, onChange }) => {
  const id = `slider-${label.toLowerCase().replace(/\s+/g, "-")}`;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Label htmlFor={id}>{label}</Label>
          {hint && (
            <TooltipProvider delayDuration={200}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-muted-foreground hover:text-foreground transition-colors cursor-help">
                    <Info className="h-4 w-4" />
                    <span className="sr-only">Field info</span>
                  </span>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs text-sm" side="top">
                  <p>{hint}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
        <span className="font-mono text-sm font-semibold text-foreground tabular-nums">
          {Number(value).toLocaleString()}
        </span>
      </div>

      <Slider
        id={id}
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([v]) => onChange(v)}
      />

    </div>
  );
};
export default SliderField;
