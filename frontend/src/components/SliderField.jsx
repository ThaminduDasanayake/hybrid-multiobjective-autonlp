import { Label } from "@/components/ui/label.jsx";
import { Slider } from "@/components/ui/slider.jsx";

const SliderField = ({ label, hint, value, min, max, step, onChange }) => {
  const id = `slider-${label.toLowerCase().replace(/\s+/g, "-")}`;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label htmlFor={id}>{label}</Label>
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

      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
};
export default SliderField;
