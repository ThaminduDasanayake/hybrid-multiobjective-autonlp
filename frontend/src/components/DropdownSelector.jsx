/**
 * DropdownSelector
 *
 * A labelled shadcn <Select> used throughout the app for job/dataset pickers.
 *
 * Props:
 *   label     – text rendered above the trigger (omit to hide)
 *   options   – string[] | { value: string; label: string }[]
 *   value     – currently selected value
 *   onChange  – (value: string) => void
 *   className – overrides the outer wrapper div class (default: "relative max-w-md")
 */

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

export default function DropdownSelector({ label, options = [], value, onChange, className }) {
  const normalised = options.map((o) => (typeof o === "string" ? { value: o, label: o } : o));

  return (
    <div className={className ?? "relative max-w-md"}>
      {label && <label className="mb-1.5 block text-sm font-medium text-foreground">{label}</label>}
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent position="popper">
          {normalised.map((opt) => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
