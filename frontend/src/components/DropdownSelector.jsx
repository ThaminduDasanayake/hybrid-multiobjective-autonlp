import { Label } from "./ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

const DropdownSelector = ({ label, options = [], value, onChange, className }) => {
  const normalised = options.map((o) => (typeof o === "string" ? { value: o, label: o } : o));

  return (
    <div className={className ?? "relative max-w-md"}>
      {label && <Label className="mb-1.5 block">{label}</Label>}
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
};

export default DropdownSelector;
