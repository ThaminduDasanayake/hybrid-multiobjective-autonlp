import { Label } from "@/components/ui/label.jsx";
import { Input } from "@/components/ui/input.jsx";

const InputField = ({ label, hint, type = "text", value, onChange, ...rest }) => {
  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <Input type={type} value={value} onChange={onChange} {...rest} />
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
};

export default InputField;
