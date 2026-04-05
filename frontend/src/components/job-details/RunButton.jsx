import { Button } from "@/components/ui/button.jsx";
import { Loader2, Play } from "lucide-react";

/** Button to queue a missing ablation. Shows "Queued…" spinner once pressed. */
const RunButton = ({ label, queued, onClick }) => {
  return (
    <Button
      variant="outline"
      onClick={onClick}
      disabled={queued}
      className="border-primary/30 bg-primary/5 text-primary hover:bg-primary/10 hover:text-primary"
    >
      {queued ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
      {queued ? "Queued…" : label}
    </Button>
  );
};
export default RunButton;
