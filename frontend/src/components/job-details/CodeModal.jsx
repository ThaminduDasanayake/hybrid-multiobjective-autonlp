// Dialog that shows the generated sklearn pipeline code for a selected solution.

import { useState, useCallback, useMemo } from "react";
import { Copy, Check } from "lucide-react";
import { Button } from "@/components/ui/button.jsx";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog.jsx";
import { generateSklearnCode } from "@/utils/generateSklearnCode.js";
import { fmt } from "@/utils/formatters.js";

const CodeModal = ({ open, onClose, solution }) => {
  const [copied, setCopied] = useState(false);

  const code = useMemo(() => (solution ? generateSklearnCode(solution) : ""), [solution]);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for non-secure contexts
      const ta = document.createElement("textarea");
      ta.value = code;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [code]);

  if (!solution) return null;

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="max-w-3xl max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-foreground">
            Pipeline Code —{" "}
            <span className="font-mono text-primary text-base">{fmt.model(solution.model)}</span>
          </DialogTitle>
          <DialogDescription>
            Scikit-Learn code to reconstruct this pipeline. Copy and paste into your Python
            environment.
          </DialogDescription>
        </DialogHeader>

        {/* Code block */}
        <div className="relative flex-1 min-h-0">
          {/* Copy button — top-right corner of code block */}
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="absolute right-5 top-2 z-10 h-8 gap-1.5 rounded-md border border-border/60 bg-background/80 px-2.5 text-xs font-medium text-muted-foreground backdrop-blur-sm hover:bg-muted hover:text-foreground"
          >
            {copied ? (
              <>
                <Check className="size-3.5 text-secondary" />
                Copied
              </>
            ) : (
              <>
                <Copy className="size-3.5" />
                Copy
              </>
            )}
          </Button>

          <div className="overflow-auto rounded-lg border border-border bg-[hsl(var(--muted)/0.4)] max-h-[55vh]">
            <pre className="p-4 pr-24 text-[13px] leading-relaxed">
              <code className="font-mono text-foreground/90 whitespace-pre">{code}</code>
            </pre>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default CodeModal;
