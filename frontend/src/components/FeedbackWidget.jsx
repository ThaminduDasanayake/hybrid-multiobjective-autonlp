import React, { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Button } from "./ui/button";
import { MessageSquare, X, Send } from "lucide-react";

export default function FeedbackWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [name, setName] = useState("");
  const [message, setMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!message.trim()) return;

    setIsSubmitting(true);
    try {
      // The proxy in vite config handles /api requests
      const response = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, message }),
      });

      if (response.ok) {
        setIsSuccess(true);
        setTimeout(() => {
          setIsOpen(false);
          setIsSuccess(false);
          setName("");
          setMessage("");
        }, 2000); // 2 seconds delay to fade/collapse
      }
    } catch (error) {
      console.error("Failed to submit feedback:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) {
    return (
      <Button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 z-50 p-0 flex items-center justify-center"
      >
        <MessageSquare className="h-6 w-6" />
      </Button>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 w-80 z-50 animate-in fade-in slide-in-from-bottom-5 duration-300">
      <Card className="shadow-2xl border-border/50">
        <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0 relative">
          <CardTitle className="text-sm font-medium">Send us your feedback</CardTitle>
          <Button
            variant="ghost"
            className="h-8 w-8 absolute right-2 top-2 p-0 flex items-center justify-center"
            onClick={() => setIsOpen(false)}
          >
            <X className="h-4 w-4" />
          </Button>
        </CardHeader>
        <CardContent>
          {isSuccess ? (
            <div className="flex flex-col items-center justify-center py-6 text-center space-y-3 text-green-600 animate-in zoom-in duration-300">
              <div className="h-10 w-10 bg-green-100 rounded-full flex items-center justify-center">
                <Send className="h-5 w-5" />
              </div>
              <p className="font-medium">Thank you for your feedback!</p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4 pt-2">
              <div className="space-y-2">
                <Input
                  placeholder="Name (Optional)"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="text-sm"
                />
              </div>
              <div className="space-y-2">
                <Textarea
                  placeholder="What's on your mind?"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  required
                  className="min-h-[100px] text-sm resize-none"
                />
              </div>
              <Button type="submit" className="w-full" disabled={isSubmitting || !message.trim()}>
                {isSubmitting ? "Sending..." : "Submit"}
              </Button>
            </form>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
