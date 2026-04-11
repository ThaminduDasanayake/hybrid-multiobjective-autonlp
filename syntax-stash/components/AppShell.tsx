"use client";

import { useState } from "react";
import Header from "@/components/Header";
import CommandMenu from "@/components/CommandMenu";
import Sidebar from "@/components/Sidebar";
import { cn } from "@/lib/utils";

export default function AppShell({ children }: { children: React.ReactNode }) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Mobile overlay — sits between sidebar (z-50) and content */}
      <div
        className={cn(
          "fixed inset-0 z-40 bg-black/50 transition-opacity duration-300 md:hidden",
          sidebarOpen
            ? "opacity-100 pointer-events-auto"
            : "opacity-0 pointer-events-none"
        )}
        onClick={() => setSidebarOpen(false)}
        aria-hidden="true"
      />

      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <main className="flex-1 overflow-hidden flex flex-col min-w-0">
        <Header
          onSearchOpen={() => setSearchOpen(true)}
          onMenuOpen={() => setSidebarOpen((prev) => !prev)}
        />
        <div className="flex-1 overflow-y-auto p-8">{children}</div>
        <CommandMenu open={searchOpen} setOpen={setSearchOpen} />
      </main>
    </div>
  );
}
