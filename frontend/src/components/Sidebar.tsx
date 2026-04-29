"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import ThemeToggle from "./ThemeToggle";
import BackendStatus from "./BackendStatus";

const NAV = [
  { href: "/", label: "Problem" },
  { href: "/classical", label: "Classical" },
  { href: "/qaoa", label: "Quantum (Sim)" },
  { href: "/landscape", label: "Landscape" },
  { href: "/hardware", label: "Hardware" },
  { href: "/trials", label: "Trials" },
  { href: "/judge", label: "Judge Mode" },
];

export default function Sidebar() {
  const path = usePathname();
  return (
    <aside
      className="w-56 shrink-0 border-r flex flex-col py-6 px-4 sticky top-0 h-screen"
      style={{ background: "var(--background-alt)" }}
    >
      <div className="px-2 mb-6">
        <div className="display text-base leading-tight">Hybrid QPO</div>
        <div className="label-cap mt-1">24h challenge</div>
      </div>

      <nav className="flex flex-col gap-1 flex-1">
        {NAV.map((item) => {
          const active =
            item.href === "/"
              ? path === "/"
              : path.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className="px-3 py-2 rounded-md text-[13px]"
              style={{
                color: active ? "var(--accent)" : "var(--foreground)",
                background: active ? "var(--card)" : "transparent",
                fontWeight: active ? 600 : 400,
                borderLeft: active ? "2px solid var(--accent)" : "2px solid transparent",
              }}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-4 px-2 flex flex-col gap-2">
        <ThemeToggle />
        <BackendStatus />
        <div className="mono text-[11px]" style={{ color: "var(--muted)" }}>
          {process.env.NEXT_PUBLIC_API_BASE ?? "127.0.0.1:8765"}
        </div>
      </div>
    </aside>
  );
}
