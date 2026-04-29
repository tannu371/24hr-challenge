"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";

type State = "checking" | "ok" | "ok_no_token" | "down";

export default function BackendStatus() {
  const [state, setState] = useState<State>("checking");

  useEffect(() => {
    let cancelled = false;
    async function ping() {
      try {
        const r = await fetch(`${API_BASE}/healthz`, { cache: "no-store" });
        if (cancelled) return;
        if (!r.ok) { setState("down"); return; }
        const j = await r.json();
        setState(j.ibm_credentials_configured ? "ok" : "ok_no_token");
      } catch {
        if (!cancelled) setState("down");
      }
    }
    ping();
    const t = setInterval(ping, 8000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  const color = {
    checking: "var(--muted)",
    ok: "var(--good)",
    ok_no_token: "var(--accent)",
    down: "var(--danger)",
  }[state];
  const label = {
    checking: "checking…",
    ok: "live · IBM ready",
    ok_no_token: "live · no IBM token",
    down: "backend offline",
  }[state];

  return (
    <div className="flex items-center gap-2">
      <span style={{
        width: 8, height: 8, borderRadius: 8,
        background: color, display: "inline-block",
        boxShadow: state === "ok" || state === "ok_no_token" ? `0 0 6px ${color}` : "none",
      }} />
      <span className="mono text-[11px]" style={{ color }}>{label}</span>
    </div>
  );
}
