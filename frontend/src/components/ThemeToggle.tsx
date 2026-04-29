"use client";

import { useEffect, useState } from "react";

const KEY = "qportf.theme.v1";

export default function ThemeToggle() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  function toggle() {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    try { window.localStorage.setItem(KEY, next ? "dark" : "light"); } catch {}
  }

  return (
    <button className="btn ghost" onClick={toggle} aria-label="toggle theme">
      <span className="mono text-[11px]">{dark ? "◐ dark" : "◑ light"}</span>
    </button>
  );
}
