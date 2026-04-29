"use client";

import { useEffect } from "react";

const KEY = "qportf.theme.v1";

export default function ThemeBoot() {
  useEffect(() => {
    const stored = window.localStorage.getItem(KEY);
    const prefersDark =
      stored === "dark" ||
      (!stored && window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.classList.toggle("dark", prefersDark);
  }, []);
  return null;
}
