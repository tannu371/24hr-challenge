"use client";

/* Shared problem-parameter state across panels.
   Persisted to localStorage so the user can refresh / open Judge Mode without
   losing the slider state. */

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ProblemParams } from "./api";

const DEFAULT: ProblemParams = {
  N: 8,
  lambda: 2.5,
  lambda_return: 1.0,
  K: 3,
  P_K: 5.0,
  P_S: 0.0,
  sector_caps: {},
  P_R: 0.5,
  theta_risk: 0.04,
  transaction_costs: [],
  seed: 7,
  csv_data: null,
};

type Ctx = {
  params: ProblemParams;
  setParams: (p: Partial<ProblemParams>) => void;
  reset: () => void;
};

const ProblemCtx = createContext<Ctx | null>(null);

const STORAGE_KEY = "qportf.problem.v1";

export function ProblemProvider({ children }: { children: React.ReactNode }) {
  const [params, setParamsState] = useState<ProblemParams>(DEFAULT);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) setParamsState({ ...DEFAULT, ...JSON.parse(raw) });
    } catch { /* ignore */ }
  }, []);

  const setParams = useCallback((delta: Partial<ProblemParams>) => {
    setParamsState((prev) => {
      const next = { ...prev, ...delta };
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch { /* ignore */ }
      return next;
    });
  }, []);

  const reset = useCallback(() => {
    setParamsState(DEFAULT);
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch { /* ignore */ }
  }, []);

  return (
    <ProblemCtx.Provider value={{ params, setParams, reset }}>
      {children}
    </ProblemCtx.Provider>
  );
}

export function useProblem(): Ctx {
  const ctx = useContext(ProblemCtx);
  if (!ctx) throw new Error("useProblem must be used inside ProblemProvider");
  return ctx;
}

/* Pinned trials helper — used by Judge Mode. */
const PIN_KEY = "qportf.pinnedTrials.v1";

export function getPinnedTrials(): number[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(PIN_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

export function setPinnedTrials(ids: number[]) {
  try { window.localStorage.setItem(PIN_KEY, JSON.stringify(ids)); }
  catch { /* ignore */ }
}

export function togglePin(trialId: number): number[] {
  const cur = getPinnedTrials();
  const next = cur.includes(trialId)
    ? cur.filter((x) => x !== trialId)
    : [...cur, trialId];
  setPinnedTrials(next);
  return next;
}

export function unpinTrial(trialId: number): number[] {
  const next = getPinnedTrials().filter((x) => x !== trialId);
  setPinnedTrials(next);
  return next;
}
