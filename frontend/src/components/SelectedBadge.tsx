"use client";

/* Renders one chip per held asset.

   `names` is expected to be either:
     • the **selected** asset names (length == selected.length, parallel array
       — what every backend response actually sends as `selected_names`), or
     • the **full** asset_names list of length N, indexed by asset position.

   We auto-detect: if names.length matches selected.length we treat it as
   the parallel sublist; otherwise we look up by asset index.
*/

type Props = { selected?: number[]; names?: string[] };

export default function SelectedBadge({ selected, names }: Props) {
  if (!selected) return null;
  const parallel = !!names && names.length === selected.length;

  return (
    <div className="flex flex-wrap gap-1">
      {selected.map((i, k) => {
        const label = parallel ? names![k] : (names?.[i] ?? `#${i}`);
        return (
          <span
            key={k}
            className="mono text-[11px] px-2 py-1 rounded"
            style={{ background: "var(--background-alt)", border: "1px solid var(--border)" }}
          >
            {label}
          </span>
        );
      })}
    </div>
  );
}
