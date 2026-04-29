"use client";

/* Light-weight SVG heatmap for Σ + landscape grids.
   We avoid recharts here because it doesn't do scalar grid heatmaps cleanly. */

type Props = {
  data: number[][];               // [rows][cols]
  rowLabels?: string[];
  colLabels?: string[];
  width?: number;
  height?: number;
  diverging?: boolean;            // negative→cyan, positive→gold
  onCellClick?: (r: number, c: number) => void;
  highlight?: { r: number; c: number };
  formatValue?: (v: number) => string;
  title?: string;
};

export default function Heatmap({
  data, rowLabels, colLabels, width = 480, height = 380,
  diverging = true, onCellClick, highlight, formatValue, title,
}: Props) {
  const rows = data.length;
  const cols = rows > 0 ? data[0].length : 0;
  if (rows === 0 || cols === 0) return null;

  let lo = Infinity, hi = -Infinity;
  for (const row of data) for (const v of row) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!isFinite(lo) || !isFinite(hi) || lo === hi) {
    lo -= 1e-9;
    hi += 1e-9;
  }

  // Pick the actually-applicable scheme: only true diverging when the data
  // crosses 0. Otherwise a single-hue ramp keyed on whether values are all
  // positive (gold) or all negative (cyan). This avoids the "legend says
  // cyan exists but you never see any" case on the Σ heatmap.
  const straddlesZero = lo < 0 && hi > 0;
  const useDiverging = diverging && straddlesZero;
  const allNegative = !straddlesZero && hi <= 0;
  const scheme = useDiverging
    ? "diverging"
    : allNegative
      ? "cyan"
      : "gold";

  const colorAt = (v: number) => {
    if (useDiverging) {
      const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
      const t = v / m; // [-1, 1]
      if (t >= 0) {
        const a = Math.min(1, t);
        return `rgba(180, 138, 44, ${0.18 + 0.82 * a})`;
      } else {
        const a = Math.min(1, -t);
        return `rgba(31, 138, 138, ${0.18 + 0.82 * a})`;
      }
    }
    // Single-hue ramp on |v| within [lo, hi].
    const t = (v - lo) / (hi - lo || 1);
    const a = 0.15 + 0.85 * t;
    return scheme === "cyan"
      ? `rgba(31, 138, 138, ${a})`
      : `rgba(180, 138, 44, ${a})`;
  };

  const PAD_L = rowLabels ? 56 : 12;
  const PAD_T = colLabels ? 28 : 12;
  const PAD_R = 12;
  const PAD_B = 12;
  const innerW = width - PAD_L - PAD_R;
  const innerH = height - PAD_T - PAD_B;
  const cellW = innerW / cols;
  const cellH = innerH / rows;

  return (
    <div className="flex flex-col gap-2">
      {title && <div className="label-cap">{title}</div>}
      <svg
        width={width}
        height={height}
        style={{ overflow: "visible", maxWidth: "100%" }}
      >
        {colLabels?.map((lab, c) => (
          <text
            key={`cl-${c}`}
            x={PAD_L + c * cellW + cellW / 2}
            y={PAD_T - 8}
            fontSize={10}
            textAnchor="middle"
            fill="var(--muted)"
            className="mono"
          >
            {lab}
          </text>
        ))}
        {rowLabels?.map((lab, r) => (
          <text
            key={`rl-${r}`}
            x={PAD_L - 6}
            y={PAD_T + r * cellH + cellH / 2 + 3}
            fontSize={10}
            textAnchor="end"
            fill="var(--muted)"
            className="mono"
          >
            {lab}
          </text>
        ))}
        {data.map((row, r) =>
          row.map((v, c) => {
            const isHL = highlight && highlight.r === r && highlight.c === c;
            return (
              <g key={`${r}-${c}`}>
                <rect
                  x={PAD_L + c * cellW}
                  y={PAD_T + r * cellH}
                  width={cellW + 0.5}
                  height={cellH + 0.5}
                  fill={colorAt(v)}
                  stroke={isHL ? "var(--accent)" : "transparent"}
                  strokeWidth={isHL ? 2 : 0}
                  cursor={onCellClick ? "pointer" : undefined}
                  onClick={onCellClick ? () => onCellClick(r, c) : undefined}
                >
                  <title>
                    {formatValue ? formatValue(v) : v.toFixed(4)}
                    {colLabels ? ` · ${colLabels[c]}` : ""}
                    {rowLabels ? ` × ${rowLabels[r]}` : ""}
                  </title>
                </rect>
              </g>
            );
          })
        )}
      </svg>
      <div className="flex justify-between items-center mono text-[10px]" style={{ color: "var(--muted)" }}>
        <span>min {lo.toExponential(2)}</span>
        <span className="flex items-center gap-1">
          {scheme === "diverging" && (
            <>
              <Swatch color="rgba(31, 138, 138, 0.9)" />cyan = −
              <Swatch color="rgba(180, 138, 44, 0.9)" style={{ marginLeft: 6 }} />gold = +
            </>
          )}
          {scheme === "gold" && (<><Swatch color="rgba(180, 138, 44, 0.9)" />gold scales magnitude</>)}
          {scheme === "cyan" && (<><Swatch color="rgba(31, 138, 138, 0.9)" />cyan scales magnitude</>)}
        </span>
        <span>max {hi.toExponential(2)}</span>
      </div>
    </div>
  );
}

function Swatch({ color, style }: { color: string; style?: React.CSSProperties }) {
  return (
    <span
      style={{
        display: "inline-block",
        width: 10, height: 10, borderRadius: 2,
        background: color,
        ...style,
      }}
    />
  );
}
