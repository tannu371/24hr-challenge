"use client";

type Props = {
  /** approx_ratio = method_cost / brute_cost (literal). 1.0 = matched optimum.
   *  Can be in (0,1] for typical good runs, > 1 if method is in violation,
   *  or negative if method and brute have opposite signs (clear failure). */
  value: number;
  label?: string;
  width?: number;
  height?: number;
};

export default function Gauge({
  value,
  label = "approx ratio (method / brute)",
  width = 280,
  height = 90,
}: Props) {
  const PAD = 14;
  const trackY = height - 28;
  const trackH = 14;
  const trackW = width - 2 * PAD;

  // Bar shows the value clamped to [0, 1] for visual scale; the numeric label
  // shows the literal value so under/over-1 failures stay visible.
  const clamped = Math.max(0, Math.min(1, value));

  // Colour-code the numeric readout.
  const color =
    value < 0
      ? "var(--danger)"
      : value > 1.001
        ? "var(--danger)"
        : Math.abs(value - 1) < 1e-9
          ? "var(--good)"
          : "var(--accent)";

  return (
    <div className="flex flex-col gap-2" style={{ width }}>
      <div className="flex justify-between items-baseline">
        <span className="label-cap">{label}</span>
        <span className="mono text-base" style={{ color }}>
          {Number.isFinite(value) ? value.toFixed(3) : "—"}
        </span>
      </div>
      <svg width={width} height={height - 14}>
        <rect
          x={PAD} y={trackY - 16} width={trackW} height={trackH}
          rx={trackH / 2} className="gauge-track"
        />
        <rect
          x={PAD} y={trackY - 16} width={Math.max(2, trackW * clamped)} height={trackH}
          rx={trackH / 2} className="gauge-fill"
        />
        {[0, 0.25, 0.5, 0.75, 1.0].map((t) => (
          <text
            key={t}
            x={PAD + t * trackW}
            y={trackY + 8}
            fontSize={9}
            textAnchor="middle"
            fill="var(--muted)"
            className="mono"
          >
            {t.toFixed(2)}
          </text>
        ))}
        {value > 1.001 && (
          <text
            x={PAD + trackW + 4} y={trackY - 8}
            fontSize={9} fill="var(--danger)" className="mono"
          >▶ in violation
          </text>
        )}
        {value < 0 && (
          <text
            x={PAD + 2} y={trackY - 8}
            fontSize={9} fill="var(--danger)" className="mono"
          >◀ wrong sign
          </text>
        )}
      </svg>
    </div>
  );
}
