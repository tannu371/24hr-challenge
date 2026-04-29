"use client";

type Props = {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  display?: (v: number) => string;
};

export default function Slider({
  label, value, onChange, min, max, step = 1, unit, display,
}: Props) {
  const shown = display ? display(value) : value.toString();
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-baseline">
        <span className="label-cap">{label}</span>
        <span className="mono text-[12px]">{shown}{unit ? ` ${unit}` : ""}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
