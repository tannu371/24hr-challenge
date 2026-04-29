"use client";

type Props = {
  eyebrow: string;
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
};

export default function PageHeader({ eyebrow, title, subtitle, actions }: Props) {
  return (
    <header className="mb-6 flex items-end justify-between gap-4">
      <div>
        <div className="label-cap">{eyebrow}</div>
        <h1 className="display text-3xl mt-1">{title}</h1>
        {subtitle && (
          <p className="mt-2 text-[14px] max-w-2xl" style={{ color: "var(--muted)" }}>
            {subtitle}
          </p>
        )}
      </div>
      {actions && <div className="flex gap-2">{actions}</div>}
    </header>
  );
}
