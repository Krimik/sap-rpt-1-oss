interface RunProgressIndicatorProps {
  stage: string;
  percent: number;
  etaSeconds?: number;
  detail?: string;
}

const formatEta = (eta?: number) => {
  if (eta === undefined || Number.isNaN(eta)) {
    return "Estimating…";
  }
  if (!Number.isFinite(eta)) {
    return "Estimating…";
  }
  if (eta < 1) {
    return "< 1s";
  }
  const minutes = Math.floor(eta / 60);
  const seconds = Math.floor(eta % 60);
  if (minutes === 0) {
    return `${seconds}s`;
  }
  return `${minutes}m ${seconds}s`;
};

export const RunProgressIndicator = ({ stage, percent, etaSeconds, detail }: RunProgressIndicatorProps) => {
  const clampedPercent = Math.max(0, Math.min(percent, 1));
  const percentLabel = `${Math.round(clampedPercent * 100)}%`;

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 shadow-lg">
      <header className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-100">Run Progress</h2>
        <span className="text-sm text-slate-300">{percentLabel}</span>
      </header>
      <p className="mt-2 text-sm text-slate-300">{stage}</p>
      {detail && <p className="text-xs text-slate-500">{detail}</p>}
      <div className="mt-3 h-2 w-full overflow-hidden rounded bg-slate-800">
        <div
          className="h-2 rounded bg-primary-500 transition-all duration-300"
          style={{ width: `${clampedPercent * 100}%` }}
        />
      </div>
      <p className="mt-3 text-xs text-slate-400">ETA: {formatEta(etaSeconds)}</p>
    </section>
  );
};


