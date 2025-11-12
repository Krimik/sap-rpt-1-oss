import { RunResult, getResultDownloadUrl } from "../api";

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const formatMetricValue = (value: unknown): string => {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    const rounded = Number.isInteger(value) ? value.toString() : value.toFixed(4);
    return rounded.replace(/\.?0+$/, "");
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) return value.map(formatMetricValue).join(", ");
  return String(value);
};

const flattenMetrics = (metrics: Record<string, unknown>): Array<{ label: string; value: string }> => {
  const rows: Array<{ label: string; value: string }> = [];

  const visit = (value: unknown, path: string) => {
    if (isPlainObject(value)) {
      Object.entries(value).forEach(([key, child]) => {
        const nextPath = path ? `${path} → ${key}` : key;
        visit(child, nextPath);
      });
    } else {
      rows.push({ label: path, value: formatMetricValue(value) });
    }
  };

  Object.entries(metrics).forEach(([key, value]) => visit(value, key));

  return rows;
};

interface ResultsDashboardProps {
  result?: RunResult;
  isLoading: boolean;
  error?: string;
}

export const ResultsDashboard = ({ result, isLoading, error }: ResultsDashboardProps) => {
  const metricRows = result ? flattenMetrics(result.metrics) : [];
  const downloadName =
    result?.source_name !== undefined && result?.source_name !== null
      ? `${result.source_name.replace(/\.[^/.]+$/, "")} - results.csv`
      : undefined;

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 shadow-lg">
      <h2 className="text-lg font-semibold text-slate-100">Results</h2>
      {isLoading && <p className="mt-2 text-sm text-slate-300">Waiting for predictions...</p>}
      {error && <p className="mt-2 text-sm text-danger">{error}</p>}
      {!result && !isLoading && !error && (
        <p className="mt-2 text-sm text-slate-400">Run the model to see results.</p>
      )}
      {result && (
        <div className="mt-4 space-y-4">
          <div className="rounded border border-slate-800 bg-slate-950/60 p-3">
            <h3 className="text-sm font-semibold text-slate-200">Metrics</h3>
            {result.source_name && (
              <p className="mt-1 text-xs text-slate-400">Dataset: {result.source_name}</p>
            )}
            {metricRows.length > 0 ? (
              <div className="mt-2 max-h-64 overflow-auto">
                <table className="w-full border-collapse text-xs text-slate-200">
                  <thead className="sticky top-0 bg-slate-900/80 text-slate-400">
                    <tr>
                      <th className="px-3 py-2 text-left font-semibold">Metric</th>
                      <th className="px-3 py-2 text-left font-semibold">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metricRows.map(({ label, value }, index) => (
                      <tr key={`${label}-${index}`} className="border-t border-slate-800">
                        <td className="px-3 py-2 align-top text-slate-300">{label}</td>
                        <td className="px-3 py-2 text-slate-100">{value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="mt-2 text-xs text-slate-400">No metrics reported.</p>
            )}
          </div>
          {result.download_path && (
            <div className="flex justify-end">
              <a
                className="inline-flex items-center rounded border border-slate-700 px-4 py-2 text-sm font-medium text-slate-200 hover:border-primary-500 hover:text-primary-300"
                href={getResultDownloadUrl(result.download_path)}
                download={downloadName}
              >
                Download predictions CSV
              </a>
            </div>
          )}
        </div>
      )}
    </section>
  );
};


