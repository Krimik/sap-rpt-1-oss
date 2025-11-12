import { ChangeEvent } from "react";

import { DatasetPreview, ExampleDataset } from "../api";

interface DatasetPanelProps {
  onFileSelected: (file: File) => void;
  onExampleSelected?: (exampleId?: string) => void;
  preview?: DatasetPreview;
  examples?: ExampleDataset[];
  selectedExampleId?: string;
  isLoading: boolean;
  isExampleLoading?: boolean;
  error?: string;
  selectedFile?: File;
}

export const DatasetPanel = ({
  onFileSelected,
  onExampleSelected,
  preview,
  examples,
  selectedExampleId,
  isLoading,
  isExampleLoading,
  error,
  selectedFile
}: DatasetPanelProps) => {
  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) onFileSelected(file);
  };

  const handleExampleChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const value = event.target.value || undefined;
    onExampleSelected?.(value);
  };

  const selectedExample = examples?.find((example) => example.id === selectedExampleId);
  const statusMessage = selectedExample
    ? `Using example dataset: ${selectedExample.name}`
    : selectedFile
      ? `Selected file: ${selectedFile.name}`
      : undefined;
  const showAnalyzing = isLoading || Boolean(isExampleLoading);

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 shadow-lg">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-slate-100">Dataset</h2>
        <div className="flex flex-wrap items-center gap-2">
          {examples && examples.length > 0 && (
            <select
              value={selectedExampleId ?? ""}
              onChange={handleExampleChange}
              className="rounded border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-100 focus:border-primary-500 focus:outline-none"
            >
              <option value="">Choose example dataset...</option>
              {examples.map((example) => (
                <option key={example.id} value={example.id}>
                  {example.name}
                </option>
              ))}
            </select>
          )}
          <label className="cursor-pointer rounded bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-500 focus:outline-none">
            {selectedFile ? "Change File" : "Upload CSV"}
            <input type="file" accept=".csv,.parquet,.json" className="hidden" onChange={handleChange} />
          </label>
        </div>
      </header>
      {statusMessage && (
        <p className="mt-2 text-sm text-slate-400">
          {statusMessage}
        </p>
      )}
      {showAnalyzing && <p className="mt-3 text-sm text-slate-300">Analyzing dataset...</p>}
      {error && <p className="mt-3 text-sm text-danger">{error}</p>}
      {preview && (
        <div className="mt-4 space-y-2 text-sm text-slate-300">
          <p>
            Rows: <span className="text-slate-100">{preview.row_count}</span> · Columns:{" "}
            <span className="text-slate-100">{preview.columns.length}</span>
          </p>
          <div className="max-h-48 overflow-auto rounded border border-slate-800 bg-slate-950/60 p-3 text-xs">
            <table className="w-full border-collapse">
              <thead className="text-slate-400">
                <tr>
                  <th className="px-2 py-1 text-left">Column</th>
                  <th className="px-2 py-1 text-left">Type</th>
                  <th className="px-2 py-1 text-left">Missing</th>
                </tr>
              </thead>
              <tbody>
                {preview.columns.map((column) => (
                  <tr key={column} className="border-t border-slate-800">
                    <td className="px-2 py-1 text-slate-200">{column}</td>
                    <td className="px-2 py-1 text-slate-400">{preview.dtypes[column]}</td>
                    <td className="px-2 py-1 text-slate-400">{preview.missing_values[column] ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
};


