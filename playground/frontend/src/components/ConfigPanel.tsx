interface ConfigPanelProps {
  availableColumns: string[];
  task: "classification" | "regression";
  setTask: (task: "classification" | "regression") => void;
  targetColumn?: string;
  setTargetColumn: (column: string) => void;
  maxContextSize: number;
  setMaxContextSize: (value: number) => void;
  bagging: number;
  setBagging: (value: number) => void;
  testSize: number;
  setTestSize: (value: number) => void;
  dropConstantColumns: boolean;
  setDropConstantColumns: (value: boolean) => void;
  disabled: boolean;
  onSubmit: () => void;
  isSubmitting: boolean;
}

export const ConfigPanel = ({
  availableColumns,
  task,
  setTask,
  targetColumn,
  setTargetColumn,
  maxContextSize,
  setMaxContextSize,
  bagging,
  setBagging,
  testSize,
  setTestSize,
  dropConstantColumns,
  setDropConstantColumns,
  disabled,
  onSubmit,
  isSubmitting
}: ConfigPanelProps) => {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 shadow-lg">
      <header className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-100">Configuration</h2>
        <div className="flex gap-2 rounded bg-slate-800 p-1 text-xs">
          <button
            type="button"
            className={`rounded px-3 py-1 ${task === "classification" ? "bg-primary-600 text-white" : "text-slate-300"}`}
            onClick={() => setTask("classification")}
          >
            Classification
          </button>
          <button
            type="button"
            className={`rounded px-3 py-1 ${task === "regression" ? "bg-primary-600 text-white" : "text-slate-300"}`}
            onClick={() => setTask("regression")}
          >
            Regression
          </button>
        </div>
      </header>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div>
          <label className="text-sm text-slate-300">
            Target Column
            <select
              className="mt-1 w-full rounded border border-slate-800 bg-slate-950 px-3 py-2 text-slate-100 focus:border-primary-500 focus:outline-none"
              value={targetColumn ?? ""}
              onChange={(event) => setTargetColumn(event.target.value)}
              disabled={disabled}
            >
              <option value="">Select column...</option>
              {availableColumns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div>
          <label className="text-sm text-slate-300">
            Max Context Size
            <input
              type="number"
              className="mt-1 w-full rounded border border-slate-800 bg-slate-950 px-3 py-2 text-slate-100 focus:border-primary-500 focus:outline-none"
              value={maxContextSize}
              min={256}
              max={8192}
              step={256}
              onChange={(event) => setMaxContextSize(Number(event.target.value))}
              disabled={disabled}
            />
          </label>
        </div>
        <div>
          <label className="text-sm text-slate-300">
            Bagging Factor
            <input
              type="number"
              className="mt-1 w-full rounded border border-slate-800 bg-slate-950 px-3 py-2 text-slate-100 focus:border-primary-500 focus:outline-none"
              value={bagging}
              min={1}
              max={16}
              onChange={(event) => setBagging(Number(event.target.value))}
              disabled={disabled}
            />
          </label>
        </div>
        <div>
          <label className="text-sm text-slate-300">
            Test Split
            <input
              type="range"
              min={0.1}
              max={0.5}
              step={0.05}
              value={testSize}
              onChange={(event) => setTestSize(Number(event.target.value))}
              disabled={disabled}
              className="mt-2 w-full accent-primary-500"
            />
            <span className="mt-1 block text-xs text-slate-400">{Math.round(testSize * 100)}%</span>
          </label>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between text-sm text-slate-300">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={dropConstantColumns}
            onChange={(event) => setDropConstantColumns(event.target.checked)}
            disabled={disabled}
            className="h-4 w-4 accent-primary-500"
          />
          Drop constant columns
        </label>
        <button
          type="button"
          onClick={onSubmit}
          disabled={disabled || !targetColumn || isSubmitting}
          className="rounded bg-primary-600 px-4 py-2 text-sm font-semibold text-white hover:bg-primary-500 disabled:cursor-not-allowed disabled:bg-slate-700"
        >
          {isSubmitting ? "Running..." : "Run Model"}
        </button>
      </div>
    </section>
  );
};


