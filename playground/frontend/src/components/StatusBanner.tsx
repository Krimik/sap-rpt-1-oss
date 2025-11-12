import { HealthResponse } from "../api";

interface StatusBannerProps {
  health?: HealthResponse;
  isLoading: boolean;
  error?: string;
}

const statusColor = (health?: HealthResponse) => {
  if (!health) return "bg-primary-900";
  return health.status === "ok" ? "bg-success/20 text-success" : "bg-warning/20 text-warning";
};

export const StatusBanner = ({ health, isLoading, error }: StatusBannerProps) => {
  const hfConnected = health?.huggingface === "connected";
  const hfStatus = hfConnected ? "Hugging Face: Connected" : "Hugging Face: Token missing";
  const gpuAvailable = Boolean(health?.model.cuda_available);
  const gpuStatus = gpuAvailable ? "GPU detected" : "CPU fallback";
  const checkpointReady = Boolean(health?.model.checkpoint_path);
  const modelStatus = checkpointReady ? "sap-rpt-1-oss cached" : "sap-rpt-1-oss pending";

  const statusItems: Array<{ label: string; variant: "success" | "warning" | "danger" }> = [
    { label: hfStatus, variant: hfConnected ? "success" : "danger" },
    { label: modelStatus, variant: checkpointReady ? "success" : "warning" },
    { label: gpuStatus, variant: gpuAvailable ? "success" : "warning" }
  ];

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 shadow-lg">
      <header className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-100">System Status</h2>
        <span className={`rounded-full px-3 py-1 text-sm font-semibold ${statusColor(health)}`}>
          {isLoading ? "Checking..." : health?.status === "ok" ? "Operational" : "Degraded"}
        </span>
      </header>
      {error && <p className="mt-2 text-sm text-danger">{error}</p>}
      <div className="mt-3 grid gap-3 text-sm text-slate-300 md:grid-cols-3">
        {statusItems.map((item) => (
          <StatusPill key={item.label} label={item.label} variant={item.variant} />
        ))}
      </div>
      {health && (
        <p className="mt-3 text-xs text-slate-400">
          Device: {health.model.device} · Torch {health.model.torch_version} · Embeddings port{" "}
          {health.model.embedding_server_port}
        </p>
      )}
    </section>
  );
};

const StatusPill = ({ label, variant }: { label: string; variant: "success" | "warning" | "danger" }) => {
  const classes =
    variant === "success"
      ? "bg-success/20 text-success"
      : variant === "warning"
        ? "bg-warning/20 text-warning"
        : "bg-danger/20 text-danger";
  return (
    <span className={`inline-flex items-center gap-2 rounded-md px-3 py-1 text-xs font-medium ${classes}`}>
      <span className="inline-block h-2 w-2 rounded-full bg-current" />
      {label}
    </span>
  );
};
