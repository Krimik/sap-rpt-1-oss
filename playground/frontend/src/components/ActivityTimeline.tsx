export interface ActivityEvent {
  timestamp: string;
  message: string;
  tone?: "info" | "success" | "warning" | "error";
}

const toneClasses: Record<string, string> = {
  info: "border-primary-500 text-primary-200",
  success: "border-success text-success",
  warning: "border-warning text-warning",
  error: "border-danger text-danger"
};

interface ActivityTimelineProps {
  events: ActivityEvent[];
}

export const ActivityTimeline = ({ events }: ActivityTimelineProps) => {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 shadow-lg">
      <h2 className="text-lg font-semibold text-slate-100">Activity</h2>
      {events.length === 0 ? (
        <p className="mt-2 text-sm text-slate-400">Progress updates will appear here.</p>
      ) : (
        <ul className="mt-3 space-y-2 text-xs">
          {events.map((event, index) => (
            <li
              key={`${event.timestamp}-${index}`}
              className={`rounded border-l-4 bg-slate-950/60 px-3 py-2 ${
                toneClasses[event.tone ?? "info"] ?? toneClasses.info
              }`}
            >
              <p className="font-mono text-slate-400">{new Date(event.timestamp).toLocaleTimeString()}</p>
              <p className="text-slate-200">{event.message}</p>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
};


