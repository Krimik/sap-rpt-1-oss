import { useCallback, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import {
  DatasetPreview as DatasetPreviewType,
  ExampleDataset,
  HealthResponse,
  RunParametersPayload,
  RunResult,
  fetchHealth,
  fetchRunResult,
  listExampleDatasets,
  previewExampleDataset,
  submitExampleRun,
  submitRun,
  uploadPreview
} from "./api";
import { StatusBanner } from "./components/StatusBanner";
import { DatasetPanel } from "./components/DatasetPanel";
import { ConfigPanel } from "./components/ConfigPanel";
import { ResultsDashboard } from "./components/ResultsDashboard";
import { ActivityEvent, ActivityTimeline } from "./components/ActivityTimeline";
import { RunProgressIndicator } from "./components/RunProgressIndicator";

type TaskType = "classification" | "regression";

type RunProgressState = {
  stage: string;
  percent: number;
  etaSeconds?: number;
  detail?: string;
};

const resolveWsUrl = (path: string): string => {
  const backend = import.meta.env.VITE_BACKEND_URL;
  if (backend) {
    const url = new URL(backend);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = path;
    return url.toString();
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
};

const App = () => {
  const [selectedFile, setSelectedFile] = useState<File>();
  const [selectedExampleId, setSelectedExampleId] = useState<string>();
  const [preview, setPreview] = useState<DatasetPreviewType>();
  const [task, setTask] = useState<TaskType>("classification");
  const [targetColumn, setTargetColumn] = useState<string>();
  const [maxContextSize, setMaxContextSize] = useState(1024);
  const [bagging, setBagging] = useState(2);
  const [testSize, setTestSize] = useState(0.2);
  const [dropConstantColumns, setDropConstantColumns] = useState(true);
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [currentTaskId, setCurrentTaskId] = useState<string>();
  const [runResult, setRunResult] = useState<RunResult>();
  const [runError, setRunError] = useState<string>();
  const [runProgress, setRunProgress] = useState<RunProgressState>();

  const healthQuery = useQuery<HealthResponse, Error>({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 30000
  });

  const examplesQuery = useQuery<ExampleDataset[], Error>({
    queryKey: ["example-datasets"],
    queryFn: listExampleDatasets,
    staleTime: 60_000
  });

  const pushEvent = useCallback((message: string, tone: ActivityEvent["tone"]) => {
    setEvents((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.message === message && last.tone === tone) {
        return prev;
      }
      return [
        ...prev,
        {
          timestamp: new Date().toISOString(),
          message,
          tone
        }
      ];
    });
  }, []);

  const previewFileMutation = useMutation({
    mutationFn: uploadPreview,
    onSuccess: (data) => {
      setPreview(data);
      setTargetColumn(data.columns[0]);
      setRunError(undefined);
    },
    onError: (error: Error) => {
      setPreview(undefined);
      setTargetColumn(undefined);
      setRunError(error.message);
    }
  });

  const previewExampleMutation = useMutation({
    mutationFn: previewExampleDataset,
    onSuccess: (data) => {
      setPreview(data);
      setTargetColumn(data.columns[0]);
      setRunError(undefined);
    },
    onError: (error: Error) => {
      setPreview(undefined);
      setTargetColumn(undefined);
      setRunError(error.message);
    }
  });

  type RunSubmission =
    | { type: "upload"; file: File; params: RunParametersPayload }
    | { type: "example"; exampleId: string; params: RunParametersPayload };

  const runMutation = useMutation({
    mutationFn: async (payload: RunSubmission) => {
      if (payload.type === "upload") {
        return submitRun({
          file: payload.file,
          ...payload.params
        });
      }
      return submitExampleRun(payload.exampleId, payload.params);
    },
    onSuccess: ({ task_id }) => {
      setRunResult(undefined);
      setRunError(undefined);
      setRunProgress({ stage: "Queued", percent: 0 });
      pushEvent("Inference job queued.", "info");
      setCurrentTaskId(task_id);
    },
    onError: (error: Error) => {
      setRunError(error.message);
      setRunProgress(undefined);
      pushEvent(`Failed to start job: ${error.message}`, "error");
    }
  });

  useEffect(() => {
    if (!currentTaskId) return;

    const ws = new WebSocket(resolveWsUrl(`/api/run/stream/${currentTaskId}`));

    ws.onmessage = async (event) => {
      const payload = JSON.parse(event.data) as {
        event: string;
        detail?: string;
        progress?: number;
        stage?: string;
        eta_seconds?: number;
      };

      if (payload.event === "progress") {
        const percent = typeof payload.progress === "number" ? Math.max(0, Math.min(payload.progress, 1)) : 0;
        const stage = payload.stage ?? "Processing";
        setRunProgress({ stage, percent, etaSeconds: payload.eta_seconds, detail: payload.detail });
        const progressMessage = `Progress ${Math.round(percent * 100)}% - ${stage}${
          payload.detail ? ` (${payload.detail})` : ""
        }`;
        pushEvent(progressMessage, "info");
        return;
      }

      if (payload.event === "queued") {
        setRunProgress({ stage: "Queued", percent: 0 });
      }

      const tone: ActivityEvent["tone"] =
        payload.event === "failed" ? "error" : payload.event === "completed" ? "success" : "info";
      pushEvent(`Event: ${payload.event}`, tone);

      if (payload.event === "completed") {
        try {
          const result = await fetchRunResult(currentTaskId);
          setRunResult(result);
        } catch (error) {
          setRunError(error instanceof Error ? error.message : "Failed to fetch results.");
        } finally {
          setRunProgress(undefined);
          setCurrentTaskId(undefined);
        }
        pushEvent("Inference job completed.", "success");
      }

      if (payload.event === "failed") {
        setRunError(payload.detail ?? "Inference failed.");
        setRunProgress(undefined);
        setCurrentTaskId(undefined);
        pushEvent(payload.detail ?? "Inference failed.", "error");
      }
    };

    ws.onerror = () => {
      pushEvent("WebSocket connection error.", "error");
      setRunProgress(undefined);
      ws.close();
    };

    return () => {
      ws.close();
    };
  }, [currentTaskId, pushEvent]);

  const examples = examplesQuery.data ?? [];
  const selectedExample = useMemo(
    () => examples.find((example) => example.id === selectedExampleId),
    [examples, selectedExampleId]
  );

  const handleFileSelected = (file: File) => {
    previewFileMutation.reset();
    setSelectedFile(file);
    setSelectedExampleId(undefined);
    previewExampleMutation.reset();
    setPreview(undefined);
    setRunResult(undefined);
    setRunError(undefined);
    setRunProgress(undefined);
    setTargetColumn(undefined);
    setEvents([
      {
        timestamp: new Date().toISOString(),
        message: `Selected dataset: ${file.name}`,
        tone: "info"
      }
    ]);
    previewFileMutation.mutate(file);
  };

  const handleExampleSelected = (exampleId?: string) => {
    previewExampleMutation.reset();
    if (!exampleId) {
      setSelectedExampleId(undefined);
      setSelectedFile(undefined);
      previewExampleMutation.reset();
      setPreview(undefined);
      setTargetColumn(undefined);
      setRunResult(undefined);
      setRunError(undefined);
      setRunProgress(undefined);
      setEvents([
        {
          timestamp: new Date().toISOString(),
          message: "Cleared example selection. Choose a dataset or upload your own.",
          tone: "info"
        }
      ]);
      return;
    }

    const example = examples.find((item) => item.id === exampleId);
    setSelectedExampleId(exampleId);
    setSelectedFile(undefined);
    previewFileMutation.reset();
    setPreview(undefined);
    setRunResult(undefined);
    setRunError(undefined);
    setRunProgress(undefined);
    setTargetColumn(undefined);
    setEvents([
      {
        timestamp: new Date().toISOString(),
        message: `Selected example dataset: ${example?.name ?? exampleId}`,
        tone: "info"
      }
    ]);
    previewExampleMutation.mutate(exampleId);
  };

  const handleRun = () => {
    if (!targetColumn || !preview) return;

    const params: RunParametersPayload = {
      task,
      target_column: targetColumn,
      max_context_size: maxContextSize,
      bagging,
      test_size: testSize,
      drop_constant_columns: dropConstantColumns
    };

    if (selectedExampleId) {
      const exampleName = selectedExample?.name ?? selectedExampleId;
      pushEvent(`Submitting ${task} job for example ${exampleName}...`, "info");
      runMutation.mutate({ type: "example", exampleId: selectedExampleId, params });
      return;
    }

    if (selectedFile) {
      pushEvent(`Submitting ${task} job for ${selectedFile.name}...`, "info");
      runMutation.mutate({ type: "upload", file: selectedFile, params });
    }
  };

  const availableColumns = useMemo(() => preview?.columns ?? [], [preview]);
  const isRunning = runMutation.isPending || Boolean(currentTaskId);
  const previewError = previewFileMutation.error?.message ?? previewExampleMutation.error?.message;

  return (
    <main className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <h1 className="text-2xl font-bold text-slate-50">SAP RPT Playground</h1>
      <StatusBanner
        health={healthQuery.data}
        isLoading={healthQuery.isLoading}
        error={healthQuery.error?.message}
      />
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-6">
          <DatasetPanel
            onFileSelected={handleFileSelected}
            onExampleSelected={handleExampleSelected}
            preview={preview}
            examples={examples}
            selectedExampleId={selectedExampleId}
            isLoading={previewFileMutation.isPending}
            isExampleLoading={previewExampleMutation.isPending}
            error={previewError}
            selectedFile={selectedFile}
          />
          <ConfigPanel
            availableColumns={availableColumns}
            task={task}
            setTask={(value) => setTask(value)}
            targetColumn={targetColumn}
            setTargetColumn={setTargetColumn}
            maxContextSize={maxContextSize}
            setMaxContextSize={setMaxContextSize}
            bagging={bagging}
            setBagging={setBagging}
            testSize={testSize}
            setTestSize={setTestSize}
            dropConstantColumns={dropConstantColumns}
            setDropConstantColumns={setDropConstantColumns}
            disabled={!preview}
            onSubmit={handleRun}
            isSubmitting={isRunning}
          />
          {runProgress && (
            <RunProgressIndicator
              stage={runProgress.stage}
              percent={runProgress.percent}
              etaSeconds={runProgress.etaSeconds}
              detail={runProgress.detail}
            />
          )}
          <ResultsDashboard result={runResult} isLoading={isRunning} error={runError} />
        </div>
        <div className="flex flex-col gap-6">
          <ActivityTimeline events={events} />
        </div>
      </div>
    </main>
  );
};

export default App;


