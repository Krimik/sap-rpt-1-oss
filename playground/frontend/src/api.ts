import axios from "axios";

const client = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_URL ?? "",
  withCredentials: false
});

export interface HealthResponse {
  status: "ok" | "degraded";
  message: string;
  huggingface: "connected" | "missing_token" | "error";
  model: {
    checkpoint_path?: string | null;
    device: string;
    dtype?: string | null;
    embedding_server_port: number;
    embedding_server_started: boolean;
    cuda_available: boolean;
    torch_version: string;
  };
}

export interface DatasetPreview {
  columns: string[];
  dtypes: Record<string, string>;
  row_count: number;
  missing_values: Record<string, number>;
}

export interface ExampleDataset {
  id: string;
  name: string;
  size_bytes: number;
}

export interface RunRequestResponse {
  task_id: string;
}

export interface RunResult {
  metrics: Record<string, unknown>;
  predictions_preview: Array<Record<string, unknown>>;
  download_path?: string | null;
  source_name?: string | null;
}

export const fetchHealth = async (): Promise<HealthResponse> => {
  const { data } = await client.get("/api/health");
  return data;
};

export const uploadPreview = async (file: File): Promise<DatasetPreview> => {
  const form = new FormData();
  form.append("file", file);
  const { data } = await client.post("/api/datasets/preview", form, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return data;
};

export const listExampleDatasets = async (): Promise<ExampleDataset[]> => {
  const { data } = await client.get("/api/datasets/examples");
  return data;
};

export const previewExampleDataset = async (exampleId: string): Promise<DatasetPreview> => {
  const { data } = await client.get(`/api/datasets/examples/${encodeURIComponent(exampleId)}/preview`);
  return data;
};

export interface RunParametersPayload {
  task: "classification" | "regression";
  target_column: string;
  max_context_size: number;
  bagging: number;
  test_size: number;
  drop_constant_columns: boolean;
}

interface RunParams extends RunParametersPayload {
  file: File;
}

export const submitRun = async (params: RunParams): Promise<RunRequestResponse> => {
  const form = new FormData();
  form.append("file", params.file);
  form.append("task", params.task);
  form.append("target_column", params.target_column);
  form.append("max_context_size", String(params.max_context_size));
  form.append("bagging", String(params.bagging));
  form.append("test_size", String(params.test_size));
  form.append("drop_constant_columns", String(params.drop_constant_columns));
  const { data } = await client.post("/api/run", form, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return data;
};

export const submitExampleRun = async (
  exampleId: string,
  params: RunParametersPayload
): Promise<RunRequestResponse> => {
  const { data } = await client.post(`/api/run/examples/${encodeURIComponent(exampleId)}`, {
    task: params.task,
    target_column: params.target_column,
    max_context_size: params.max_context_size,
    bagging: params.bagging,
    test_size: params.test_size,
    drop_constant_columns: params.drop_constant_columns
  });
  return data;
};

export const fetchRunResult = async (taskId: string): Promise<RunResult> => {
  const { data } = await client.get(`/api/run/${taskId}/result`);
  return data;
};

export const getResultDownloadUrl = (token: string): string => {
  const backend = import.meta.env.VITE_BACKEND_URL?.replace(/\/$/, "");
  if (backend) {
    return `${backend}/api/results/${token}/download`;
  }
  return `/api/results/${token}/download`;
};


