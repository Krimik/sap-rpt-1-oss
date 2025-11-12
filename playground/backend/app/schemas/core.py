"""Core Pydantic models for playground APIs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class HFStatus(str, Enum):
    connected = "connected"
    missing_token = "missing_token"
    error = "error"


class ModelStatusSchema(BaseModel):
    checkpoint_path: Optional[str] = None
    device: str
    dtype: Optional[str] = None
    embedding_server_port: int
    embedding_server_started: bool
    cuda_available: bool
    torch_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    message: str
    huggingface: HFStatus
    model: ModelStatusSchema


class DatasetPreview(BaseModel):
    columns: list[str]
    dtypes: dict[str, str]
    row_count: int
    missing_values: dict[str, int]


class ExampleDataset(BaseModel):
    id: str
    name: str
    size_bytes: int


class RunParameters(BaseModel):
    task: Literal["classification", "regression"]
    target_column: str = Field(..., description="Name of the target column in the dataset.")
    max_context_size: int = Field(1024, ge=128, le=8192)
    bagging: int = Field(2, ge=1, le=16)
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    drop_constant_columns: bool = True


class RunRequest(BaseModel):
    task_id: str


class RunStatus(BaseModel):
    task_id: str
    state: Literal["queued", "running", "completed", "failed"]
    detail: Optional[str] = None
    progress: Optional[float] = None


class RunResult(BaseModel):
    metrics: dict[str, Any]
    predictions_preview: list[dict[str, Any]]
    download_path: Optional[str] = None
    source_name: Optional[str] = None


