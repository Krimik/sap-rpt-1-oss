"""Core API routes for the playground backend."""

from __future__ import annotations

import asyncio
import io
import logging
import math
import secrets
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from playground.backend.app.dependencies import get_settings_dependency
from playground.backend.app.jobs import JOB_REGISTRY, ProgressStreamer, run_inference
from playground.backend.app.schemas.core import (
    DatasetPreview,
    ExampleDataset,
    HFStatus,
    HealthResponse,
    ModelStatusSchema,
    RunParameters,
    RunRequest,
    RunResult,
    RunStatus,
)
from playground.backend.app.services.model_manager import ModelManager, get_model_manager
from playground.backend.config import Settings

router = APIRouter(tags=["core"])
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResultFileInfo:
    path: Path
    download_name: str


RESULT_FILES: Dict[str, ResultFileInfo] = {}
ALLOWED_DATASET_EXTENSIONS = {".csv", ".parquet", ".json"}


def _register_result_file(token: str, info: ResultFileInfo):
    RESULT_FILES[token] = info


def _resolve_result_file(token: str) -> ResultFileInfo:
    info = RESULT_FILES.get(token)
    if not info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result file not found.")
    if not info.path.exists():
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Result file expired.")
    return info


def _examples_root(settings: Settings) -> Path:
    return settings.examples_dir.resolve()


def _collect_examples(settings: Settings) -> List[ExampleDataset]:
    directory = _examples_root(settings)
    if not directory.exists():
        return []

    examples: List[ExampleDataset] = []
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix.lower() in ALLOWED_DATASET_EXTENSIONS:
            try:
                stat = entry.stat()
            except OSError:
                continue
            examples.append(ExampleDataset(id=entry.name, name=entry.stem, size_bytes=stat.st_size))

    return sorted(examples, key=lambda item: item.name.lower())


def _resolve_example_path(example_id: str, settings: Settings) -> Path:
    directory = _examples_root(settings)
    if not directory.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Examples directory not configured.")

    candidate = (directory / example_id).resolve()
    try:
        candidate.relative_to(directory)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid example reference.")

    if not candidate.exists() or not candidate.is_file() or candidate.suffix.lower() not in ALLOWED_DATASET_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Example dataset not found.")

    return candidate


def _load_example_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".json":
            df = pd.read_json(path)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format for example '{path.name}'.",
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load example dataset '%s': %s", path, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load example dataset. Verify the file contents.",
        ) from exc
    return df


def _huggingface_status(settings: Settings) -> HFStatus:
    if settings.huggingface_token:
        return HFStatus.connected
    return HFStatus.missing_token


def _load_dataframe(file: UploadFile) -> pd.DataFrame:
    filename = file.filename or "dataset"
    file.file.seek(0)
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    buffer = io.BytesIO(content)
    lower = filename.lower()
    try:
        if lower.endswith(".csv"):
            df = pd.read_csv(buffer)
        elif lower.endswith(".parquet"):
            df = pd.read_parquet(buffer)
        elif lower.endswith(".json"):
            df = pd.read_json(buffer)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Please upload CSV, Parquet, or JSON.",
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to parse uploaded dataset: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to parse uploaded dataset. Please verify the file contents.",
        ) from exc

    return df


def _validate_filesize(file: UploadFile, limit_mb: int) -> None:
    position = file.file.tell()
    file.file.seek(0, io.SEEK_END)
    size_bytes = file.file.tell()
    file.file.seek(position)
    if size_bytes > limit_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds the {limit_mb} MB size limit.",
        )


def _preview_dataframe(df: pd.DataFrame) -> DatasetPreview:
    return DatasetPreview(
        columns=list(df.columns),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        row_count=int(len(df)),
        missing_values={col: int(df[col].isna().sum()) for col in df.columns},
    )


def _schedule_run(
    df: pd.DataFrame,
    params: RunParameters,
    settings: Settings,
    source_name: Optional[str],
) -> RunRequest:
    manager = get_model_manager()
    task_id, stream = JOB_REGISTRY.create_job()

    async def orchestrate() -> None:
        coro = run_inference(
            task_id,
            _run_model,
            df,
            params,
            manager,
            settings=settings,
            run_token=task_id,
            stream=stream,
            source_name=source_name,
        )
        await _stream_progress(task_id, stream, coro)

    asyncio.create_task(orchestrate())
    return RunRequest(task_id=task_id)


def _split_dataset(
    df: pd.DataFrame, params: RunParameters
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if params.target_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{params.target_column}' was not found in the dataset.",
        )

    X = df.drop(columns=[params.target_column])
    y = df[params.target_column]
    stratify = y if params.task == "classification" else None
    if params.task == "classification":
        class_counts = y.value_counts()
        if class_counts.min() < 2 or class_counts[class_counts >= 2].shape[0] < 2:
            stratify = None
    else:
        if len(y) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Regression datasets must have at least two rows.",
            )

    try:
        return train_test_split(
            X,
            y,
            test_size=params.test_size,
            random_state=42,
            stratify=stratify,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unable to split dataset. For classification, ensure each class has enough samples; "
                "for regression verify the dataset has more than one row."
            ),
        ) from exc


def _to_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _persist_result(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: list,
    params: RunParameters,
    settings: Settings | None,
    run_token: str,
    source_name: Optional[str],
) -> str:
    try:
        base_dir = Path(settings.cache_dir) if settings else Path(tempfile.gettempdir()) / "sap_rpt_results"
        result_dir = base_dir / "runs"
        result_dir.mkdir(parents=True, exist_ok=True)
        file_path = result_dir / f"{run_token}.csv"
        df_out = X_test.reset_index(drop=True).copy()
        df_out[f"{params.target_column}_actual"] = pd.Series(y_test).reset_index(drop=True)
        df_out["prediction"] = pd.Series(predictions).reset_index(drop=True)
        df_out.to_csv(file_path, index=False)
        if source_name:
            source_stem = Path(source_name).stem or source_name
            download_name = f"{source_stem} - results.csv"
        else:
            download_name = f"{run_token} - results.csv"
        _register_result_file(run_token, ResultFileInfo(path=file_path, download_name=download_name))
        return run_token
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to persist run result: %s", exc)
        return ""


def _run_model(
    df: pd.DataFrame,
    params: RunParameters,
    manager: ModelManager,
    settings: Settings | None = None,
    run_token: str | None = None,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    source_name: Optional[str] = None,
) -> RunResult:
    X_train, X_test, y_train, y_test = _split_dataset(df, params)
    estimator = manager.get_estimator(params.task)

    start_time = time.time()

    def emit(stage: str, progress: float, detail: Optional[str] = None, eta: Optional[float] = None) -> None:
        if not progress_callback:
            return
        payload: dict[str, Any] = {
            "event": "progress",
            "stage": stage,
            "progress": float(max(0.0, min(progress, 1.0))),
        }
        if detail:
            payload["detail"] = detail
        if eta is not None:
            payload["eta_seconds"] = max(0.0, float(eta))
        progress_callback(payload)

    estimator.max_context_size = params.max_context_size
    estimator.bagging = params.bagging
    estimator.drop_constant_columns = params.drop_constant_columns

    emit("Preparing dataset", 0.05, "Splitting data")

    estimator.fit(X_train, y_train)
    emit("Training model", 0.30, "Model fit complete")

    total_chunks = max(1, math.ceil(max(1, len(X_test)) / estimator.test_chunk_size))
    predictions_list: list[Any] = []
    probability_batches = []
    regression_batches = []

    predict_start = time.time()
    for chunk_index, start in enumerate(range(0, len(X_test), estimator.test_chunk_size), start=1):
        end = start + estimator.test_chunk_size
        chunk_df = X_test.iloc[start:end]
        if params.task == "classification":
            batch_probs = estimator._predict(chunk_df)
            probability_batches.append(batch_probs)
        else:
            batch_preds = estimator._predict(chunk_df)
            regression_batches.append(batch_preds)

        elapsed = time.time() - predict_start
        remaining_chunks = total_chunks - chunk_index
        eta = (elapsed / chunk_index) * remaining_chunks if chunk_index > 0 else None
        progress_fraction = 0.30 + 0.60 * (chunk_index / total_chunks)
        emit("Generating predictions", progress_fraction, detail=f"Chunk {chunk_index}/{total_chunks}", eta=eta)

    if len(X_test) == 0:
        emit("Generating predictions", 0.90, detail="No evaluation rows detected", eta=0.0)

    if params.task == "classification":
        if probability_batches:
            probs = torch.cat(probability_batches)
            preds_idx = probs.argmax(dim=-1).numpy()
            predictions_list = [estimator.classes_[p] for p in preds_idx]
        else:
            predictions_list = []
    else:
        if regression_batches:
            preds_array = np.concatenate(regression_batches)
            predictions_list = preds_array.tolist()
        else:
            predictions_list = []

    preview_records = []
    for features, target, prediction in zip(X_test.itertuples(index=False), y_test, predictions_list):
        preview_records.append({"target": _to_python_scalar(target), "prediction": _to_python_scalar(prediction)})
        if len(preview_records) >= 5:
            break

    if params.task == "classification":
        accuracy = accuracy_score(y_test, predictions_list)
        metrics = {
            "accuracy": float(accuracy),
            "report": classification_report(y_test, predictions_list, zero_division=0, output_dict=True),
        }
    else:
        predictions_array = np.asarray(predictions_list, dtype=float)
        r2 = r2_score(y_test, predictions_array)
        mse = mean_squared_error(y_test, predictions_array)
        metrics = {"r2": float(r2), "rmse": float(np.sqrt(mse))}

    token = run_token or secrets.token_hex(8)
    emit("Finalizing", 0.95, "Saving predictions")
    download_token = _persist_result(X_test, y_test, predictions_list, params, settings, token, source_name)
    emit("Completed", 1.0, "Run complete", eta=0.0)

    return RunResult(
        metrics=metrics,
        predictions_preview=preview_records,
        download_path=download_token or None,
        source_name=source_name,
    )


async def _stream_progress(task_id: str, stream: ProgressStreamer, coro) -> None:
    try:
        await stream.publish({"event": "queued"})
        result = await coro
        await stream.publish({"event": "completed"})
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference job %s failed: %s", task_id, exc)
        await stream.publish({"event": "failed", "detail": str(exc)})
        raise


@router.get("/health", response_model=HealthResponse, summary="Health status")
async def health_check(settings: Annotated[Settings, Depends(get_settings_dependency)]) -> HealthResponse:
    """Return Hugging Face and model readiness information."""

    manager = get_model_manager()
    model_status = manager.status()

    status_value = "ok" if model_status.embedding_server_started or model_status.checkpoint_path else "degraded"
    message = (
        "Model ready."
        if model_status.checkpoint_path
        else "Model checkpoint has not been fetched yet; first run may take longer."
    )

    huggingface_status = _huggingface_status(settings)

    return HealthResponse(
        status=status_value,
        message=message,
        huggingface=huggingface_status,
        model=ModelStatusSchema(**model_status.__dict__),
    )


@router.post(
    "/datasets/preview",
    response_model=DatasetPreview,
    summary="Upload dataset and receive schema preview.",
)
async def preview_dataset(
    settings: Annotated[Settings, Depends(get_settings_dependency)],
    file: UploadFile = File(...),
) -> DatasetPreview:
    _validate_filesize(file, settings.max_upload_mb)
    df = _load_dataframe(file)
    return _preview_dataframe(df)


@router.get(
    "/datasets/examples",
    response_model=list[ExampleDataset],
    summary="List datasets available in the examples directory.",
)
async def list_example_datasets(
    settings: Annotated[Settings, Depends(get_settings_dependency)],
) -> list[ExampleDataset]:
    return _collect_examples(settings)


@router.get(
    "/datasets/examples/{example_id}/preview",
    response_model=DatasetPreview,
    summary="Preview an example dataset without uploading.",
)
async def preview_example_dataset(
    example_id: str,
    settings: Annotated[Settings, Depends(get_settings_dependency)],
) -> DatasetPreview:
    path = _resolve_example_path(example_id, settings)
    df = _load_example_dataframe(path)
    return _preview_dataframe(df)


@router.post(
    "/run",
    response_model=RunRequest,
    summary="Kick off a model run on uploaded data.",
)
async def run_dataset(
    settings: Annotated[Settings, Depends(get_settings_dependency)],
    target_column: Annotated[str, Form(...)],
    file: UploadFile = File(...),
    task: Annotated[str, Form()] = "classification",
    max_context_size: Annotated[int, Form()] = 1024,
    bagging: Annotated[int, Form()] = 2,
    test_size: Annotated[float, Form()] = 0.2,
    drop_constant_columns: Annotated[bool, Form()] = True,
) -> RunRequest:
    _validate_filesize(file, settings.max_upload_mb)
    df = _load_dataframe(file)
    source_name = file.filename

    params = RunParameters(
        task=task,  # type: ignore[arg-type]
        target_column=target_column,
        max_context_size=max_context_size,
        bagging=bagging,
        test_size=test_size,
        drop_constant_columns=drop_constant_columns,
    )

    return _schedule_run(df, params, settings, source_name)


@router.post(
    "/run/examples/{example_id}",
    response_model=RunRequest,
    summary="Kick off a model run using a dataset from the examples directory.",
)
async def run_example_dataset(
    example_id: str,
    params: RunParameters,
    settings: Annotated[Settings, Depends(get_settings_dependency)],
) -> RunRequest:
    path = _resolve_example_path(example_id, settings)
    df = _load_example_dataframe(path)
    return _schedule_run(df, params, settings, source_name=path.name)


@router.get(
    "/run/{task_id}",
    response_model=RunStatus,
    summary="Check status of a submitted run.",
)
async def get_run_status(task_id: str) -> RunStatus:
    result = JOB_REGISTRY.get_result(task_id)
    if result is None:
        stream = JOB_REGISTRY.get_stream(task_id)
        if not stream:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
        return RunStatus(task_id=task_id, state="running")
    return RunStatus(task_id=task_id, state="completed")


@router.get(
    "/run/{task_id}/result",
    response_model=RunResult,
    summary="Retrieve finished run results.",
)
async def get_run_result(task_id: str) -> RunResult:
    result = JOB_REGISTRY.get_result(task_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run result not ready.")
    return result


@router.get(
    "/results/{token}/download",
    summary="Download a generated prediction CSV by token.",
)
async def download_result_file(token: str) -> StreamingResponse:
    file_info = _resolve_result_file(token)
    headers = {"Content-Disposition": f'attachment; filename="{file_info.download_name}"'}
    return StreamingResponse(file_info.path.open("rb"), media_type="text/csv", headers=headers)


@router.websocket("/run/stream/{task_id}")
async def run_stream(task_id: str, websocket: WebSocket) -> None:
    stream = JOB_REGISTRY.get_stream(task_id)
    if not stream:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    try:
        async for payload in stream.stream():
            await websocket.send_json(payload)
            if payload.get("event") in {"completed", "failed"}:
                break
    except WebSocketDisconnect:
        logger.info("Client disconnected from task %s stream.", task_id)
    finally:
        JOB_REGISTRY.clear_stream(task_id)
        try:
            await websocket.close()
        except RuntimeError:
            logger.debug("WebSocket for task %s already closed before server close.", task_id)

