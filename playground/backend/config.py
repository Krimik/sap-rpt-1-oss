"""
Configuration utilities for the sap-rpt-1-oss playground backend.

Loads environment variables, establishes logging defaults, and exposes
helpers that other modules can import without triggering side effects.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Settings:
    """Backend environment configuration."""

    huggingface_token: Optional[str]
    cache_dir: Path
    log_level: str
    zmq_port: int
    max_upload_mb: int
    examples_dir: Path


def _coerce_path(value: Optional[str], *, default: Path) -> Path:
    target = Path(value) if value else default
    target.mkdir(parents=True, exist_ok=True)
    return target


def _coerce_int(value: Optional[str], *, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        logging.getLogger(__name__).warning("Invalid integer value '%s', using default %s", value, default)
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings derived from environment variables."""

    huggingface_token = os.getenv("HUGGINGFACE_API_KEY")
    backend_root = Path(__file__).resolve().parent
    project_root = backend_root.parent.parent
    cache_dir = _coerce_path(os.getenv("PLAYGROUND_CACHE_DIR"), default=project_root / ".cache")
    examples_default = project_root / "example_datasets"
    examples_dir = _coerce_path(os.getenv("PLAYGROUND_EXAMPLES_DIR"), default=examples_default)
    log_level = os.getenv("PLAYGROUND_LOG_LEVEL", "INFO").upper()
    zmq_port = _coerce_int(os.getenv("PLAYGROUND_ZMQ_PORT"), default=5655)
    max_upload_mb = _coerce_int(os.getenv("PLAYGROUND_MAX_UPLOAD_MB"), default=50)

    return Settings(
        huggingface_token=huggingface_token,
        cache_dir=cache_dir,
        log_level=log_level,
        zmq_port=zmq_port,
        max_upload_mb=max_upload_mb,
        examples_dir=examples_dir,
    )


def configure_logging() -> None:
    """Configure root logging once."""

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


