"""Entrypoint for the sap-rpt-1-oss playground backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from playground.backend.app.routes import api_router


def create_app() -> FastAPI:
    """Construct the FastAPI application."""

    app = FastAPI(
        title="SAP RPT Playground Backend",
        version="0.1.0",
        summary="API surface for the sap-rpt-1-oss interactive playground.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    return app


app = create_app()


