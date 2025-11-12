"""Router package initialisation."""

from fastapi import APIRouter

from playground.backend.app.routes import base

api_router = APIRouter()
api_router.include_router(base.router, prefix="/api")

__all__ = ["api_router"]


