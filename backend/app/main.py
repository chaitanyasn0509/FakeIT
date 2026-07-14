"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import auth, download, history, predict, upload
from backend.app.core.config import get_settings
from backend.app.db.session import init_db


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    settings = get_settings()
    app = FastAPI(
        title="UNCLOUD IT API",
        version="0.1.0",
        description="Cloud detection and generative cloud removal for LISS-IV satellite imagery.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth.router)
    app.include_router(upload.router)
    app.include_router(predict.router)
    app.include_router(download.router)
    app.include_router(history.router)

    @app.on_event("startup")
    def on_startup() -> None:
        """Initialize database tables when the app starts."""
        init_db()

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return health status for load balancers and Docker Compose."""
        return {"status": "ok"}

    return app


app = create_app()
