"""Database engine, sessions, and dependency wiring."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import get_settings
from backend.app.db.models import Base

settings = get_settings()
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create database tables for local development and container startup."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session for FastAPI route dependencies."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
