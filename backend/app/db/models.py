"""SQLAlchemy ORM models for users, jobs, model versions, and datasets."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base class for application tables."""


def new_uuid() -> str:
    """Generate a UUID4 string for primary keys."""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class User(Base):
    """Application user allowed to submit inference jobs."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    jobs: Mapped[list["Job"]] = relationship(back_populates="user")


class ModelVersion(Base):
    """Registered model artifact used for inference or evaluation."""

    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(120), index=True)
    checkpoint_path: Mapped[str] = mapped_column(Text)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Job(Base):
    """User-submitted cloud detection or removal job."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="uploaded")
    input_uri: Mapped[str] = mapped_column(Text)
    mask_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version_id: Mapped[str | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    user: Mapped[User | None] = relationship(back_populates="jobs")


class DownloadedDataset(Base):
    """Downloaded source dataset product tracked for reproducibility."""

    __tablename__ = "downloaded_datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    provider: Mapped[str] = mapped_column(String(80), index=True)
    collection_id: Mapped[str] = mapped_column(String(160), index=True)
    item_id: Mapped[str] = mapped_column(String(240), index=True)
    local_uri: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
