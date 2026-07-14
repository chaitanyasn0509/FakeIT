"""Local and S3-compatible file storage backends."""

from __future__ import annotations

import shutil
from urllib.parse import urlparse
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import uuid4

import boto3

from backend.app.core.config import Settings


class StorageBackend(Protocol):
    """Protocol implemented by storage adapters."""

    def save_upload(self, fileobj: BinaryIO, filename: str) -> str:
        """Persist an uploaded file and return its URI."""
        ...

    def save_file(self, source: str | Path, key_prefix: str) -> str:
        """Persist a local file and return its URI."""
        ...

    def resolve(self, uri: str) -> Path:
        """Resolve a URI to a local path when available."""
        ...


class LocalStorage:
    """Filesystem-backed storage for development and single-node deployments."""

    def __init__(self, root: str | Path) -> None:
        """Create local storage under a root directory."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_upload(self, fileobj: BinaryIO, filename: str) -> str:
        """Save an uploaded file stream into local storage."""
        safe_name = Path(filename).name
        destination = self.root / "uploads" / f"{uuid4()}_{safe_name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as output:
            shutil.copyfileobj(fileobj, output)
        return str(destination)

    def save_file(self, source: str | Path, key_prefix: str) -> str:
        """Copy a local file into storage and return its path URI."""
        src = Path(source)
        destination = self.root / key_prefix / src.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, destination)
        return str(destination)

    def resolve(self, uri: str) -> Path:
        """Resolve a local storage URI to a filesystem path."""
        return Path(uri)


class S3Storage:
    """S3-compatible storage adapter for object-store deployments."""

    def __init__(self, settings: Settings) -> None:
        """Create an S3 client from environment settings."""
        if not settings.s3_bucket:
            raise ValueError("S3_BUCKET must be configured for S3 storage.")
        self.bucket = settings.s3_bucket
        self.cache_root = Path(settings.local_storage_root) / "s3-cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name=settings.s3_region,
        )

    def save_upload(self, fileobj: BinaryIO, filename: str) -> str:
        """Upload a file stream to S3-compatible storage."""
        key = f"uploads/{uuid4()}_{Path(filename).name}"
        self.client.upload_fileobj(fileobj, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def save_file(self, source: str | Path, key_prefix: str) -> str:
        """Upload a local file to S3-compatible storage."""
        key = f"{key_prefix.rstrip('/')}/{Path(source).name}"
        self.client.upload_file(str(source), self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def resolve(self, uri: str) -> Path:
        """Download an S3 URI into a local cache and return its path."""
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ValueError(f"Unsupported S3 URI: {uri}")
        key = parsed.path.lstrip("/")
        destination = self.cache_root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            self.client.download_file(self.bucket, key, str(destination))
        return destination


def create_storage(settings: Settings) -> StorageBackend:
    """Create the configured storage backend."""
    if settings.storage_backend.lower() == "s3":
        return S3Storage(settings)
    return LocalStorage(settings.local_storage_root)
