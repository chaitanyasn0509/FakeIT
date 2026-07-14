"""Backend API smoke tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_endpoint() -> None:
    """The API exposes a basic health endpoint."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
