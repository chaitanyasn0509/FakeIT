"""Password hashing and JWT helpers for backend authentication."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import jwt


def hash_password(password: str) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a random salt."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"pbkdf2_sha256${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, encoded: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2 hash."""
    try:
        algorithm, salt_b64, digest_b64 = encoded.split("$", 2)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    salt = base64.b64decode(salt_b64)
    expected = base64.b64decode(digest_b64)
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return hmac.compare_digest(actual, expected)


def create_access_token(subject: str, secret_key: str, expires_minutes: int) -> str:
    """Create a signed JWT access token for a user subject."""
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload: dict[str, Any] = {"sub": subject, "exp": expires_at}
    return jwt.encode(payload, secret_key, algorithm="HS256")


def decode_access_token(token: str, secret_key: str) -> dict[str, Any]:
    """Decode and validate a JWT access token."""
    return jwt.decode(token, secret_key, algorithms=["HS256"])
