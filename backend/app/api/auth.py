"""Authentication routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.core.security import create_access_token, hash_password, verify_password
from backend.app.db.models import User
from backend.app.db.session import get_db
from backend.app.schemas import TokenResponse, UserCreate

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
def register(
    payload: UserCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """Create a user and return an access token."""
    existing = db.scalar(select(User).where(User.email == payload.email))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    user = User(email=payload.email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.id, settings.secret_key, settings.access_token_expire_minutes)
    return TokenResponse(access_token=token)


@router.post("/token", response_model=TokenResponse)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """Authenticate a user and return a bearer token."""
    user = db.scalar(select(User).where(User.email == form.username))
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(user.id, settings.secret_key, settings.access_token_expire_minutes)
    return TokenResponse(access_token=token)
