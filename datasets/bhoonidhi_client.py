"""Official Bhoonidhi API client for authentication, STAC search, and downloads."""

from __future__ import annotations

import logging
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import httpx
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class BhoonidhiError(RuntimeError):
    """Raised when the Bhoonidhi API returns an unrecoverable response."""


@dataclass(slots=True)
class BhoonidhiSettings:
    """Runtime settings for the Bhoonidhi API client."""

    base_url: str
    username: str
    password: str
    timeout_seconds: float = 60.0
    max_retries: int = 5
    max_concurrent_downloads: int = 2

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BhoonidhiSettings":
        """Create client settings from the project YAML configuration."""
        section = config.get("bhoonidhi", {})
        return cls(
            base_url=str(section.get("base_url", "https://bhoonidhi-api.nrsc.gov.in")),
            username=str(section.get("username", "")),
            password=str(section.get("password", "")),
            timeout_seconds=float(section.get("timeout_seconds", 60)),
            max_retries=int(section.get("max_retries", 5)),
            max_concurrent_downloads=int(section.get("max_concurrent_downloads", 2)),
        )


@dataclass(slots=True)
class TokenState:
    """In-memory OAuth token state for the current Bhoonidhi session."""

    access_token: str = ""
    refresh_token: str = ""
    expires_at: float = 0.0
    token_type: str = "Bearer"

    def is_valid(self, safety_margin_seconds: int = 60) -> bool:
        """Return true when the access token can be reused safely."""
        return bool(self.access_token) and time.time() < self.expires_at - safety_margin_seconds


class BhoonidhiClient:
    """Client for the official Bhoonidhi API documented by NRSC."""

    def __init__(self, settings: BhoonidhiSettings) -> None:
        """Initialize the HTTP client and token cache."""
        if not settings.username or not settings.password:
            raise BhoonidhiError("Bhoonidhi credentials must be provided through environment config.")
        self.settings = settings
        self.token = TokenState()
        self._client = httpx.Client(timeout=settings.timeout_seconds, follow_redirects=True)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "BhoonidhiClient":
        """Enter a context-managed client session."""
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Exit a context-managed client session and release connections."""
        self.close()

    def _url(self, path: str) -> str:
        """Build an absolute API URL from an endpoint path."""
        return f"{self.settings.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _auth_headers(self) -> dict[str, str]:
        """Return a bearer token header, refreshing the token when needed."""
        self.ensure_authenticated()
        return {"Authorization": f"{self.token.token_type} {self.token.access_token}"}

    def authenticate(self) -> TokenState:
        """Authenticate with userId/password and cache access and refresh tokens."""
        payload = {
            "userId": self.settings.username,
            "password": self.settings.password,
            "grant_type": "password",
        }
        response = self._request("POST", "/auth/token", json=payload, auth_required=False)
        self._update_token(response.json())
        return self.token

    def refresh_access_token(self) -> TokenState:
        """Refresh the access token using the current refresh token."""
        if not self.token.refresh_token:
            return self.authenticate()
        payload = {
            "userId": self.settings.username,
            "refresh_token": self.token.refresh_token,
            "grant_type": "refresh_token",
        }
        response = self._request("POST", "/auth/token", json=payload, auth_required=False)
        self._update_token(response.json())
        return self.token

    def logout(self) -> None:
        """Revoke the current Bhoonidhi refresh token if one is available."""
        if not self.token.refresh_token:
            return
        headers = {"Authorization": f"Bearer {self.token.refresh_token}"}
        self._request("POST", "/auth/logout", headers=headers, auth_required=False)
        self.token = TokenState()

    def ensure_authenticated(self) -> None:
        """Ensure the current access token exists and has not nearly expired."""
        if self.token.is_valid():
            return
        if self.token.refresh_token:
            self.refresh_access_token()
        else:
            self.authenticate()

    def _update_token(self, payload: dict[str, Any]) -> None:
        """Parse Bhoonidhi token response fields into TokenState."""
        try:
            expires_in = int(payload.get("expires_in", 0))
            self.token = TokenState(
                access_token=str(payload["access_token"]),
                refresh_token=str(payload.get("refresh_token", self.token.refresh_token)),
                expires_at=time.time() + expires_in,
                token_type=str(payload.get("token_type", "Bearer")),
            )
        except KeyError as exc:
            raise BhoonidhiError(f"Token response is missing required field: {exc}") from exc

    def _request(
        self,
        method: str,
        path: str,
        *,
        auth_required: bool = True,
        stream: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        """Send a request with Bhoonidhi-aware retries and token refresh."""
        headers = dict(kwargs.pop("headers", {}) or {})
        if auth_required:
            headers.update(self._auth_headers())
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries):
            try:
                response = self._client.request(
                    method,
                    self._url(path),
                    headers=headers,
                    **kwargs,
                )
                if response.status_code == 401 and auth_required:
                    self.refresh_access_token()
                    headers.update(self._auth_headers())
                    continue
                if response.status_code in {408, 412, 429, 500, 502, 503, 504}:
                    self._sleep_before_retry(response, attempt)
                    continue
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                self._sleep_before_retry(None, attempt)
        raise BhoonidhiError(f"Bhoonidhi request failed after retries: {method} {path}") from last_error

    def _sleep_before_retry(self, response: httpx.Response | None, attempt: int) -> None:
        """Sleep with exponential backoff and honor Retry-After when present."""
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after and retry_after.isdigit():
            delay = float(retry_after)
        else:
            delay = min(60.0, 2.0**attempt)
        LOGGER.warning("Bhoonidhi request throttled or unavailable; retrying in %.1fs", delay)
        time.sleep(delay)

    def list_collections(self) -> dict[str, Any]:
        """Return all STAC collections available to the authenticated user."""
        return self._request("GET", "/data/collections").json()

    def get_collection(self, collection_id: str) -> dict[str, Any]:
        """Return metadata for a single Bhoonidhi STAC collection."""
        path = f"/data/collections/{quote(collection_id, safe='')}"
        return self._request("GET", path).json()

    def list_items(self, collection_id: str, *, limit: int = 100) -> dict[str, Any]:
        """Return STAC items for a collection using Bhoonidhi's items endpoint."""
        path = f"/data/collections/{quote(collection_id, safe='')}/items"
        return self._request("GET", path, params={"limit": min(limit, 500)}).json()

    def get_item(self, collection_id: str, item_id: str) -> dict[str, Any]:
        """Return metadata for one STAC item within a collection."""
        collection = quote(collection_id, safe="")
        item = quote(item_id, safe="")
        return self._request("GET", f"/data/collections/{collection}/items/{item}").json()

    def search(self, query: dict[str, Any]) -> dict[str, Any]:
        """Search Bhoonidhi STAC data with POST body parameters."""
        bounded = dict(query)
        if "limit" in bounded:
            bounded["limit"] = min(int(bounded["limit"]), 500)
        return self._request("POST", "/data/search", json=bounded).json()

    def search_all(self, query: dict[str, Any], *, max_pages: int | None = None) -> list[dict[str, Any]]:
        """Collect all STAC features exposed through result links or paging context."""
        items: list[dict[str, Any]] = []
        page_query = dict(query)
        pages = 0
        while True:
            payload = self.search(page_query)
            features = payload.get("features", [])
            items.extend(features)
            pages += 1
            next_link = self._next_link(payload)
            if not next_link or (max_pages is not None and pages >= max_pages):
                break
            page_query = dict(query)
            page_query.update(next_link)
        return items

    def online_filter(self) -> dict[str, Any]:
        """Return a CQL2 filter for Bhoonidhi products with Online equal to Y."""
        return {"args": [{"property": "Online"}, "Y"], "op": "eq"}

    def download(
        self,
        *,
        item_id: str,
        collection_id: str,
        output_dir: str | Path,
        filename: str | None = None,
    ) -> Path:
        """Stream a Bhoonidhi product to disk using the official download endpoint."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        response = self._request(
            "GET",
            "/download",
            params={"id": item_id, "collection": collection_id},
        )
        resolved = output_path / (filename or self._resolve_download_name(response, item_id))
        total = int(response.headers.get("Content-Length", "0") or 0)
        with resolved.open("wb") as stream:
            with tqdm(total=total, unit="B", unit_scale=True, desc=resolved.name) as progress:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    stream.write(chunk)
                    progress.update(len(chunk))
        return resolved

    def download_many(
        self,
        items: Iterable[dict[str, Any]],
        output_dir: str | Path,
    ) -> list[Path]:
        """Download online STAC items sequentially while honoring Bhoonidhi throttling guidance."""
        downloaded: list[Path] = []
        for item in items:
            properties = item.get("properties", {})
            if properties.get("Online") != "Y":
                LOGGER.info("Skipping non-online item %s", item.get("id"))
                continue
            collection_id = item.get("collection") or item.get("collection_id")
            if not collection_id:
                LOGGER.warning("Skipping item without collection id: %s", item.get("id"))
                continue
            downloaded.append(
                self.download(
                    item_id=str(item["id"]),
                    collection_id=str(collection_id),
                    output_dir=output_dir,
                )
            )
        return downloaded

    def _resolve_download_name(self, response: httpx.Response, fallback_id: str) -> str:
        """Infer a filename from response headers and content type."""
        disposition = response.headers.get("Content-Disposition", "")
        if "filename=" in disposition:
            return disposition.split("filename=", 1)[1].strip('" ')
        extension = mimetypes.guess_extension(response.headers.get("Content-Type", "")) or ".zip"
        return f"{fallback_id}{extension}"

    def _next_link(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Extract Bhoonidhi/STAC paging parameters when a next link is present."""
        for link in payload.get("links", []):
            if link.get("rel") == "next":
                body = link.get("body")
                if isinstance(body, dict):
                    return body
        return None
